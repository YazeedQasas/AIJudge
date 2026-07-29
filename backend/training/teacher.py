"""Teachers: draft or repair a training answer, with a swappable model backend.

Mirrors eval/judge.py's registry idiom: a Teacher ABC, a TEACHERS dict, get_teacher().
Swapping the drafting model is: add a subclass, register it below, select it by name.
No change to build_dataset.py, which will depend only on Teacher.draft(...).

Two drafting modes (full rules in training/teacher_style.md):
  write - draft the answer from scratch, given only the question and sources.
  edit  - (the default, and the better one) minimally repair a base model's own
          answer from a Stage A eval run record. Keeps the training target close to
          the student's own distribution: easier to learn, less catastrophic
          forgetting, and a direct defense against the LoRA learning the teacher's
          Arabic voice instead of the student's own.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from app.config import settings
from app.llm_client import get_completion

STYLE_PATH = Path(__file__).parent / "teacher_style.md"


def load_style() -> str:
    """The teacher's style guide, read fresh each call so edits while developing
    don't need a process restart."""
    return STYLE_PATH.read_text(encoding="utf-8")


_WRITE_TEMPLATE = """Draft the answer to this question, using only the numbered sources given.

المصادر:
{sources_block}

السؤال: {question}

الإجابة:
"""

_EDIT_TEMPLATE = """Below is a base model's draft answer to the question, using the same \
numbered sources. Repair it minimally per the edit-mode rules; do not rewrite it from scratch.

المصادر:
{sources_block}

السؤال: {question}

مسودة النموذج الأساسي:
{base_answer}

الإجابة المصححة:
"""


def build_task_prompt(
    *, mode: str, question: str, sources_block: str, base_answer: str | None
) -> str:
    """The task-specific half of the prompt - style rules are composed in separately
    by each Teacher (system message for Claude, prepended text for local LM Studio),
    so this function stays agnostic to which backend is asking."""
    if mode == "write":
        return _WRITE_TEMPLATE.format(sources_block=sources_block, question=question)
    if mode == "edit":
        if base_answer is None:
            raise ValueError("edit mode needs base_answer - pass the Stage A run record's answer")
        return _EDIT_TEMPLATE.format(
            sources_block=sources_block, question=question, base_answer=base_answer
        )
    raise ValueError(f"Unknown teacher mode {mode!r}. Use 'write' or 'edit'.")


class Teacher(ABC):
    """A model that drafts or repairs a training answer. The plug-and-play seam."""

    name: str

    @abstractmethod
    async def draft(
        self,
        *,
        question: str,
        sources_block: str,
        mode: str = "edit",
        base_answer: str | None = None,
    ) -> str:
        ...


class LocalLLMTeacher(Teacher):
    """Drafts using a local LM Studio model (defaults to the app's served model).

    CAVEAT: if this points at the same model family being fine-tuned, it teaches the
    student its own quirks - near-zero signal lift (see the plan's teacher-choice
    table). Useful as a free, offline path for smoke-testing build_dataset.py before
    spending Claude tokens; ClaudeTeacher (next) is what the real dataset uses.
    """

    name = "local"

    def __init__(self, model: str | None = None):
        self.model = model or settings.lm_studio_model

    async def draft(
        self,
        *,
        question: str,
        sources_block: str,
        mode: str = "edit",
        base_answer: str | None = None,
    ) -> str:
        task = build_task_prompt(
            mode=mode, question=question, sources_block=sources_block, base_answer=base_answer
        )
        # No system role over LM Studio's /chat/completions as this codebase calls it
        # (see app/llm_client.get_completion) - the style guide has to ride in the
        # same user turn as the task.
        prompt = f"{load_style()}\n\n{task}"
        return await get_completion(prompt, {"temperature": 0.3}, model=self.model)


class ClaudeTeacher(Teacher):
    """Drafts using Claude via the Anthropic API - the real teacher for this project.

    Distilling a much stronger teacher into a 4B student on style, format, and
    calibration is the well-founded case for this (distilling *reasoning* would not
    be - which is exactly why knowledge injection was scoped out). See the plan's
    teacher-choice table for why local-on-local was rejected: same family teaches
    the student its own quirks, near-zero signal lift.

    Setup: pip install -r requirements-data.txt, then either export ANTHROPIC_API_KEY
    or run `ant auth login`.

    Same two traps as eval.judge.ClaudeJudge, for the same reason - both hit the same
    API:

    1. NO temperature / top_p / top_k. Removed on this model generation; sending one
       returns a 400. LocalLLMTeacher above passes temperature - correct for LM
       Studio, fatal here.
    2. Check stop_reason before reading content. A refusal returns HTTP 200 with an
       empty content list - indexing content[0] blindly raises an unrelated
       IndexError several frames from the real cause.

    Unlike LocalLLMTeacher, the style guide rides in a real `system` message here
    rather than being prepended to the task prompt - the Anthropic API supports it
    natively, LM Studio's /chat/completions as this codebase calls it doesn't.
    """

    name = "claude"

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 4096):
        # Imported lazily so the local teacher keeps working without the extra dependency.
        try:
            from anthropic import AsyncAnthropic
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The 'claude' teacher needs the anthropic SDK, which is deliberately not a "
                "runtime dependency. Install it with:\n"
                "    pip install -r requirements-data.txt\n"
                "then authenticate with either `export ANTHROPIC_API_KEY=...` or `ant auth login`."
            ) from exc

        self.model = model
        self.max_tokens = max_tokens
        # No arguments: resolves ANTHROPIC_API_KEY, then ANTHROPIC_AUTH_TOKEN, then an
        # `ant auth login` profile. Never hardcode a key, and don't put one in
        # app/config.py - that is runtime config, and this is an offline tool.
        self._client = AsyncAnthropic()

    async def draft(
        self,
        *,
        question: str,
        sources_block: str,
        mode: str = "edit",
        base_answer: str | None = None,
    ) -> str:
        task = build_task_prompt(
            mode=mode, question=question, sources_block=sources_block, base_answer=base_answer
        )
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=load_style(),
            messages=[{"role": "user", "content": task}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude teacher declined to draft: {response.stop_details}")
        return next((b.text for b in response.content if b.type == "text"), "")


# Registry: adding a teacher is one line here (plus its class above).
TEACHERS: dict[str, type[Teacher]] = {
    "local": LocalLLMTeacher,
    "claude": ClaudeTeacher,
}


def get_teacher(name: str = "claude", **kwargs) -> Teacher:
    """Return a teacher by name. Default is 'claude' - the real dataset's drafting
    model; pass name='local' for a free offline smoke test of the surrounding
    pipeline (build_dataset.py etc.) without spending API credits."""
    if name not in TEACHERS:
        raise ValueError(f"Unknown teacher {name!r}. Available: {', '.join(TEACHERS)}")
    return TEACHERS[name](**kwargs)
