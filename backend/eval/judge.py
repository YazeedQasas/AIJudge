"""LLM-as-a-judge: score a judge answer on a rubric, with a swappable model backend.

The plug-and-play seam is the `Judge` abstract base. The eval harness depends only on
`Judge.score(...)`; it never names a concrete model. Swapping to a different judge model
(e.g. Claude) is: add a Judge subclass, register it in JUDGES, select it by name. No
change to the harness or the rubric.

The rubric (DIMENSIONS) lives here, once, so every judge scores on the same criteria.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.config import settings
from app.llm_client import get_completion

# What "good" means, shared by every judge. Edit here to change the yardstick everywhere.
DIMENSIONS: dict[str, str] = {
    "faithfulness": "Every claim is supported by the cited sources; no invented or outside law.",
    "relevance": "Directly addresses the question that was asked.",
    "completeness": "Uses the relevant sources and covers the key legal points.",
    "clarity": "The ruling is clear, well-structured, and actionable.",
}

# Fixed so the judge itself is reproducible run-to-run.
JUDGE_SEED = 7

_JUDGE_PROMPT = """You are a strict evaluator of an AI legal judge's answers. You are given a
QUESTION, the numbered SOURCES the AI judge was given, and its ANSWER.

Score the ANSWER on each criterion from 1 (poor) to 5 (excellent):
{criteria}

Evaluate ONLY against the given sources; do not use outside knowledge. Be critical:
reserve 5 for genuinely excellent answers.

Respond with a SINGLE JSON object and nothing else, exactly in this shape:
{{"faithfulness": <1-5>, "relevance": <1-5>, "completeness": <1-5>, "clarity": <1-5>, "reasoning": "<one or two sentences>"}}

QUESTION:
{question}

SOURCES:
{sources_block}

ANSWER:
{answer}
"""


@dataclass
class RubricScore:
    """One judge's scores for one answer."""

    scores: dict[str, int]  # dimension -> 1..5
    reasoning: str

    @property
    def average(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0


def build_judge_prompt(question: str, sources_block: str, answer: str) -> str:
    criteria = "\n".join(f"- {name}: {desc}" for name, desc in DIMENSIONS.items())
    return _JUDGE_PROMPT.format(
        criteria=criteria, question=question, sources_block=sources_block, answer=answer
    )


def parse_rubric(raw: str) -> RubricScore:
    """Pull the JSON rubric out of the judge's raw reply, tolerating surrounding text."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        raise ValueError(f"No JSON object found in judge reply: {raw[:200]!r}")
    data = json.loads(match.group(0))
    scores = {}
    for dim in DIMENSIONS:
        if dim not in data:
            raise ValueError(f"Judge reply missing dimension '{dim}': {data}")
        scores[dim] = max(1, min(5, int(data[dim])))  # clamp to 1..5
    return RubricScore(scores=scores, reasoning=str(data.get("reasoning", "")))


class Judge(ABC):
    """A model that scores an answer on the shared rubric. The plug-and-play seam."""

    name: str

    @abstractmethod
    async def score(self, *, question: str, sources_block: str, answer: str) -> RubricScore:
        ...


class LocalLLMJudge(Judge):
    """Scores using a local LM Studio model (defaults to the app's served model).

    CAVEAT, and it is not a small one: by default this is the same model being judged.
    A model shares its own blind spots - it cannot penalise a fabrication it would also
    have made, and it tends to rate its own phrasing highly. Useful as a cheap,
    offline, reproducible signal; not useful as the deciding evidence in a comparison
    where one of the candidates is a fine-tune of the judge. That is what ClaudeJudge
    is for.
    """

    name = "local"

    def __init__(self, model: str | None = None):
        self.model = model or settings.lm_studio_model

    async def score(self, *, question: str, sources_block: str, answer: str) -> RubricScore:
        prompt = build_judge_prompt(question, sources_block, answer)
        # temperature 0 + fixed seed => the judge is as consistent as the backend allows.
        raw = await get_completion(prompt, {"temperature": 0, "seed": JUDGE_SEED}, model=self.model)
        return parse_rubric(raw)


def _rubric_schema() -> dict:
    """JSON schema for the rubric, derived from DIMENSIONS so it can't drift from it.

    Uses enum rather than minimum/maximum: numeric range constraints aren't part of the
    supported schema subset for structured outputs, whereas an explicit enum is.
    """
    return {
        "type": "object",
        "properties": {
            **{dim: {"type": "integer", "enum": [1, 2, 3, 4, 5]} for dim in DIMENSIONS},
            "reasoning": {"type": "string"},
        },
        "required": [*DIMENSIONS, "reasoning"],
        "additionalProperties": False,
    }


class ClaudeJudge(Judge):
    """Scores using Claude via the Anthropic API - an INDEPENDENT judge.

    Independence is the entire point. When the comparison is "base model vs. a LoRA of
    that same base model", a local judge from the same family is scoring two variants
    of itself, and its agreement with either tells you less than it appears to.

    Setup: pip install -r requirements-data.txt, then either export ANTHROPIC_API_KEY
    or run `ant auth login` (the SDK finds a stored profile with no code change).

    Three API details worth knowing, because two of them are traps:

    1. NO temperature / top_p / top_k. Those parameters were removed on this model
       generation and sending one returns a 400. LocalLLMJudge above passes
       temperature 0 and a seed - correct for LM Studio, fatal here. Determinism comes
       from the prompt and the schema instead, so a Claude-judged run is less bit-wise
       reproducible than a local one. That is the trade for an unbiased judge; note it
       when comparing scorecards produced by different judges.

    2. Check stop_reason before reading content. A refusal returns HTTP 200 with an
       empty content list, so indexing content[0] blindly raises an unrelated
       IndexError several frames from the real cause.

    3. No fallback model, deliberately. The API can re-run a refused request on a
       different model automatically, which is usually what you want and is exactly
       wrong here: a judge is a measuring instrument, and silently swapping the
       instrument partway through a run makes the resulting column incomparable.
       Better to fail loudly on the case and leave the rest of the scorecard honest.
    """

    name = "claude"

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 2048):
        # Imported lazily so the local judge keeps working without the extra dependency.
        try:
            from anthropic import AsyncAnthropic
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The 'claude' judge needs the anthropic SDK, which is deliberately not a "
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

    async def score(self, *, question: str, sources_block: str, answer: str) -> RubricScore:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": _rubric_schema()}},
            messages=[
                {"role": "user", "content": build_judge_prompt(question, sources_block, answer)}
            ],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude judge declined to score: {response.stop_details}")
        text = next((b.text for b in response.content if b.type == "text"), "")
        # Still routed through the shared parser: the schema guarantees the shape, and
        # parse_rubric keeps the clamping and the missing-dimension check in one place
        # for every judge.
        return parse_rubric(text)


# Registry: adding a judge is one line here (plus its class above).
JUDGES: dict[str, type[Judge]] = {
    "local": LocalLLMJudge,
    "claude": ClaudeJudge,
}


def get_judge(name: str = "local", **kwargs) -> Judge:
    """Return a judge by name. Swapping models later = register a class and pass its name."""
    if name not in JUDGES:
        raise ValueError(f"Unknown judge {name!r}. Available: {', '.join(JUDGES)}")
    return JUDGES[name](**kwargs)
