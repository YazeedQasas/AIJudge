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
    """Scores using a local LM Studio model (defaults to the app's served model)."""

    name = "local"

    def __init__(self, model: str | None = None):
        self.model = model or settings.lm_studio_model

    async def score(self, *, question: str, sources_block: str, answer: str) -> RubricScore:
        prompt = build_judge_prompt(question, sources_block, answer)
        # temperature 0 + fixed seed => the judge is as consistent as the backend allows.
        raw = await get_completion(prompt, {"temperature": 0, "seed": JUDGE_SEED}, model=self.model)
        return parse_rubric(raw)


# Registry: adding a judge is one line here (plus its class above).
JUDGES: dict[str, type[Judge]] = {
    "local": LocalLLMJudge,
}


def get_judge(name: str = "local", **kwargs) -> Judge:
    """Return a judge by name. Swapping models later = register a class and pass its name."""
    if name not in JUDGES:
        raise ValueError(f"Unknown judge {name!r}. Available: {', '.join(JUDGES)}")
    return JUDGES[name](**kwargs)
