"""Scorers: turn one judge answer into pass/fail signals.

These are PURE functions — no network, no model. They take a case (its expectation)
and a result (what the pipeline produced) and return True/False. That purity is why
we can unit-test them offline, before any real generation happens.

Two kinds of scorer:
  - "quality"  metrics: higher pass-rate is unambiguously better for any version.
  - "behavior" probes:  descriptive — they characterize what a version DOES (e.g. does
                        it emit v2's ruling line), not whether it's better or worse.
"""

import re
from dataclasses import dataclass, field
from typing import Callable

CITATION_RE = re.compile(r"\[(\d+)\]")
RULING_MARKER = "الحكم المقترح:"


@dataclass
class CaseResult:
    """What the pipeline produced for one case."""

    answer: str
    sources: list[dict]
    invalid_citations: list[int]

    @property
    def refused(self) -> bool:
        """A refusal returns no sources (matches the backend's refusal path)."""
        return len(self.sources) == 0


# --- Quality metrics (higher pass-rate is better) ---------------------------------

def refusal_correct(case: dict, result: CaseResult) -> bool:
    """Did the judge refuse exactly when it should have (and answer when it should)?"""
    return result.refused == case["should_refuse"]


def citations_valid(case: dict, result: CaseResult) -> bool:
    """No citations to sources that don't exist. Trivially true for a refusal."""
    return len(result.invalid_citations) == 0


def has_citation(case: dict, result: CaseResult) -> bool:
    """When answering, the answer grounds itself in at least one [n] citation.

    N/A for cases meant to be refused — we don't penalize a (correct) refusal for
    not citing anything, so it passes.
    """
    if case["should_refuse"]:
        return True
    return bool(CITATION_RE.search(result.answer))


# --- Behavior probes (descriptive, not better/worse) ------------------------------

def ruling_line_present(case: dict, result: CaseResult) -> bool:
    """Does the answer include a labelled 'الحكم المقترح:' line? (v2's designed behavior.)"""
    if case["should_refuse"]:
        return True  # N/A for refusals
    return RULING_MARKER in result.answer


def _answered_only(case: dict) -> bool:
    """A scorer that only makes sense for cases the judge is meant to answer."""
    return not case["should_refuse"]


@dataclass
class Scorer:
    name: str
    kind: str  # "quality" | "behavior"
    description: str
    fn: Callable[[dict, CaseResult], bool]
    # Which cases this scorer counts against. Scorers that are N/A for refusals only
    # apply to answered cases, so the scorecard denominator stays honest (3, not 5).
    applies: Callable[[dict], bool] = field(default=lambda case: True)


SCORERS: list[Scorer] = [
    Scorer("refusal_correct", "quality", "Refuses iff the question is out-of-corpus", refusal_correct),
    Scorer("citations_valid", "quality", "No citations to nonexistent sources", citations_valid),
    Scorer("has_citation", "quality", "Answered cases cite at least one source", has_citation, _answered_only),
    Scorer("ruling_line_present", "behavior", "Answer emits a labelled ruling line", ruling_line_present, _answered_only),
]
