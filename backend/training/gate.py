"""The auto-gate: does a drafted example pass the same rules a model answer would?

Reuses eval/scorers.py wholesale rather than a parallel copy of citation/ruling-line
logic - the rules that grade a model's answer at inference time are exactly the
rules a *training target* has to satisfy, or the LoRA learns to imitate whatever
slipped through. One accepted example with an invalid citation, or a missing
ruling line, teaches the opposite of the goal.
"""

from app.generation import find_invalid_citations
from eval.scorers import SCORERS, CaseResult
from training.schema import Example

# How a training `kind` maps onto the eval's three-way `expected`, for scoring
# purposes only. Training's kind vocabulary is finer (it also encodes *why* an
# answer should look that way), but the scorers only need to know its shape:
#   answered   -> answered  : cites sources, has a ruling line
#   partial    -> answered  : SAME shape - cites what's supported, still rules
#                             (plan B6 construction #4: partial coverage is not a
#                             decline, it's a smaller answered case)
#   declined   -> declined  : no citations, no ruling line
#   mismatched -> declined  : same shape as declined (sources present, irrelevant)
_EXPECTED_FOR_KIND = {
    "answered": "answered",
    "partial": "answered",
    "declined": "declined",
    "mismatched": "declined",
}


def run_auto_gate(example: Example) -> dict[str, bool]:
    """Run every applicable eval scorer against a drafted example's answer."""
    expected = _EXPECTED_FOR_KIND.get(example.kind, "answered")
    case = {"expected": expected}
    result = CaseResult(
        answer=example.answer,
        sources=example.sources,
        invalid_citations=find_invalid_citations(example.answer, len(example.sources)),
    )
    return {s.name: s.fn(case, result) for s in SCORERS if s.applies(case)}


def passes_gate(checks: dict[str, bool]) -> bool:
    """An example is usable only if every check that applied to it passed - one
    False (e.g. a citation to a source that doesn't exist) rejects the record."""
    return all(checks.values())
