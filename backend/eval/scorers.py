"""Scorers: turn one judge answer into pass/fail signals.

These are PURE functions - no network, no model. They take a case (its expectation)
and a result (what the pipeline produced) and return True/False. That purity is why
we can unit-test them offline, before any real generation happens.

Two kinds of scorer:
  - "quality"  metrics: higher pass-rate is unambiguously better for any variant.
  - "behavior" probes:  descriptive - they characterize what a variant DOES (e.g. does
                        it emit v2's ruling line), not whether it's better or worse.

WHY THERE ARE THREE EXPECTATIONS, NOT A BOOLEAN
-----------------------------------------------
A case's `expected` is one of "answered" / "declined" / "refused", because two very
different things were previously both called "refusing":

  refused  - the RETRIEVAL GATE fired. Top chunk scored below min_relevance_score, so
             api/ask.py returned a canned string and no model ever ran. This is code
             behaviour. It is identical across every prompt and every model, and no
             amount of fine-tuning can change it.

  declined - the gate PASSED (sources came back above threshold) but they don't
             actually cover the question, and the right answer is to say so. This is
             model behaviour, and it is the hard part: the model has plausible-looking
             context in front of it and has to not use it.

Collapsing those into one boolean made the second invisible. A weak-but-passing case
had should_refuse: false, so has_citation and ruling_line_present both *penalised* a
correct decline - the eval actively rewarded answering from sources that didn't
support an answer. Any work aimed at calibration was unmeasurable by construction.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

CITATION_RE = re.compile(r"\[(\d+)\]")
RULING_MARKER = "الحكم المقترح:"

# Arabic diacritics (tashkeel) and tatweel. Models emit these inconsistently: the same
# word appears as "تتضمّن" or "تتضمن" depending on the sample. Matching raw substrings
# against undiacritised markers silently fails on the diacritised half of the output,
# so normalise both sides before comparing.
_TASHKEEL_RE = re.compile(r"[ً-ْٰـ]")
_ALEF_RE = re.compile(r"[آأإ]")  # آ أ إ -> ا


def normalize_arabic(text: str) -> str:
    """Strip diacritics/tatweel and unify alef forms, for robust substring matching."""
    text = unicodedata.normalize("NFC", text)
    text = _TASHKEEL_RE.sub("", text)
    return _ALEF_RE.sub("ا", text)


# Phrases that mark "the sources don't support an answer". Drawn from the wording the
# prompt templates ask for and the canned string api/ask.py returns on a gate refusal.
#
# HEURISTIC, and known to be one. Validate it against real output: after a baseline
# run, grep eval/results/*.jsonl for answers where decline_correct failed and see
# whether the model phrased a decline some way this list misses. Tighten it there,
# where there's evidence, rather than guessing more phrases now.
DECLINE_MARKERS = [normalize_arabic(m) for m in (
    "لا تتضمّن",     # "[the sources] do not include"
    "لا تتوفّر",      # "not available"
    "لا تكفي",       # "not sufficient"
    "غير كافية",     # "insufficient"
    "لا تغطّي",      # "do not cover"
    "لا تتناول",     # "do not address"
)]


@dataclass
class CaseResult:
    """What the pipeline produced for one case."""

    answer: str
    sources: list[dict]
    invalid_citations: list[int]

    @property
    def refused(self) -> bool:
        """True when the retrieval gate fired: no sources, so no generation happened."""
        return len(self.sources) == 0

    @property
    def declined(self) -> bool:
        """True when the model, given sources, said they don't support an answer."""
        return any(m in normalize_arabic(self.answer) for m in DECLINE_MARKERS)

    @property
    def citations(self) -> list[int]:
        return [int(n) for n in CITATION_RE.findall(self.answer)]


def expectation(case: dict) -> str:
    """This case's expected outcome: "answered" | "declined" | "refused".

    Falls back to the older boolean field so a hand-written case using should_refuse
    still loads. New cases should use `expected` - it's the only way to express
    "declined", which is the interesting third state.
    """
    if "expected" in case:
        return case["expected"]
    return "refused" if case["should_refuse"] else "answered"


# --- Quality metrics (higher pass-rate is better) ---------------------------------

def gate_correct(case: dict, result: CaseResult) -> bool:
    """Did the RETRIEVAL GATE fire exactly on the out-of-corpus cases?

    This scores app/api/ask.py's min_relevance_score threshold, not the model. It is
    expected to be identical across every variant - if it ever differs between two
    models, something is wrong with the harness, not with the models. Keep it in the
    scorecard precisely as that regression check.
    """
    return result.refused == (expectation(case) == "refused")


def decline_correct(case: dict, result: CaseResult) -> bool:
    """Given sources, did the model decline exactly when they don't support an answer?

    This is the calibration metric. It is symmetric on purpose: it fails both a model
    that answers anyway from weak sources AND a model that declines when the sources
    were fine. A one-sided "did it decline?" check would be trivially satisfied by a
    model that declines everything.
    """
    return result.declined == (expectation(case) == "declined")


def citations_valid(case: dict, result: CaseResult) -> bool:
    """No citations to sources that don't exist. Trivially true for a refusal."""
    return len(result.invalid_citations) == 0


def has_citation(case: dict, result: CaseResult) -> bool:
    """When answering, the answer grounds itself in at least one [n] citation."""
    return bool(CITATION_RE.search(result.answer))


def no_citation_when_declining(case: dict, result: CaseResult) -> bool:
    """A decline cites nothing - there is no supporting source to point at.

    Citing while declining is a specific, common failure: the model hedges ("the
    sources [1] do not address...") and half-grounds a non-answer.
    """
    return not result.citations


# --- Behavior probes (descriptive, not better/worse) ------------------------------

def ruling_line_present(case: dict, result: CaseResult) -> bool:
    """Does the answer include a labelled 'الحكم المقترح:' line? (v2's designed behavior.)"""
    return RULING_MARKER in result.answer


# --- Which cases each scorer counts against ---------------------------------------
#
# The denominator has to stay honest. A scorer that is meaningless for a refusal must
# not silently pass on it - that inflates the rate and hides the cases it does cover.

def _gate_passed(case: dict) -> bool:
    """Cases where a model actually ran, so model behaviour is on the hook."""
    return expectation(case) in ("answered", "declined")


def _answered_only(case: dict) -> bool:
    return expectation(case) == "answered"


def _declined_only(case: dict) -> bool:
    return expectation(case) == "declined"


@dataclass
class Scorer:
    name: str
    kind: str  # "quality" | "behavior"
    description: str
    fn: Callable[[dict, CaseResult], bool]
    # Which cases this scorer counts against. Scorers that are N/A for some outcomes
    # only apply where they mean something, so the denominator reflects reality.
    applies: Callable[[dict], bool] = field(default=lambda case: True)


SCORERS: list[Scorer] = [
    Scorer("gate_correct", "quality",
           "Retrieval gate fires iff the question is out-of-corpus", gate_correct),
    Scorer("decline_correct", "quality",
           "Declines iff the retrieved sources don't support an answer",
           decline_correct, _gate_passed),
    Scorer("citations_valid", "quality",
           "No citations to nonexistent sources", citations_valid),
    Scorer("has_citation", "quality",
           "Answered cases cite at least one source", has_citation, _answered_only),
    Scorer("no_citation_when_declining", "quality",
           "Declines cite nothing", no_citation_when_declining, _declined_only),
    Scorer("ruling_line_present", "behavior",
           "Answer emits a labelled ruling line", ruling_line_present, _answered_only),
]
