"""The contamination guard: training questions must never overlap eval questions.

eval/cases.yaml is the project's permanent yardstick (see its own header comment).
Training on a question that's identical, or nearly identical, to an eval case would
make a later scorecard measure memorization of the test, not improvement -
silently, because the model would score *better* for exactly the wrong reason.
Checked here in code rather than left to careful reading: at ~380 training
questions against 59 eval cases, manual review doesn't scale and a missed overlap
is invisible until someone notices a suspiciously perfect score.
"""

import re
import unicodedata
from dataclasses import dataclass

from eval.run_eval import load_cases
from eval.scorers import normalize_arabic

# Above this trigram-overlap ratio, two questions are similar enough to warrant a
# human look even though they're not byte-identical after normalization. Not
# empirically tuned yet - tighten with evidence from real overlaps observed, same
# spirit as eval/scorers.py's DECLINE_MARKERS.
TRIGRAM_WARN_THRESHOLD = 0.6

_WHITESPACE_RE = re.compile(r"\s+")
_HAMZA_RE = re.compile(r"[ؤئ]")  # hamza-on-carrier -> bare hamza
_TA_MARBUTA_RE = re.compile(r"ة")  # ta marbuta -> ha, a common free-variation pair


def normalize_for_contamination(text: str) -> str:
    """Collapse the free variation contamination-checking needs to see through:
    diacritics/tatweel/alef forms (via eval.scorers.normalize_arabic), plus hamza
    carriers, ta-marbuta, and whitespace - none of which should be enough
    difference to call two questions distinct."""
    text = normalize_arabic(text)
    text = _HAMZA_RE.sub("ء", text)
    text = _TA_MARBUTA_RE.sub("ه", text)
    text = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _trigrams(text: str) -> set[str]:
    return {text[i : i + 3] for i in range(len(text) - 2)} if len(text) >= 3 else {text}


def trigram_overlap(a: str, b: str) -> float:
    """Jaccard similarity of character trigrams - a cheap, language-agnostic
    near-duplicate signal that needs no tokenizer and no embedding call."""
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class ContaminationHit:
    """One training question that matches or nearly matches an eval case."""

    question_id: str
    question: str
    eval_case_id: str
    eval_question: str
    exact: bool  # exact -> hard error for the caller; not exact -> warn only
    overlap: float


def check_contamination(questions: dict[str, str]) -> list[ContaminationHit]:
    """questions: {id: question text}. Compares every one against every eval case.

    Returns exact matches (the caller should hard-error on these - see B3 in the
    plan) and high-trigram-overlap near-matches (the caller should warn on these
    and let a human judge whether they're a real problem).
    """
    eval_cases = load_cases()
    eval_norm = [
        (c["id"], c["question"], normalize_for_contamination(c["question"])) for c in eval_cases
    ]

    hits: list[ContaminationHit] = []
    for qid, question in questions.items():
        norm = normalize_for_contamination(question)
        for case_id, eval_question, eval_norm_q in eval_norm:
            if norm == eval_norm_q:
                hits.append(ContaminationHit(qid, question, case_id, eval_question, True, 1.0))
                continue
            overlap = trigram_overlap(norm, eval_norm_q)
            if overlap >= TRIGRAM_WARN_THRESHOLD:
                hits.append(ContaminationHit(qid, question, case_id, eval_question, False, overlap))
    return hits
