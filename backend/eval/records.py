"""Persist what an eval run actually produced, one JSON object per case, per variant.

Until now the eval printed a scorecard and threw everything else away. That is fine
while the only question is "which of these two prompts is better right now", and
becomes a problem the moment you want to answer any of these:

  - "What did the model ACTUALLY say on the case that failed?" Rates tell you a case
    failed; only the answer tells you why.
  - "Is this run better than last week's?" A number with nothing to compare it to
    isn't a baseline, it's a screenshot.
  - "Did that scorer heuristic actually work?" DECLINE_MARKERS in scorers.py is a
    guess about Arabic phrasing. The records are the evidence that tunes it.
  - "What do we train on?" A record holds the question, the exact sources block the
    model saw, and the answer it gave. That triple is most of a training example -
    the dataset builder reads these rather than re-running retrieval.

The file format is JSONL (one JSON object per line) rather than one big JSON array,
so a run can append as it goes and an interrupted run still leaves valid, readable
data behind.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class RunRecord:
    """One case, run through one variant, with everything needed to explain the score."""

    # --- which run, and what produced this ---
    run_id: str
    timestamp: str
    variant: str  # "v2@base"
    prompt_version: str
    model_id: str
    model: str  # the resolved identifier that went out on the wire
    generation: dict[str, Any]  # exact sampling params sent, seed included
    seed: int

    # --- what was asked ---
    case_id: str
    question: str
    expected: str  # "answered" | "declined" | "refused"

    # --- what the pipeline retrieved ---
    # Trimmed source metadata, not raw chunks: the text is already in sources_block,
    # and duplicating it would multiply the file size for no extra information.
    sources: list[dict[str, Any]]
    # THE EXACT CONTEXT THE MODEL SAW. Load-bearing for two reasons: it is what makes
    # a failure reproducible without re-running retrieval, and it is what the training
    # dataset needs so a training example carries byte-identical context to inference.
    sources_block: str

    # --- what came back ---
    answer: str
    invalid_citations: list[int]
    refused: bool
    elapsed_seconds: float

    # --- how it scored ---
    scorers: dict[str, bool] = field(default_factory=dict)
    rubric: dict[str, Any] | None = None


def new_run_id() -> str:
    """Timestamp-based id, sortable as a string: '20260801-143022'."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_path(run_id: str, label: str = "run") -> Path:
    """Where a run's records live, e.g. results/baseline-20260801-143022.jsonl."""
    return RESULTS_DIR / f"{label}-{run_id}.jsonl"


def write_records(path: Path, records: list[RunRecord]) -> Path:
    """Append records to a JSONL file, creating the directory if needed.

    Append rather than overwrite so a run can flush after each variant and survive
    being interrupted halfway through a long comparison.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            # ensure_ascii=False keeps the Arabic readable when you open the file.
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return path


def load_records(path: Path) -> list[RunRecord]:
    """Read a JSONL file back into RunRecords."""
    with path.open(encoding="utf-8") as handle:
        return [RunRecord(**json.loads(line)) for line in handle if line.strip()]


def latest_run() -> Path | None:
    """The most recent results file, or None if there aren't any yet."""
    files = sorted(RESULTS_DIR.glob("*.jsonl"))
    return files[-1] if files else None
