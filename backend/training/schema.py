"""The shape of one training example, and JSONL helpers to read/write them.

Mirrors eval/records.py's RunRecord: a stdlib dataclass, not a dict, so every
downstream script (teacher.py, build_dataset.py, review.py, split.py) imports one
authoritative shape instead of five scripts agreeing on dict keys by convention.
Defined before any of those exist - renaming a field later is a one-line change
here, not a grep across the whole subsystem.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"

# The four negative constructions from the plan (B6), plus the positive case.
# Not enforced as a Literal/Enum here - kept a plain str so seed_questions.yaml
# and the dataset builder can introduce values without touching this file.
#   answered   - corpus covers it; cite sources, emit the ruling line
#   declined   - weak-but-passing sources; say so, cite nothing, no ruling line
#   mismatched - sources exist but are from the wrong document (deliberate)
#   partial    - some chunks on-topic, some not; answer only what's supported
#   adversarial / multi-doc - format traps and multi-source synthesis
KINDS = ("answered", "declined", "mismatched", "partial", "adversarial", "multi-doc")


@dataclass
class Example:
    """One training example: a question, the context a teacher saw, and its answer."""

    # --- identity and provenance ---
    id: str
    kind: str
    source_docs: list[str]

    # --- the retrieval context, same shape eval/records.py stores ---
    # Byte-identical to what the live API produces, because build_dataset.py calls
    # the same app.generation functions /ask does - that's what keeps a training
    # example from teaching the model a prompt shape it never sees at inference.
    question: str
    prompt_version: str
    sources: list[dict[str, Any]]
    sources_block: str
    prompt: str

    # --- what the teacher produced ---
    answer: str
    teacher: str  # "local" | "claude"
    teacher_model: str  # the resolved model identifier
    teacher_mode: str  # "write" | "edit"
    base_answer: str | None = None  # the answer edit mode repaired, if any

    # --- gates, populated by later stages ---
    checks: dict[str, bool] = field(default_factory=dict)  # B6: auto-gate scorer results
    review: dict[str, Any] | None = None  # B10: human accept/edit/reject decision


def dataset_path(version: str, name: str) -> Path:
    """Where a dataset file lives, e.g. data/v1/drafts.jsonl."""
    return DATA_DIR / version / name


def write_examples(path: Path, examples: list[Example]) -> Path:
    """Append examples to a JSONL file, creating the directory if needed.

    Append rather than overwrite so a long draft or review run can flush as it
    goes and an interrupted run doesn't lose what's already been produced.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
    return path


def load_examples(path: Path) -> list[Example]:
    """Read a JSONL file back into Examples.

    Example(**line) is the structural check: a missing required field or an
    unexpected key raises TypeError immediately, rather than silently loading
    a malformed record.
    """
    with path.open(encoding="utf-8") as handle:
        return [Example(**json.loads(line)) for line in handle if line.strip()]
