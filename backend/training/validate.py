"""The dataset gate: exits nonzero unless a JSONL file is a well-formed, clean set
of Examples.

Structure, contamination, and the auto-gate are checked here independently of
build_dataset.py, on purpose - build_dataset.py's own checks run once, at draft
time; this is the thing you re-run after a manual edit in review.py, after
merging drafts from separate runs, or just before split.py, to confirm the
guarantees still hold instead of trusting that they did. Split integrity and
manifest hashes (B11) land here too once there's something to check.

Usage: python -m training.validate data/v1/drafts.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from training.contamination import check_contamination
from training.gate import passes_gate, run_auto_gate
from training.schema import Example


def validate(path: Path) -> list[str]:
    """Return a list of error strings; empty means the file is well-formed and clean."""
    errors: list[str] = []
    ids_seen: dict[str, int] = {}
    examples: dict[str, Example] = {}

    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON ({exc})")
                continue

            try:
                example = Example(**raw)
            except TypeError as exc:
                errors.append(f"line {line_no}: doesn't match Example schema ({exc})")
                continue

            if example.id in ids_seen:
                errors.append(
                    f"line {line_no}: duplicate id '{example.id}' "
                    f"(first seen on line {ids_seen[example.id]})"
                )
            else:
                ids_seen[example.id] = line_no
                examples[example.id] = example

    # Contamination: re-checked against the FULL file, independent of whatever
    # subset any single build_dataset.py run happened to check at draft time.
    hits = check_contamination({eid: ex.question for eid, ex in examples.items()})
    for hit in hits:
        prefix = "CONTAMINATION" if hit.exact else "contamination warning"
        errors_list = errors if hit.exact else None  # exact -> error, near -> print only
        message = (
            f"{hit.question_id} {'==' if hit.exact else f'{hit.overlap:.0%} similar to'} "
            f"eval case {hit.eval_case_id!r}: {hit.question!r}"
        )
        if errors_list is not None:
            errors.append(f"{prefix}: {message}")
        else:
            print(f"  {prefix}: {message}")

    # Auto-gate: recomputed from each example's own answer/sources, not trusted
    # from whatever `checks` happens to hold on disk - this is the single source
    # of truth for "is this record usable," not a cache of it.
    for eid, example in examples.items():
        checks = run_auto_gate(example)
        if not passes_gate(checks):
            failed = [name for name, ok in checks.items() if not ok]
            errors.append(f"{eid}: fails auto-gate {failed}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSONL dataset file to validate")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"no such file: {args.path}", file=sys.stderr)
        sys.exit(1)

    errors = validate(args.path)
    total = sum(1 for line in args.path.open(encoding="utf-8") if line.strip())

    if errors:
        print(f"{args.path}: {len(errors)} error(s) across {total} record(s)")
        for error in errors:
            print(f"  {error}")
        sys.exit(1)

    print(f"{args.path}: {total} record(s), all valid")


if __name__ == "__main__":
    main()
