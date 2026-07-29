"""Turn seed questions into training Examples, through the real inference pipeline.

Usage (from the backend/ directory, with LM Studio + Qdrant running):
    python -m training.build_dataset --limit 5                        # wiring smoke test, free
    python -m training.build_dataset --construction mismatched         # a negative construction
    python -m training.build_dataset --teacher claude                  # the real drafting run

Four constructions (see the plan's B6 table; construction #1, true out-of-corpus,
produces no example at all - the retrieval gate fires before any of this runs):

  answered    seed_questions.yaml, natural retrieval on content that's really there.
  declined    declined_seeds.yaml, natural retrieval that clears the gate but on a
              figure the document conspicuously omits (construction #2).
  mismatched  seed_questions.yaml, retrieval FORCED onto a document that doesn't
              cover the question at all (construction #3).
  partial     seed_questions.yaml, retrieval mixing 2 genuinely on-topic chunks
              with 3 off-topic ones (construction #4).

Per seed, once chunks are obtained (differently for each construction above):
    chunks -> build_sources_block(...)        # app.generation
           -> build_prompt(...)                # app.generation + app.prompt_registry
           -> [edit mode only] get_completion(...) with the BASE model
                                                # the "base_answer" the teacher repairs
           -> teacher.draft(...)                # training.teacher
           -> Example -> run_auto_gate(...)     # training.gate
           -> data/v1/drafts.jsonl (or auto_rejected.drafts.jsonl)

Two gates run before an example reaches the real dataset file:
  - training.contamination, BEFORE any teacher call: hard-errors if a seed
    question exactly matches an eval/cases.yaml question, warns on high overlap.
    Checked first because it's free and an exact hit means the seed itself is
    wrong, independent of anything downstream.
  - training.gate, AFTER drafting: reuses eval/scorers.py wholesale to check the
    drafted answer's citations and ruling line. Anything that fails is written to
    a sibling auto_rejected.jsonl instead of the real dataset - visible for
    tuning teacher_style.md or the seed, never silently dropped.

WHY EDIT MODE GENERATES base_answer LIVE, NOT FROM AN EVAL RUN RECORD:
the plan's B1 describes edit mode reading `answer` + `sources_block` straight out
of an eval/results/*.jsonl run record - true for the 59 FIXED eval cases, which
already have baseline runs. Seed questions are new and were never run through
eval/run_eval.py (they can't be - eval/cases.yaml and training data must never
overlap, see the contamination note in seed_questions.yaml). So for a seed
question there is no existing run record to read; this script produces the
equivalent of one on the spot, by pushing the question through the exact same
retrieval + prompt + base-model call eval/run_eval.run_case() and api/ask.py both
use. Same shape, same guarantee (byte-identical context to inference), just
computed fresh instead of read off disk.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import yaml

from app.config import settings
from app.embedding_client import embed_texts
from app.generation import build_prompt, build_sources_block
from app.llm_client import get_completion
from app.model_registry import load_model
from app.prompt_registry import load_prompt
from app.retrieval import retrieve_chunks
from app.vector_store import search
from training.contamination import check_contamination
from training.gate import passes_gate, run_auto_gate
from training.schema import Example, dataset_path, write_examples
from training.teacher import get_teacher

SEEDS_PATH = Path(__file__).parent / "seed_questions.yaml"
DECLINED_SEEDS_PATH = Path(__file__).parent / "declined_seeds.yaml"
# Must match eval/run_eval.SEARCH_LIMIT and AskRequest.limit's default (both 5
# today). A mismatch here would silently train on a different context shape than
# inference ever produces - see the plan's risk notes on this exact number.
SEARCH_LIMIT = 5


def load_seeds() -> list[dict]:
    return yaml.safe_load(SEEDS_PATH.read_text(encoding="utf-8"))["seeds"]


def load_declined_seeds() -> list[dict]:
    return yaml.safe_load(DECLINED_SEEDS_PATH.read_text(encoding="utf-8"))["seeds"]


def _source_summary(chunks: list[dict]) -> list[dict[str, Any]]:
    """Trimmed source metadata for the record; the text already lives in sources_block."""
    return [
        {
            "number": i + 1,
            "source": chunk["payload"]["source"],
            "chunk_index": chunk["payload"]["chunk_index"],
            "score": chunk["score"],
        }
        for i, chunk in enumerate(chunks)
    ]


def _all_docs(seeds: list[dict]) -> list[str]:
    """Unique source docs across a seed set, first-seen order - the fixed ring
    _wrong_doc walks to pick a deterministic 'wrong' document, so constructions
    #3/#4 are reproducible run to run without needing a random seed."""
    seen: list[str] = []
    for seed in seeds:
        if seed["source_doc"] not in seen:
            seen.append(seed["source_doc"])
    return seen


def _wrong_doc(own_doc: str, all_docs: list[str], offset: int) -> str:
    """A different corpus document than `own_doc`, `offset` steps around the ring."""
    i = all_docs.index(own_doc)
    return all_docs[(i + offset) % len(all_docs)]


async def _draft_and_gate(
    *,
    id_: str,
    kind: str,
    source_docs: list[str],
    question: str,
    chunks: list[dict],
    prompt_version_id: str,
    base_model_id: str,
    mode: str,
    teacher,
) -> Example:
    """Shared tail for every construction: render the prompt, get a base answer
    (edit mode), draft with the teacher, package as an Example, run the auto-gate.

    What differs between constructions is only how `chunks` was obtained - this is
    everything that happens once you have them, so it lives in one place instead
    of four near-identical copies that could drift.
    """
    sources_block = build_sources_block(chunks)
    prompt_version = load_prompt(prompt_version_id)
    model_version = load_model(base_model_id)
    generation = {**prompt_version.generation, **model_version.generation_overrides}
    rendered = build_prompt(question, chunks, prompt_version)

    base_answer: str | None = None
    if mode == "edit":
        base_answer = await get_completion(rendered, generation, model=model_version.model)

    answer = await teacher.draft(
        question=question, sources_block=sources_block, mode=mode, base_answer=base_answer
    )

    example = Example(
        id=id_,
        kind=kind,
        source_docs=source_docs,
        question=question,
        prompt_version=prompt_version.version,
        sources=_source_summary(chunks),
        sources_block=sources_block,
        prompt=rendered,
        answer=answer,
        teacher=teacher.name,
        teacher_model=teacher.model,
        teacher_mode=mode,
        base_answer=base_answer,
    )
    example.checks = run_auto_gate(example)
    return example


async def build_answered_example(
    seed: dict, *, prompt_version_id: str, base_model_id: str, mode: str, teacher
) -> Example | None:
    """Natural retrieval on a seed written from real corpus content.

    None if the gate refuses - the corpus doesn't actually cover this seed as
    expected (or embeddings drifted since it was written), so it's not a training
    example, just a bad seed to flag and fix.
    """
    question = seed["question"]
    chunks = await retrieve_chunks(question, limit=SEARCH_LIMIT)
    if not chunks or chunks[0]["score"] < settings.min_relevance_score:
        return None
    return await _draft_and_gate(
        id_=f"{seed['id']}-answered",
        kind="answered",
        source_docs=[seed["source_doc"]],
        question=question,
        chunks=chunks,
        prompt_version_id=prompt_version_id,
        base_model_id=base_model_id,
        mode=mode,
        teacher=teacher,
    )


async def build_declined_example(
    seed: dict, *, prompt_version_id: str, base_model_id: str, mode: str, teacher
) -> Example | None:
    """Construction #2 (weak-but-passing): natural retrieval on a
    declined_seeds.yaml question, deliberately written to ask for a figure its
    document doesn't give. None if retrieval doesn't clear the gate this run -
    scores were verified live when the seed was written, but embeddings or the
    corpus can drift.
    """
    question = seed["question"]
    chunks = await retrieve_chunks(question, limit=SEARCH_LIMIT)
    if not chunks or chunks[0]["score"] < settings.min_relevance_score:
        return None
    return await _draft_and_gate(
        id_=f"{seed['id']}-declined",
        kind="declined",
        source_docs=[seed["source_doc"]],
        question=question,
        chunks=chunks,
        prompt_version_id=prompt_version_id,
        base_model_id=base_model_id,
        mode=mode,
        teacher=teacher,
    )


async def build_mismatched_example(
    seed: dict, all_docs: list[str], *, prompt_version_id: str, base_model_id: str, mode: str, teacher
) -> Example | None:
    """Construction #3: force retrieval onto a document that doesn't cover this
    question at all. Same corpus, so scores plausibly still clear the gate - the
    point is teaching "sources present != sources relevant", not testing the gate
    (construction #1 already covers that, and produces no example).
    """
    question = seed["question"]
    wrong_doc = _wrong_doc(seed["source_doc"], all_docs, offset=1)
    vectors = await embed_texts([question])
    chunks = await search(vectors[0], limit=SEARCH_LIMIT, source=wrong_doc)
    if not chunks or chunks[0]["score"] < settings.min_relevance_score:
        return None
    return await _draft_and_gate(
        id_=f"{seed['id']}-mismatched",
        kind="mismatched",
        source_docs=[wrong_doc],
        question=question,
        chunks=chunks,
        prompt_version_id=prompt_version_id,
        base_model_id=base_model_id,
        mode=mode,
        teacher=teacher,
    )


async def build_partial_example(
    seed: dict, all_docs: list[str], *, prompt_version_id: str, base_model_id: str, mode: str, teacher
) -> Example | None:
    """Construction #4: partial coverage. Two genuinely on-topic chunks from the
    seed's own document, padded to SEARCH_LIMIT with off-topic chunks from an
    unrelated one. The target is NOT a decline: cite only the on-topic sources,
    name the gap, still give a ruling line (see teacher_style.md's
    partial-coverage section - this is what distinguishes "partial" from
    "declined").
    """
    question = seed["question"]
    own_doc = seed["source_doc"]
    filler_doc = _wrong_doc(own_doc, all_docs, offset=2)

    vectors = await embed_texts([question])
    relevant = await search(vectors[0], limit=2, source=own_doc)
    if not relevant or relevant[0]["score"] < settings.min_relevance_score:
        return None
    filler = await search(vectors[0], limit=SEARCH_LIMIT - len(relevant), source=filler_doc)
    if not filler:
        return None

    chunks = relevant + filler  # on-topic sources numbered first: [1]-[2], then filler
    return await _draft_and_gate(
        id_=f"{seed['id']}-partial",
        kind="partial",
        source_docs=[own_doc, filler_doc],
        question=question,
        chunks=chunks,
        prompt_version_id=prompt_version_id,
        base_model_id=base_model_id,
        mode=mode,
        teacher=teacher,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--construction",
        default="answered",
        choices=["answered", "declined", "mismatched", "partial"],
        help="which construction to draft (default: answered)",
    )
    parser.add_argument("--limit", type=int, default=None, help="only draft the first N seeds")
    parser.add_argument(
        "--teacher",
        default="local",
        choices=["local", "claude"],
        help="drafting model (default: local - free, for wiring smoke tests; "
        "use claude for the real dataset)",
    )
    parser.add_argument("--mode", default="edit", choices=["write", "edit"])
    parser.add_argument(
        "--prompt-version",
        default="v2",
        help="prompt version to retrieve/generate against (default: v2, the version "
        "the format objective was measured on in the A9 baseline)",
    )
    parser.add_argument("--base-model-id", default="base", help="model card id for the base model")
    parser.add_argument("--out", type=Path, default=dataset_path("v1", "drafts.jsonl"))
    args = parser.parse_args()

    # mismatched/partial need the full 95-seed ring to pick a deterministic
    # "wrong" document, regardless of --limit or which file this run drafts from.
    all_docs = _all_docs(load_seeds())

    seeds = load_declined_seeds() if args.construction == "declined" else load_seeds()
    if args.limit is not None:
        seeds = seeds[: args.limit]

    # Contamination check runs BEFORE any teacher call - free, and an exact hit
    # means a seed is wrong regardless of what a teacher would draft for it.
    hits = check_contamination({seed["id"]: seed["question"] for seed in seeds})
    exact_hits = [h for h in hits if h.exact]
    warn_hits = [h for h in hits if not h.exact]
    if exact_hits:
        print("CONTAMINATION: exact match(es) against eval/cases.yaml - fix the seed(s) and rerun:")
        for h in exact_hits:
            print(f"  {h.question_id} == eval case {h.eval_case_id!r}: {h.question!r}")
        sys.exit(1)
    for h in warn_hits:
        print(f"  contamination warning: {h.question_id} is {h.overlap:.0%} trigram-similar to "
              f"eval case {h.eval_case_id!r} ({h.eval_question!r}) - review it")

    teacher = get_teacher(args.teacher)
    print(f"Drafting {len(seeds)} seed(s) | construction: {args.construction} | "
          f"teacher: {teacher.name} ({teacher.model}) | mode: {args.mode} | prompt: {args.prompt_version}")

    accepted: list[Example] = []
    rejected: list[Example] = []
    skipped: list[str] = []
    common = dict(
        prompt_version_id=args.prompt_version,
        base_model_id=args.base_model_id,
        mode=args.mode,
        teacher=teacher,
    )
    for seed in seeds:
        if args.construction == "answered":
            example = await build_answered_example(seed, **common)
        elif args.construction == "declined":
            example = await build_declined_example(seed, **common)
        elif args.construction == "mismatched":
            example = await build_mismatched_example(seed, all_docs, **common)
        else:
            example = await build_partial_example(seed, all_docs, **common)

        if example is None:
            skipped.append(seed["id"])
            print(f"  {seed['id']:<25} SKIPPED (gate refused)")
            continue

        if passes_gate(example.checks):
            accepted.append(example)
            print(f"  {seed['id']:<25} drafted ({len(example.answer)} chars)")
        else:
            rejected.append(example)
            failed = [name for name, ok in example.checks.items() if not ok]
            print(f"  {seed['id']:<25} AUTO-REJECTED: failed {failed}")

    write_examples(args.out, accepted)
    if rejected:
        rejected_path = args.out.parent / f"auto_rejected.{args.out.name}"
        write_examples(rejected_path, rejected)
        print(f"\n{len(rejected)} example(s) auto-rejected -> {rejected_path} (tune teacher_style.md "
              "or the seed, don't just re-run and hope)")
    print(f"{len(accepted)} example(s) written to {args.out}"
          f"{f' | {len(skipped)} gate-skipped: {skipped}' if skipped else ''}")


if __name__ == "__main__":
    asyncio.run(main())
