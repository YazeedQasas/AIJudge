"""Run the eval dataset through one or more (prompt version x model) variants.

Usage (from the backend/ directory, with LM Studio + Qdrant running):
    python -m eval.run_eval                          # all prompts, active model
    python -m eval.run_eval --prompts v1 v2          # only these prompt versions
    python -m eval.run_eval --models base lora_v1    # compare models
    python -m eval.run_eval --judge local            # pick judge backend (see eval/judge.py)
    python -m eval.run_eval --no-judge               # guardrails only, skip rubric scoring

Two reports are printed:
  - Guardrails: binary pass/fail hygiene checks (is the answer broken?).
  - Rubric:     1-5 quality scores from an LLM judge (which variant is better?).

Each case runs through the SAME pipeline as POST /ask, plus a fixed seed for
reproducibility. What varies is the variant; everything else is held still. That is
the whole point - if retrieval, seed, or the case set moved between two runs, a
difference in the scorecard tells you nothing about the thing you changed.
"""

import argparse
import asyncio
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.config import settings
from app.embedding_client import embed_texts
from app.generation import build_prompt, build_sources_block, find_invalid_citations
from app.llm_client import get_completion
from app.model_registry import ModelVersion, load_model
from app.prompt_registry import PromptVersion, list_prompts, load_prompt
from app.vector_store import search
from eval.judge import DIMENSIONS, Judge, RubricScore, get_judge
from eval.records import RunRecord, new_run_id, record_path, utc_now, write_records
from eval.scorers import SCORERS, CaseResult, expectation

CASES_PATH = Path(__file__).parent / "cases.yaml"
EVAL_SEED = 42  # fixed so the only thing that varies between runs is the variant
SEARCH_LIMIT = 5


@dataclass(frozen=True)
class Variant:
    """One (prompt version x model) combination - the unit an eval scores.

    Prompt version and model are separate axes because they fail differently. A
    wording change alters what the model is asked for; a weights change alters what
    it is inclined to do. Scoring their combination is also how you measure the
    coupling between them - a LoRA trained on v2's wording is expected to do worse
    on v1, and the matrix is where that shows up instead of being assumed away.
    """

    prompt_version: str
    model_id: str

    @property
    def label(self) -> str:
        return f"{self.prompt_version}@{self.model_id}"


# (case, pipeline output, judge rubric or None) for one evaluated case.
CaseRun = tuple[dict, CaseResult, RubricScore | None]


def load_cases() -> list[dict]:
    return yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))["cases"]


def resolve(variant: Variant) -> tuple[PromptVersion, ModelVersion, dict[str, Any]]:
    """Load the prompt and model a variant names, and the sampling params it runs with.

    Same merge order as the API's _resolve(): the prompt owns sampling, a model card
    may override it, and the fixed eval seed wins over both. Resolved once per variant
    rather than per case - the API resolves per request, but within a variant these
    never change, and rereading the files 60 times would only invite them to drift.
    """
    prompt = load_prompt(variant.prompt_version)
    model = load_model(variant.model_id)
    generation = {**prompt.generation, **model.generation_overrides, "seed": EVAL_SEED}
    return prompt, model, generation


async def run_case(
    case: dict, prompt: PromptVersion, model: ModelVersion, generation: dict[str, Any]
) -> tuple[CaseResult, str]:
    """Push one case through the real retrieval + generation pipeline.

    Returns the result and the exact sources block the model saw - the caller needs
    that both for judging and for the run record, and rebuilding it later from the
    chunks would risk it differing from what was actually sent.
    """
    [vector] = await embed_texts([case["question"]])
    chunks = await search(vector, limit=SEARCH_LIMIT)

    # Same refusal gate as the API: no chunks or weak top hit -> refuse (no generation).
    # This fires before any model runs, so it is identical across variants - it is
    # code, not model behaviour, and no amount of fine-tuning can change it.
    if not chunks or chunks[0]["score"] < settings.min_relevance_score:
        return CaseResult(answer="<refused>", sources=[], invalid_citations=[]), ""

    sources_block = build_sources_block(chunks)
    rendered = build_prompt(case["question"], chunks, prompt)
    answer = await get_completion(rendered, generation, model=model.model)
    invalid = find_invalid_citations(answer, len(chunks))
    return CaseResult(answer=answer, sources=chunks, invalid_citations=invalid), sources_block


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


def _score_case(case: dict, result: CaseResult) -> dict[str, bool]:
    """Every scorer that applies to this case, by name. Scorers that don't apply are
    omitted rather than recorded as passing - absent and true are different facts."""
    return {s.name: s.fn(case, result) for s in SCORERS if s.applies(case)}


async def evaluate(
    variant: Variant, cases: list[dict], judge: Judge | None, run_id: str
) -> tuple[list[CaseRun], list[RunRecord]]:
    """Run all cases for a variant; score answered ones with the judge if provided."""
    prompt, model, generation = resolve(variant)
    runs: list[CaseRun] = []
    records: list[RunRecord] = []

    for case in cases:
        started = time.monotonic()
        result, sources_block = await run_case(case, prompt, model, generation)

        rubric: RubricScore | None = None
        if not result.refused and judge is not None:
            try:
                rubric = await judge.score(
                    question=case["question"], sources_block=sources_block, answer=result.answer
                )
            except Exception as exc:  # a bad judge parse shouldn't abort the whole run
                print(f"    judge error on {case['id']}: {exc}")

        elapsed = time.monotonic() - started
        verdict = "refused" if result.refused else ("declined" if result.declined else "answered")
        avg = f" | rubric {rubric.average:.2f}" if rubric else ""
        flag = "" if verdict == expectation(case) else f"  <- expected {expectation(case)}"
        print(f"  [{variant.label}] {case['id']:<26} {verdict:<9} ({elapsed:5.1f}s){avg}{flag}")

        runs.append((case, result, rubric))
        records.append(
            RunRecord(
                run_id=run_id,
                timestamp=utc_now(),
                variant=variant.label,
                prompt_version=variant.prompt_version,
                model_id=variant.model_id,
                model=model.model,
                generation=generation,
                seed=EVAL_SEED,
                case_id=case["id"],
                question=case["question"],
                expected=expectation(case),
                sources=_source_summary(result.sources),
                sources_block=sources_block,
                answer=result.answer,
                invalid_citations=result.invalid_citations,
                refused=result.refused,
                elapsed_seconds=round(elapsed, 2),
                scorers=_score_case(case, result),
                rubric={"scores": rubric.scores, "reasoning": rubric.reasoning} if rubric else None,
            )
        )
    return runs, records


def guardrail_card(runs: list[CaseRun]) -> dict[str, tuple[int, int]]:
    """{scorer_name: (passes, applicable_total)} for the binary hygiene checks."""
    card: dict[str, tuple[int, int]] = {}
    for scorer in SCORERS:
        applicable = [(c, r) for (c, r, _) in runs if scorer.applies(c)]
        passes = sum(1 for (c, r) in applicable if scorer.fn(c, r))
        card[scorer.name] = (passes, len(applicable))
    return card


def rubric_card(runs: list[CaseRun]) -> dict[str, float | None]:
    """{dimension: avg score} plus '_overall' and '_n', averaged over judged answers."""
    rubrics = [rub for (_, _, rub) in runs if rub is not None]
    card: dict[str, float | None] = {}
    for dim in DIMENSIONS:
        card[dim] = sum(r.scores[dim] for r in rubrics) / len(rubrics) if rubrics else None
    card["_overall"] = sum(r.average for r in rubrics) / len(rubrics) if rubrics else None
    card["_n"] = float(len(rubrics))
    return card


# Variant labels ("v2@lora_v1") are longer than bare versions were, so columns widened.
_COL = 14
# Wide enough for the longest scorer name ("no_citation_when_declining", 26).
_NAME_COL = 30


def print_guardrail_report(variants: list[Variant], cards: dict[str, dict]) -> None:
    # ASCII only in report output: the Windows console is cp1252 by default, where an
    # em dash renders as a replacement char. This text gets pasted into the README.
    print(f"\n{'=' * 72}\nGuardrails - binary pass/fail (seed={EVAL_SEED})\n{'=' * 72}")
    header = f"{'metric':<{_NAME_COL}}{'kind':<10}" + "".join(f"{v.label:>{_COL}}" for v in variants)
    print(header + "\n" + "-" * len(header))
    for scorer in SCORERS:
        cells = ""
        for v in variants:
            passes, total = cards[v.label][scorer.name]
            cells += f"{f'{passes}/{total}':>{_COL}}"
        print(f"{scorer.name:<{_NAME_COL}}{scorer.kind:<10}{cells}")


def print_rubric_report(variants: list[Variant], cards: dict[str, dict]) -> None:
    print(f"\n{'=' * 72}\nRubric - LLM judge, 1-5 (higher is better)\n{'=' * 72}")
    header = f"{'dimension':<32}" + "".join(f"{v.label:>{_COL}}" for v in variants)
    print(header + "\n" + "-" * len(header))
    for dim in list(DIMENSIONS) + ["_overall"]:
        label = "OVERALL" if dim == "_overall" else dim
        cells = ""
        for v in variants:
            val = cards[v.label][dim]
            cells += f"{(f'{val:.2f}' if val is not None else '-'):>{_COL}}"
        print(f"{label:<32}{cells}")
    n = {v.label: int(cards[v.label]["_n"]) for v in variants}
    print(f"\n(averaged over judged answers per variant: {n})")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate (prompt version x model) variants.")
    parser.add_argument("--prompts", nargs="*", help="prompt versions to evaluate (default: all)")
    parser.add_argument(
        "--models",
        nargs="*",
        help=f"model ids to evaluate (default: the active model, {settings.active_model_id!r})",
    )
    parser.add_argument("--judge", default="local", help="judge backend name (see eval/judge.py)")
    parser.add_argument("--no-judge", action="store_true", help="skip rubric scoring")
    parser.add_argument(
        "--label",
        default="run",
        help="prefix for the results file, e.g. --label baseline (default: run)",
    )
    parser.add_argument("--no-save", action="store_true", help="don't write a results file")
    args = parser.parse_args()

    prompt_versions = args.prompts or [p.version for p in list_prompts()]
    # Deliberately NOT every model by default. The full cartesian product is cases x
    # prompts x models generations, each one a local inference - comparing models is
    # something you ask for, not something you trip over.
    model_ids = args.models or [settings.active_model_id]

    # Models outer, prompts inner: LM Studio JIT-loads a model on first use, so this
    # order loads each model once instead of thrashing between them on every case.
    variants = [
        Variant(prompt_version=p, model_id=m)
        for m, p in itertools.product(model_ids, prompt_versions)
    ]

    judge = None if args.no_judge else get_judge(args.judge)
    cases = load_cases()
    judge_label = "none" if judge is None else judge.name
    labels = ", ".join(v.label for v in variants)
    run_id = new_run_id()
    path = record_path(run_id, args.label)
    print(f"Evaluating {len(cases)} cases | variants: {labels} | judge: {judge_label}")
    print(f"run_id: {run_id}" + ("" if args.no_save else f" -> {path}"))

    all_runs: dict[str, list[CaseRun]] = {}
    for variant in variants:
        print(f"\n--- running {variant.label} ---")
        runs, records = await evaluate(variant, cases, judge, run_id)
        all_runs[variant.label] = runs
        # Flush per variant, not at the end: a long comparison that dies on variant 3
        # should still leave variants 1 and 2 on disk.
        if not args.no_save:
            write_records(path, records)

    print_guardrail_report(variants, {v.label: guardrail_card(all_runs[v.label]) for v in variants})
    if judge is not None:
        print_rubric_report(variants, {v.label: rubric_card(all_runs[v.label]) for v in variants})
    if not args.no_save:
        print(f"\nRecords: {path}")


if __name__ == "__main__":
    asyncio.run(main())
