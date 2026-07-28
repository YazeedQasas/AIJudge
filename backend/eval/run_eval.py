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

import yaml

from app.config import settings
from app.embedding_client import embed_texts
from app.generation import build_prompt, build_sources_block, find_invalid_citations
from app.llm_client import get_completion
from app.model_registry import load_model
from app.prompt_registry import list_prompts, load_prompt
from app.vector_store import search
from eval.judge import DIMENSIONS, Judge, RubricScore, get_judge
from eval.scorers import SCORERS, CaseResult

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


async def run_case(case: dict, variant: Variant) -> CaseResult:
    """Push one case through the real retrieval + generation pipeline for a variant."""
    [vector] = await embed_texts([case["question"]])
    chunks = await search(vector, limit=SEARCH_LIMIT)

    # Same refusal gate as the API: no chunks or weak top hit -> decline (no generation).
    # Note this fires before any model runs, so it is identical across variants - it is
    # code, not model behaviour, and no amount of fine-tuning can change it.
    if not chunks or chunks[0]["score"] < settings.min_relevance_score:
        return CaseResult(answer="<refused>", sources=[], invalid_citations=[])

    prompt = load_prompt(variant.prompt_version)
    model = load_model(variant.model_id)
    rendered = build_prompt(case["question"], chunks, prompt)
    # Same merge order as the API's _resolve(): prompt owns sampling, model may
    # override it, and the fixed seed wins over both for reproducibility.
    generation = {**prompt.generation, **model.generation_overrides, "seed": EVAL_SEED}
    answer = await get_completion(rendered, generation, model=model.model)
    invalid = find_invalid_citations(answer, len(chunks))
    return CaseResult(answer=answer, sources=chunks, invalid_citations=invalid)


async def evaluate(variant: Variant, cases: list[dict], judge: Judge | None) -> list[CaseRun]:
    """Run all cases for a variant; score answered ones with the judge if provided."""
    runs: list[CaseRun] = []
    for case in cases:
        started = time.monotonic()
        result = await run_case(case, variant)

        rubric: RubricScore | None = None
        if not result.refused and judge is not None:
            sources_block = build_sources_block(result.sources)
            try:
                rubric = await judge.score(
                    question=case["question"], sources_block=sources_block, answer=result.answer
                )
            except Exception as exc:  # a bad judge parse shouldn't abort the whole run
                print(f"    judge error on {case['id']}: {exc}")

        elapsed = time.monotonic() - started
        verdict = "refused" if result.refused else "answered"
        avg = f" | rubric {rubric.average:.2f}" if rubric else ""
        print(f"  [{variant.label}] {case['id']:<24} {verdict:<9} ({elapsed:5.1f}s){avg}")
        runs.append((case, result, rubric))
    return runs


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


def print_guardrail_report(variants: list[Variant], cards: dict[str, dict]) -> None:
    # ASCII only in report output: the Windows console is cp1252 by default, where an
    # em dash renders as a replacement char. This text gets pasted into the README.
    print(f"\n{'=' * 72}\nGuardrails - binary pass/fail (seed={EVAL_SEED})\n{'=' * 72}")
    header = f"{'metric':<22}{'kind':<10}" + "".join(f"{v.label:>{_COL}}" for v in variants)
    print(header + "\n" + "-" * len(header))
    for scorer in SCORERS:
        cells = ""
        for v in variants:
            passes, total = cards[v.label][scorer.name]
            cells += f"{f'{passes}/{total}':>{_COL}}"
        print(f"{scorer.name:<22}{scorer.kind:<10}{cells}")


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
    print(f"Evaluating {len(cases)} cases | variants: {labels} | judge: {judge_label}")

    all_runs: dict[str, list[CaseRun]] = {}
    for variant in variants:
        print(f"\n--- running {variant.label} ---")
        all_runs[variant.label] = await evaluate(variant, cases, judge)

    print_guardrail_report(variants, {v.label: guardrail_card(all_runs[v.label]) for v in variants})
    if judge is not None:
        print_rubric_report(variants, {v.label: rubric_card(all_runs[v.label]) for v in variants})


if __name__ == "__main__":
    asyncio.run(main())
