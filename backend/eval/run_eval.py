"""Run the eval dataset through one or more prompt versions and print scorecards.

Usage (from the backend/ directory, with LM Studio + Qdrant running):
    python -m eval.run_eval                 # all versions, local LLM judge
    python -m eval.run_eval v1 v2           # only these versions
    python -m eval.run_eval --judge local   # pick judge backend (see eval/judge.py)
    python -m eval.run_eval --no-judge      # guardrails only, skip rubric scoring

Two reports are printed:
  - Guardrails: binary pass/fail hygiene checks (is the answer broken?).
  - Rubric:     1-5 quality scores from an LLM judge (which version is better?).

Each case runs through the SAME pipeline as POST /ask, plus a fixed seed for reproducibility.
"""

import argparse
import asyncio
import time
from pathlib import Path

import yaml

from app.config import settings
from app.embedding_client import embed_texts
from app.generation import build_prompt, build_sources_block, find_invalid_citations
from app.llm_client import get_completion
from app.prompt_registry import list_prompts, load_prompt
from app.vector_store import search
from eval.judge import DIMENSIONS, Judge, RubricScore, get_judge
from eval.scorers import SCORERS, CaseResult

CASES_PATH = Path(__file__).parent / "cases.yaml"
EVAL_SEED = 42  # fixed so the only thing that varies between runs is the prompt version
SEARCH_LIMIT = 5

# (case, pipeline output, judge rubric or None) for one evaluated case.
CaseRun = tuple[dict, CaseResult, RubricScore | None]


def load_cases() -> list[dict]:
    return yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))["cases"]


async def run_case(case: dict, version: str) -> CaseResult:
    """Push one case through the real retrieval + generation pipeline for a version."""
    [vector] = await embed_texts([case["question"]])
    chunks = await search(vector, limit=SEARCH_LIMIT)

    # Same refusal gate as the API: no chunks or weak top hit -> decline (no generation).
    if not chunks or chunks[0]["score"] < settings.min_relevance_score:
        return CaseResult(answer="<refused>", sources=[], invalid_citations=[])

    prompt = load_prompt(version)
    rendered = build_prompt(case["question"], chunks, prompt)
    # Fold a fixed seed into this version's own sampling params for reproducibility.
    generation = {**prompt.generation, "seed": EVAL_SEED}
    answer = await get_completion(rendered, generation)
    invalid = find_invalid_citations(answer, len(chunks))
    return CaseResult(answer=answer, sources=chunks, invalid_citations=invalid)


async def evaluate(version: str, cases: list[dict], judge: Judge | None) -> list[CaseRun]:
    """Run all cases for a version; score answered ones with the judge if provided."""
    runs: list[CaseRun] = []
    for case in cases:
        started = time.monotonic()
        result = await run_case(case, version)

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
        print(f"  [{version}] {case['id']:<24} {verdict:<9} ({elapsed:5.1f}s){avg}")
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


def print_guardrail_report(versions: list[str], cards: dict[str, dict]) -> None:
    print(f"\n{'=' * 60}\nGuardrails — binary pass/fail (seed={EVAL_SEED})\n{'=' * 60}")
    header = f"{'metric':<22}{'kind':<10}" + "".join(f"{v:>8}" for v in versions)
    print(header + "\n" + "-" * len(header))
    for scorer in SCORERS:
        cells = "".join(f"{f'{cards[v][scorer.name][0]}/{cards[v][scorer.name][1]}':>8}" for v in versions)
        print(f"{scorer.name:<22}{scorer.kind:<10}{cells}")


def print_rubric_report(versions: list[str], cards: dict[str, dict]) -> None:
    print(f"\n{'=' * 60}\nRubric — LLM judge, 1-5 (higher is better)\n{'=' * 60}")
    header = f"{'dimension':<32}" + "".join(f"{v:>8}" for v in versions)
    print(header + "\n" + "-" * len(header))
    for dim in list(DIMENSIONS) + ["_overall"]:
        label = "OVERALL" if dim == "_overall" else dim
        cells = ""
        for v in versions:
            val = cards[v][dim]
            cells += f"{(f'{val:.2f}' if val is not None else '—'):>8}"
        print(f"{label:<32}{cells}")
    n = {v: int(cards[v]["_n"]) for v in versions}
    print(f"\n(averaged over judged answers per version: {n})")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prompt versions.")
    parser.add_argument("versions", nargs="*", help="versions to evaluate (default: all)")
    parser.add_argument("--judge", default="local", help="judge backend name (see eval/judge.py)")
    parser.add_argument("--no-judge", action="store_true", help="skip rubric scoring")
    args = parser.parse_args()

    versions = args.versions or [p.version for p in list_prompts()]
    judge = None if args.no_judge else get_judge(args.judge)
    cases = load_cases()
    judge_label = "none" if judge is None else judge.name
    print(f"Evaluating {len(cases)} cases | versions: {', '.join(versions)} | judge: {judge_label}")

    all_runs: dict[str, list[CaseRun]] = {}
    for version in versions:
        print(f"\n--- running {version} ---")
        all_runs[version] = await evaluate(version, cases, judge)

    print_guardrail_report(versions, {v: guardrail_card(all_runs[v]) for v in versions})
    if judge is not None:
        print_rubric_report(versions, {v: rubric_card(all_runs[v]) for v in versions})


if __name__ == "__main__":
    asyncio.run(main())
