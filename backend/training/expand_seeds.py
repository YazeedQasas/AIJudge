"""Paraphrase each hand-written seed into further phrasing variants (plan B8).

seed_questions.yaml's own header reserves TWO axes of variation: kind
(definitional/scenario, fixed when the seed was hand-written) and phrasing style
(formal/colloquial, short/multi-part - NOT fixed, and this script's whole job).
Expanding the corpus-topic axis is explicitly out of scope - that's what the
seeds themselves already cover, one per corpus section; this script only varies
HOW a given topic is asked about, never WHAT topic.

Per seed, 3 variants are requested in one call: a formal rephrasing (same
register, different wording), a colloquial one (as a non-lawyer would ask), and
one that flips the seed's `kind` - a scenario seed gets a short direct variant,
a definitional seed gets a short-scenario variant. That is "short vs multi-part"
without needing to stitch separate seeds together.

Two checks run on every candidate variant before it's kept, because paraphrasing
can silently drift a question off its original topic or dilute it enough to miss
retrieval:
  - it must still retrieve its seed's own source_doc on top, above the relevance
    gate (same reasoning as build_dataset.py's own gate check, applied here to a
    candidate QUESTION before any answer is drafted for it)
  - training.contamination, same as every other seed file

Usage (from backend/, with LM Studio + Qdrant running):
    python -m training.expand_seeds --limit 3           # smoke test, a few seeds
    python -m training.expand_seeds                      # the real expansion, all 95
"""

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

from app.config import settings
from app.embedding_client import embed_texts
from app.llm_client import get_completion
from app.vector_store import search
from training.build_dataset import load_seeds
from training.contamination import check_contamination

OUT_PATH = Path(__file__).parent / "data" / "v1" / "expanded_questions.yaml"

_PROMPT = """أعد صياغة السؤال القانوني التالي بثلاثة أساليب مختلفة، مع الحفاظ التام على
نفس الموضوع القانوني والمعنى بالضبط - لا تُغيّر الموضوع، ولا تُضِف أو تحذف أي مطلب.

السؤال الأصلي:
{question}

اكتب ثلاثة أسطر فقط، سطر لكل صياغة، دون ترقيم ودون أي نص إضافي قبلها أو بعدها:
1. صياغة رسمية فصيحة بكلمات مختلفة عن السؤال الأصلي، بنفس الطول تقريباً.
2. صياغة عامية أقرب لأسلوب شخص عادي يسأل محامياً، تحمل نفس السؤال بالضبط.
3. {third_instruction}
"""

_THIRD_FOR_KIND = {
    "scenario": "صياغة مختصرة جداً بصيغة سؤال تعريفي مباشر، دون سرد وقائع، عن نفس القاعدة القانونية.",
    "definitional": "صياغة على هيئة واقعة أو سيناريو قصير يسأل عن نفس القاعدة القانونية، بدلاً من سؤال تعريفي مباشر.",
}


def _variant_kind(original_kind: str, slot: int) -> str:
    """Slot 3 deliberately flips the seed's kind; slots 1-2 keep it (only the
    register changes, not the framing)."""
    if slot == 3:
        return "definitional" if original_kind == "scenario" else "scenario"
    return original_kind


async def expand_one(seed: dict) -> list[str]:
    """Ask the local model for 3 paraphrases of one seed; return raw lines.

    declined_seeds.yaml has no `kind` field (every entry is the same flavor -
    an omitted-figure question), so it defaults to "definitional" there.
    """
    kind = seed.get("kind", "definitional")
    prompt = _PROMPT.format(
        question=seed["question"], third_instruction=_THIRD_FOR_KIND[kind]
    )
    raw = await get_completion(prompt, {"temperature": 0.7}, model=settings.lm_studio_model)
    lines = [line.strip().lstrip("123.-) ").strip() for line in raw.strip().splitlines() if line.strip()]
    return lines[:3]


async def verify_variant(question: str, own_doc: str) -> float | None:
    """Top retrieval score if `own_doc` is still the top hit and clears the gate,
    else None - a paraphrase that drifted off-topic or diluted past the gate
    isn't a usable variant of this seed."""
    vectors = await embed_texts([question])
    results = await search(vectors[0], limit=1)
    if not results:
        return None
    top = results[0]
    if top["payload"]["source"] != own_doc or top["score"] < settings.min_relevance_score:
        return None
    return top["score"]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, default=None, help="override the seeds file")
    parser.add_argument("--limit", type=int, default=None, help="only expand the first N seeds")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    seeds = (
        yaml.safe_load(args.seeds.read_text(encoding="utf-8"))["seeds"]
        if args.seeds
        else load_seeds()
    )
    if args.limit is not None:
        seeds = seeds[: args.limit]

    print(f"Expanding {len(seeds)} seed(s) x 3 variants, verifying each against Qdrant...")

    variants: list[dict] = []
    dropped = 0
    for seed in seeds:
        try:
            lines = await expand_one(seed)
        except Exception as exc:  # a bad paraphrase call shouldn't abort the whole run
            print(f"  {seed['id']:<20} EXPAND ERROR: {exc}")
            continue

        kept = 0
        for slot, question in enumerate(lines, start=1):
            score = await verify_variant(question, seed["source_doc"])
            if score is None:
                dropped += 1
                continue
            variants.append(
                {
                    "id": f"{seed['id']}-x{slot}",
                    "source_doc": seed["source_doc"],
                    "kind": _variant_kind(seed.get("kind", "definitional"), slot),
                    "question": question,
                }
            )
            kept += 1
        print(f"  {seed['id']:<20} {kept}/{len(lines)} variant(s) kept")

    # Contamination check over the whole batch at once - cheaper than per-variant,
    # and an expansion run either passes as a whole or gets fixed as a whole.
    hits = check_contamination({v["id"]: v["question"] for v in variants})
    exact_hits = [h for h in hits if h.exact]
    if exact_hits:
        print("\nCONTAMINATION: a paraphrase landed on an eval question - dropping it:")
        exact_ids = set()
        for h in exact_hits:
            print(f"  {h.question_id} == eval case {h.eval_case_id!r}: {h.question!r}")
            exact_ids.add(h.question_id)
        variants = [v for v in variants if v["id"] not in exact_ids]
    for h in hits:
        if not h.exact:
            print(f"  contamination warning: {h.question_id} is {h.overlap:.0%} similar to "
                  f"eval case {h.eval_case_id!r} - review it")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"seeds": variants}, handle, allow_unicode=True, sort_keys=False)

    print(f"\n{len(variants)} variant(s) kept, {dropped} dropped (off-topic or below gate) -> {args.out}")
    print("Human pass still needed: skim for paraphrases that drifted in MEANING even "
          "though retrieval didn't notice (a score check can't catch that).")


if __name__ == "__main__":
    asyncio.run(main())
