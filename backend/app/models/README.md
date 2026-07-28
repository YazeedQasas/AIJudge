# Model cards

Each model the app can be pointed at is **one Markdown file** named `<id>.md`
(e.g. `base.md`, `lora_v1.md`). Same file shape as `app/prompts/`: a **YAML
frontmatter** block between `---` fences, then free prose below.

A card answers one question the code can't: when we say `lora_v1`, what string
does LM Studio actually need in the request body — and where did that model come
from?

## Why this is separate from `lm_studio_model` in config

Config names *one* model, the one currently serving. A registry lets "which
model" become an **axis the eval varies**, alongside prompt version. Once there
are two models, every eval run has to record which one produced each answer, and
every fine-tune has to record enough provenance that a later reader can tell
whether the comparison was fair. That's what a card carries.

## Metadata fields

| Field | Meaning |
|---|---|
| `id` | Card id. Must match the filename. This is what config and the eval CLI use. |
| `name` | Human-readable label for scorecards and the testing page. |
| `model` | **The exact identifier LM Studio serves this under.** Verify with `curl localhost:1234/v1/models`. |
| `kind` | `base` (stock model) or `adapter` (a fine-tune of one). |
| `order` | Optional int controlling position in listings. Cards without one sort last. |
| `description` | What this model is and when to use it. |
| `generation_overrides` | Optional sampling params that win over the prompt version's. Normally absent — the prompt owns sampling. |

### Additional fields for `kind: adapter`

| Field | Meaning |
|---|---|
| `base_model` | Which card's model this was fine-tuned from. |
| `prompt_version` | **The prompt version it was trained against.** Load-bearing — see below. |
| `training_run` | The `training/runs/{id}/` directory that produced it. |
| `dataset` / `dataset_sha256` | Which dataset, and its hash, so a run is reproducible. |
| `quantization` | e.g. `Q4_K_M`, `Q8_0`. A small LoRA delta can be erased by aggressive quantization, so this is part of the identity. |
| `created` | ISO date. |
| `changelog` | What changed vs. the previous adapter, newest last. |

## Why `prompt_version` matters on an adapter

A LoRA is trained on fully rendered prompts — including the template's wording.
It learns to respond to *that* wording. Pair it with a different prompt version
and it underperforms, quietly, for a reason nobody remembers six weeks later.
Recording the pairing here is what makes that discoverable.

This is also why the eval varies **(prompt version × model)** rather than either
alone: the matrix is how you measure the coupling instead of assuming it away.

## Adding a model

1. Load it in LM Studio, then `curl localhost:1234/v1/models` and copy the
   identifier **exactly**.
2. Write `<id>.md` with that string in `model:`.
3. Compare it: `python -m eval.run_eval --models base <id>`.
4. If it wins, point `active_model_id` at it. That's a config change, not a code
   change — same rollback story as prompt versions.
