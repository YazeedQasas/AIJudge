"""Loads model cards from app/models/ and exposes which served model each one names.

A model card is one Markdown file (e.g. base.md) with a YAML frontmatter block on
top and prose notes below. It answers one question: when this app says "lora_v1",
what string does LM Studio actually need in the request body?

Why a registry at all, when config already has lm_studio_model: because comparing
models is the point. Once there's a fine-tune, "which model" becomes an axis the
eval varies alongside prompt version, and every run needs to record not just the
identifier but where it came from — base model, training run, which prompt version
it was trained against. That provenance is what the card carries.

This mirrors prompt_registry.py on purpose: same frontmatter mechanic (shared in
app/frontmatter.py), same "files are data, code just loads them" split. It diverges
in one way, and the reason is worth knowing. Prompt versions form a chain
(v1 -> v2 -> v3), each superseding the last, so they sort numerically. Models are a
set of named candidates (base, lora_v1, qwen_base) that get compared against each
other and coexist indefinitely. So a card's filename is a free identifier, and
display order is an explicit `order:` field rather than something parsed out of it.

Like prompt_registry, this does NOT decide which model is active — that's config.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.frontmatter import split_frontmatter

MODELS_DIR = Path(__file__).parent / "models"

# Files in models/ that document the format rather than describe a model.
_NOT_A_CARD = {"README"}


@dataclass(frozen=True)
class ModelVersion:
    """One loaded model card: what to call it, what to send, and where it came from."""

    id: str
    name: str
    model: str  # the exact identifier LM Studio serves this under
    notes: str
    metadata: dict[str, Any]

    @property
    def kind(self) -> str:
        """"base" for a stock model, "adapter" for a fine-tune of one."""
        return self.metadata.get("kind", "base")

    @property
    def is_adapter(self) -> bool:
        return self.kind == "adapter"

    @property
    def prompt_version(self) -> str | None:
        """The prompt version this model was trained against, if it's a fine-tune.

        Load-bearing for adapters. A LoRA trained on judge_v2's wording is coupled to
        it and will underperform on v1 or a future v3 — recording that here is what
        stops a later reader from silently pairing them.
        """
        return self.metadata.get("prompt_version")

    @property
    def generation_overrides(self) -> dict[str, Any]:
        """Sampling params this model needs regardless of what the prompt asks for.

        Normally empty — the prompt version owns sampling, because the same wording
        behaves differently at different settings. The exception is a fine-tune that
        legitimately wants different settings than the wording was tuned at (a model
        trained to be decisive may want a lower temperature than the prompt declares).
        Merged over the prompt's generation dict, so these win on conflict.
        """
        return self.metadata.get("generation_overrides", {})


def _card_ids() -> list[str]:
    """Every available model id, for error messages and listing."""
    return sorted(p.stem for p in MODELS_DIR.glob("*.md") if p.stem not in _NOT_A_CARD)


def _load_file(path: Path) -> ModelVersion:
    """Parse one model card into a ModelVersion."""
    metadata, notes = split_frontmatter(path.read_text(encoding="utf-8"))
    model = metadata.get("model")
    if not model:
        raise ValueError(
            f"Model card {path.name} has no `model:` field. That string is what gets "
            "sent to LM Studio, so a card without one can't be served."
        )
    return ModelVersion(
        id=metadata.get("id", path.stem),
        name=metadata.get("name", path.stem),
        model=model,
        notes=notes,
        metadata=metadata,
    )


def load_model(model_id: str) -> ModelVersion:
    """Load the model card app/models/{model_id}.md and parse it."""
    path = MODELS_DIR / f"{model_id}.md"
    if not path.exists():
        # Unlike prompt versions, model ids aren't guessable from a pattern, so say
        # what's actually there rather than only what's missing.
        available = ", ".join(_card_ids()) or "(none)"
        raise FileNotFoundError(f"No model card at {path}. Available: {available}")
    return _load_file(path)


def list_models() -> list[ModelVersion]:
    """Load every model card, sorted by the optional `order:` field then by id.

    There's no natural numeric order the way prompt versions have one, so `order:`
    is how a card controls where it lands in a comparison table. Cards without one
    sort to the end.
    """
    models = [_load_file(MODELS_DIR / f"{model_id}.md") for model_id in _card_ids()]
    return sorted(models, key=lambda m: (m.metadata.get("order", 10_000), m.id))
