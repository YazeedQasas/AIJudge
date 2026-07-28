from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.model_registry import list_models

router = APIRouter()


class ModelInfo(BaseModel):
    """Public metadata about one model card, for the model-picker UI.

    Mirrors PromptInfo in api/prompts.py — same idea, other axis. The testing page
    pairs the two dropdowns so you can ask "which prompt, on which model?".
    """

    id: str
    name: str
    model: str
    kind: str
    description: str | None = None
    prompt_version: str | None = None
    generation_overrides: dict[str, Any] = {}


@router.get("/models")
def get_models() -> list[ModelInfo]:
    """List available models and their metadata (for the testing page)."""
    return [
        ModelInfo(
            id=m.id,
            name=m.name,
            model=m.model,
            kind=m.kind,
            description=m.metadata.get("description"),
            prompt_version=m.prompt_version,
            generation_overrides=m.generation_overrides,
        )
        for m in list_models()
    ]
