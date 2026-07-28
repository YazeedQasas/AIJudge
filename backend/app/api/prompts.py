from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.prompt_registry import list_prompts

router = APIRouter()


class PromptInfo(BaseModel):
    """Public metadata about one prompt version, for the version-picker UI."""

    version: str
    name: str
    model: str | None = None
    description: str | None = None
    generation: dict[str, Any] = {}


@router.get("/prompts")
def get_prompts() -> list[PromptInfo]:
    """List available prompt versions and their metadata (for the testing page)."""
    return [
        PromptInfo(
            version=p.version,
            name=p.name,
            model=p.metadata.get("model"),
            description=p.metadata.get("description"),
            generation=p.generation,
        )
        for p in list_prompts()
    ]
