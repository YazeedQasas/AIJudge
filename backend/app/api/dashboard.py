import asyncio

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from app.llm_client import is_lm_studio_healthy
from app.vector_store import is_qdrant_healthy, scroll_all_points

router = APIRouter()


async def _scroll_all_points_or_empty() -> list[dict]:
    """Same as scroll_all_points(), but degrades to an empty list if Qdrant is unreachable
    rather than crashing the whole dashboard when is_qdrant_healthy() already reports it."""
    try:
        return await scroll_all_points()
    except httpx.RequestError:
        return []


class CorpusStats(BaseModel):
    document_count: int
    chunk_count: int


class HealthStatus(BaseModel):
    qdrant: bool
    lm_studio: bool


class DashboardResponse(BaseModel):
    corpus: CorpusStats
    health: HealthStatus


@router.get("/dashboard")
async def dashboard() -> DashboardResponse:
    points, qdrant_ok, lm_studio_ok = await asyncio.gather(
        _scroll_all_points_or_empty(),
        is_qdrant_healthy(),
        is_lm_studio_healthy(),
    )

    sources = {point["payload"]["source"] for point in points}

    return DashboardResponse(
        corpus=CorpusStats(document_count=len(sources), chunk_count=len(points)),
        health=HealthStatus(qdrant=qdrant_ok, lm_studio=lm_studio_ok),
    )
