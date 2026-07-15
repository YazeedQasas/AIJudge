import httpx

from app.config import settings


async def ensure_collection() -> None:
    """Create the configured Qdrant collection if it doesn't already exist."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        existing = await client.get(f"{settings.qdrant_url}/collections/{settings.qdrant_collection}")
        if existing.status_code == 200:
            return

        response = await client.put(
            f"{settings.qdrant_url}/collections/{settings.qdrant_collection}",
            json={"vectors": {"size": settings.embedding_dimension, "distance": "Cosine"}},
        )
        response.raise_for_status()


async def upsert_points(points: list[dict]) -> None:
    """Write a batch of points (id, vector, payload) into the collection."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.put(
            f"{settings.qdrant_url}/collections/{settings.qdrant_collection}/points",
            json={"points": points},
        )
        response.raise_for_status()


async def search(vector: list[float], limit: int = 5, source: str | None = None) -> list[dict]:
    """Find the nearest stored points to a query vector, optionally restricted to one source document."""
    body: dict = {"vector": vector, "limit": limit, "with_payload": True}
    if source is not None:
        body["filter"] = {"must": [{"key": "source", "match": {"value": source}}]}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{settings.qdrant_url}/collections/{settings.qdrant_collection}/points/search",
            json=body,
        )
        response.raise_for_status()
        return response.json()["result"]


async def scroll_all_points(limit: int = 1000) -> list[dict]:
    """List every point in the collection (no similarity search involved)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{settings.qdrant_url}/collections/{settings.qdrant_collection}/points/scroll",
            json={"limit": limit, "with_payload": True, "with_vector": False},
        )
        response.raise_for_status()
        return response.json()["result"]["points"]


async def is_qdrant_healthy() -> bool:
    """Check whether Qdrant is reachable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(settings.qdrant_url)
            return response.status_code == 200
    except httpx.RequestError:
        return False
