import httpx

from app.config import settings


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning one vector per input, in the same order."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.lm_studio_base_url}/embeddings",
            json={"model": settings.lm_studio_embedding_model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()

    ordered = sorted(data["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in ordered]
