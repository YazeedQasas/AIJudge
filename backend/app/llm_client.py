import json
from collections.abc import AsyncIterator

import httpx

from app.config import settings


async def get_completion(prompt: str) -> str:
    """Send a single-turn prompt to LM Studio and return the model's reply text."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.lm_studio_base_url}/chat/completions",
            json={
                "model": settings.lm_studio_model,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def stream_completion(prompt: str) -> AsyncIterator[str]:
    """Stream a completion from LM Studio, yielding text deltas as they're generated."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            f"{settings.lm_studio_base_url}/chat/completions",
            json={
                "model": settings.lm_studio_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line.removeprefix("data: ")
                if payload == "[DONE]":
                    break
                delta = json.loads(payload)["choices"][0]["delta"].get("content")
                if delta:
                    yield delta


async def is_lm_studio_healthy() -> bool:
    """Check whether LM Studio's server is reachable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{settings.lm_studio_base_url}/models")
            return response.status_code == 200
    except httpx.RequestError:
        return False
