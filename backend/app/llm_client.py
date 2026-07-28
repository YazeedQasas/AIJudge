import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import settings


def _build_payload(
    prompt: str, generation: dict[str, Any] | None, *, stream: bool, model: str | None = None
) -> dict[str, Any]:
    """Assemble the LM Studio request body, folding in per-version sampling params.

    `generation` typically holds temperature/top_p/top_k from the active prompt version.
    top_p/temperature are OpenAI-standard; top_k is an LM Studio extra it passes through.
    `model` overrides the default served model — resolved from a model card (see
    app/model_registry.py), or set directly by the eval judge to target another model.
    """
    payload: dict[str, Any] = {
        "model": model or settings.lm_studio_model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if stream:
        payload["stream"] = True
    if generation:
        payload.update(generation)
    return payload


async def get_completion(
    prompt: str, generation: dict[str, Any] | None = None, model: str | None = None
) -> str:
    """Send a single-turn prompt to LM Studio and return the model's reply text."""
    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        response = await client.post(
            f"{settings.lm_studio_base_url}/chat/completions",
            json=_build_payload(prompt, generation, stream=False, model=model),
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def stream_completion(
    prompt: str, generation: dict[str, Any] | None = None, model: str | None = None
) -> AsyncIterator[str]:
    """Stream a completion from LM Studio, yielding text deltas as they're generated.

    Takes the same `model` override as get_completion — the two paths must be able to
    run the same request, or the Ask page and the eval could silently disagree about
    which model answered.
    """
    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        async with client.stream(
            "POST",
            f"{settings.lm_studio_base_url}/chat/completions",
            json=_build_payload(prompt, generation, stream=True, model=model),
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
