import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.generation import build_prompt, find_invalid_citations
from app.llm_client import get_completion, stream_completion
from app.model_registry import ModelVersion, load_model
from app.prompt_registry import PromptVersion, load_prompt
from app.retrieval import embed_query, retrieve_chunks, search_query

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    limit: int = 5
    # Optional per-request overrides. When None, the settings defaults are used.
    # These are the two axes an answer varies along: which wording, which weights.
    prompt_version: str | None = None
    model_id: str | None = None


class Source(BaseModel):
    number: int
    source: str
    chunk_index: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    invalid_citations: list[int] = []
    # What produced this answer. Both None for a refusal — the gate fires before any
    # generation happens, so no prompt and no model were involved.
    prompt_version: str | None = None
    model_id: str | None = None


def _resolve(request: AskRequest) -> tuple[PromptVersion, ModelVersion, dict[str, Any]]:
    """Resolve which prompt and which model this request runs on, plus its sampling params.

    Sampling normally belongs to the prompt version, because the same wording behaves
    differently at different settings. A model card can override it (a fine-tune may
    want different settings than the wording was tuned at), which is why the merge
    order is model-wins-over-prompt and not the reverse.
    """
    prompt_version = load_prompt(request.prompt_version or settings.active_prompt_version)
    model_version = load_model(request.model_id or settings.active_model_id)
    generation = {**prompt_version.generation, **model_version.generation_overrides}
    return prompt_version, model_version, generation


def _build_sources(chunks: list[dict]) -> list[Source]:
    return [
        Source(
            number=i + 1,
            source=chunk["payload"]["source"],
            chunk_index=chunk["payload"]["chunk_index"],
            score=chunk["score"],
        )
        for i, chunk in enumerate(chunks)
    ]


@router.post("/ask")
async def ask(request: AskRequest) -> AskResponse:
    chunks = await retrieve_chunks(request.question, limit=request.limit)

    if not chunks or chunks[0]["score"] < settings.min_relevance_score:
        return AskResponse(
            answer="لا تتوفّر لديّ معلومات كافية في المصادر للإجابة عن هذا السؤال.",
            sources=[],
        )

    prompt_version, model_version, generation = _resolve(request)
    # The *full* question goes to the model, never the segmented form. Segmenting
    # decides what to retrieve; the model still needs every fact to reason about.
    prompt = build_prompt(request.question, chunks, prompt_version)
    answer = await get_completion(prompt, generation, model=model_version.model)
    invalid_citations = find_invalid_citations(answer, len(chunks))

    return AskResponse(
        answer=answer,
        sources=_build_sources(chunks),
        invalid_citations=invalid_citations,
        prompt_version=prompt_version.version,
        model_id=model_version.id,
    )


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event. The trailing blank line is part of the SSE
    wire format itself -- without it, the client never sees the event as complete."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _ask_stream_events(request: AskRequest) -> AsyncIterator[str]:
    yield _sse("stage", {"stage": "embedding"})
    vectors = await embed_query(request.question)

    yield _sse("stage", {"stage": "searching"})
    chunks = await search_query(vectors, limit=request.limit)

    if not chunks or chunks[0]["score"] < settings.min_relevance_score:
        yield _sse(
            "refused",
            {"answer": "لا تتوفّر لديّ معلومات كافية في المصادر للإجابة عن هذا السؤال."},
        )
        return

    yield _sse("stage", {"stage": "generating"})
    prompt_version, model_version, generation = _resolve(request)
    prompt = build_prompt(request.question, chunks, prompt_version)

    full_answer = ""
    async for delta in stream_completion(prompt, generation, model=model_version.model):
        full_answer += delta
        yield _sse("token", {"text": delta})

    invalid_citations = find_invalid_citations(full_answer, len(chunks))
    sources = [source.model_dump() for source in _build_sources(chunks)]
    yield _sse(
        "done",
        {
            "sources": sources,
            "invalid_citations": invalid_citations,
            # Report the exact prompt, model, and params that were just used, so the UI
            # shows what actually ran — not a separately-fetched list that can drift.
            "prompt_version": prompt_version.version,
            "model_id": model_version.id,
            # The resolved served identifier, i.e. the string that went out on the wire.
            # This used to read the prompt frontmatter's `model:`, which only records
            # what the wording was *tuned against* — a declared value, free to drift
            # from the model actually answering.
            "model": model_version.model,
            "generation": generation,
        },
    )


@router.post("/ask/stream")
async def ask_stream(request: AskRequest) -> StreamingResponse:
    return StreamingResponse(_ask_stream_events(request), media_type="text/event-stream")
