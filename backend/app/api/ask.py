import json
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.embedding_client import embed_texts
from app.generation import build_prompt, find_invalid_citations
from app.llm_client import get_completion, stream_completion
from app.vector_store import search

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    limit: int = 5


class Source(BaseModel):
    number: int
    source: str
    chunk_index: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    invalid_citations: list[int] = []


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
    [vector] = await embed_texts([request.question])
    chunks = await search(vector, limit=request.limit)

    if not chunks or chunks[0]["score"] < settings.min_relevance_score:
        return AskResponse(
            answer="لا تتوفّر لديّ معلومات كافية في المصادر للإجابة عن هذا السؤال.",
            sources=[],
        )

    prompt = build_prompt(request.question, chunks)
    answer = await get_completion(prompt)
    invalid_citations = find_invalid_citations(answer, len(chunks))

    return AskResponse(answer=answer, sources=_build_sources(chunks), invalid_citations=invalid_citations)


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event. The trailing blank line is part of the SSE
    wire format itself -- without it, the client never sees the event as complete."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _ask_stream_events(request: AskRequest) -> AsyncIterator[str]:
    yield _sse("stage", {"stage": "embedding"})
    [vector] = await embed_texts([request.question])

    yield _sse("stage", {"stage": "searching"})
    chunks = await search(vector, limit=request.limit)

    if not chunks or chunks[0]["score"] < settings.min_relevance_score:
        yield _sse(
            "refused",
            {"answer": "لا تتوفّر لديّ معلومات كافية في المصادر للإجابة عن هذا السؤال."},
        )
        return

    yield _sse("stage", {"stage": "generating"})
    prompt = build_prompt(request.question, chunks)

    full_answer = ""
    async for delta in stream_completion(prompt):
        full_answer += delta
        yield _sse("token", {"text": delta})

    invalid_citations = find_invalid_citations(full_answer, len(chunks))
    sources = [source.model_dump() for source in _build_sources(chunks)]
    yield _sse("done", {"sources": sources, "invalid_citations": invalid_citations})


@router.post("/ask/stream")
async def ask_stream(request: AskRequest) -> StreamingResponse:
    return StreamingResponse(_ask_stream_events(request), media_type="text/event-stream")
