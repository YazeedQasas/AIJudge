from fastapi import APIRouter
from pydantic import BaseModel

from app.embedding_client import embed_texts
from app.vector_store import search

router = APIRouter()


class RetrieveRequest(BaseModel):
    query: str
    limit: int = 5
    source: str | None = None


class RetrievedChunk(BaseModel):
    score: float
    text: str
    source: str
    chunk_index: int


class RetrieveResponse(BaseModel):
    results: list[RetrievedChunk]


@router.post("/retrieve")
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    [vector] = await embed_texts([request.query])
    raw_results = await search(vector, limit=request.limit, source=request.source)

    results = [
        RetrievedChunk(
            score=item["score"],
            text=item["payload"]["text"],
            source=item["payload"]["source"],
            chunk_index=item["payload"]["chunk_index"],
        )
        for item in raw_results
    ]
    return RetrieveResponse(results=results)
