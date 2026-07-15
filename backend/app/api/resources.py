from fastapi import APIRouter
from pydantic import BaseModel

from app.vector_store import scroll_all_points

router = APIRouter()


class ChunkOut(BaseModel):
    id: int
    chunk_index: int
    text: str


class DocumentOut(BaseModel):
    source: str
    chunks: list[ChunkOut]


class ResourcesResponse(BaseModel):
    documents: list[DocumentOut]


@router.get("/resources")
async def list_resources() -> ResourcesResponse:
    points = await scroll_all_points()

    chunks_by_source: dict[str, list[ChunkOut]] = {}
    for point in points:
        payload = point["payload"]
        source = payload["source"]
        chunk = ChunkOut(id=point["id"], chunk_index=payload["chunk_index"], text=payload["text"])
        chunks_by_source.setdefault(source, []).append(chunk)

    documents = [
        DocumentOut(source=source, chunks=sorted(chunks, key=lambda c: c.chunk_index))
        for source, chunks in sorted(chunks_by_source.items())
    ]
    return ResourcesResponse(documents=documents)
