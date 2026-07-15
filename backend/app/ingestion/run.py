import asyncio
from pathlib import Path

from app.config import settings
from app.embedding_client import embed_texts
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_documents
from app.vector_store import ensure_collection, upsert_points

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "sample_corpus"


async def main() -> None:
    documents = load_documents(CORPUS_DIR)
    print(f"Loaded {len(documents)} documents from {CORPUS_DIR}")

    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document))
    print(f"Split into {len(chunks)} chunks")

    vectors = await embed_texts([chunk.text for chunk in chunks])
    print(f"Embedded {len(vectors)} chunks")

    points = [
        {
            "id": i,
            "vector": vector,
            "payload": {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.index,
            },
        }
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    await ensure_collection()
    await upsert_points(points)
    print(f"Upserted {len(points)} points into '{settings.qdrant_collection}'")


if __name__ == "__main__":
    asyncio.run(main())
