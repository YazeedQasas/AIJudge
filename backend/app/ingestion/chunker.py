from dataclasses import dataclass

from app.ingestion.loader import Document


@dataclass
class Chunk:
    source: str
    text: str
    index: int


def chunk_document(document: Document, chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split a document's text into fixed-size, overlapping chunks."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    index = 0
    while start < len(document.text):
        end = start + chunk_size
        chunk_text = document.text[start:end]
        chunks.append(Chunk(source=document.source, text=chunk_text, index=index))
        start += chunk_size - overlap
        index += 1
    return chunks
