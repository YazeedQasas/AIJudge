from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    source: str
    text: str


def load_documents(directory: Path) -> list[Document]:
    """Read every .txt file in a directory into a Document, tagged with its filename."""
    documents = []
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        documents.append(Document(source=path.name, text=text))
    return documents
