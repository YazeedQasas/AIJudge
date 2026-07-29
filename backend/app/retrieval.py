"""Turning a question into the chunks that answer it.

Split out of the API layer because a long question is not one query. Embedding a
1,200-character case narrative produces the *average* of everything in it — names,
dates, amounts, procedural history — and that average is a weak match for every
chunk in the corpus. A real case filed against this pipeline scored 0.488 as one
blob and was refused by the relevance gate, even though the top hit was the
correct document. Embedding its paragraphs separately and keeping the best hit
per chunk scored 0.684 on the same corpus.

Short questions are left strictly alone (see `segment_query`), so the everyday
one-line question follows exactly the same path it always has.
"""

import asyncio
import re

from app.embedding_client import embed_texts
from app.vector_store import search

# Below this length a question is treated as a single query. A one-line question
# has nothing to dilute, so splitting it would only add embedding calls and give
# the merge a chance to reorder results that are already correct.
MULTI_QUERY_MIN_CHARS = 400

# A segment longer than this is split again on sentence boundaries. Wall-of-text
# input with no blank lines would otherwise come back as one diluted segment,
# which is the exact problem this module exists to solve.
SEGMENT_CHAR_LIMIT = 600

# Ceiling on segments per question, and so on searches per question. Guards against
# a pathologically long paste turning one request into hundreds of vector searches.
MAX_SEGMENTS = 12

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
# Split *after* a sentence ender so the punctuation stays with its own sentence.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?؟؛\n])\s*")


def _hard_split(text: str, limit: int) -> list[str]:
    """Last-resort slicing for text with no sentence punctuation at all."""
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def _split_long(text: str, limit: int) -> list[str]:
    """Pack an over-long paragraph into sentence-aligned windows of at most `limit`."""
    if len(text) <= limit:
        return [text]

    windows: list[str] = []
    current = ""
    for sentence in _SENTENCE_BREAK.split(text):
        if not sentence:
            continue
        # A single sentence over the limit can't be packed; slice it and move on.
        if len(sentence) > limit:
            if current:
                windows.append(current)
                current = ""
            windows.extend(_hard_split(sentence, limit))
            continue

        if current and len(current) + 1 + len(sentence) > limit:
            windows.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence

    if current:
        windows.append(current)
    return windows


def segment_query(question: str) -> list[str]:
    """Break a question into the independent queries it should actually be searched as.

    Deliberately structural — blank lines, then sentence boundaries. An earlier
    design looked for a trailing "المطلوب:" marker and searched only that, which
    fails twice over: narratives that never state a closing question retrieve
    nothing, and the discarded facts turn out to carry legal issues the closing
    question doesn't name. On the case above, the paragraph describing a delay in
    seeking treatment retrieved the contributory-negligence chunk that scored
    highest of anything in the merge — a marker-based reading would have dropped it.
    """
    text = question.strip()
    if len(text) <= MULTI_QUERY_MIN_CHARS:
        return [text] if text else []

    segments: list[str] = []
    seen: set[str] = set()
    for paragraph in _PARAGRAPH_BREAK.split(text):
        stripped = paragraph.strip()
        if not stripped:
            continue
        for segment in _split_long(stripped, SEGMENT_CHAR_LIMIT):
            segment = segment.strip()
            # Repeated boilerplate (headings, party captions) would otherwise be
            # embedded and searched more than once for the same result.
            if segment and segment not in seen:
                seen.add(segment)
                segments.append(segment)

    # Falling back to the whole text keeps this total: no input can produce zero queries.
    return segments[:MAX_SEGMENTS] or [text]


def merge_by_max(result_sets: list[list[dict]], limit: int) -> list[dict]:
    """Union several searches, scoring each chunk by its *best* segment match.

    Max rather than sum, because a chunk that answers one paragraph decisively is
    more useful than one weakly related to several. Summing would reward chunks
    that match the narrative's generic legal vocabulary everywhere and nothing in
    particular.

    The `limit` cut is what keeps this from trading a diluted embedding for a
    diluted prompt: the model still sees at most `limit` chunks, exactly as before.
    """
    best: dict[str | int, dict] = {}
    for results in result_sets:
        for chunk in results:
            existing = best.get(chunk["id"])
            if existing is None or chunk["score"] > existing["score"]:
                best[chunk["id"]] = chunk

    ranked = sorted(best.values(), key=lambda chunk: chunk["score"], reverse=True)
    return ranked[:limit]


async def embed_query(question: str) -> list[list[float]]:
    """Segment a question and embed every segment in one batched call.

    Separate from `search_query` only so the streaming endpoint can report the two
    phases independently — embedding is the slow one when the embedding model has
    to be loaded first, and it would otherwise look like a hang.
    """
    segments = segment_query(question)
    return await embed_texts(segments) if segments else []


async def search_query(vectors: list[list[float]], limit: int = 5) -> list[dict]:
    """Search one vector per segment concurrently, then merge into a single ranking."""
    if not vectors:
        return []
    result_sets = await asyncio.gather(*(search(vector, limit=limit) for vector in vectors))
    return merge_by_max(list(result_sets), limit)


async def retrieve_chunks(question: str, limit: int = 5) -> list[dict]:
    """Retrieve the chunks for a question, searching it as one query or several.

    Single-segment questions return exactly what a plain embed-and-search returns,
    so this is a no-op for everything the eval set covers.
    """
    return await search_query(await embed_query(question), limit)
