import re

from app.prompt_registry import PromptVersion

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def build_sources_block(chunks: list[dict]) -> str:
    """Assemble the numbered sources text from retrieved chunks.

    Source numbers stay as ASCII [1], [2], ... so citations match the numbered
    `sources` list returned to the frontend.
    """
    return "\n\n".join(
        f"[{i + 1}] (من {chunk['payload']['source']})\n{chunk['payload']['text']}"
        for i, chunk in enumerate(chunks)
    )


def build_prompt(question: str, chunks: list[dict], prompt: PromptVersion) -> str:
    """Render the given prompt version into a final prompt: sources block + question filled in.

    The wording lives in the version's template file (app/prompts/), not here — this
    function only supplies the dynamic pieces.
    """
    sources_block = build_sources_block(chunks)
    return prompt.render(sources_block=sources_block, question=question)


def find_invalid_citations(answer: str, num_sources: int) -> list[int]:
    """Return citation numbers the model used that don't correspond to a real numbered source."""
    cited = {int(n) for n in CITATION_PATTERN.findall(answer)}
    return sorted(n for n in cited if n < 1 or n > num_sources)
