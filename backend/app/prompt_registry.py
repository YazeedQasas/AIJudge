"""Loads versioned prompt files from app/prompts/ and exposes their template + metadata.

A prompt file is one Markdown file (e.g. judge_v1.md) with a YAML frontmatter block
on top and the prompt template below. This module is the bridge between those files
and the code: given a version, it returns a PromptVersion holding both halves.

It does NOT decide which version is active (that's config) and does NOT touch the
request path. Its only job is: read file -> split frontmatter from template -> return.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Matches a leading YAML frontmatter block delimited by '---' fences, then the body.
# Group 1 = the YAML between the fences; Group 2 = everything after the closing fence.
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class PromptVersion:
    """One loaded prompt version: its metadata and its (still-unfilled) template."""

    version: str
    name: str
    template: str
    metadata: dict[str, Any]

    @property
    def generation(self) -> dict[str, Any]:
        """Sampling params (temperature, top_p, top_k) this version runs with, if any."""
        return self.metadata.get("generation", {})

    def render(self, *, sources_block: str, question: str) -> str:
        """Fill the template's placeholders to produce the final prompt string."""
        return self.template.format(sources_block=sources_block, question=question)


def _split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """Split a prompt file's raw text into (metadata dict, template string)."""
    match = _FRONTMATTER_RE.match(raw)
    if match is None:
        raise ValueError("Prompt file is missing a '---' YAML frontmatter block.")
    metadata = yaml.safe_load(match.group(1)) or {}
    template = match.group(2).strip()
    return metadata, template


def _load_file(path: Path) -> PromptVersion:
    """Parse one prompt file into a PromptVersion."""
    metadata, template = _split_frontmatter(path.read_text(encoding="utf-8"))
    return PromptVersion(
        version=metadata.get("version", path.stem),
        name=metadata.get("name", path.stem),
        template=template,
        metadata=metadata,
    )


def load_prompt(version: str, name: str = "judge") -> PromptVersion:
    """Load the prompt file app/prompts/{name}_{version}.md and parse it."""
    path = PROMPTS_DIR / f"{name}_{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"No prompt file at {path}")
    return _load_file(path)


def list_prompts(name: str = "judge") -> list[PromptVersion]:
    """Load every version of the given prompt family, sorted by version number."""
    paths = PROMPTS_DIR.glob(f"{name}_v*.md")
    # Sort numerically by the digits after "_v" so v10 lands after v2, not before it.
    def version_number(path: Path) -> int:
        match = re.search(r"_v(\d+)$", path.stem)
        return int(match.group(1)) if match else 0

    return [_load_file(path) for path in sorted(paths, key=version_number)]
