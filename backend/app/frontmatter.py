"""Parsing for the '--- YAML --- body' file format used by the registries.

Two directories use the same file shape: app/prompts/*.md (prompt versions) and
app/models/*.md (model cards). Each is a YAML block between '---' fences on top,
with free text below. That shared mechanic lives here so neither registry has to
reach into the other for it.

This module knows nothing about prompts or models — it only splits a string.
"""

import re
from typing import Any

import yaml

# Matches a leading YAML frontmatter block delimited by '---' fences, then the body.
# Group 1 = the YAML between the fences; Group 2 = everything after the closing fence.
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


def split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """Split raw file text into (metadata dict, body string).

    Raises ValueError if the file has no frontmatter block — a silent empty dict
    would turn a typo'd fence into a mysteriously unconfigured prompt or model.
    """
    match = _FRONTMATTER_RE.match(raw)
    if match is None:
        raise ValueError("File is missing a '---' YAML frontmatter block.")
    metadata = yaml.safe_load(match.group(1)) or {}
    body = match.group(2).strip()
    return metadata, body
