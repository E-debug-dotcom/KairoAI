"""
utils/helpers.py — Shared utility functions used across modules.
"""

import re
import unicodedata
from typing import Any


def clean_text(text: str) -> str:
    """
    Normalize and clean raw text extracted from documents.
    - Strips extra whitespace and blank lines
    - Normalizes unicode (e.g., smart quotes → ASCII)
    - Removes non-printable characters
    """
    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)

    # Replace common non-ASCII punctuation
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Remove non-printable characters (keep newlines and tabs)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)

    # Collapse multiple spaces and blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def truncate_text(text: str, max_chars: int = 8000) -> str:
    """
    Truncate text to fit within LLM context limits.
    Trims at a sentence boundary where possible.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Try to end at last sentence boundary
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1]

    return truncated + "..."


def extract_json_block(text: str) -> str:
    """
    Extract JSON content from an LLM response that may include
    surrounding explanation text or markdown code fences.
    """
    # Try to find JSON fenced block first
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fence_match:
        return fence_match.group(1).strip()

    # Try to find raw JSON object or array
    json_match = re.search(r"(\{[\s\S]+\}|\[[\s\S]+\])", text)
    if json_match:
        return json_match.group(1).strip()

    return text.strip()


def safe_dict_get(d: dict, *keys: str, default: Any = None) -> Any:
    """
    Safe nested dictionary access.
    Example: safe_dict_get(data, "results", "score", default=0)
    """
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def format_list_as_bullets(items: list[str]) -> str:
    """Convert a list of strings to a markdown bullet list."""
    return "\n".join(f"• {item}" for item in items)
