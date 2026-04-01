"""
Single source of truth for LlamaParse tier and version.

Used by the async parser, parse cache keys, job index.json, and progress events
so audit metadata always matches the values passed to the LlamaCloud API.
"""

from __future__ import annotations

import os


def effective_parse_tier() -> str:
    """LlamaParse tier string passed to ``parsing.parse`` (default: ``fast``)."""
    return os.getenv("FRY9C_PARSE_TIER", "fast").strip().lower()


def effective_parse_version() -> str:
    """LlamaParse API version label (default: ``latest``)."""
    return os.getenv("FRY9C_PARSE_VERSION", "latest").strip()
