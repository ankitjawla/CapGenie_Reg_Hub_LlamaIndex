"""
LlamaExtract configuration aligned with the FRY9C Extraction Guide (LlamaCloud).

Model / mode resolution
-----------------------
LLAMA_EXTRACT_MODE env  → MULTIMODAL (default) or PREMIUM / BALANCED / FAST
LLAMA_EXTRACT_MODEL env → explicit slug (only effective in PREMIUM mode)
                          Falls back to AZURE_OPENAI_DEPLOYMENT, then guide default.

Best practices applied:
  - PREMIUM mode when a model is explicitly configured (only mode that allows it)
  - MULTIMODAL when no model override (handles visually rich FR Y-9C schedules)
  - confidence_scores + cite_sources enabled for auditability
  - chunk_mode: SECTION for instruction PDFs, PAGE for form PDFs
  - num_pages_context per schedule size class (see num_pages_context_for_schedule)
  - Field-level hints belong in schema descriptions; system_prompt scopes the task

See: https://developers.llamaindex.ai/python/cloud/llamaextract/features/options/
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

DEFAULT_GUIDE_EXTRACT_MODEL = "openai-gpt-4-1"
_VALID_MODES = {"FAST", "BALANCED", "MULTIMODAL", "PREMIUM"}

FORM_SYSTEM_PROMPT = (
    "You are a regulatory data extraction specialist. Extract every line item from this "
    "FR Y-9C form schedule exactly as printed. Preserve the exact reference number "
    "formatting including dots, parentheses, and memoranda prefixes (e.g. 'M.'). "
    "Include all rows — individual items and totals alike. Do not skip footnote-gated items."
)

INSTRUCTION_SYSTEM_PROMPT = (
    "You are a regulatory compliance analyst. Extract structured instruction entries from "
    "this FR Y-9C instruction schedule. Each entry starts with a bold heading such as "
    "'Item 1.a.' or 'Line Item M9(g)'. Extract only top-level instruction headings — do not "
    "decompose include/exclude sub-lists into separate items. Capture the full instruction "
    "text for each heading without truncation."
)


def resolve_extract_mode() -> str:
    """Return the extraction mode from env; defaults to MULTIMODAL."""
    mode = os.getenv("LLAMA_EXTRACT_MODE", "MULTIMODAL").strip().upper()
    if mode not in _VALID_MODES:
        mode = "MULTIMODAL"
    return mode


def resolve_extract_model() -> str:
    """
    Resolve ``extract_model`` slug.

    Order: ``LLAMA_EXTRACT_MODEL`` → ``AZURE_OPENAI_DEPLOYMENT`` → guide default.
    Only meaningful when extraction_mode is PREMIUM.
    """
    explicit = os.environ.get("LLAMA_EXTRACT_MODEL", "").strip()
    if explicit:
        return explicit
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    if deployment:
        return deployment
    return DEFAULT_GUIDE_EXTRACT_MODEL


def _effective_mode() -> str:
    """
    Return the mode that will actually be used:
    - If an explicit extract model is configured → PREMIUM (only mode that accepts it)
    - Otherwise → whatever LLAMA_EXTRACT_MODE says (default MULTIMODAL)
    """
    model_explicitly_set = bool(
        os.environ.get("LLAMA_EXTRACT_MODEL", "").strip()
        or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    )
    if model_explicitly_set:
        return "PREMIUM"
    return resolve_extract_mode()


def build_extract_config(kind: Literal["form", "instruction"]) -> dict:
    """
    Build an ``ExtractConfigParam``-compatible dict for LlamaExtract.

    Uses PREMIUM mode when an extract model is explicitly configured, MULTIMODAL otherwise.
    Differences by kind:
      - form:        chunk_mode=PAGE (table-heavy; each page processed independently)
      - instruction: chunk_mode=SECTION (narrative; semantic section boundaries)
    """
    mode = _effective_mode()
    system_prompt = FORM_SYSTEM_PROMPT if kind == "form" else INSTRUCTION_SYSTEM_PROMPT

    cfg: dict = {
        "extraction_target": "PER_TABLE_ROW",
        "extraction_mode": mode,
        "high_resolution_mode": True,
        "use_reasoning": True,
        "confidence_scores": True,
        "cite_sources": True,
        "num_pages_context": 1,
        "chunk_mode": "PAGE" if kind == "form" else "SECTION",
        "system_prompt": system_prompt,
    }

    if mode == "PREMIUM":
        cfg["extract_model"] = resolve_extract_model()

    return cfg


def num_pages_context_for_schedule(schedule_label: str) -> int:
    """
    Return a sensible num_pages_context for the given schedule.

    HI / HI-A schedules are dense ordered lists → 1 page context is sufficient.
    HC sub-schedules span multiple pages with cross-page tables → use 2.
    Unclassified → 1.
    """
    if not schedule_label.startswith("Schedule_"):
        return 1
    s = schedule_label.replace("Schedule_", "")
    if s in ("HI", "HI-A", "HI-B", "HI-C"):
        return 1
    if s.startswith("HC"):
        return 2
    return 1


def extract_config_fingerprint(kind: Literal["form", "instruction"]) -> str:
    """Stable hash of the active extract config (incl. resolved mode/model) for cache keys."""
    cfg = build_extract_config(kind)
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
