"""
LlamaExtract configuration aligned with the FRY9C Extraction Guide (LlamaCloud).

Model / mode resolution
-----------------------
LlamaCloud PREMIUM mode uses its own model routing with LlamaCloud-specific slugs
(e.g. "openai-gpt-4-1"). Azure OpenAI deployment names (e.g. "gpt-5.4") are NOT
valid LlamaCloud extract model identifiers and will be rejected server-side.

Resolution order for extract_model:
  1. ``LLAMA_EXTRACT_MODEL`` — must be a valid LlamaCloud slug (e.g. "openai-gpt-4-1")
  2. Fall back to ``LLAMACLOUD_DEFAULT_EXTRACT_MODEL`` ("openai-gpt-4-1")

Mode resolution:
  - PREMIUM when ``LLAMA_EXTRACT_MODEL`` is set or ``LLAMA_EXTRACT_MODE=PREMIUM``
  - Otherwise MULTIMODAL (LlamaCloud default; handles visually-rich FR Y-9C schedules)

Azure OpenAI credentials in the environment (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
AZURE_OPENAI_DEPLOYMENT) are for direct Azure OpenAI usage elsewhere in the app and are
deliberately NOT used to configure LlamaCloud extraction.

Best practices applied:
  - PREMIUM mode with openai-gpt-4-1 (extract) + anthropic-haiku-4.5 (parse) for dense tables
  - MULTIMODAL for visually-rich schedules when no extract model is explicitly configured
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

# Default LlamaCloud extract model slug for PREMIUM mode.
# Valid options: "openai-gpt-4-1", "openai-gpt-5-mini", "openai-gpt-5"
# NOTE: Azure deployment names (e.g. "gpt-5.4") are NOT valid here.
LLAMACLOUD_DEFAULT_EXTRACT_MODEL = "openai-gpt-4-1"

# Default LlamaCloud parse model slug for PREMIUM mode (no extra credits).
# This is the PREMIUM default — provides advanced OCR + complex table detection.
LLAMACLOUD_DEFAULT_PARSE_MODEL = "anthropic-haiku-4.5"

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
    Resolve the LlamaCloud extract model slug.

    Resolution order:
      1. ``LLAMA_EXTRACT_MODEL`` env var — must be a valid LlamaCloud slug
         (e.g. "openai-gpt-4-1", "openai-gpt-5-mini", "openai-gpt-5")
      2. ``LLAMACLOUD_DEFAULT_EXTRACT_MODEL`` ("openai-gpt-4-1")

    Azure OpenAI deployment names (e.g. "gpt-5.4") are NOT valid here.
    """
    explicit = os.environ.get("LLAMA_EXTRACT_MODEL", "").strip()
    if explicit:
        return explicit
    return LLAMACLOUD_DEFAULT_EXTRACT_MODEL


def _effective_mode() -> str:
    """
    Return the mode that will actually be used:
    - PREMIUM when ``LLAMA_EXTRACT_MODEL`` is explicitly set (extract model implies PREMIUM)
    - PREMIUM when ``LLAMA_EXTRACT_MODE=PREMIUM`` is explicitly set
    - Otherwise → MULTIMODAL (LlamaCloud default; good for visually-rich FR Y-9C schedules)
    """
    if os.environ.get("LLAMA_EXTRACT_MODEL", "").strip():
        return "PREMIUM"
    configured = os.getenv("LLAMA_EXTRACT_MODE", "").strip().upper()
    if configured == "PREMIUM":
        return "PREMIUM"
    return resolve_extract_mode()


def build_extract_config(kind: Literal["form", "instruction"]) -> dict:
    """
    Build an ``ExtractConfigParam``-compatible dict for LlamaExtract.

    Uses PREMIUM mode (with openai-gpt-4-1 + anthropic-haiku-4.5) when
    ``LLAMA_EXTRACT_MODEL`` or ``LLAMA_EXTRACT_MODE=PREMIUM`` is configured,
    otherwise MULTIMODAL.

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
        cfg["parse_model"] = LLAMACLOUD_DEFAULT_PARSE_MODEL

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
