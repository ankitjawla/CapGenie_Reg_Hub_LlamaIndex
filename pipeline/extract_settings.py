"""
LlamaExtract configuration aligned with the FRY9C Extraction Guide (LlamaCloud).

Model resolution (no hardcoded extract model in call sites):
  1. LLAMA_EXTRACT_MODEL — explicit LlamaCloud extract_model slug if set.
  2. AZURE_OPENAI_DEPLOYMENT — deployment name from .env when (1) is unset.
  3. openai-gpt-4-1 — guide default when neither is set.

See also: FRY9C_Package_for_Ankit-2 / FRY9C_Extraction_Guide.pdf
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

DEFAULT_GUIDE_EXTRACT_MODEL = "openai-gpt-4-1"

FORM_SYSTEM_PROMPT = (
    "You are extracting line items from a regulatory form schedule."
)

INSTRUCTION_SYSTEM_PROMPT = (
    "You are extracting structured line item instructions from an FRY9C regulatory "
    "filing instruction document. Each instruction section begins with a bold heading "
    "like 'Line Item 1(a)' or 'Line Item M9(g)' followed by a title and detailed reporting "
    "guidance. Extract only top-level instruction headings. Do not extract numbered "
    "sub-points within include/exclude lists as separate line items. Capture the full "
    "instruction text without truncation."
)


def resolve_extract_model() -> str:
    """
    Resolve ``extract_model`` for ``ExtractConfig``.

    Order: ``LLAMA_EXTRACT_MODEL`` → ``AZURE_OPENAI_DEPLOYMENT`` → guide default.
    """
    explicit = os.environ.get("LLAMA_EXTRACT_MODEL", "").strip()
    if explicit:
        return explicit
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    if deployment:
        return deployment
    return DEFAULT_GUIDE_EXTRACT_MODEL


def build_extract_config(kind: Literal["form", "instruction"]):
    """
    Build ``ExtractConfig`` per FRY9C guide: PER_TABLE_ROW, BALANCED, SECTION chunking,
    high resolution, reasoning, 1 page context, guide system prompts.
    """
    from llama_cloud.types import (  # type: ignore[import]
        DocumentChunkMode,
        ExtractConfig,
        ExtractMode,
        ExtractTarget,
    )

    system_prompt = FORM_SYSTEM_PROMPT if kind == "form" else INSTRUCTION_SYSTEM_PROMPT
    model = resolve_extract_model()

    return ExtractConfig(
        extraction_target=ExtractTarget.PER_TABLE_ROW,
        extraction_mode=ExtractMode.BALANCED,
        extract_model=model,
        chunk_mode=DocumentChunkMode.SECTION,
        high_resolution_mode=True,
        use_reasoning=True,
        num_pages_context=1,
        system_prompt=system_prompt,
    )


def extract_config_fingerprint(kind: Literal["form", "instruction"]) -> str:
    """Stable hash of the active extract config (including resolved model) for cache keys."""
    cfg = build_extract_config(kind)
    if hasattr(cfg, "model_dump"):
        data = cfg.model_dump(mode="json", exclude_none=True)
    else:
        data = cfg.dict(exclude_none=True)
    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
