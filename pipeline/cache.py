"""
Disk-based SHA-256 content-addressed cache for expensive pipeline steps.

Cache layout
------------
.cache/
  parse/
    <sha256_of_pdf+parse_profile>.txt        ← LlamaParse page-separated text
    <sha256_of_pdf+parse_profile>.meta.json
  extract/
    <sha256_of_pdf+schema+extract_cfg>.json  ← list of extracted records
    <sha256_of_pdf+schema+extract_cfg>.meta.json

Parse keys version when result_type, fast_mode, or parse Azure options change.
Extract keys version when schema or serialized LlamaExtract config (incl. model) changes.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).parent.parent / ".cache"
PARSE_CACHE = CACHE_DIR / "parse"
EXTRACT_CACHE = CACHE_DIR / "extract"


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def parse_options_fingerprint() -> str:
    """
    Stable token for current LlamaParse options (invalidates parse cache when changed).

    Default: markdown + fast_mode (FRY9C_PARSE_FAST_MODE=true).
    Set FRY9C_PARSE_FAST_MODE=false for plain text / non-fast experiments.
    Includes whether Azure OpenAI env is present for parse (LlamaParse may route there).
    """
    fast_md = os.getenv("FRY9C_PARSE_FAST_MODE", "true").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    ep = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    key = os.getenv("AZURE_OPENAI_API_KEY", "").strip() or os.getenv("AZURE_OPENAI_KEY", "").strip()
    parse_azure = bool(ep and key)
    payload = json.dumps(
        {
            "result_type": "markdown" if fast_md else "text",
            "fast_mode": fast_md,
            "parse_azure": parse_azure,
        },
        sort_keys=True,
    )
    return _sha256_str(payload)


# ---------------------------------------------------------------------------
# Parse cache
# ---------------------------------------------------------------------------

def parse_cache_key(pdf_path: Path) -> str:
    return _sha256_str(_sha256_file(pdf_path) + "|" + parse_options_fingerprint())


def get_cached_parse(pdf_path: Path) -> tuple[str, dict] | None:
    """
    Return (page_text, meta) if *pdf_path* is cached, else None.
    """
    key = parse_cache_key(pdf_path)
    txt = PARSE_CACHE / f"{key}.txt"
    meta_path = PARSE_CACHE / f"{key}.meta.json"
    if txt.exists() and meta_path.exists():
        return txt.read_text(encoding="utf-8"), json.loads(meta_path.read_text())
    return None


def save_parse_cache(pdf_path: Path, text: str, pages: int) -> dict:
    """Persist *text* to the parse cache and return the written meta dict."""
    PARSE_CACHE.mkdir(parents=True, exist_ok=True)
    key = parse_cache_key(pdf_path)
    meta = {
        "file": pdf_path.name,
        "size_bytes": os.path.getsize(pdf_path),
        "size_mb": round(os.path.getsize(pdf_path) / 1_048_576, 2),
        "pages": pages,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "cache_key": key,
        "parse_profile": parse_options_fingerprint(),
    }
    (PARSE_CACHE / f"{key}.txt").write_text(text, encoding="utf-8")
    (PARSE_CACHE / f"{key}.meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# Extract cache
# ---------------------------------------------------------------------------

def extract_cache_key(pdf_path: Path, schema_path: Path, extract_config_fp: str) -> str:
    """Cache key = SHA-256 of pdf bytes + schema text + extract config fingerprint."""
    pdf_hash = _sha256_file(pdf_path)
    schema_hash = _sha256_str(schema_path.read_text())
    return _sha256_str(pdf_hash + schema_hash + extract_config_fp)


def get_cached_extract(
    pdf_path: Path,
    schema_path: Path,
    extract_config_fp: str,
) -> tuple[list[dict], dict] | None:
    """
    Return (records, meta) if this (pdf, schema, extract config) triple is cached.
    """
    key = extract_cache_key(pdf_path, schema_path, extract_config_fp)
    data_path = EXTRACT_CACHE / f"{key}.json"
    meta_path = EXTRACT_CACHE / f"{key}.meta.json"
    if data_path.exists() and meta_path.exists():
        return json.loads(data_path.read_text()), json.loads(meta_path.read_text())
    return None


def save_extract_cache(
    pdf_path: Path,
    schema_path: Path,
    records: list[dict[str, Any]],
    extract_config_fp: str,
) -> dict:
    """Persist *records* to the extract cache and return the written meta dict."""
    EXTRACT_CACHE.mkdir(parents=True, exist_ok=True)
    key = extract_cache_key(pdf_path, schema_path, extract_config_fp)
    meta = {
        "file": pdf_path.name,
        "schema": schema_path.name,
        "items": len(records),
        "size_bytes": os.path.getsize(pdf_path),
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "cache_key": key,
        "extract_config_fingerprint": extract_config_fp,
    }
    (EXTRACT_CACHE / f"{key}.json").write_text(json.dumps(records, indent=2))
    (EXTRACT_CACHE / f"{key}.meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# Cache stats helper
# ---------------------------------------------------------------------------

def cache_stats() -> dict:
    """Return a summary of how many entries exist in each cache."""
    parse_count = len(list(PARSE_CACHE.glob("*.txt"))) if PARSE_CACHE.exists() else 0
    extract_count = len(list(EXTRACT_CACHE.glob("*.json"))) if EXTRACT_CACHE.exists() else 0
    return {
        "parse_entries": parse_count,
        "extract_entries": extract_count // 2,
        "cache_dir": str(CACHE_DIR),
    }
