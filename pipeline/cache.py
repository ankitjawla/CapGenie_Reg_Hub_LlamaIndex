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

Parse keys encode: pdf content + parse tier + parse version + Azure endpoint hash.
Extract keys encode: pdf content + schema text + extract config fingerprint.

Set CACHE_MAX_AGE_DAYS to evict stale entries (default 0 = never evict locally).
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.parse_config import effective_parse_tier, effective_parse_version

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


def _cache_max_age_days() -> float:
    """Return configured max age in days; 0 means no TTL."""
    try:
        return float(os.getenv("CACHE_MAX_AGE_DAYS", "0"))
    except ValueError:
        return 0.0


def _is_stale(cached_at_iso: str) -> bool:
    """Return True if the entry is older than CACHE_MAX_AGE_DAYS (when > 0)."""
    max_age = _cache_max_age_days()
    if max_age <= 0:
        return False
    try:
        cached_at = datetime.fromisoformat(cached_at_iso)
        age = (datetime.now(timezone.utc) - cached_at).total_seconds() / 86400
        return age > max_age
    except Exception:
        return False


def parse_options_fingerprint() -> str:
    """
    Stable token for current LlamaParse options (invalidates parse cache when changed).

    Encodes: parse tier, parse version, and actual Azure endpoint value (if set).
    Defaults match pipeline.parse_config (same as the live parse API call).
    """
    tier = effective_parse_tier()
    version = effective_parse_version()
    ep = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    dep = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    ver = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
    payload = json.dumps(
        {
            "tier": tier,
            "version": version,
            "azure_endpoint": _sha256_str(ep)[:12] if ep else "",
            "azure_deployment": dep,
            "azure_api_version": ver,
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
    """Return ``(page_text, meta)`` if cached and not stale, else ``None``."""
    key = parse_cache_key(pdf_path)
    txt = PARSE_CACHE / f"{key}.txt"
    meta_path = PARSE_CACHE / f"{key}.meta.json"
    if txt.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if _is_stale(meta.get("cached_at", "")):
            return None
        return txt.read_text(encoding="utf-8"), meta
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
    """Return ``(records, meta)`` if cached and not stale, else ``None``."""
    key = extract_cache_key(pdf_path, schema_path, extract_config_fp)
    data_path = EXTRACT_CACHE / f"{key}.json"
    meta_path = EXTRACT_CACHE / f"{key}.meta.json"
    if data_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if _is_stale(meta.get("cached_at", "")):
            return None
        return json.loads(data_path.read_text()), meta
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
# Cache migration / integrity
# ---------------------------------------------------------------------------

def _purge_orphaned_entries() -> dict[str, int]:
    """
    Remove cache entries written by older code versions that are invisible to
    the current key-computation logic.

    Parse entries without ``parse_profile`` use the old (no-fingerprint) key
    and will never match the current ``parse_cache_key()``.

    Extract entries without ``extract_config_fingerprint`` use the old key
    and will never match the current ``extract_cache_key()``.

    Returns counts of deleted entries.
    """
    deleted: dict[str, int] = {"parse": 0, "extract": 0}

    if PARSE_CACHE.exists():
        for meta_f in list(PARSE_CACHE.glob("*.meta.json")):
            try:
                m = json.loads(meta_f.read_text())
                if "parse_profile" not in m:
                    stem = meta_f.stem.replace(".meta", "")
                    txt = PARSE_CACHE / f"{stem}.txt"
                    meta_f.unlink(missing_ok=True)
                    txt.unlink(missing_ok=True)
                    deleted["parse"] += 1
            except Exception:
                pass

    if EXTRACT_CACHE.exists():
        for meta_f in list(EXTRACT_CACHE.glob("*.meta.json")):
            try:
                m = json.loads(meta_f.read_text())
                if "extract_config_fingerprint" not in m:
                    stem = meta_f.stem.replace(".meta", "")
                    data = EXTRACT_CACHE / f"{stem}.json"
                    meta_f.unlink(missing_ok=True)
                    data.unlink(missing_ok=True)
                    deleted["extract"] += 1
            except Exception:
                pass

    return deleted


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
        "max_age_days": _cache_max_age_days(),
    }
