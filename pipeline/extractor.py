"""
PDF extraction utilities using the LlamaCloud SDK (llama-cloud>=1.0).

LlamaParse: AsyncLlamaCloud.parsing.parse() — tier from FRY9C_PARSE_TIER (default fast), per-page text.
LlamaExtract: AsyncLlamaCloud.extraction.extract() — concurrent across schedules.

Both steps are backed by a versioned disk cache (parse tier/version + extract config).
The SDK handles retries internally (configurable via max_retries). parse_pdf runs
asyncio.run() from a worker thread (see app.py), not nested under uvicorn's loop.

IMPORTANT: Exceptions in _extract_batch_async are now surfaced via the progress
callback and logged to stderr. Failures appear as "error" status events in
status.json instead of being silently dropped as empty results.
"""

import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from dotenv import load_dotenv

from pipeline.cache import (
    get_cached_extract,
    get_cached_parse,
    save_extract_cache,
    save_parse_cache,
)
from pipeline.extract_settings import (
    build_extract_config,
    extract_config_fingerprint,
    num_pages_context_for_schedule,
)
from pipeline.parse_config import effective_parse_tier, effective_parse_version

load_dotenv()

_PAGE_SEP = "\n\n---\n\n"


def _resolve_api_key() -> str:
    """
    Resolve the LlamaCloud API key.
    Accepts LLAMA_CLOUD_API_KEY (new), LLAMA_PARSE_API_KEY, or legacy LLAMA_API_KEY.
    """
    for var in ("LLAMA_CLOUD_API_KEY", "LLAMA_PARSE_API_KEY", "LLAMA_API_KEY"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    raise RuntimeError(
        "No LlamaCloud API key found. Set LLAMA_CLOUD_API_KEY in your .env file."
    )


LLAMA_API_KEY = _resolve_api_key()


@dataclass
class ParseResult:
    text: str
    pages: int
    size_mb: float
    elapsed_s: float
    from_cache: bool


@dataclass
class ExtractResult:
    records: list[dict[str, Any]]
    items: int
    elapsed_s: float
    from_cache: bool


async def _parse_async(pdf_path: Path) -> ParseResult:
    """Parse *pdf_path* via LlamaCloud async client (tier from effective_parse_tier())."""
    from llama_cloud import AsyncLlamaCloud  # type: ignore[import]

    t0 = time.perf_counter()
    size_mb = round(os.path.getsize(pdf_path) / 1_048_576, 2)

    cached = get_cached_parse(pdf_path)
    if cached:
        text, _meta = cached
        pages = text.count(_PAGE_SEP) + 1
        return ParseResult(
            text=text,
            pages=pages,
            size_mb=size_mb,
            elapsed_s=round(time.perf_counter() - t0, 2),
            from_cache=True,
        )

    client = AsyncLlamaCloud(api_key=LLAMA_API_KEY)

    with open(pdf_path, "rb") as fh:
        file_obj = await client.files.create(file=fh, purpose="parse")

    result = await client.parsing.parse(
        file_id=file_obj.id,
        tier=effective_parse_tier(),
        version=effective_parse_version(),
        expand=["text"],
    )

    if result.text and result.text.pages:
        text_pages = [p.text or "" for p in result.text.pages]
    else:
        text_pages = []
    text = _PAGE_SEP.join(text_pages)
    pages = len(text_pages)

    if pages > 0:
        save_parse_cache(pdf_path, text, pages)

    return ParseResult(
        text=text,
        pages=pages,
        size_mb=size_mb,
        elapsed_s=round(time.perf_counter() - t0, 2),
        from_cache=False,
    )


def parse_pdf(pdf_path: Path) -> ParseResult:
    """
    Parse *pdf_path* with LlamaParse.

    Default tier is ``fast`` (see ``pipeline.parse_config.effective_parse_tier``).
    Set ``FRY9C_PARSE_TIER`` to ``agentic`` or ``cost_effective`` when you need
    different accuracy/cost tradeoffs.
    """
    return asyncio.run(_parse_async(pdf_path))


def _load_schema(schema_path: Path) -> dict:
    with open(schema_path) as fh:
        return json.load(fh)


async def _extract_one_async(
    pdf_path: Path,
    schema_path: Path,
    kind: Literal["form", "instruction"],
    schedule_label: str,
) -> ExtractResult:
    """Extract a single schedule PDF. Called concurrently by extract_schedules_batch."""
    from llama_cloud import AsyncLlamaCloud  # type: ignore[import]

    t0 = time.perf_counter()
    cfg_fp = extract_config_fingerprint(kind)

    cached = get_cached_extract(pdf_path, schema_path, cfg_fp)
    if cached:
        records, _meta = cached
        return ExtractResult(
            records=records,
            items=len(records),
            elapsed_s=round(time.perf_counter() - t0, 2),
            from_cache=True,
        )

    schema = _load_schema(schema_path)
    # Override num_pages_context per schedule for better accuracy
    cfg = build_extract_config(kind)
    cfg["num_pages_context"] = num_pages_context_for_schedule(schedule_label)

    client = AsyncLlamaCloud(api_key=LLAMA_API_KEY)

    with open(pdf_path, "rb") as fh:
        file_obj = await client.files.create(file=fh, purpose="extract")

    run = await client.extraction.extract(
        file_id=file_obj.id,
        data_schema=schema,
        config=cfg,
        timeout=3600.0,
    )

    records = _unwrap_run_data(run)

    if records:
        save_extract_cache(pdf_path, schema_path, records, cfg_fp)

    return ExtractResult(
        records=records,
        items=len(records),
        elapsed_s=round(time.perf_counter() - t0, 2),
        from_cache=False,
    )


async def _extract_batch_async(
    tasks: list[tuple[Path, Path, Literal["form", "instruction"], str]],
    progress_cb: Callable[[str, str, "ExtractResult | Exception"], None] | None = None,
) -> dict[str, ExtractResult]:
    """
    Run extraction for all (pdf, schema, kind, label) tuples concurrently.
    Returns {schedule_label: ExtractResult}.

    Exceptions are now surfaced via progress_cb and logged to stderr rather than
    silently dropped. Each failed schedule is stored as an empty ExtractResult so
    the pipeline can continue, but the error is visible in status.json.
    """
    async def _run_one(pdf: Path, schema: Path, kind, label: str):
        try:
            result = await _extract_one_async(pdf, schema, kind, label)
            if progress_cb:
                progress_cb(label, kind, result)
            return label, result
        except Exception as exc:
            tb = traceback.format_exc()
            print(
                f"[LlamaExtract ERROR] {label} ({kind}): {exc}\n{tb}",
                file=sys.stderr,
                flush=True,
            )
            if progress_cb:
                progress_cb(label, kind, exc)
            return label, ExtractResult(records=[], items=0, elapsed_s=0.0, from_cache=False)

    coros = [_run_one(pdf, schema, kind, label) for pdf, schema, kind, label in tasks]
    results = await asyncio.gather(*coros)

    out: dict[str, ExtractResult] = {}
    for label, result in results:
        out[label] = result
    return out


def extract_schedules_batch(
    tasks: list[tuple[Path, Path, Literal["form", "instruction"], str]],
    progress_cb: Callable[[str, str, "ExtractResult | Exception"], None] | None = None,
) -> dict[str, ExtractResult]:
    """
    Extract multiple schedule PDFs concurrently.

    tasks = list of (pdf_path, schema_path, kind, schedule_label).
    Returns {schedule_label: ExtractResult}.

    Uses asyncio.gather for concurrent API calls; cache hits are immediate.
    """
    return asyncio.run(_extract_batch_async(tasks, progress_cb))


def extract_schedule(
    pdf_path: Path,
    schema_path: Path,
    kind: Literal["form", "instruction"],
    schedule_label: str = "",
    progress_cb: Callable[[str], None] | None = None,
) -> ExtractResult:
    """
    Extract a single schedule PDF (convenience wrapper over the batch path).
    Kept for backward compatibility; prefer extract_schedules_batch for speed.
    """
    if progress_cb:
        progress_cb(f"  Calling LlamaExtract: {pdf_path.name}")
    results = extract_schedules_batch([(pdf_path, schema_path, kind, schedule_label or pdf_path.stem)])
    result = results.get(schedule_label or pdf_path.stem)
    if result is None:
        result = ExtractResult(records=[], items=0, elapsed_s=0.0, from_cache=False)
    if progress_cb and result.from_cache:
        progress_cb(f"  Cache hit: {pdf_path.name} ({result.items} items)")
    return result


def _list_of_dict_records(v: Any) -> list[dict[str, Any]] | None:
    """If *v* is a non-empty list of dict-like objects, return plain dicts; else None."""
    if not isinstance(v, list) or not v:
        return None
    if not all(isinstance(x, dict) for x in v):
        return None
    return [_to_dict(item) for item in v]


def _records_from_mapping(d: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Find the first list-of-dicts in common LlamaExtract response shapes."""
    for key in (
        "items",
        "line_items",
        "records",
        "data",
        "rows",
        "results",
        "output",
        "table_rows",
        "extracted_data",
    ):
        if key in d:
            got = _list_of_dict_records(d[key])
            if got is not None:
                return got
    # Nested object (e.g. {"extraction": {"rows": [...]}})
    for v in d.values():
        if isinstance(v, dict):
            nested = _records_from_mapping(v)
            if nested:
                return nested
        got = _list_of_dict_records(v)
        if got is not None:
            return got
    return None


def _unwrap_run_data(run: Any) -> list[dict[str, Any]]:
    """
    Unwrap ``JobGetResultResponse`` (or dict) to a flat list of line-item dicts.

    Primary field is ``data`` (list or object with known array keys). Some API
    versions place row arrays under ``extraction_metadata`` instead; we scan
    both to avoid silently saving empty extractions.
    """
    if isinstance(run, dict):
        data = run.get("data")
        meta = run.get("extraction_metadata")
    else:
        data = getattr(run, "data", None)
        meta = getattr(run, "extraction_metadata", None)

    if data is None and meta is None:
        return []

    if isinstance(data, list):
        out: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                out.append(_to_dict(item))
            elif hasattr(item, "model_dump") or hasattr(item, "dict"):
                out.append(_to_dict(item))
            # Skip scalars / malformed entries — avoids TypeError from vars(str).
        return out

    if isinstance(data, dict):
        from_data = _records_from_mapping(data)
        if from_data:
            return from_data
        # Single extracted object (not PER_TABLE_ROW batch) — keep as one row
        return [data]

    if data is not None:
        return [_to_dict(data)]

    if isinstance(meta, dict):
        from_meta = _records_from_mapping(meta)
        if from_meta:
            return from_meta

    return []


def _to_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return vars(obj)
