"""
PDF extraction utilities using LlamaParse and LlamaExtract.

LlamaParse converts the full PDF to page-separated text or markdown (default:
markdown + fast_mode per FRY9C guide). LlamaExtract pulls structured JSON from
per-schedule PDFs using guide-aligned ExtractConfig.

Both steps use a disk cache (versioned by parse profile and extract config).
"""

import asyncio
import json
import os
import time
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
)

load_dotenv()

LLAMA_API_KEY = os.environ["LLAMA_API_KEY"]

_PAGE_SEP = "\n\n---\n\n"

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


def _llama_parse_kwargs() -> dict[str, Any]:
    """Build LlamaParse kwargs; profile must stay in sync with cache.parse_options_fingerprint."""
    from llama_cloud_services.parse.utils import ResultType

    fast_md = os.getenv("FRY9C_PARSE_FAST_MODE", "true").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    kw: dict[str, Any] = {
        "api_key": LLAMA_API_KEY,
        "result_type": ResultType.MD if fast_md else ResultType.TXT,
        "split_by_page": True,
        "verbose": False,
        "fast_mode": fast_md,
    }
    ep = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    key = os.getenv("AZURE_OPENAI_API_KEY", "").strip() or os.getenv("AZURE_OPENAI_KEY", "").strip()
    if ep and key:
        kw["azure_openai_endpoint"] = ep
        kw["azure_openai_key"] = key
        dep = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        if dep:
            kw["azure_openai_deployment_name"] = dep
        ver = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
        if ver:
            kw["azure_openai_api_version"] = ver
    return kw


async def _parse_async(pdf_path: Path) -> ParseResult:
    from llama_cloud_services import LlamaParse

    t0 = time.perf_counter()
    size_mb = round(os.path.getsize(pdf_path) / 1_048_576, 2)

    cached = get_cached_parse(pdf_path)
    if cached:
        text, meta = cached
        pages = text.count(_PAGE_SEP) + 1
        return ParseResult(
            text=text,
            pages=pages,
            size_mb=size_mb,
            elapsed_s=round(time.perf_counter() - t0, 2),
            from_cache=True,
        )

    parser = LlamaParse(**_llama_parse_kwargs())
    docs = await parser.aload_data(str(pdf_path))
    text = _PAGE_SEP.join(doc.text for doc in docs)
    pages = len(docs)

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
    Parse *pdf_path* with LlamaParse (markdown + fast_mode by default).

    Set FRY9C_PARSE_FAST_MODE=false for plain-text parsing experiments.
    """
    return asyncio.run(_parse_async(pdf_path))


def _load_schema(schema_path: Path) -> dict:
    with open(schema_path) as fh:
        return json.load(fh)


def extract_schedule(
    pdf_path: Path,
    schema_path: Path,
    kind: Literal["form", "instruction"],
    progress_cb: Callable[[str], None] | None = None,
) -> ExtractResult:
    """
    Run LlamaExtract on *pdf_path* with FRY9C guide-aligned config for *kind*.
    """
    from llama_cloud_services import LlamaExtract

    t0 = time.perf_counter()
    cfg_fp = extract_config_fingerprint(kind)

    cached = get_cached_extract(pdf_path, schema_path, cfg_fp)
    if cached:
        records, _meta = cached
        if progress_cb:
            progress_cb(f"  Cache hit: {pdf_path.name} ({len(records)} items)")
        return ExtractResult(
            records=records,
            items=len(records),
            elapsed_s=round(time.perf_counter() - t0, 2),
            from_cache=True,
        )

    if progress_cb:
        progress_cb(f"  Calling LlamaExtract: {pdf_path.name}")

    schema = _load_schema(schema_path)
    extractor = LlamaExtract(api_key=LLAMA_API_KEY)
    config = build_extract_config(kind)

    run = extractor.extract(
        data_schema=schema,
        config=config,
        files=str(pdf_path),
    )
    records = _unwrap_run_data(run)
    save_extract_cache(pdf_path, schema_path, records, cfg_fp)

    return ExtractResult(
        records=records,
        items=len(records),
        elapsed_s=round(time.perf_counter() - t0, 2),
        from_cache=False,
    )


def _unwrap_run_data(run: Any) -> list[dict[str, Any]]:
    """Unwrap ExtractRun (or list) to a flat list of plain dicts."""
    runs = run if isinstance(run, list) else [run]
    records: list[dict] = []

    for r in runs:
        data = getattr(r, "data", None)
        if data is None:
            continue
        if isinstance(data, list):
            for item in data:
                records.append(_to_dict(item))
        elif isinstance(data, dict):
            inner = None
            for key in ("items", "line_items", "records", "data"):
                if key in data and isinstance(data[key], list):
                    inner = data[key]
                    break
            if inner is not None:
                records.extend(_to_dict(i) for i in inner)
            else:
                records.append(data)
        else:
            records.append(_to_dict(data))

    return records


def _to_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return vars(obj)
