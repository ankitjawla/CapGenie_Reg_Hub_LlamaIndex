"""
Pipeline orchestrator for FRY9C extraction.

Runs all 5 steps in sequence for a given job:
  1. Parse  – LlamaParse (agentic tier by default; versioned cache)
  2. Split  – per-schedule PDFs (hybrid header+footer or footer_only)
  3. Extract Forms     – LlamaExtract with form schema (cached per schedule)
  4. Extract Instructions – LlamaExtract with instruction schema (cached)
  5. Match  – join form + instruction records

Every significant event is emitted via a ProgressEvent callback so the
FastAPI layer can stream it over SSE.  Events carry a ``detail`` dict with
structured data (timing, page counts, cache hits, match rates) so the
frontend can render rich per-step information.
"""

import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import os
from pipeline.extractor import extract_schedules_batch, parse_pdf, ExtractResult
from pipeline.extract_settings import resolve_extract_model, resolve_extract_mode
from pipeline.matcher import build_combined_output, save_combined_output
from pipeline.splitter import get_schedule_names, split_pdf_by_schedule

# ---------------------------------------------------------------------------
# Progress event model
# ---------------------------------------------------------------------------

@dataclass
class ProgressEvent:
    step: int           # 1-5  (0 = error)
    step_name: str
    schedule: str       # "" for global/summary events
    status: str         # "running" | "done" | "error" | "cached"
    message: str
    pct: float          # 0.0 – 100.0
    detail: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


ProgressCallback = Callable[[ProgressEvent], None]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"
FORM_SCHEMA = SCHEMAS_DIR / "form_line_item_schema.json"
INSTR_SCHEMA = SCHEMAS_DIR / "instruction_line_item_schema.json"


def _write_status(job_dir: Path, events: list[dict]) -> None:
    with open(job_dir / "status.json", "w") as fh:
        json.dump(events, fh, indent=2)


def _fmt_s(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    job_dir: Path,
    form_pdf: Path,
    instr_pdf: Path,
    on_progress: ProgressCallback | None = None,
) -> None:
    """
    Execute the full 5-step FRY9C extraction pipeline.

    Parameters
    ----------
    job_dir:
        Root directory for this job.  Sub-dirs ``splits/``, ``extractions/``,
        and ``results/`` are created automatically.
    form_pdf:
        Path to the uploaded Form PDF (e.g. FR_Y-9C…_f.pdf).
    instr_pdf:
        Path to the uploaded Instruction PDF (e.g. FR_Y-9C…_i.pdf).
    on_progress:
        Called with a :class:`ProgressEvent` after each significant step.
    """
    events: list[dict] = []
    pipeline_start = time.perf_counter()

    def emit(
        step: int,
        step_name: str,
        schedule: str,
        status: str,
        message: str,
        pct: float,
        detail: dict[str, Any] | None = None,
    ) -> None:
        evt = ProgressEvent(
            step=step,
            step_name=step_name,
            schedule=schedule,
            status=status,
            message=message,
            pct=pct,
            detail=detail or {},
        )
        events.append(evt.__dict__)
        _write_status(job_dir, events)
        if on_progress:
            on_progress(evt)

    splits_dir = job_dir / "splits"
    form_splits_dir = splits_dir / "form"
    instr_splits_dir = splits_dir / "instructions"
    extractions_dir = job_dir / "extractions"
    results_dir = job_dir / "results"

    for d in (form_splits_dir, instr_splits_dir, extractions_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    import os
    form_size_mb = round(os.path.getsize(form_pdf) / 1_048_576, 2)
    instr_size_mb = round(os.path.getsize(instr_pdf) / 1_048_576, 2)

    try:
        # ==================================================================
        # STEP 1 – Parse
        # ==================================================================
        step_start = time.perf_counter()

        emit(1, "Parse", "", "running", f"Parsing Form PDF ({form_size_mb} MB) with LlamaParse…", 0.0, {
            "file": form_pdf.name,
            "size_mb": form_size_mb,
            "api": f"LlamaParse (tier={os.environ.get('FRY9C_PARSE_TIER','agentic')})",
            "what": "Per FRY9C Extraction Guide: LlamaParse agentic tier returns one page per document. Set FRY9C_PARSE_TIER=cost_effective or fast to reduce credit usage.",
        })

        form_parse = parse_pdf(form_pdf)
        (job_dir / "form_parsed.txt").write_text(form_parse.text, encoding="utf-8")

        emit(1, "Parse", "", "running",
             f"{'[CACHED] ' if form_parse.from_cache else ''}Form PDF parsed: {form_parse.pages} pages in {_fmt_s(form_parse.elapsed_s)}. "
             f"Now parsing Instruction PDF ({instr_size_mb} MB)…",
             5.0, {
                 "file": form_pdf.name,
                 "pages": form_parse.pages,
                 "size_mb": form_size_mb,
                 "elapsed_s": form_parse.elapsed_s,
                 "from_cache": form_parse.from_cache,
             })

        instr_parse = parse_pdf(instr_pdf)
        (job_dir / "instr_parsed.txt").write_text(instr_parse.text, encoding="utf-8")

        step_elapsed = round(time.perf_counter() - step_start, 2)
        emit(1, "Parse", "", "done",
             f"Both PDFs parsed in {_fmt_s(step_elapsed)}. "
             f"Form: {form_parse.pages} pages, Instructions: {instr_parse.pages} pages.",
             10.0, {
                 "form": {
                     "file": form_pdf.name, "pages": form_parse.pages,
                     "size_mb": form_size_mb, "elapsed_s": form_parse.elapsed_s,
                     "from_cache": form_parse.from_cache,
                 },
                 "instructions": {
                     "file": instr_pdf.name, "pages": instr_parse.pages,
                     "size_mb": instr_size_mb, "elapsed_s": instr_parse.elapsed_s,
                     "from_cache": instr_parse.from_cache,
                 },
                 "total_elapsed_s": step_elapsed,
                 "what": "LlamaParse reads each PDF page (markdown + fast_mode by default). Pages are output in sequence separated by '---' markers.",
             })

        # ==================================================================
        # STEP 2 – Split
        # ==================================================================
        step_start = time.perf_counter()

        emit(2, "Split", "", "running",
             f"Classifying {form_parse.pages} form pages by schedule…", 10.0, {
                 "what": "Default hybrid: header in the first ~600 characters, then footer codes in the last ~400. Set FRY9C_SPLIT_MODE=footer_only for package-style footer-only rules. One PDF per schedule.",
                 "why": "Extraction accuracy improves when the model sees one schedule at a time.",
                 "split_mode": os.environ.get("FRY9C_SPLIT_MODE", "hybrid"),
                 "note": "To use footer-only classification set FRY9C_SPLIT_MODE=footer_only",
             })

        form_mapping = split_pdf_by_schedule(form_parse.text, form_pdf, form_splits_dir)

        emit(2, "Split", "", "running",
             f"Classifying {instr_parse.pages} instruction pages by schedule…", 15.0, {
                 "what": "Same classifier as the form PDF (hybrid or footer_only per FRY9C_SPLIT_MODE).",
             })

        instr_mapping = split_pdf_by_schedule(instr_parse.text, instr_pdf, instr_splits_dir)

        form_schedules = get_schedule_names(form_mapping)
        instr_schedules = get_schedule_names(instr_mapping)
        step_elapsed = round(time.perf_counter() - step_start, 2)

        # Build per-schedule page detail for the UI
        form_sched_detail = [
            {"label": k, "pages": len(v), "pdf": f"{k}.pdf"}
            for k, v in sorted(form_mapping.items())
            if k.startswith("Schedule_")
        ]
        instr_sched_detail = [
            {"label": k, "pages": len(v), "pdf": f"{k}.pdf"}
            for k, v in sorted(instr_mapping.items())
            if k.startswith("Schedule_")
        ]

        emit(2, "Split", "", "done",
             f"Split complete in {_fmt_s(step_elapsed)}: {len(form_schedules)} form schedules, "
             f"{len(instr_schedules)} instruction schedules.",
             20.0, {
                 "form_schedules": form_sched_detail,
                 "instr_schedules": instr_sched_detail,
                 "form_other_sections": [k for k in form_mapping if not k.startswith("Schedule_")],
                 "instr_other_sections": [k for k in instr_mapping if not k.startswith("Schedule_")],
                 "total_elapsed_s": step_elapsed,
             })

        # ==================================================================
        # STEP 3 – Extract Forms
        # ==================================================================
        step_start = time.perf_counter()
        total_schedules = len(form_schedules) or 1

        _extract_model = resolve_extract_model()
        emit(3, "Extract Forms", "", "running",
             f"Extracting structured data from {total_schedules} form schedule PDFs…", 20.0, {
                 "what": "LlamaExtract (guide): PER_TABLE_ROW + BALANCED, SECTION chunking, high_resolution_mode, use_reasoning, num_pages_context=1, form system prompt. Schema: line_item_number, description, mdrm_code, mdrm_prefix, data_type, parent_line_item, is_total_or_subtotal, schedule_name, section, reporting_threshold, footnotes.",
                 "schema": "form_line_item_schema.json",
                 "extract_model": _extract_model,
                 "extract_mode": resolve_extract_mode(),
                 "total_schedules": total_schedules,
             })

        form_extraction: dict[str, list[dict]] = {}
        form_step_rows: list[dict] = []
        total_form_items = 0
        cache_hits_form = 0

        # Build tasks list — skip Unclassified PDFs (saves API credits)
        form_tasks = []
        for sched_label in form_schedules:
            pdf_path = form_splits_dir / f"{sched_label}.pdf"
            if not pdf_path.exists():
                continue
            if sched_label == "Unclassified":
                emit(3, "Extract Forms", sched_label, "running",
                     "Skipping Unclassified pages — no schedule identity detected.", 20.5, {
                         "schedule": sched_label, "skipped": True,
                     })
                continue
            form_tasks.append((pdf_path, FORM_SCHEMA, "form", sched_label))

        emit(3, "Extract Forms", "", "running",
             f"Submitting {len(form_tasks)} form schedule PDFs concurrently to LlamaExtract…", 21.0, {
                 "batch_size": len(form_tasks),
             })

        def _form_progress(label, kind, result):
            row = {
                "schedule": label.replace("Schedule_", ""),
                "items": result.items if not isinstance(result, Exception) else 0,
                "elapsed_s": result.elapsed_s if not isinstance(result, Exception) else 0,
                "from_cache": result.from_cache if not isinstance(result, Exception) else False,
                "status": "done" if not isinstance(result, Exception) else "error",
            }
            cache_tag = " [CACHED]" if (not isinstance(result, Exception) and result.from_cache) else ""
            items = result.items if not isinstance(result, Exception) else 0
            elapsed = result.elapsed_s if not isinstance(result, Exception) else 0
            emit(3, "Extract Forms", label,
                 "cached" if (not isinstance(result, Exception) and result.from_cache) else "done",
                 f"{label}{cache_tag}: {items} items in {_fmt_s(elapsed)}", 30.0, {"row": row})

        form_batch = extract_schedules_batch(form_tasks, _form_progress)

        for idx, sched_label in enumerate(form_schedules):
            pdf_path = form_splits_dir / f"{sched_label}.pdf"
            if not pdf_path.exists() or sched_label == "Unclassified":
                continue
            result = form_batch.get(sched_label)
            if result is None:
                result = ExtractResult(records=[], items=0, elapsed_s=0.0, from_cache=False)
            form_extraction[sched_label] = result.records
            total_form_items += result.items
            if result.from_cache:
                cache_hits_form += 1

            out_path = extractions_dir / f"form_{sched_label}.json"
            with open(out_path, "w") as fh:
                json.dump(result.records, fh, indent=2)

            row = {
                "schedule": sched_label.replace("Schedule_", ""),
                "items": result.items,
                "elapsed_s": result.elapsed_s,
                "from_cache": result.from_cache,
                "status": "done",
            }
            form_step_rows.append(row)
            pct_done = 20.0 + ((idx + 1) / total_schedules) * 25.0
            cache_tag = " [CACHED]" if result.from_cache else ""
            emit(3, "Extract Forms", sched_label, "cached" if result.from_cache else "done",
                 f"{sched_label}{cache_tag}: {result.items} items in {_fmt_s(result.elapsed_s)}",
                 pct_done, {"row": row, "running_total": total_form_items})

        step_elapsed = round(time.perf_counter() - step_start, 2)
        emit(3, "Extract Forms", "", "done",
             f"Form extraction complete in {_fmt_s(step_elapsed)}: {total_form_items} total items "
             f"across {len(form_schedules)} schedules ({cache_hits_form} cache hits).",
             45.0, {
                 "rows": form_step_rows,
                 "total_items": total_form_items,
                 "total_schedules": len(form_schedules),
                 "cache_hits": cache_hits_form,
                 "total_elapsed_s": step_elapsed,
             })

        # ==================================================================
        # STEP 4 – Extract Instructions
        # ==================================================================
        step_start = time.perf_counter()
        total_instr = len(instr_schedules) or 1

        emit(4, "Extract Instructions", "", "running",
             f"Extracting structured data from {total_instr} instruction schedule PDFs…", 45.0, {
                 "what": "Same extract settings as forms with an instruction-specific system prompt (top-level headings; full instruction text). Schema: line_item_number, mdrm_code, mdrm_prefix, line_item_title, instruction_text, schedule_name, cross_references, reporting_guidance, effective_date.",
                 "schema": "instruction_line_item_schema.json",
                 "extract_model": _extract_model,
                 "extract_mode": resolve_extract_mode(),
                 "total_schedules": total_instr,
             })

        instr_extraction: dict[str, list[dict]] = {}
        instr_step_rows: list[dict] = []
        total_instr_items = 0
        cache_hits_instr = 0

        # Build tasks list — skip Unclassified PDFs
        instr_tasks = []
        for sched_label in instr_schedules:
            pdf_path = instr_splits_dir / f"{sched_label}.pdf"
            if not pdf_path.exists():
                continue
            if sched_label == "Unclassified":
                emit(4, "Extract Instructions", sched_label, "running",
                     "Skipping Unclassified pages — no schedule identity detected.", 45.5, {
                         "schedule": sched_label, "skipped": True,
                     })
                continue
            instr_tasks.append((pdf_path, INSTR_SCHEMA, "instruction", sched_label))

        emit(4, "Extract Instructions", "", "running",
             f"Submitting {len(instr_tasks)} instruction schedule PDFs concurrently to LlamaExtract…", 46.0, {
                 "batch_size": len(instr_tasks),
             })

        def _instr_progress(label, kind, result):
            row = {
                "schedule": label.replace("Schedule_", ""),
                "items": result.items if not isinstance(result, Exception) else 0,
                "elapsed_s": result.elapsed_s if not isinstance(result, Exception) else 0,
                "from_cache": result.from_cache if not isinstance(result, Exception) else False,
                "status": "done" if not isinstance(result, Exception) else "error",
            }
            cache_tag = " [CACHED]" if (not isinstance(result, Exception) and result.from_cache) else ""
            items = result.items if not isinstance(result, Exception) else 0
            elapsed = result.elapsed_s if not isinstance(result, Exception) else 0
            emit(4, "Extract Instructions", label,
                 "cached" if (not isinstance(result, Exception) and result.from_cache) else "done",
                 f"{label}{cache_tag}: {items} items in {_fmt_s(elapsed)}", 60.0, {"row": row})

        instr_batch = extract_schedules_batch(instr_tasks, _instr_progress)

        for idx, sched_label in enumerate(instr_schedules):
            pdf_path = instr_splits_dir / f"{sched_label}.pdf"
            if not pdf_path.exists() or sched_label == "Unclassified":
                continue
            result = instr_batch.get(sched_label)
            if result is None:
                result = ExtractResult(records=[], items=0, elapsed_s=0.0, from_cache=False)
            instr_extraction[sched_label] = result.records
            total_instr_items += result.items
            if result.from_cache:
                cache_hits_instr += 1

            out_path = extractions_dir / f"instructions_{sched_label}.json"
            with open(out_path, "w") as fh:
                json.dump(result.records, fh, indent=2)

            row = {
                "schedule": sched_label.replace("Schedule_", ""),
                "items": result.items,
                "elapsed_s": result.elapsed_s,
                "from_cache": result.from_cache,
                "status": "done",
            }
            instr_step_rows.append(row)
            pct_done = 45.0 + ((idx + 1) / total_instr) * 30.0
            cache_tag = " [CACHED]" if result.from_cache else ""
            emit(4, "Extract Instructions", sched_label, "cached" if result.from_cache else "done",
                 f"{sched_label}{cache_tag}: {result.items} instruction items in {_fmt_s(result.elapsed_s)}",
                 pct_done, {"row": row, "running_total": total_instr_items})

        step_elapsed = round(time.perf_counter() - step_start, 2)
        emit(4, "Extract Instructions", "", "done",
             f"Instruction extraction complete in {_fmt_s(step_elapsed)}: {total_instr_items} total items "
             f"across {len(instr_schedules)} schedules ({cache_hits_instr} cache hits).",
             75.0, {
                 "rows": instr_step_rows,
                 "total_items": total_instr_items,
                 "total_schedules": len(instr_schedules),
                 "cache_hits": cache_hits_instr,
                 "total_elapsed_s": step_elapsed,
             })

        # ==================================================================
        # STEP 5 – Match
        # ==================================================================
        step_start = time.perf_counter()
        all_schedules = sorted(set(form_extraction.keys()) | set(instr_extraction.keys()))
        total_match = len(all_schedules) or 1

        emit(5, "Match", "", "running",
             f"Matching form items with instruction text across {total_match} schedules…", 75.0, {
                 "what": "A normalisation function converts both notation styles — form-style '1.a.(1)(a)' and instruction-style '1(a)(1)(a)' — to a canonical form, then tries exact matches. Unmatched form items fall back to parent-level matching (e.g. item '1.a.(1)(a)' inherits from '1.a.(1)' if no exact match exists).",
                 "total_schedules": total_match,
             })

        schedule_summary: list[dict] = []
        match_rows: list[dict] = []

        for idx, sched_label in enumerate(all_schedules):
            form_items = form_extraction.get(sched_label, [])
            instr_items = instr_extraction.get(sched_label, [])
            pct = 75.0 + (idx / total_match) * 24.0

            emit(5, "Match", sched_label, "running",
                 f"Matching {sched_label} ({idx+1}/{total_match}): "
                 f"{len(form_items)} form items + {len(instr_items)} instruction items…",
                 pct, {"schedule": sched_label, "form_items": len(form_items), "instr_items": len(instr_items)})

            output = build_combined_output(sched_label, form_items, instr_items)
            out_path = results_dir / f"{sched_label}_combined.json"
            save_combined_output(output, out_path)

            match_rate = (output["matched"] / output["total_line_items"] * 100) if output["total_line_items"] else 0

            row = {
                "schedule": output["schedule"],
                "total": output["total_line_items"],
                "matched": output["matched"],
                "unmatched": output["unmatched_form_items"],
                "instr_only": output["instruction_only_items"],
                "match_rate_pct": round(match_rate, 1),
                "status": "done",
            }
            match_rows.append(row)

            summary_entry = {
                "label": sched_label,
                "schedule": output["schedule"],
                "title": output["title"],
                "total_line_items": output["total_line_items"],
                "matched": output["matched"],
                "unmatched_form_items": output["unmatched_form_items"],
                "instruction_only_items": output["instruction_only_items"],
                "match_rate_pct": round(match_rate, 1),
                "result_file": out_path.name,
            }
            schedule_summary.append(summary_entry)

            pct_done = 75.0 + ((idx + 1) / total_match) * 24.0
            emit(5, "Match", sched_label, "done",
                 f"{sched_label}: {output['matched']}/{output['total_line_items']} matched ({match_rate:.0f}%)",
                 pct_done, {"row": row})

        pipeline_elapsed = round(time.perf_counter() - pipeline_start, 2)
        step_elapsed = round(time.perf_counter() - step_start, 2)

        total_matched = sum(r["matched"] for r in match_rows)
        total_items = sum(r["total"] for r in match_rows)
        overall_match_rate = (total_matched / total_items * 100) if total_items else 0

        # Write top-level index
        index = {
            "job_id": job_dir.name,
            "form_pdf": form_pdf.name,
            "instr_pdf": instr_pdf.name,
            "form_pages": form_parse.pages,
            "instr_pages": instr_parse.pages,
            "extract_model": resolve_extract_model(),
            "schedules": schedule_summary,
            "total_line_items": total_items,
            "total_matched": total_matched,
            "overall_match_rate_pct": round(overall_match_rate, 1),
            "pipeline_elapsed_s": pipeline_elapsed,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(job_dir / "index.json", "w") as fh:
            json.dump(index, fh, indent=2)

        emit(5, "Match", "", "done",
             f"Pipeline complete in {_fmt_s(pipeline_elapsed)}. "
             f"{len(all_schedules)} schedules · {total_items} line items · "
             f"{total_matched} matched ({overall_match_rate:.0f}%).",
             100.0, {
                 "rows": match_rows,
                 "total_items": total_items,
                 "total_matched": total_matched,
                 "overall_match_rate_pct": round(overall_match_rate, 1),
                 "total_schedules": len(all_schedules),
                 "pipeline_elapsed_s": pipeline_elapsed,
                 "step_elapsed_s": step_elapsed,
             })

    except Exception:
        tb = traceback.format_exc()
        emit(0, "Error", "", "error", f"Pipeline failed:\n{tb}", 0.0, {"traceback": tb})
        raise
