"""
FRY9C Extraction Web App – FastAPI backend.

Endpoints
---------
POST  /api/upload
    Accept Form PDF + Instruction PDF, start background pipeline,
    return {job_id}.

GET   /api/jobs/{job_id}/stream
    Server-Sent Events: real-time progress events while the pipeline runs.

GET   /api/jobs/{job_id}/status
    Return the latest status event list (for polling fallback).

GET   /api/jobs/{job_id}/results
    Return index.json (schedule list + summary stats).

GET   /api/jobs/{job_id}/results/{schedule_label}
    Return the combined JSON for one schedule.

GET   /api/jobs/{job_id}/export
    Download the full combined JSON for all schedules.

GET   /
    Serve the single-page frontend.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import AsyncIterator

import aiofiles
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pipeline.cache import _purge_orphaned_entries
from pipeline.processor import ProgressEvent, run_pipeline

load_dotenv()


def _llama_cloud_key_configured() -> bool:
    """True if any supported LlamaCloud API key env var is non-empty (no SDK import)."""
    for var in ("LLAMA_CLOUD_API_KEY", "LLAMA_PARSE_API_KEY", "LLAMA_API_KEY"):
        if os.environ.get(var, "").strip():
            return True
    return False


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
STATIC_DIR = BASE_DIR / "static"

for _d in (UPLOADS_DIR, RESULTS_DIR, STATIC_DIR):
    _d.mkdir(exist_ok=True)

# Purge any cache entries written by older code versions (mismatched key format).
# This is idempotent and fast (only reads meta JSON files).
_purged = _purge_orphaned_entries()
if _purged["parse"] or _purged["extract"]:
    import logging as _logging
    _logging.getLogger(__name__).info(
        "Cache migration: removed %d orphaned parse + %d orphaned extract entries.",
        _purged["parse"],
        _purged["extract"],
    )

# ---------------------------------------------------------------------------
# In-memory event bus (job_id → list of SSE strings)
# ---------------------------------------------------------------------------

# Maps job_id -> list of JSON-encoded event strings already flushed
_event_store: dict[str, list[str]] = {}
# Maps job_id -> asyncio.Event signalled when new events arrive
_event_signals: dict[str, asyncio.Event] = {}
_event_lock = threading.Lock()
# Uvicorn's running loop — set on startup so worker threads can wake SSE waiters.
_main_loop: asyncio.AbstractEventLoop | None = None


def _health_body() -> dict:
    """Payload for liveness/readiness probes (no external API calls)."""
    writable = os.access(RESULTS_DIR, os.W_OK)
    return {
        "status": "ok" if writable and _llama_cloud_key_configured() else "degraded",
        "llama_cloud_key_configured": _llama_cloud_key_configured(),
        "results_dir": str(RESULTS_DIR),
        "results_dir_writable": writable,
        "event_loop_captured": _main_loop is not None,
    }


# Schedule labels match pipeline output: Schedule_HC, Schedule_HC-B, etc.
_SCHEDULE_LABEL_RE = re.compile(r"^Schedule_[A-Za-z0-9_.-]+$")


def _signal_new_event(job_id: str, event_json: str) -> None:
    """Thread-safe: append event and wake any waiting SSE generators."""
    with _event_lock:
        _event_store.setdefault(job_id, []).append(event_json)
    signal = _event_signals.get(job_id)
    if signal is None:
        return
    loop = _main_loop
    if loop is not None and loop.is_running():
        try:
            loop.call_soon_threadsafe(signal.set)
        except RuntimeError:
            pass


def _make_progress_callback(job_id: str):
    def callback(evt: ProgressEvent) -> None:
        payload = json.dumps(evt.__dict__)
        _signal_new_event(job_id, payload)
    return callback


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline_thread(job_id: str, job_dir: Path, form_pdf: Path, instr_pdf: Path) -> None:
    cb = _make_progress_callback(job_id)
    try:
        run_pipeline(job_dir, form_pdf, instr_pdf, on_progress=cb)
    except Exception:
        # run_pipeline emits step 0 "error" before re-raising; log for server operators.
        logging.getLogger(__name__).exception("Pipeline failed for job %s", job_id)


def _resolve_default_project_pdfs() -> tuple[Path, Path] | None:
    """
    Find Form + Instruction PDFs in the project root.

    Prefers ``FR_Y-9C20260310_f.pdf`` / ``FR_Y-9C20260310_i.pdf``, then any
    ``*_f.pdf`` + ``*_i.pdf`` pair in ``BASE_DIR``.
    """
    preferred_f = BASE_DIR / "FR_Y-9C20260310_f.pdf"
    preferred_i = BASE_DIR / "FR_Y-9C20260310_i.pdf"
    if preferred_f.is_file() and preferred_i.is_file():
        return preferred_f, preferred_i
    forms = sorted(BASE_DIR.glob("*_f.pdf"))
    instrs = sorted(BASE_DIR.glob("*_i.pdf"))
    if forms and instrs:
        return forms[0], instrs[0]
    return None


def _enqueue_pipeline_job(form_path: Path, instr_pdf: Path) -> str:
    """Create job dir, copy PDFs in, start background thread. Returns job_id."""
    job_id = str(uuid.uuid4())
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True)

    dest_form = job_dir / f"form_{form_path.name}"
    dest_instr = job_dir / f"instr_{instr_pdf.name}"
    shutil.copy2(form_path, dest_form)
    shutil.copy2(instr_pdf, dest_instr)

    _event_store[job_id] = []
    _event_signals[job_id] = asyncio.Event()

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(job_id, job_dir, dest_form, dest_instr),
        daemon=True,
    )
    thread.start()
    return job_id


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="FRY9C Extraction API", version="1.0.0")


@app.on_event("startup")
async def _capture_main_event_loop() -> None:
    """Required for SSE: pipeline runs in a thread and must signal this loop."""
    global _main_loop
    _main_loop = asyncio.get_running_loop()


@app.get("/health")
@app.get("/api/health")
async def health():
    """
    Liveness / readiness for load balancers and operators.

    Does not call LlamaCloud. Always returns HTTP 200 if the process is up.
    ``status`` is ``ok`` when an API key is configured and ``results/`` is writable;
    otherwise ``degraded`` (use for alerts; avoids restart loops on liveness probes).
    """
    return JSONResponse(_health_body())


# ---------------------------------------------------------------------------
# Default project PDFs (no browser upload)
# ---------------------------------------------------------------------------


def _safe_upload_name(filename: str | None, fallback: str) -> str:
    """Use basename only — rejects path components like ../."""
    name = (filename or fallback).strip()
    base = Path(name).name
    if not base or base in (".", ".."):
        return fallback
    return base

@app.get("/api/default-files")
async def default_files():
    """
    Report whether default FR Y-9C PDFs exist next to ``app.py``.

    The UI can call ``POST /api/start-default`` to run the pipeline on them.
    """
    pair = _resolve_default_project_pdfs()
    if not pair:
        return JSONResponse(
            {
                "available": False,
                "form": None,
                "instr": None,
                "hint": "Place FR_Y-9C20260310_f.pdf and FR_Y-9C20260310_i.pdf in the project root.",
            }
        )
    form_p, instr_p = pair
    return JSONResponse(
        {
            "available": True,
            "form": form_p.name,
            "instr": instr_p.name,
            "form_bytes": form_p.stat().st_size,
            "instr_bytes": instr_p.stat().st_size,
        }
    )


@app.post("/api/start-default")
async def start_with_default_files():
    """Copy default project PDFs into a new job and start the pipeline."""
    pair = _resolve_default_project_pdfs()
    if not pair:
        raise HTTPException(
            status_code=404,
            detail="No default PDFs found. Add FR_Y-9C*_f.pdf and FR_Y-9C*_i.pdf to the project directory.",
        )
    form_p, instr_p = pair
    job_id = _enqueue_pipeline_job(form_p, instr_p)
    return JSONResponse(
        {
            "job_id": job_id,
            "message": "Pipeline started from project PDFs.",
            "stream_url": f"/api/jobs/{job_id}/stream",
            "form": form_p.name,
            "instr": instr_p.name,
        }
    )


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    form_pdf: UploadFile = File(..., description="FR Y-9C Form PDF"),
    instr_pdf: UploadFile = File(..., description="FR Y-9C Instruction PDF"),
):
    """Accept the two PDFs, persist them, and kick off the pipeline."""
    # Validate content type
    for upload in (form_pdf, instr_pdf):
        ct = upload.content_type or ""
        if "pdf" not in ct.lower() and not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"'{upload.filename}' must be a PDF file.")

    job_id = str(uuid.uuid4())
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True)

    form_name = _safe_upload_name(form_pdf.filename, "form.pdf")
    instr_name = _safe_upload_name(instr_pdf.filename, "instr.pdf")
    form_path = job_dir / f"form_{form_name}"
    instr_path = job_dir / f"instr_{instr_name}"

    for upload, dest in ((form_pdf, form_path), (instr_pdf, instr_path)):
        async with aiofiles.open(dest, "wb") as fh:
            content = await upload.read()
            await fh.write(content)

    _event_store[job_id] = []
    _event_signals[job_id] = asyncio.Event()

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(job_id, job_dir, form_path, instr_path),
        daemon=True,
    )
    thread.start()

    return JSONResponse(
        {"job_id": job_id, "message": "Pipeline started.", "stream_url": f"/api/jobs/{job_id}/stream"}
    )


# ---------------------------------------------------------------------------
# SSE stream endpoint
# ---------------------------------------------------------------------------

async def _sse_generator(job_id: str) -> AsyncIterator[str]:
    """Yield SSE-formatted progress events for the given job."""
    if job_id not in _event_store:
        yield "data: {\"error\": \"job not found\"}\n\n"
        return

    sent = 0
    signal = _event_signals.get(job_id)
    if signal is None:
        return

    while True:
        # Yield any buffered events we haven't sent yet
        with _event_lock:
            current = list(_event_store[job_id])

        for payload in current[sent:]:
            yield f"data: {payload}\n\n"
            sent += 1

        # Check if pipeline is done (last event has status "done" at step 5 or "error")
        if current:
            last = json.loads(current[-1])
            if (last.get("step") == 5 and last.get("status") == "done") or last.get("status") == "error":
                break

        # Wait for the next event signal (with timeout to avoid hanging forever)
        signal.clear()
        try:
            await asyncio.wait_for(asyncio.shield(asyncio.ensure_future(_wait_signal(signal))), timeout=30.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass


async def _wait_signal(signal: asyncio.Event) -> None:
    await signal.wait()


@app.get("/api/jobs/{job_id}/stream")
async def stream_progress(job_id: str):
    """Server-Sent Events endpoint for real-time progress."""
    if job_id not in _event_store:
        raise HTTPException(status_code=404, detail="Job not found.")
    return StreamingResponse(
        _sse_generator(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Status polling endpoint (fallback for environments without SSE support)
# ---------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}/status")
async def get_status(job_id: str):
    events = _event_store.get(job_id)
    if events is None:
        # Try loading from disk (server restart scenario)
        status_file = RESULTS_DIR / job_id / "status.json"
        if status_file.exists():
            return JSONResponse(json.loads(status_file.read_text()))
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse([json.loads(e) for e in events])


# ---------------------------------------------------------------------------
# Results endpoints
# ---------------------------------------------------------------------------

@app.get("/api/jobs/{job_id}/results")
async def get_results_index(job_id: str):
    """Return schedule index with summary stats."""
    index_path = RESULTS_DIR / job_id / "index.json"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Results not ready yet or job not found.")
    return JSONResponse(json.loads(index_path.read_text()))


@app.get("/api/jobs/{job_id}/results/{schedule_label}")
async def get_schedule_result(job_id: str, schedule_label: str):
    """Return the combined JSON for a specific schedule."""
    if not _SCHEDULE_LABEL_RE.match(schedule_label):
        raise HTTPException(status_code=400, detail="Invalid schedule label.")
    root = RESULTS_DIR.resolve()
    job_dir = (RESULTS_DIR / job_id).resolve()
    try:
        job_dir.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job not found.")
    subdir = (job_dir / "results").resolve()
    result_path = (subdir / f"{schedule_label}_combined.json").resolve()
    try:
        result_path.relative_to(subdir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path.")
    if not result_path.is_file():
        raise HTTPException(status_code=404, detail=f"Result for '{schedule_label}' not found.")
    return JSONResponse(json.loads(result_path.read_text()))


@app.get("/api/jobs/{job_id}/export")
async def export_all(job_id: str):
    """Download a ZIP of all combined JSON results for this job."""
    job_results_dir = RESULTS_DIR / job_id / "results"
    if not job_results_dir.exists():
        raise HTTPException(status_code=404, detail="Results not ready yet or job not found.")

    zip_path = RESULTS_DIR / job_id / "export.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(job_results_dir))

    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=f"FRY9C_{job_id[:8]}_results.zip",
    )


# ---------------------------------------------------------------------------
# Jobs list (for resuming previous jobs)
# ---------------------------------------------------------------------------

@app.get("/api/jobs")
async def list_jobs():
    """Return summary of all completed jobs found on disk."""
    jobs = []
    for job_dir in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if not job_dir.is_dir():
            continue
        index_path = job_dir / "index.json"
        if index_path.exists():
            data = json.loads(index_path.read_text())
            jobs.append({
                "job_id": job_dir.name,
                "form_pdf": data.get("form_pdf"),
                "instr_pdf": data.get("instr_pdf"),
                "schedules": len(data.get("schedules", [])),
                "completed_at": data.get("completed_at"),
            })
    return JSONResponse(jobs)


# ---------------------------------------------------------------------------
# Serve static frontend
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
