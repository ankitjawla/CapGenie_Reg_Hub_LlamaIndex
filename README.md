# CapGenie Reg Hub — FRY9C LlamaCloud pipeline

FastAPI web app that ingests **FR Y-9C Form** and **Instruction** PDFs, runs **LlamaParse** (agentic tier, per-page text), splits by schedule, runs **LlamaExtract** concurrently against JSON schemas, then matches form line items to instruction text. Progress is streamed over **Server-Sent Events (SSE)**.

The FRY9C reference package (Extraction Guide, walkthrough, sample PDFs/JSON) is **not** in this repository. Add a local folder such as `FRY9C_Package_for_Ankit-2/` if you use those materials (that path is gitignored).

**How it works (architecture, pipeline, caching):** see [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

---

## What's new (latest iteration)

| Area | Improvement |
|---|---|
| SDK | Migrated from deprecated `llama-cloud-services` → `llama-cloud>=1.0` (AsyncLlamaCloud) |
| Parse | **Agentic tier** by default (`FRY9C_PARSE_TIER=agentic`); change to `cost_effective`/`fast` to save credits |
| Extract mode | **PREMIUM** when an extract model is explicitly configured; **MULTIMODAL** (default) otherwise |
| Extract model | `LLAMA_EXTRACT_MODEL` → `AZURE_OPENAI_DEPLOYMENT` → `gpt-5.4` (if Azure endpoint+key set) → `openai-gpt-4-1` |
| Accuracy | `confidence_scores=True`, `cite_sources=True`, `use_reasoning=True`, `high_resolution_mode=True` |
| Chunk mode | `PAGE` for form schedules (dense tables); `SECTION` for instruction PDFs (narrative text) |
| Context window | `num_pages_context` auto-sized: 1 for HI/HI-A/HI-B; 2 for HC sub-schedules |
| Schemas | Enriched field descriptions with FRY9C-specific extraction hints and examples |
| Concurrency | All schedule PDFs submitted to LlamaExtract **concurrently** via `asyncio.gather` |
| Event loop | Pipeline runs in a background thread with `asyncio.run()` (no nesting under uvicorn uvloop) |
| Cache keys | Parse fingerprint now encodes full Azure endpoint/deployment/version values |
| Cache TTL | `CACHE_MAX_AGE_DAYS` env var evicts stale cache entries (default: no TTL) |
| Unclassified PDFs | Skipped in extraction steps with a SSE warning event (saves API credits) |

---

## Quick start

```bash
git clone https://github.com/ankitjawla/CapGenie_Reg_Hub_LlamaIndex.git
cd CapGenie_Reg_Hub_LlamaIndex
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set LLAMA_CLOUD_API_KEY (required)

uvicorn app:app --reload
# Open http://localhost:8000
```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `LLAMA_CLOUD_API_KEY` | Yes | LlamaCloud API key. Also accepts `LLAMA_PARSE_API_KEY` or legacy `LLAMA_API_KEY`. |
| `LLAMA_EXTRACT_MODEL` | No | Override extract model / deployment (e.g. `gpt-5.4` on Azure or `openai-gpt-4-1` on LlamaCloud). |
| `LLAMA_EXTRACT_MODE` | No | Override extraction mode: `PREMIUM`, `MULTIMODAL` (default), `BALANCED`, `FAST`. |
| `FRY9C_PARSE_TIER` | No | LlamaParse tier: `agentic` (default), `cost_effective`, `fast`. |
| `FRY9C_PARSE_VERSION` | No | LlamaParse model version, default `latest`. |
| `FRY9C_SPLIT_MODE` | No | Schedule classification: `hybrid` (default) or `footer_only`. |
| `CACHE_MAX_AGE_DAYS` | No | Evict cache entries older than this many days. `0` = keep forever (default). |
| `AZURE_OPENAI_ENDPOINT` | No | Used as extract model fallback when `LLAMA_EXTRACT_MODEL` is unset. |
| `AZURE_OPENAI_DEPLOYMENT` | No | Azure deployment name used as extract model fallback. |
| `AZURE_OPENAI_API_VERSION` | No | Azure OpenAI API version. |

---

## Project layout

```
app.py                       # FastAPI entrypoint, SSE streaming, job management
docs/
  HOW_IT_WORKS.md            # Architecture and pipeline walkthrough
pipeline/
  cache.py                   # Disk-based SHA-256 content-addressed cache (parse + extract)
  extract_settings.py        # LlamaExtract config: mode/model resolution, system prompts
  extractor.py               # AsyncLlamaCloud parse + extract, batch concurrency, cache wiring
  processor.py               # Pipeline orchestrator (5 steps); emits ProgressEvent objects
  splitter.py                # Regex-based PDF → per-schedule PDF splitting
  matcher.py                 # Join form + instruction extracted records
schemas/
  form_line_item_schema.json           # JSON schema for form line items (enriched hints)
  instruction_line_item_schema.json    # JSON schema for instruction entries (enriched hints)
static/                      # Frontend HTML/JS/CSS
results/                     # Job output dirs (gitignored)
.cache/                      # Parse + extract cache (gitignored)
uploads/                     # Uploaded PDFs (gitignored)
```

---

## API highlights

| Endpoint | Method | Description |
|---|---|---|
| `GET /health` , `GET /api/health` | GET | Liveness: no LlamaCloud calls. JSON includes `status` (`ok` or `degraded`), `llama_cloud_key_configured`, `results_dir_writable`, `event_loop_captured`. |
| `POST /api/upload` | POST | Upload form + instruction PDFs; returns `{job_id}`. |
| `POST /api/start-default` | POST | Run pipeline on project-root default PDFs if present. |
| `GET /api/jobs/{job_id}/stream` | GET | SSE stream of `ProgressEvent` objects for real-time UI updates. |
| `GET /api/jobs/{job_id}/status` | GET | Same events as JSON (polling fallback). |
| `GET /api/jobs/{job_id}/results` | GET | Job `index.json` (schedule list + totals). |
| `GET /api/jobs/{job_id}/results/{schedule_label}` | GET | One schedule combined JSON. |
| `GET /api/jobs/{job_id}/export` | GET | ZIP of all schedule JSON files. |
| `GET /api/jobs` | GET | List completed jobs from disk. |
| `GET /api/default-files` | GET | Whether default project PDFs exist. |

---

## Pipeline steps

```
Upload PDF(s)
     │
     ▼
Step 1 — LlamaParse (agentic tier)
     │   Converts each PDF page to plain text; pages separated by '---'
     ▼
Step 2 — Schedule splitter
     │   Regex classifies each page (hybrid: header-first + footer, or footer_only)
     │   Writes one PDF per schedule label to form_splits/ and instr_splits/
     ▼
Steps 3 & 4 — LlamaExtract (concurrent, per schedule)
     │   Skips Unclassified pages
     │   All schedules submitted concurrently via asyncio.gather
     │   Results cached per (pdf_sha256 + schema + extract_config fingerprint)
     ▼
Step 5 — Matcher
         Normalises reference numbers; joins form ↔ instruction records
         Writes combined JSON + per-schedule JSON to results/<job_id>/
```

---

## Remote repository

https://github.com/ankitjawla/CapGenie_Reg_Hub_LlamaIndex
