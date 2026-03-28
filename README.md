# CapGenie Reg Hub — FRY9C LlamaCloud pipeline

FastAPI web app that ingests **FR Y-9C Form** and **Instruction** PDFs, runs **LlamaParse** (page-level text/markdown), splits by schedule, runs **LlamaExtract** against JSON schemas, then matches form line items to instruction text. Progress is streamed over **Server-Sent Events (SSE)**.

The FRY9C reference package (Extraction Guide, walkthrough, sample PDFs/JSON) is **not** in this repository. Add a local folder such as `FRY9C_Package_for_Ankit-2/` if you use those materials (that path is gitignored).

## Requirements

- Python 3.11+ (recommended)
- Active [LlamaCloud](https://cloud.llamaindex.ai/) account and API key (parsing and extraction consume credits)

## Setup

```bash
cd CapGenie_Llama_Index
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set at least LLAMA_API_KEY
```

### Default project PDFs (optional)

To use **Run with project PDFs** in the UI, copy your filing PDFs next to `app.py`:

- `FR_Y-9C20260310_f.pdf` — Form  
- `FR_Y-9C20260310_i.pdf` — Instructions  

These filenames are gitignored so filings are not pushed to the remote.

## Run the server

```bash
source .venv/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Upload both PDFs or start from default files when present.

## Environment variables

See [`.env.example`](.env.example). Summary:

| Variable | Required | Purpose |
|----------|----------|---------|
| `LLAMA_API_KEY` | Yes | LlamaParse and LlamaExtract |
| `LLAMA_EXTRACT_MODEL` | No | Explicit extract model slug |
| `AZURE_OPENAI_*` | No | Extract model fallback / optional parse Azure settings |

Optional toggles: `FRY9C_PARSE_FAST_MODE`, `FRY9C_SPLIT_MODE` (`hybrid` or `footer_only`).

## Project layout

| Path | Role |
|------|------|
| `app.py` | FastAPI app, upload, SSE, static UI |
| `pipeline/` | Parse, split, extract, match, cache |
| `schemas/` | LlamaExtract JSON schemas |
| `static/` | Single-page frontend |
| `.cache/` | Local parse/extract cache (gitignored) |
| `results/` | Per-job outputs (gitignored) |

## API highlights

- `POST /api/upload` — upload Form + Instruction PDFs, returns `job_id`
- `GET /api/jobs/{job_id}/stream` — SSE progress
- `GET /api/jobs/{job_id}/results` — summary `index.json`
- `GET /api/default-files` / `POST /api/start-default` — default root PDFs

## License

Add a license file if you distribute this repository publicly.

## Remote repository

Primary GitHub remote: [https://github.com/ankitjawla/CapGenie_Reg_Hub_LlamaIndex](https://github.com/ankitjawla/CapGenie_Reg_Hub_LlamaIndex)
