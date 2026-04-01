"""
Schedule splitter for FRY9C PDFs.

Adapted from FRY9C_Package_for_Ankit-2/split_by_schedule.py.

Classification (default ``hybrid`` — override with ``FRY9C_SPLIT_MODE``):

  - **hybrid** (default): header in the first ~600 chars, then footer codes in the
    last ~400 chars, then a footer-region schedule header repeat. Aligns with the
    guide when headers are visible; fixes Form PDF pages that lack footer codes.
  - **footer_only**: footer patterns only (same as FRY9C_Package split_by_schedule.py).

  Cover / glossary / GEN / notes / edit-check rules apply in both modes.

The source PDFs are then split into per-schedule PDFs using pypdf.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

from pypdf import PdfReader, PdfWriter


def _split_mode() -> str:
    """hybrid (default) or footer_only — package-style footer rules only (FRY9C_SPLIT_MODE)."""
    return os.getenv("FRY9C_SPLIT_MODE", "hybrid").strip().lower()


# ---------------------------------------------------------------------------
# Page classifier
# ---------------------------------------------------------------------------

def classify_page_footer_only(text: str) -> str:
    """
    Package-style classification: footer patterns only (see split_by_schedule.py).
    Form PDF pages without footer codes often land in ``Unclassified``.
    """
    if ("Instructions for Preparation of" in text and "Reporting Form FR Y-9C" in text):
        return "Cover"

    tail = text.strip()[-400:]

    m = re.search(r'FR\s+Y-9C:\s+(CHK|EDIT)-\d+\s+of\s+\d+', tail)
    if m:
        return f"Edit_Checks_{m.group(1)}"

    if re.search(r'\bGL-\d+\b', tail):
        return "Glossary"

    if re.search(r'Contents[-]?\d+', tail):
        return "Contents"

    if re.search(r'GEN[-]?\d+', tail):
        return "General_Instructions"

    if re.search(r'ISnotes[-]+P[-]?\d+', tail):
        return "ISnotes_Predecessor"
    if re.search(r'ISnotes[-]+\d+', tail):
        return "ISnotes_Other"

    if re.search(r'BSnotes[-]+P[-]?\d+', tail):
        return "BSnotes_Predecessor"
    if re.search(r'BSnotes[-]+\d+', tail):
        return "BSnotes_Other"

    m = re.search(r'(HI-[A-C]|HC-[A-Z])[-]\d+', tail)
    if m:
        return f"Schedule_{m.group(1)}"

    m = re.search(r'\b(HI|HC)[-]\d+\b', tail)
    if m:
        return f"Schedule_{m.group(1)}"

    return "Unclassified"


def classify_page_hybrid(text: str) -> str:
    """
    Header-first (first ~600 chars), then footer rules, then tail header repeat.

    Matches the FRY9C guide recommendation to use headers when visible; improves
    Form PDF schedule grouping when footers are absent.
    """
    if ("Instructions for Preparation of" in text and "Reporting Form FR Y-9C" in text):
        return "Cover"

    head = text[:600]
    m_head = re.search(r'Schedule\s+(H[IC](?:-[A-Z0-9]+)*)', head)
    if m_head:
        label = m_head.group(1).strip()
        return f"Schedule_{label}"

    tail = text.strip()[-400:]

    m = re.search(r'FR\s+Y-9C:\s+(CHK|EDIT)-\d+\s+of\s+\d+', tail)
    if m:
        return f"Edit_Checks_{m.group(1)}"

    if re.search(r'\bGL-\d+\b', tail):
        return "Glossary"

    if re.search(r'Contents[-]?\d+', tail):
        return "Contents"

    if re.search(r'GEN[-]?\d+', tail):
        return "General_Instructions"

    if re.search(r'ISnotes[-]+P[-]?\d+', tail):
        return "ISnotes_Predecessor"
    if re.search(r'ISnotes[-]+\d+', tail):
        return "ISnotes_Other"

    if re.search(r'BSnotes[-]+P[-]?\d+', tail):
        return "BSnotes_Predecessor"
    if re.search(r'BSnotes[-]+\d+', tail):
        return "BSnotes_Other"

    m = re.search(r'(HI-[A-C]|HC-[A-Z])[-]\d+', tail)
    if m:
        return f"Schedule_{m.group(1)}"

    m = re.search(r'\b(HI|HC)[-]\d+\b', tail)
    if m:
        return f"Schedule_{m.group(1)}"

    m_tail = re.search(r'Schedule\s+(H[IC](?:-[A-Z0-9]+)*)', tail)
    if m_tail:
        return f"Schedule_{m_tail.group(1).strip()}"

    return "Unclassified"


def classify_page(text: str) -> str:
    """
    Return a schedule label for one page of parsed PDF text.

    Mode ``hybrid`` (default): header in the first ~600 chars, then footer codes.
    Mode ``footer_only``: set ``FRY9C_SPLIT_MODE=footer_only`` to match the
    reference package (footer patterns only).

    Labels are ``Schedule_HI``, ``Schedule_HI-A``, etc.; special sections
    ``Cover``, ``Glossary``, …; else ``Unclassified``.
    """
    if _split_mode() == "footer_only":
        return classify_page_footer_only(text)
    return classify_page_hybrid(text)


# ---------------------------------------------------------------------------
# PDF splitter
# ---------------------------------------------------------------------------

def split_pdf_by_schedule(
    parsed_text: str,
    source_pdf_path: Path,
    output_dir: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, list[int]]:
    """
    Split *source_pdf_path* into per-schedule PDFs inside *output_dir*.

    Parameters
    ----------
    parsed_text:
        Full text from LlamaParse (pages separated by ``\\n\\n---\\n\\n``).
    source_pdf_path:
        Path to the original PDF file.
    output_dir:
        Directory where per-schedule PDF files are written.
    progress_cb:
        Optional callable receiving status strings during processing.

    Returns
    -------
    dict mapping schedule label → list of 1-based page numbers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    text_pages = parsed_text.split("\n\n---\n\n")
    reader = PdfReader(str(source_pdf_path))
    pdf_page_count = len(reader.pages)

    if progress_cb:
        progress_cb(f"Text pages: {len(text_pages)}, PDF pages: {pdf_page_count}")

    page_count = min(len(text_pages), pdf_page_count)

    schedule_pages: dict[str, list[int]] = defaultdict(list)
    for i in range(page_count):
        label = classify_page(text_pages[i])
        schedule_pages[label].append(i)

    # Write one PDF per label
    for label, pages in sorted(schedule_pages.items()):
        writer = PdfWriter()
        for page_num in pages:
            writer.add_page(reader.pages[page_num])
        out_path = output_dir / f"{label}.pdf"
        with open(out_path, "wb") as fh:
            writer.write(fh)
        if progress_cb:
            progress_cb(f"  Written: {out_path.name} ({len(pages)} pages)")

    mapping = {label: [p + 1 for p in pages] for label, pages in sorted(schedule_pages.items())}
    mapping_path = output_dir / "schedule_page_mapping.json"
    with open(mapping_path, "w") as fh:
        json.dump(mapping, fh, indent=2)

    return mapping


def get_schedule_names(mapping: dict[str, list[int]]) -> list[str]:
    """Return only the ``Schedule_*`` labels from a page-mapping dict."""
    return sorted(k for k in mapping if k.startswith("Schedule_"))


def build_split_diagnostics(
    parsed_text: str,
    schedule_page_mapping: dict[str, list[int]],
    *,
    max_snippet_pages: int = 15,
    max_pages_in_list: int = 500,
) -> dict:
    """
    Summarize Unclassified pages for debugging split quality.

    *schedule_page_mapping* must match ``split_pdf_by_schedule`` output: values are
    **1-based** page numbers per label.
    """
    text_pages = parsed_text.split("\n\n---\n\n")
    unc_1based = sorted(schedule_page_mapping.get("Unclassified") or [])
    listed = unc_1based[:max_pages_in_list]
    truncated = len(unc_1based) > len(listed)

    snippets: list[dict[str, str | int]] = []
    for p1 in unc_1based[:max_snippet_pages]:
        idx = p1 - 1
        if 0 <= idx < len(text_pages):
            t = text_pages[idx]
            head = t[:120].replace("\n", " ").strip()
            tail_chunk = t.strip()[-400:] if len(t) > 400 else t.strip()
            tail = tail_chunk.replace("\n", " ").strip()
            snippets.append({"page": p1, "head": head, "tail": tail})

    return {
        "unclassified_count": len(unc_1based),
        "unclassified_pages_1based": listed,
        "unclassified_pages_truncated": truncated,
        "unclassified_snippets": snippets,
    }
