"""
Form + Instruction matcher for FRY9C schedules.

Adapted from FRY9C_Package_for_Ankit-2/match_form_instructions.py.
Normalises reference number formats and joins form line items with
their corresponding instruction text records.

This mirrors the FRY9C guide: normalise form vs instruction line references (e.g. HI / HI-A)
and join on the canonical key; parent fallback covers sub-items without a direct instruction row.
"""

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Reference normalisation
# ---------------------------------------------------------------------------

def normalize_ref(ref: str) -> str:
    """
    Convert form-style refs (``1.a.(1)(a)``) and instruction-style refs
    (``1(a)(1)(a)``) to a single canonical form for matching.
    """
    s = ref.strip().rstrip(".")

    tokens: list[tuple[str, str]] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == ".":
            i += 1
        elif c == "(":
            j = s.index(")", i)
            tokens.append(("paren", s[i + 1 : j]))
            i = j + 1
        elif c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(("num", s[i:j]))
            i = j
        elif c.isalpha():
            j = i
            while j < len(s) and s[j].isalpha():
                j += 1
            tokens.append(("alpha", s[i:j]))
            i = j
        elif c == " ":
            i += 1
        else:
            i += 1

    parts: list[str] = []
    for _idx, (typ, val) in enumerate(tokens):
        if typ == "alpha" and val == "M":
            parts.append("M.")
        elif typ == "alpha" and val == "TEXT":
            parts.append("TEXT ")
        elif typ == "num":
            parts.append(val + ".")
        elif typ == "alpha" and len(val) == 1 and val.islower():
            if parts and not parts[-1].endswith("."):
                parts.append(".")
            parts.append(val + ".")
        elif typ == "paren":
            if len(val) == 1 and val.islower():
                if parts and not parts[-1].endswith("."):
                    parts.append(".")
                parts.append(val + ".")
            else:
                parts.append("(" + val + ")")
        else:
            parts.append(val)

    return "".join(parts).rstrip(".")


def _find_parent_ref(norm_ref: str) -> str | None:
    """Strip the last segment from a normalised ref to get the parent."""
    patterns = [
        r"\(\d+\)$",
        r"\.\w$",
        r"\.\d+$",
    ]
    for pat in patterns:
        shortened = re.sub(pat, "", norm_ref).rstrip(".")
        if shortened != norm_ref.rstrip(".") and shortened:
            return shortened
    return None


# ---------------------------------------------------------------------------
# JSON loading helpers
# ---------------------------------------------------------------------------

def _load_extraction_json(path: Path) -> list[dict]:
    """Load a LlamaExtract JSON; handles wrapped ``{run_id, data}`` and flat arrays."""
    with open(path) as fh:
        raw = json.load(fh)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    return [raw]


# ---------------------------------------------------------------------------
# Core matching logic
# ---------------------------------------------------------------------------

def match_schedule(
    form_items: list[dict],
    instr_items: list[dict],
    schedule_name: str,
) -> tuple[list[dict], list[dict]]:
    """
    Match *form_items* against *instr_items* and return
    ``(combined_records, instruction_only_records)``.

    Form items are expected to have keys produced by LlamaExtract with the
    form schema: ``line_item_number``, ``description``, ``mdrm_code``,
    ``mdrm_prefix``, ``data_type``, ``is_total_or_subtotal``, etc.

    Instruction items are expected to have keys from the instruction schema:
    ``line_item_number`` (or ``line_item_reference`` for legacy JSONs),
    ``line_item_title`` (or ``title``), ``instruction_text``.
    """
    # Build normalised lookup for instructions
    instr_lookup: dict[str, dict] = {}
    for item in instr_items:
        ref = item.get("line_item_number") or item.get("line_item_reference", "")
        norm = normalize_ref(ref)
        instr_lookup[norm] = item

    combined: list[dict] = []
    matched_refs: set[str] = set()

    for form_item in form_items:
        ref = form_item.get("line_item_number") or form_item.get("reference_number", "")
        norm = normalize_ref(ref)

        instr = instr_lookup.get(norm)
        match_type = "exact"

        if not instr:
            candidate = norm
            while candidate:
                parent = _find_parent_ref(candidate)
                if parent and parent in instr_lookup:
                    instr = instr_lookup[parent]
                    match_type = f"parent:{parent}"
                    break
                candidate = parent

        title = None
        text = None
        if instr:
            title = instr.get("line_item_title") or instr.get("title")
            text = instr.get("instruction_text")
            matched_refs.add(normalize_ref(
                instr.get("line_item_number") or instr.get("line_item_reference", "")
            ))

        record = {
            "reference_number": ref,
            "normalized_ref": norm,
            "description": (
                form_item.get("description")
                or form_item.get("full_description", "")
            ),
            "mdrm_prefix": form_item.get("mdrm_prefix", "BHCK"),
            "mdrm_code": (
                form_item.get("mdrm_code")
                or form_item.get("bhck_code", "")
            ),
            "data_type": form_item.get("data_type"),
            "is_total_or_subtotal": form_item.get("is_total_or_subtotal"),
            "section": form_item.get("section"),
            "reporting_threshold": form_item.get("reporting_threshold"),
            "footnotes": form_item.get("footnotes"),
            "instruction_title": title,
            "instruction_text": text,
            "cross_references": instr.get("cross_references") if instr else None,
            "reporting_guidance": instr.get("reporting_guidance") if instr else None,
            "match_status": f"matched ({match_type})" if instr else "no_instruction_found",
            "schedule_name": schedule_name,
        }
        combined.append(record)

    # Collect instruction records that had no corresponding form line item
    instruction_only: list[dict] = []
    for item in instr_items:
        ref = item.get("line_item_number") or item.get("line_item_reference", "")
        norm = normalize_ref(ref)
        if norm not in matched_refs:
            instruction_only.append({
                "reference_number": ref,
                "normalized_ref": norm,
                "instruction_title": item.get("line_item_title") or item.get("title"),
                "instruction_text": item.get("instruction_text"),
                "match_status": "instruction_only_no_form_item",
                "schedule_name": schedule_name,
            })

    return combined, instruction_only


def build_combined_output(
    schedule_label: str,
    form_items: list[dict],
    instr_items: list[dict],
) -> dict:
    """
    Build the full combined output dict for one schedule.

    Parameters
    ----------
    schedule_label:
        E.g. ``"Schedule_HI"`` or ``"HI"``.
    form_items:
        Extracted form line items (list of dicts).
    instr_items:
        Extracted instruction line items (list of dicts).

    Returns
    -------
    Combined output dict with keys ``schedule``, ``title``,
    ``total_line_items``, ``matched``, ``unmatched_form_items``,
    ``instruction_only_items``, ``line_items``, ``instruction_only``.
    """
    # Normalise label
    schedule_name = schedule_label.replace("Schedule_", "Schedule ").replace("_", "-")
    if not schedule_name.startswith("Schedule "):
        schedule_name = f"Schedule {schedule_name}"

    combined, instruction_only = match_schedule(form_items, instr_items, schedule_name)

    matched_count = sum(
        1 for r in combined if r["match_status"].startswith("matched")
    )

    return {
        "schedule": schedule_label.replace("Schedule_", ""),
        "title": schedule_name,
        "total_line_items": len(combined),
        "matched": matched_count,
        "unmatched_form_items": len(combined) - matched_count,
        "instruction_only_items": len(instruction_only),
        "line_items": combined,
        "instruction_only": instruction_only,
    }


def save_combined_output(output: dict, path: Path) -> None:
    """Write *output* dict to *path* as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(output, fh, indent=2)
