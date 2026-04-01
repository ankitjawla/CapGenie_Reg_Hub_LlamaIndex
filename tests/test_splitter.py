"""Regression tests for schedule page classification (no LlamaCloud calls)."""

import os

import pytest

from pipeline.splitter import (
    build_split_diagnostics,
    classify_page_footer_only,
    classify_page_hybrid,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Instructions for Preparation of\nReporting Form FR Y-9C\nfoo",
            "Cover",
        ),
        (
            "Schedule HI-A\nSome body\n" + "x" * 100 + "\nfooter HI-A-3 more",
            "Schedule_HI-A",
        ),
        (
            "No header here\n" + "y" * 200 + "\npage end with GL-12 tail",
            "Glossary",
        ),
        (
            "x" * 100 + "\n" + "z" * 100 + "\ntrailing HC-B-7 footer",
            "Schedule_HC-B",
        ),
    ],
)
def test_classify_page_hybrid(text: str, expected: str) -> None:
    assert classify_page_hybrid(text) == expected


def test_classify_page_footer_only_schedule() -> None:
    text = "ignored header\n" + "p" * 300 + "\nstuff HI-2 end"
    assert classify_page_footer_only(text) == "Schedule_HI"


def test_build_split_diagnostics_pages_1based() -> None:
    parsed = "page one\n\n---\n\npage two\n\n---\n\npage three"
    mapping = {"Schedule_HI": [1], "Unclassified": [2, 3]}
    d = build_split_diagnostics(parsed, mapping, max_snippet_pages=5, max_pages_in_list=10)
    assert d["unclassified_count"] == 2
    assert d["unclassified_pages_1based"] == [2, 3]
    assert len(d["unclassified_snippets"]) == 2
    assert d["unclassified_snippets"][0]["page"] == 2
