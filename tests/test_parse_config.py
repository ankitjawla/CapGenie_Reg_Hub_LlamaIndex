import os

from pipeline.parse_config import effective_parse_tier, effective_parse_version


def test_effective_parse_tier_default(monkeypatch) -> None:
    monkeypatch.delenv("FRY9C_PARSE_TIER", raising=False)
    assert effective_parse_tier() == "fast"


def test_effective_parse_tier_override(monkeypatch) -> None:
    monkeypatch.setenv("FRY9C_PARSE_TIER", "AGENTIC")
    assert effective_parse_tier() == "agentic"


def test_effective_parse_version(monkeypatch) -> None:
    monkeypatch.setenv("FRY9C_PARSE_VERSION", "  v2  ")
    assert effective_parse_version() == "v2"
