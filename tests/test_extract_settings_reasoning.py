
import pytest

from pipeline.extract_settings import build_extract_config, extract_config_fingerprint


@pytest.fixture(autouse=True)
def clear_reasoning_env(monkeypatch):
    monkeypatch.delenv("LLAMA_EXTRACT_USE_REASONING", raising=False)


def test_use_reasoning_default_true(monkeypatch) -> None:
    monkeypatch.delenv("LLAMA_EXTRACT_USE_REASONING", raising=False)
    cfg = build_extract_config("form")
    assert cfg["use_reasoning"] is True


def test_use_reasoning_false(monkeypatch) -> None:
    monkeypatch.setenv("LLAMA_EXTRACT_USE_REASONING", "false")
    cfg = build_extract_config("form")
    assert cfg["use_reasoning"] is False


def test_fingerprint_changes_with_use_reasoning(monkeypatch) -> None:
    monkeypatch.setenv("LLAMA_EXTRACT_USE_REASONING", "true")
    fp_on = extract_config_fingerprint("form")
    monkeypatch.setenv("LLAMA_EXTRACT_USE_REASONING", "false")
    fp_off = extract_config_fingerprint("form")
    assert fp_on != fp_off
