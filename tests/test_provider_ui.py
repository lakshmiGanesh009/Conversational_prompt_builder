"""Offline AppTest smoke coverage for provider status and configuration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from tests.mock_llm import MockLlmClient

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
CLIENT_SYMBOL = "prompt_builder.llm_client.OpenAiLlmClient"
_CONFIG_NAMES = (
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "GROQ_BASE_URL",
)


def _visible_text(at: AppTest) -> str:
    collections = (
        at.caption,
        at.markdown,
        at.error,
        at.warning,
        at.info,
        at.success,
    )
    return " ".join(
        str(element.value)
        for collection in collections
        for element in collection
    )


@pytest.mark.parametrize(
    ("provider", "key_name", "key", "model", "expected_label"),
    [
        (
            "openai",
            "OPENAI_API_KEY",
            "fake-openai-provider-key",
            "gpt-4o-mini",
            "OpenAI · gpt-4o-mini",
        ),
        (
            "groq",
            "GROQ_API_KEY",
            "fake-groq-provider-key",
            "openai/gpt-oss-120b",
            "Groq · openai/gpt-oss-120b",
        ),
    ],
)
def test_provider_app_smoke_uses_mock_and_never_displays_key(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    key_name: str,
    key: str,
    model: str,
    expected_label: str,
) -> None:
    """OpenAI and Groq render and process a turn without network access."""
    for name in _CONFIG_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv(key_name, key)
    monkeypatch.setenv("LLM_MODEL", model)

    client = MockLlmClient(
        default={"goal": "draft a launch plan", "confidence": 0.95, "parameters": {}}
    )
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    with patch(CLIENT_SYMBOL, return_value=client):
        at.run()
        assert not at.exception
        assert expected_label in _visible_text(at)
        assert key not in _visible_text(at)

        at.chat_input[0].set_value("Draft a launch plan").run()

    assert not at.exception
    assert client.call_count == 1
    assert expected_label in _visible_text(at)
    assert key not in _visible_text(at)


def test_invalid_provider_status_is_safe_and_non_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _CONFIG_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "private-invalid-provider")
    monkeypatch.setenv("LLM_API_KEY", "private-key-value")

    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    visible = _visible_text(at)

    assert not at.exception
    assert "LLM configuration unavailable" in visible
    assert "private-invalid-provider" not in visible
    assert "private-key-value" not in visible
