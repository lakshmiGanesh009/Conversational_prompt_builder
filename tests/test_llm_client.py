"""Provider configuration and OpenAI-compatible client tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st

from prompt_builder import llm_client
from prompt_builder.llm_client import (
    DEFAULT_MODEL,
    GROQ_DEFAULT_BASE_URL,
    GROQ_DEFAULT_MODEL,
    LlmError,
    LlmRequest,
    OpenAiLlmClient,
    provider_status_label,
    resolve_provider_config,
)

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


@pytest.fixture(autouse=True)
def _clear_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CONFIG_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets={}))


def _set_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    env: dict[str, str] | None = None,
    secrets: dict[str, str] | None = None,
) -> None:
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setitem(
        sys.modules, "streamlit", SimpleNamespace(secrets=secrets or {})
    )


def test_default_openai_resolution_keeps_existing_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(monkeypatch, env={"OPENAI_API_KEY": "openai-key"})

    config = resolve_provider_config()

    assert config.provider == "openai"
    assert config.api_key == "openai-key"
    assert config.model == DEFAULT_MODEL
    assert config.base_url is None


def test_groq_defaults_include_official_base_url_and_production_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(
        monkeypatch,
        env={"LLM_PROVIDER": "groq", "GROQ_API_KEY": "groq-key"},
    )

    config = resolve_provider_config()

    assert config.provider == "groq"
    assert config.base_url == GROQ_DEFAULT_BASE_URL
    assert config.model == GROQ_DEFAULT_MODEL


def test_environment_wins_over_streamlit_secret_for_the_same_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(
        monkeypatch,
        env={"LLM_API_KEY": "environment-key"},
        secrets={"LLM_API_KEY": "secret-key"},
    )

    assert resolve_provider_config().api_key == "environment-key"


def test_streamlit_secret_is_used_when_environment_name_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(
        monkeypatch,
        secrets={"LLM_PROVIDER": "groq", "GROQ_API_KEY": "secret-groq-key"},
    )

    config = resolve_provider_config()

    assert config.provider == "groq"
    assert config.api_key == "secret-groq-key"


def test_provider_neutral_values_win_over_provider_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(
        monkeypatch,
        env={
            "LLM_PROVIDER": "groq",
            "LLM_API_KEY": "neutral-key",
            "LLM_MODEL": "neutral-model",
            "LLM_BASE_URL": "https://neutral.example/v1",
            "GROQ_API_KEY": "groq-key",
            "GROQ_MODEL": "groq-model",
            "GROQ_BASE_URL": "https://groq.example/v1",
            "OPENAI_API_KEY": "legacy-key",
        },
    )

    config = resolve_provider_config()

    assert config.api_key == "neutral-key"
    assert config.model == "neutral-model"
    assert config.base_url == "https://neutral.example/v1"


def test_explicit_arguments_win_over_all_configured_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(
        monkeypatch,
        env={
            "LLM_PROVIDER": "openai",
            "LLM_API_KEY": "configured-key",
            "LLM_MODEL": "configured-model",
            "LLM_BASE_URL": "https://configured.example/v1",
        },
    )

    config = resolve_provider_config(
        provider="groq",
        api_key="explicit-key",
        model="explicit-model",
        base_url="https://explicit.example/v1",
    )

    assert config.provider == "groq"
    assert config.api_key == "explicit-key"
    assert config.model == "explicit-model"
    assert config.base_url == "https://explicit.example/v1"


# Property: explicit resolver settings always dominate configured aliases.
# **Validates: Provider configuration resolution rules 1-4**
@settings(max_examples=100)
@given(
    explicit_model=st.text(min_size=1).filter(lambda value: bool(value.strip())),
    explicit_base=st.text(min_size=1).filter(lambda value: bool(value.strip())),
)
def test_explicit_model_and_base_precedence_property(
    explicit_model: str,
    explicit_base: str,
) -> None:
    config = resolve_provider_config(
        provider="groq",
        api_key="explicit-key",
        model=explicit_model,
        base_url=explicit_base,
    )

    assert config.model == explicit_model
    assert config.base_url == explicit_base


def test_unknown_provider_fails_before_sdk_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(
        monkeypatch,
        env={"LLM_PROVIDER": "unknown", "LLM_API_KEY": "must-not-appear"},
    )

    with pytest.raises(LlmError, match="Unsupported LLM provider") as caught:
        OpenAiLlmClient()

    assert "unknown" not in str(caught.value).lower()
    assert "must-not-appear" not in str(caught.value)


@pytest.mark.parametrize(
    ("provider", "other_value"),
    [("openai", "groq-secret-value"), ("groq", "base-url-secret-value")],
)
def test_missing_key_error_is_provider_specific_and_does_not_expose_values(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    other_value: str,
) -> None:
    _set_sources(
        monkeypatch,
        env={
            "LLM_PROVIDER": provider,
            "LLM_BASE_URL": other_value,
            "LLM_MODEL": "private-model-value",
        },
    )

    provider_name = "OpenAI" if provider == "openai" else "Groq"
    with pytest.raises(LlmError, match=f"No API key configured for {provider_name}") as caught:
        resolve_provider_config()

    message = str(caught.value)
    assert other_value not in message
    assert "private-model-value" not in message


def test_groq_uses_openai_key_only_as_last_compatibility_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_sources(
        monkeypatch,
        env={"LLM_PROVIDER": "groq", "OPENAI_API_KEY": "legacy-openai-key"},
    )

    assert resolve_provider_config().api_key == "legacy-openai-key"


def test_provider_config_repr_and_status_never_include_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "super-secret-provider-key"
    _set_sources(
        monkeypatch,
        env={"LLM_PROVIDER": "groq", "GROQ_API_KEY": key},
    )

    config = resolve_provider_config()
    label = provider_status_label()

    assert key not in repr(config)
    assert label == f"Groq · {GROQ_DEFAULT_MODEL}"
    assert key not in label


def test_status_is_safe_for_missing_or_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert provider_status_label() == "LLM configuration unavailable"
    monkeypatch.setenv("LLM_PROVIDER", "invalid-private-value")
    assert provider_status_label() == "LLM configuration unavailable"


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"goal": "draft"}'),
                    finish_reason="stop",
                )
            ],
            model=kwargs["model"],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )


class _FakeOpenAI:
    instances: list["_FakeOpenAI"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=_FakeCompletions())
        self.__class__.instances.append(self)


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeOpenAI.instances.clear()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))


def test_model_base_url_request_settings_and_telemetry_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sdk(monkeypatch)
    client = OpenAiLlmClient(
        api_key="explicit-key",
        model="configured-model",
        base_url="https://compatible.example/v1",
    )

    response = client.complete(
        LlmRequest(
            prompt="Interpret this",
            system="Return JSON",
            structured=True,
            temperature=0.4,
            max_tokens=512,
            top_p=0.9,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            stop=["END"],
        )
    )

    sdk = _FakeOpenAI.instances[0]
    assert sdk.kwargs == {
        "api_key": "explicit-key",
        "base_url": "https://compatible.example/v1",
    }
    request = sdk.chat.completions.calls[0]
    assert request["model"] == "configured-model"
    assert request["response_format"] == {"type": "json_object"}
    assert request["temperature"] == 0.4
    assert request["max_tokens"] == 512
    assert request["top_p"] == 0.9
    assert request["frequency_penalty"] == 0.2
    assert request["presence_penalty"] == 0.1
    assert request["stop"] == ["END"]
    assert not ({"logprobs", "logit_bias", "top_logprobs", "n"} & request.keys())
    assert response.structured == {"goal": "draft"}
    assert (response.model, response.prompt_tokens, response.completion_tokens) == (
        "configured-model",
        11,
        7,
    )
    assert response.finish_reason == "stop"


def test_groq_configuration_smoke_uses_fake_sdk_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sdk(monkeypatch)
    _set_sources(
        monkeypatch,
        env={"LLM_PROVIDER": "groq", "GROQ_API_KEY": "fake-groq-key"},
    )

    client = OpenAiLlmClient()
    response = client.complete(
        LlmRequest(
            prompt="hello",
            temperature=0.3,
            max_tokens=1024,
            top_p=0.8,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            stop=["END"],
        )
    )

    sdk = _FakeOpenAI.instances[0]
    assert client.provider == "groq"
    assert client.model == GROQ_DEFAULT_MODEL
    assert sdk.kwargs["base_url"] == GROQ_DEFAULT_BASE_URL
    request = sdk.chat.completions.calls[0]
    assert request["model"] == GROQ_DEFAULT_MODEL
    assert request["temperature"] == 0.3
    assert request["max_tokens"] == 1024
    assert request["top_p"] == 0.8
    assert request["stop"] == ["END"]
    assert "frequency_penalty" not in request
    assert "presence_penalty" not in request
    assert response.text == '{"goal": "draft"}'


def test_request_temperature_zero_is_forwarded_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sdk(monkeypatch)
    client = OpenAiLlmClient(provider="groq", api_key="fake-key")

    client.complete(LlmRequest(prompt="hello", temperature=0))

    request = _FakeOpenAI.instances[0].chat.completions.calls[0]
    assert request["temperature"] == 0
