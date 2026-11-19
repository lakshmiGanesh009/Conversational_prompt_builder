"""Provider configuration and the app's only network-touching LLM client.

The app supports OpenAI and Groq through the OpenAI Python SDK's Chat
Completions interface. Configuration is resolved at construction time and never
logged or rendered. Importing this module performs no network access.

Resolution order for every setting is:

1. An explicit constructor/resolver argument.
2. The provider-neutral ``LLM_*`` name.
3. The selected provider's compatibility alias.
4. The provider default, where one exists.

Each named configuration value is read from the OS environment first and then
from Streamlit Secrets. OpenAI keeps the existing ``OPENAI_*`` setup. Groq uses
``GROQ_*`` aliases and accepts ``OPENAI_API_KEY`` only as a final key fallback
for backwards compatibility.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

__all__ = [
    "LlmClient",
    "LlmRequest",
    "LlmResponse",
    "LlmError",
    "ProviderConfig",
    "OpenAiLlmClient",
    "resolve_provider_config",
    "provider_status_label",
    "API_KEY_ENV_VAR",
    "BASE_URL_ENV_VAR",
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "GROQ_DEFAULT_BASE_URL",
    "GROQ_DEFAULT_MODEL",
]

# Existing public constants remain available for backwards compatibility.
API_KEY_ENV_VAR = "OPENAI_API_KEY"
BASE_URL_ENV_VAR = "OPENAI_BASE_URL"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_PROVIDER = "openai"
GROQ_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
_SUPPORTED_PROVIDERS = frozenset({"openai", "groq"})


def _read_config(name: str) -> Optional[str]:
    """Read one named value from the environment, then Streamlit Secrets.

    Streamlit access is best-effort so this module remains importable and
    testable without an active Streamlit runtime or a secrets file.
    """
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st

        secret = st.secrets.get(name)  # type: ignore[attr-defined]
        if secret:
            return str(secret)
    except Exception:  # noqa: BLE001 - unavailable secrets mean no value
        return None
    return None


def _first_config(*names: str) -> Optional[str]:
    """Return the first configured name, preserving the documented precedence."""
    for name in names:
        value = _read_config(name)
        if value is not None:
            return value
    return None


class LlmError(RuntimeError):
    """Raised when configuration or an LLM request cannot be completed."""


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved provider settings.

    The key is excluded from ``repr`` so accidental object logging cannot reveal
    it. Callers must never render, export, or log ``api_key``.
    """

    provider: str
    model: str
    base_url: Optional[str]
    api_key: Optional[str] = field(repr=False)

    @property
    def display_name(self) -> str:
        return "OpenAI" if self.provider == "openai" else "Groq"


def resolve_provider_config(
    *,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    require_api_key: bool = True,
) -> ProviderConfig:
    """Resolve provider configuration without making a network request.

    Explicit arguments win, followed by provider-neutral names and then
    provider-specific aliases. For each name, the OS environment wins over the
    same Streamlit secret.

    OpenAI defaults to ``gpt-4o-mini`` and the SDK's standard base URL. Groq
    defaults to ``https://api.groq.com/openai/v1`` and
    ``openai/gpt-oss-120b``. Unknown providers and missing keys fail with safe
    messages that do not echo configuration values.
    """
    provider_value = provider if provider is not None else _read_config("LLM_PROVIDER")
    resolved_provider = (provider_value or DEFAULT_PROVIDER).strip().lower()
    if resolved_provider not in _SUPPORTED_PROVIDERS:
        raise LlmError("Unsupported LLM provider. Use 'openai' or 'groq'.")

    if api_key is not None:
        resolved_key = api_key
    elif resolved_provider == "openai":
        resolved_key = _first_config("LLM_API_KEY", "OPENAI_API_KEY")
    else:
        resolved_key = _first_config(
            "LLM_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"
        )

    if model is not None:
        resolved_model = model
    elif resolved_provider == "openai":
        resolved_model = _first_config("LLM_MODEL", "OPENAI_MODEL") or DEFAULT_MODEL
    else:
        resolved_model = _first_config("LLM_MODEL", "GROQ_MODEL") or GROQ_DEFAULT_MODEL

    if base_url is not None:
        resolved_base_url = base_url
    elif resolved_provider == "openai":
        resolved_base_url = _first_config("LLM_BASE_URL", "OPENAI_BASE_URL")
    else:
        resolved_base_url = (
            _first_config("LLM_BASE_URL", "GROQ_BASE_URL")
            or GROQ_DEFAULT_BASE_URL
        )

    if require_api_key and (not resolved_key or not resolved_key.strip()):
        provider_name = "OpenAI" if resolved_provider == "openai" else "Groq"
        raise LlmError(
            f"No API key configured for {provider_name}. Add a provider key "
            "to the environment or Streamlit Secrets."
        )

    return ProviderConfig(
        provider=resolved_provider,
        api_key=resolved_key,
        model=resolved_model,
        base_url=resolved_base_url,
    )


def provider_status_label() -> str:
    """Return a compact, non-secret provider/model label for the UI.

    Invalid providers and missing keys deliberately collapse to one safe status;
    raw configuration values and exception details are never displayed.
    """
    try:
        config = resolve_provider_config(require_api_key=False)
    except LlmError:
        return "LLM configuration unavailable"
    if not config.api_key or not config.api_key.strip():
        return "LLM configuration unavailable"
    return f"{config.display_name} · {config.model}"


@dataclass
class LlmRequest:
    """A single OpenAI-compatible Chat Completions request."""

    prompt: str
    system: Optional[str] = None
    structured: bool = False
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    operation: str = "unknown"


@dataclass
class LlmResponse:
    """Text, parsed JSON, and optional provider telemetry from one completion."""

    text: str
    structured: Optional[dict[str, Any]] = None
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


@runtime_checkable
class LlmClient(Protocol):
    """The interface used by the intent analyzer and prompt generator."""

    def complete(self, request: LlmRequest) -> LlmResponse:
        ...


class OpenAiLlmClient:
    """OpenAI SDK client configured for OpenAI or Groq.

    The class name is retained to preserve existing imports and mocks. Groq is
    reached through its official OpenAI-compatible base URL.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        config = resolve_provider_config(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
        self._provider = config.provider
        self._model = config.model
        self._temperature = temperature

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - only without dependency
            raise LlmError(
                "The 'openai' package is required for OpenAiLlmClient. Install "
                "it with 'pip install openai'."
            ) from exc

        client_kwargs: dict[str, Any] = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        try:
            self._client = OpenAI(**client_kwargs)
        except Exception as exc:  # noqa: BLE001 - preserve cause without exposing it
            raise LlmError("Could not initialize the LLM client.") from exc

    @property
    def provider(self) -> str:
        """The resolved non-secret provider name."""
        return self._provider

    @property
    def model(self) -> str:
        """The resolved non-secret default model name."""
        return self._model

    def complete(self, request: LlmRequest) -> LlmResponse:
        """Call the OpenAI-compatible Chat Completions endpoint."""
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        kwargs: dict[str, Any] = {
            "model": request.model or self._model,
            "messages": messages,
            "temperature": (
                request.temperature
                if request.temperature is not None
                else self._temperature
            ),
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        # Groq's OpenAI-compatible endpoint does not currently support either
        # penalty field. Keep the existing OpenAI behavior, but omit these two
        # settings for Groq rather than sending a request the provider rejects.
        if self._provider != "groq" and request.frequency_penalty is not None:
            kwargs["frequency_penalty"] = request.frequency_penalty
        if self._provider != "groq" and request.presence_penalty is not None:
            kwargs["presence_penalty"] = request.presence_penalty
        if request.stop:
            kwargs["stop"] = request.stop
        if request.structured:
            # Both selected providers support JSON-object response format on
            # their OpenAI-compatible Chat Completions endpoint. No speculative
            # retry that changes parsing semantics is used.
            kwargs["response_format"] = {"type": "json_object"}

        try:
            completion = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - preserve cause without exposing it
            raise LlmError("LLM request failed.") from exc

        text = self._extract_text(completion)
        structured = self._parse_structured(text) if request.structured else None

        usage = getattr(completion, "usage", None)
        choices = getattr(completion, "choices", None) or [None]
        return LlmResponse(
            text=text,
            structured=structured,
            model=getattr(completion, "model", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=getattr(choices[0], "finish_reason", None),
        )

    @staticmethod
    def _extract_text(completion: Any) -> str:
        try:
            choice = completion.choices[0]
            content = choice.message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LlmError("Unexpected LLM response shape.") from exc
        return content or ""

    @staticmethod
    def _parse_structured(text: str) -> Optional[dict[str, Any]]:
        if not text or not text.strip():
            return None
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
