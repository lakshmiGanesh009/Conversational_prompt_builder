"""The single network-touching module for the Conversational Prompt Builder.

Everything deterministic in this app runs offline. The one exception is talking
to the language model, and that lives here and *only* here: the Intent_Analyzer
and the Prompt_Generator call this module, and nothing else does (design "LLM
Client", "Architecture"). Keeping the network behind one small interface is what
lets the rest of the code be tested with a mock client (``tests/mock_llm.py``,
task 10.2).

The public shape is three pieces:

- :class:`LlmClient` - a :class:`~typing.Protocol` describing the one method the
  rest of the app depends on, ``complete(request) -> LlmResponse`` (design "LLM
  Client"). Callers type against this protocol, so the real client and the test
  mock are interchangeable.
- :class:`LlmRequest` / :class:`LlmResponse` - plain dataclasses carrying what we
  send (a system instruction plus the user content, with optional model name,
  temperature, and a structured-output hint) and what we get back (the returned
  text and an optional parsed/structured payload).
- :class:`OpenAiLlmClient` - the concrete client. It reads the API key (and an
  optional base URL for OpenAI-compatible hosted APIs) from the environment at
  construction time, calls the chat-completions endpoint, and returns an
  :class:`LlmResponse`.

Failures are never swallowed. Any problem - a missing API key, an auth error, a
network error, or an API error - surfaces as an :class:`LlmError` so the calling
component can leave the Session unchanged and offer a retry (design "Error
Handling", Req 4.5, 5.4, 8.1-8.4).

Configuration (README "Configuration"):

- ``OPENAI_API_KEY`` (required) - the API key for the hosted, OpenAI-compatible
  model. Read in :meth:`OpenAiLlmClient.__init__`; a clear :class:`LlmError` is
  raised when it is unset.
- ``OPENAI_BASE_URL`` (optional) - a custom base URL for OpenAI-compatible hosted
  APIs. When unset, the OpenAI SDK's default endpoint is used. The README does
  not currently document this variable; it follows the OpenAI SDK's own
  convention so compatible hosts can be targeted without code changes.

Importing this module does no I/O and needs no API key: the key is read in
:meth:`OpenAiLlmClient.__init__`, and the ``openai`` SDK is imported lazily
inside that constructor, so ``import prompt_builder.llm_client`` works offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

__all__ = [
    "LlmClient",
    "LlmRequest",
    "LlmResponse",
    "LlmError",
    "OpenAiLlmClient",
    "API_KEY_ENV_VAR",
    "BASE_URL_ENV_VAR",
    "DEFAULT_MODEL",
]

# The environment variable the API key is read from (README "Configuration").
API_KEY_ENV_VAR = "OPENAI_API_KEY"

# The environment variable for an optional custom base URL. Follows the OpenAI
# SDK's own convention; lets an OpenAI-compatible hosted API be targeted without
# code changes. Not currently documented in the README.
BASE_URL_ENV_VAR = "OPENAI_BASE_URL"

# A sensible default chat model. Callers can override per request via
# ``LlmRequest.model`` or globally via ``OpenAiLlmClient(model=...)``.
DEFAULT_MODEL = "gpt-4o-mini"


def _read_config(name: str) -> Optional[str]:
    """Read a config value from the environment, then Streamlit secrets.

    Resolution order:

    1. The OS environment variable ``name`` (how the app is configured locally,
       documented in the README).
    2. Streamlit's ``st.secrets[name]`` when running inside a Streamlit app (how
       secrets are supplied on Streamlit Community Cloud, where OS environment
       variables cannot be set).

    The Streamlit lookup is best-effort and fully guarded: importing
    ``streamlit`` or accessing ``st.secrets`` can fail (Streamlit not installed,
    no ``secrets.toml`` present, or not running under a Streamlit runtime), and
    any such failure simply falls through to ``None``. This keeps
    ``llm_client`` free of a hard dependency on Streamlit and leaves the offline
    tests unaffected.
    """
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st  # local import; optional dependency at this layer

        secret = st.secrets.get(name)  # type: ignore[attr-defined]
        if secret:
            return str(secret)
    except Exception:  # noqa: BLE001 - any failure means "no secret available"
        return None
    return None


class LlmError(RuntimeError):
    """Raised when an LLM request cannot be completed.

    Covers every failure mode behind the one client: a missing/empty API key, a
    failure constructing the underlying SDK client, and any auth, network, or
    API error raised while calling the model. Callers catch this, leave the
    Session content and Prompt_Specification unchanged, and surface a retryable
    error state (design "Error Handling", Req 4.5, 5.4, 8.1-8.4).

    The originating exception (when there is one) is attached via ``raise ...
    from`` so the underlying cause is preserved for logging.
    """


@dataclass
class LlmRequest:
    """A single request to the language model.

    Attributes:
        prompt: The user-facing content to send (the message plus any rendered
            context the caller assembled).
        system: An optional system instruction that frames the task (for example
            "interpret this message" or "write a prompt"). Sent as the system
            message when present.
        structured: A hint that the caller wants a structured (JSON) object back.
            When ``True``, the client asks the model for a JSON response and the
            returned :class:`LlmResponse` carries the parsed object in
            ``structured`` when parsing succeeds. Defaults to ``False``.
        model: An optional model name overriding the client's default for this
            request.
        temperature: An optional sampling temperature overriding the client's
            default for this request. Controls randomness; keep low for factual /
            structured work, raise for creative work. (promptingguide.ai: LLM
            Settings - Temperature.)
        max_tokens: An optional hard cap on the number of tokens the model may
            generate for this response. Prevents truncated or runaway outputs and
            controls cost. (promptingguide.ai: Max Length.)
        top_p: An optional nucleus-sampling cutoff - only the smallest set of
            tokens whose cumulative probability reaches ``top_p`` is considered.
            Lower is more focused/deterministic. The guide recommends tuning
            temperature OR top_p, not both. (promptingguide.ai: Top P.)
        frequency_penalty: An optional penalty (typically -2.0..2.0) applied in
            proportion to how often a token has already appeared, reducing
            verbatim repetition. (promptingguide.ai: Frequency Penalty.)
        presence_penalty: An optional flat penalty (typically -2.0..2.0) on any
            token that has already appeared, nudging the model toward new topics.
            The guide recommends tuning frequency OR presence penalty, not both.
            (promptingguide.ai: Presence Penalty.)
        stop: An optional stop sequence or list of sequences that halt generation
            when produced, giving precise control over output length/structure.
            (promptingguide.ai: Stop Sequences.)
        operation: A short label naming what this call is for (``"interpret"``,
            ``"generate"``, or ``"revise"``). It is never sent to the model; the
            observability layer uses it to attribute latency, errors, and token
            usage to the right agent step. Defaults to ``"unknown"``.
    """

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
    """The model's answer to an :class:`LlmRequest`.

    Attributes:
        text: The returned text. Always populated (empty string when the model
            returned no content). String-only callers (such as the
            Prompt_Generator) read this directly.
        structured: The parsed structured payload when the request asked for
            structured output and the text parsed as a JSON object; otherwise
            ``None``. Structured-aware callers (such as the Intent_Analyzer)
            prefer this when present and fall back to parsing ``text``.
        model: The model that actually served the request, when the provider
            reports it. ``None`` otherwise.
        prompt_tokens: Input tokens billed for the request, when reported.
        completion_tokens: Output tokens generated, when reported.
        finish_reason: Why generation stopped (for example ``"stop"`` or
            ``"length"``). ``"length"`` means the output hit ``max_tokens`` and
            was truncated, which the observability layer tracks as an SLI.

    The four usage fields are optional and default to ``None`` so test doubles
    that only supply ``text``/``structured`` keep working unchanged.
    """

    text: str
    structured: Optional[dict[str, Any]] = None
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


@runtime_checkable
class LlmClient(Protocol):
    """The one interface the rest of the app depends on (design "LLM Client").

    A client takes a structured :class:`LlmRequest` and returns an
    :class:`LlmResponse`, raising on failure. The rest of the code types against
    this protocol, so the real :class:`OpenAiLlmClient` and the test mock
    (``tests/mock_llm.py``) are interchangeable.
    """

    def complete(self, request: LlmRequest) -> LlmResponse:
        """Send ``request`` to the model and return its response.

        Raises:
            Exception: on any failure (the concrete client raises
                :class:`LlmError`).
        """
        ...


class OpenAiLlmClient:
    """Concrete :class:`LlmClient` backed by the official OpenAI Python SDK.

    Talks to a hosted, OpenAI-compatible chat-completions API. The API key is
    read from the environment (:data:`API_KEY_ENV_VAR`) at construction time and
    an optional custom base URL from :data:`BASE_URL_ENV_VAR`, so an
    OpenAI-compatible host can be targeted without code changes. This is the only
    object in the app that touches the network.

    The ``openai`` package is imported lazily inside ``__init__`` so that merely
    importing this module stays offline and dependency-light.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Construct the client and read configuration from the environment.

        Args:
            model: The default chat model to use when a request does not specify
                one. Defaults to :data:`DEFAULT_MODEL`.
            temperature: The default sampling temperature used when a request
                does not specify one.
            api_key: An explicit API key. When ``None`` (the usual case), the key
                is read from the :data:`API_KEY_ENV_VAR` environment variable.
            base_url: An explicit base URL for an OpenAI-compatible host. When
                ``None``, the :data:`BASE_URL_ENV_VAR` environment variable is
                used if set; otherwise the SDK's default endpoint is used.

        Raises:
            LlmError: if no API key is available (neither ``api_key`` nor the
                environment variable), or if the underlying SDK client cannot be
                constructed.
        """
        key = api_key if api_key is not None else _read_config(API_KEY_ENV_VAR)
        if not key or not key.strip():
            raise LlmError(
                f"No API key found. Set the {API_KEY_ENV_VAR} environment "
                "variable (locally) or add it to Streamlit secrets (on Streamlit "
                "Community Cloud) to a valid key for the hosted, OpenAI-compatible API."
            )

        resolved_base_url = (
            base_url if base_url is not None else _read_config(BASE_URL_ENV_VAR)
        )

        self._model = model
        self._temperature = temperature

        # Import the SDK lazily so importing this module needs no network access,
        # no API key, and does not hard-require ``openai`` to be installed until a
        # real client is actually constructed.
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the SDK
            raise LlmError(
                "The 'openai' package is required for OpenAiLlmClient. Install "
                "it with 'pip install openai'."
            ) from exc

        # Only pass base_url through when one was provided, so the SDK default
        # endpoint is used otherwise.
        client_kwargs: dict[str, Any] = {"api_key": key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        try:
            self._client = OpenAI(**client_kwargs)
        except Exception as exc:  # noqa: BLE001 - surface any construction failure as LlmError
            raise LlmError(f"Could not initialize the OpenAI client: {exc}") from exc

    def complete(self, request: LlmRequest) -> LlmResponse:
        """Call the chat-completions API and return its answer (raises on failure).

        Builds the message list from the request's optional system instruction
        and its prompt, calls the model, and wraps the result in an
        :class:`LlmResponse`. When ``request.structured`` is set, the model is
        asked for a JSON object and the returned text is parsed into
        ``LlmResponse.structured`` on success (a parse failure leaves
        ``structured`` as ``None`` but still returns the raw text).

        Args:
            request: The request to send.

        Returns:
            The :class:`LlmResponse` carrying the returned text and, when
            applicable, the parsed structured payload.

        Raises:
            LlmError: on any auth, network, or API failure, or if the response
                has an unexpected shape.
        """
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
        # Pass through the optional sampling / length controls only when the
        # caller set them, so unset values fall back to the provider defaults
        # (promptingguide.ai LLM Settings: max length, top_p, frequency/presence
        # penalties, and stop sequences).
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            kwargs["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            kwargs["presence_penalty"] = request.presence_penalty
        if request.stop:
            kwargs["stop"] = request.stop
        if request.structured:
            # Ask the OpenAI-compatible API for a JSON object so the text parses
            # cleanly into LlmResponse.structured.
            kwargs["response_format"] = {"type": "json_object"}

        try:
            completion = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - any API/network error becomes LlmError
            raise LlmError(f"LLM request failed: {exc}") from exc

        text = self._extract_text(completion)

        structured: Optional[dict[str, Any]] = None
        if request.structured:
            structured = self._parse_structured(text)

        # Usage / finish reason feed the observability layer. Every read is
        # defensive: an OpenAI-compatible host may omit any of these, and a
        # missing value must never turn a successful call into a failure.
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
        """Pull the assistant message text out of a chat-completion response.

        Returns an empty string when the model produced no content. Raises
        :class:`LlmError` when the response does not have the expected shape, so
        a surprising payload is treated as a (retryable) failure rather than
        silently yielding nothing.
        """
        try:
            choice = completion.choices[0]
            content = choice.message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LlmError(
                f"Unexpected LLM response shape: {completion!r}"
            ) from exc
        return content or ""

    @staticmethod
    def _parse_structured(text: str) -> Optional[dict[str, Any]]:
        """Parse ``text`` as a JSON object, returning ``None`` on failure.

        Structured output is best-effort: a non-object or unparseable body simply
        yields ``None`` so the caller can fall back to the raw ``text`` rather
        than failing the whole request.
        """
        import json

        if not text or not text.strip():
            return None
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
