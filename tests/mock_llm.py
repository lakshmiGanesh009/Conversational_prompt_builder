"""A deterministic, offline mock LLM client for the test suite.

Task 10.2: the real network-touching client lives in
``prompt_builder/llm_client.py`` (task 10.1) and is the only module that calls
the hosted language-model API. The property and unit tests, however, must run
fast, deterministically, and completely offline (design "Testing Strategy":
"LLM calls are replaced with a mock LLM client in these tests"). This module
provides that mock.

The real client exposes a tiny structural interface (design "LLM Client"):

    class LlmClient(Protocol):
        def complete(self, request: LlmRequest) -> LlmResponse: ...  # raises on failure

Because task 10.1 is being implemented concurrently and its concrete
``LlmRequest`` / ``LlmResponse`` shapes may still be in flux, this mock is kept
self-contained: it accepts *any* request object (it only ever reads a best-effort
``.prompt`` / ``.text`` / ``.content`` attribute, and otherwise ignores the
request) and returns its own lightweight :class:`MockLlmResponse`. That response
carries both a ``text`` string and an optional ``structured`` payload, so it is
structurally compatible with whatever ``LlmResponse`` the real client settles on
(callers that read ``response.text`` keep working). Nothing here imports the real
client, so the two files can be developed independently without conflict.

Three behaviours are supported, matching what the surrounding tests need:

1. **Canned structured responses** - construct the mock with a queue of
   :class:`MockLlmResponse` objects (or plain strings / dicts, which are coerced)
   and each ``complete`` call pops the next one. This drives interpretation and
   question-phrasing tests with known outputs.
2. **Echoing structured content** - the assembly tests (Property 13) need every
   collected parameter to appear verbatim in the generated text. An echo-mode
   client turns the structured content handed to it into a deterministic text
   blob, so the test can assert each collected value is present.
3. **Raising on demand** - error-path tests (generation/revision/response
   failure, Req 4.5, 5.4, 6.4, 8.1-8.4) need the client to fail. A failing mock
   raises :class:`MockLlmError` from ``complete``.

The mock also records every request it received in ``calls`` so tests can assert
on what was sent (for example, that retained session messages were passed as
context, Req 7.3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union


class MockLlmError(RuntimeError):
    """Raised by a failing mock LLM client to simulate an LLM/network failure.

    The real :class:`~prompt_builder.llm_client.LlmClient` "raises on failure";
    callers catch that exception and turn it into a retryable error state
    (design "Error Handling", Req 4.5 / 5.4 / 8.1-8.4). Tests use this concrete
    type so they can assert the failure path precisely.
    """


@dataclass
class MockLlmResponse:
    """A canned response returned by the mock client.

    Attributes:
        text: The free-text body. Callers that treat the LLM result as a plain
            string (for example the Prompt_Generator) read this.
        structured: An optional structured payload (for example extracted
            parameters, a goal, and a confidence level for the Intent_Analyzer).
            Kept as a plain ``dict`` so tests can shape it freely.
    """

    text: str = ""
    structured: Optional[dict[str, Any]] = None


# What a test may hand to the mock as a canned response: a ready response, a bare
# string (becomes the ``text``), or a dict (becomes the ``structured`` payload).
CannedResponse = Union[MockLlmResponse, str, dict]


def _coerce_response(value: CannedResponse) -> MockLlmResponse:
    """Coerce a test-supplied canned value into a :class:`MockLlmResponse`."""
    if isinstance(value, MockLlmResponse):
        return value
    if isinstance(value, str):
        return MockLlmResponse(text=value)
    if isinstance(value, dict):
        # Preserve the dict as structured content and also expose a readable
        # text form so string-only callers still get something meaningful.
        return MockLlmResponse(text=_stringify_structured(value), structured=value)
    raise TypeError(f"Unsupported canned response type: {type(value)!r}")


def _extract_prompt_text(request: Any) -> str:
    """Best-effort read of the human-readable prompt text out of a request.

    The real ``LlmRequest`` shape is still being finalised in task 10.1, so this
    tries the most likely attribute names and falls back to ``str(request)``.
    Used only for echo mode and for recording calls; the mock never depends on a
    specific request shape.
    """
    for attr in ("prompt", "text", "content", "message"):
        value = getattr(request, attr, None)
        if isinstance(value, str):
            return value
    if isinstance(request, str):
        return request
    return str(request)


def _stringify_structured(structured: dict[str, Any]) -> str:
    """Render a structured payload as deterministic, readable text.

    Keys are sorted so the output is stable across runs (important for
    deterministic tests). Every value appears verbatim, which is what the
    assembly property test relies on to assert each collected parameter is
    present in the generated prompt (Property 13).
    """
    lines: list[str] = []
    for key in sorted(structured):
        value = structured[key]
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, sort_keys=True, ensure_ascii=False)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines)


class MockLlmClient:
    """A deterministic, offline stand-in for the real ``LlmClient``.

    Structurally compatible with the design's ``LlmClient`` Protocol: it exposes
    a ``complete(request) -> MockLlmResponse`` method that raises on failure. It
    never touches the network.

    Construct it one of three ways (see the module docstring):

    - ``MockLlmClient(responses=[...])`` - pop a canned response per call.
    - ``MockLlmClient.echoing()`` - echo the request's structured content so
      assembly tests can assert collected values are present.
    - ``MockLlmClient.failing()`` - raise :class:`MockLlmError` on every call.

    Every call is recorded in ``calls`` (the raw request objects) so tests can
    assert on what was sent.
    """

    def __init__(
        self,
        responses: Optional[list[CannedResponse]] = None,
        *,
        default: Optional[CannedResponse] = None,
        fail: bool = False,
        error: Optional[Exception] = None,
        echo: bool = False,
        responder: Optional[Callable[[Any], MockLlmResponse]] = None,
    ) -> None:
        """Create a mock client.

        Args:
            responses: A queue of canned responses, consumed in order, one per
                ``complete`` call. When the queue is exhausted, ``default`` (if
                given) is returned on every subsequent call; otherwise calling
                again raises :class:`AssertionError` so tests notice an
                unexpected extra call.
            default: A fallback response returned once ``responses`` is empty.
            fail: When ``True``, every ``complete`` call raises.
            error: The specific exception to raise when failing. Defaults to a
                :class:`MockLlmError`.
            echo: When ``True``, ``complete`` echoes the request's structured
                content back as the response text (assembly tests).
            responder: An optional callable mapping a request to a
                :class:`MockLlmResponse`, for fully custom behaviour. Takes
                precedence over ``responses``/``default``/``echo`` when provided.
        """
        self._responses: list[MockLlmResponse] = [
            _coerce_response(r) for r in (responses or [])
        ]
        self._default: Optional[MockLlmResponse] = (
            _coerce_response(default) if default is not None else None
        )
        self._fail = fail
        self._error = error
        self._echo = echo
        self._responder = responder
        # A record of every request passed to ``complete``, in call order.
        self.calls: list[Any] = []

    # --- Convenience constructors -------------------------------------------

    @classmethod
    def with_responses(cls, *responses: CannedResponse) -> "MockLlmClient":
        """Build a client that returns the given canned responses in order."""
        return cls(responses=list(responses))

    @classmethod
    def echoing(cls) -> "MockLlmClient":
        """Build an echo-mode client for assembly tests (Property 13).

        Each ``complete`` call returns a response whose ``text`` is the request's
        structured content rendered deterministically, so every collected
        parameter value appears verbatim and can be asserted present.
        """
        return cls(echo=True)

    @classmethod
    def failing(cls, error: Optional[Exception] = None) -> "MockLlmClient":
        """Build a client that raises on every call (error-path tests)."""
        return cls(fail=True, error=error)

    # --- The LlmClient interface --------------------------------------------

    def complete(self, request: Any) -> MockLlmResponse:
        """Return the next canned/echoed response, or raise if configured to fail.

        Mirrors ``LlmClient.complete``: returns an ``LlmResponse``-compatible
        object and raises on failure. Records ``request`` in ``calls`` before
        doing anything else, so even a failing call is observable by tests.
        """
        self.calls.append(request)

        if self._fail:
            raise self._error if self._error is not None else MockLlmError(
                "mock LLM client configured to fail"
            )

        if self._responder is not None:
            return self._responder(request)

        if self._echo:
            return MockLlmResponse(text=_extract_prompt_text(request))

        if self._responses:
            return self._responses.pop(0)

        if self._default is not None:
            return self._default

        raise AssertionError(
            "MockLlmClient.complete called more times than canned responses "
            "were provided (and no default was set)"
        )

    @property
    def call_count(self) -> int:
        """How many times ``complete`` has been called."""
        return len(self.calls)
