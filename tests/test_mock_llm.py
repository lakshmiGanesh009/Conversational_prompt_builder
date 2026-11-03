"""Unit tests for the deterministic mock LLM client (task 10.2).

These checks make sure the test-support double behaves as the surrounding
property and unit tests rely on: canned responses come back in order, structured
content is echoed verbatim (so assembly tests can assert collected values are
present), failures raise, and every request is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tests.mock_llm import (
    MockLlmClient,
    MockLlmError,
    MockLlmResponse,
    _stringify_structured,
)


@dataclass
class _FakeRequest:
    """A stand-in request object with the most likely attribute name."""

    prompt: str


def test_canned_responses_returned_in_order():
    client = MockLlmClient.with_responses("first", "second")

    assert client.complete(_FakeRequest("a")).text == "first"
    assert client.complete(_FakeRequest("b")).text == "second"


def test_dict_canned_response_is_structured_and_stringified():
    client = MockLlmClient.with_responses({"goal": "write a poem", "confidence": 0.9})

    response = client.complete(_FakeRequest("x"))

    assert response.structured == {"goal": "write a poem", "confidence": 0.9}
    # Every value appears verbatim in the text form.
    assert "write a poem" in response.text
    assert "0.9" in response.text


def test_default_used_after_queue_exhausted():
    client = MockLlmClient(responses=["only"], default="fallback")

    assert client.complete(_FakeRequest("a")).text == "only"
    assert client.complete(_FakeRequest("b")).text == "fallback"
    assert client.complete(_FakeRequest("c")).text == "fallback"


def test_overcall_without_default_raises_assertion():
    client = MockLlmClient.with_responses("one")

    client.complete(_FakeRequest("a"))
    with pytest.raises(AssertionError):
        client.complete(_FakeRequest("b"))


def test_echo_mode_returns_structured_content_verbatim():
    client = MockLlmClient.echoing()
    # Simulate the structured prompt the Prompt_Generator would send.
    structured_prompt = _stringify_structured(
        {"audience": "students", "tone": "friendly", "domain": "biology"}
    )

    response = client.complete(_FakeRequest(structured_prompt))

    # Every collected value must survive into the generated text (Property 13).
    for value in ("students", "friendly", "biology"):
        assert value in response.text


def test_failing_client_raises_mock_error():
    client = MockLlmClient.failing()

    with pytest.raises(MockLlmError):
        client.complete(_FakeRequest("a"))


def test_failing_client_raises_custom_error():
    boom = ValueError("custom boom")
    client = MockLlmClient.failing(error=boom)

    with pytest.raises(ValueError, match="custom boom"):
        client.complete(_FakeRequest("a"))


def test_calls_are_recorded_even_on_failure():
    client = MockLlmClient.failing()
    request = _FakeRequest("remember me")

    with pytest.raises(MockLlmError):
        client.complete(request)

    assert client.call_count == 1
    assert client.calls == [request]


def test_responder_takes_precedence():
    client = MockLlmClient(
        responder=lambda req: MockLlmResponse(text=f"saw: {req.prompt}")
    )

    assert client.complete(_FakeRequest("hello")).text == "saw: hello"


def test_plain_string_request_is_handled_in_echo_mode():
    client = MockLlmClient.echoing()

    # Even a bare string request works (no specific request shape required).
    assert client.complete("raw prompt text").text == "raw prompt text"
