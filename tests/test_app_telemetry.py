"""App-level instrumentation: user actions produce the right telemetry.

These drive the real Streamlit app through ``AppTest`` (offline, with the mock
LLM client injected) and then assert exactly what the process-wide metrics store
recorded. The store is emptied around every test by ``conftest.py``.

Injection point: as in ``test_app.py``, the tests patch
``prompt_builder.llm_client.OpenAiLlmClient`` before ``.run()`` so the freshly
executed ``app.py`` binds the patched class and ``_build_llm_client`` wraps the
mock in the instrumenting client.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from prompt_builder.llm_client import LlmError
from prompt_builder.telemetry import get_store
from tests.mock_llm import MockLlmClient

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
CLIENT_SYMBOL = "prompt_builder.llm_client.OpenAiLlmClient"


def _turn_client() -> MockLlmClient:
    """First call interprets (fills both required params); later calls generate."""
    return MockLlmClient(
        responses=[
            {
                "goal": "Write a product description",
                "confidence": 0.95,
                "parameters": {"audience": "developers", "domain": "software"},
            }
        ],
        default="A polished generated prompt.",
    )


def _submit(client, text: str) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    with patch(CLIENT_SYMBOL, return_value=client):
        at.run()
        at.chat_input[0].set_value(text).run()
    assert not at.exception
    return at


def _names(events):
    return [(e.name, e.success) for e in events]


def test_a_successful_turn_records_both_agent_calls_and_the_turn() -> None:
    _submit(_turn_client(), "I want a product description")

    store = get_store()
    assert [(c.operation, c.success) for c in store.calls()] == [
        ("interpret", True),
        ("generate", True),
    ]
    assert ("prompt_generated", True) in _names(store.events())  # both required params present
    [turn] = [e for e in store.events() if e.name == "turn"]
    assert turn.success is True
    assert turn.duration_s is not None and turn.duration_s >= 0


def test_a_failed_model_call_is_recorded_and_the_turn_is_marked_failed() -> None:
    _submit(MockLlmClient.failing(error=LlmError("upstream down")), "write a post")

    store = get_store()
    [call] = store.calls()
    assert (call.operation, call.success, call.error_type) == ("interpret", False, "LlmError")
    assert _names(store.events()) == [("turn", False)]


def test_a_missing_api_key_counts_as_a_failed_client_init_call() -> None:
    """No key means the client cannot even be built; that must still be visible."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    with patch("prompt_builder.llm_client._read_config", return_value=None):
        at.run()
        at.chat_input[0].set_value("write a post").run()
    assert not at.exception

    [call] = get_store().calls()
    assert (call.operation, call.success) == ("client_init", False)
    assert "No API key" in call.error_message
    assert ("turn", False) in _names(get_store().events())


def test_a_rejected_input_is_recorded_without_any_model_call() -> None:
    _submit(MockLlmClient.with_responses(), "   ")

    store = get_store()
    assert store.calls() == []
    assert _names(store.events()) == [("validation_rejected", None)]


def test_opening_a_demo_example_is_recorded_without_any_model_call() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    next(b for b in at.button if b.key and b.key.startswith("demo_")).click().run()
    assert not at.exception

    store = get_store()
    assert store.calls() == []
    assert _names(store.events()) == [("demo_loaded", None)]


def test_user_text_and_prompts_are_never_stored_in_telemetry() -> None:
    """Telemetry holds timings, counts and tokens only: no message or prompt content."""
    secret = "SECRET-MARKER-98765"
    _submit(_turn_client(), f"I want a product description {secret}")

    store = get_store()
    dumped = repr(store.calls()) + repr(store.events())
    assert secret not in dumped
    assert "polished generated prompt" not in dumped
