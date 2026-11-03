"""Integration timing tests for the Conversational Prompt Builder (task 16.1).

The design's timing-related acceptance criteria say several operations must
resolve *promptly* and that the in-progress indicator must appear and then clear
on resolution. These are wall-clock, end-to-end concerns rather than universal
properties, so this module uses a handful of representative examples (1-3 each)
instead of Hypothesis, and measures elapsed time with ``time.perf_counter``.

Everything here runs completely offline using the mock LLM client
(``tests/mock_llm.py``), so each LLM-backed step is effectively instantaneous.
That makes the timing assertions meaningful in a specific way: a generous
upper bound (:data:`BUDGET_SECONDS`) still *fails loudly* if an operation hangs,
blocks, or accidentally performs real network I/O, while never being flaky for a
fast offline run. We are asserting "resolves promptly and does not hang", which
is exactly what the timing criteria require.

Requirement coverage (one or two representative examples each):

- Req 2.1 - interpretation completes within its budget
  (``test_interpretation_completes_within_budget``).
- Req 4.3 - the generated prompt is produced (ready to render) promptly after
  production (``test_generated_prompt_ready_within_budget``).
- Req 5.2 - revision completes within its budget
  (``test_revision_completes_within_budget``).
- Req 6.3 - the in-progress indicator appears quickly and clears on resolution
  (``test_in_progress_indicator_clears_on_resolution``), exercised through the
  Streamlit app with ``AppTest`` and a patched mock client.
- Req 7.5 - the new-session view clears within its budget
  (``test_new_session_clears_within_budget`` and
  ``test_new_session_button_clears_view_within_budget``).
- Req 8.3 - retry re-executes and resolves within budget
  (``test_retry_resolves_within_budget`` for the eventual-success path and
  ``test_retry_cap_resolves_within_budget`` for the capped-failure path).
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest import mock

import pytest

from prompt_builder import conversation_manager
from prompt_builder.intent_analyzer import IntentAnalyzer
from prompt_builder.models import GeneratedPrompt, Message, PromptSpecification
from prompt_builder.prompt_generator import PromptGenerator
from prompt_builder.retry_controller import (
    COULD_NOT_COMPLETE_MESSAGE,
    RetryController,
)
from tests.mock_llm import MockLlmClient, MockLlmError

# A generous wall-clock upper bound, reused by every timing assertion. The mock
# LLM client is instantaneous, so any single operation should finish in well
# under this. The budget is intentionally loose: it is not a benchmark, it is a
# guard that fails if an operation hangs or does real I/O.
BUDGET_SECONDS = 2.0

# The Streamlit entry point, resolved to an absolute path from this file's
# location (``tests/``) up to the project root. AppTest.from_file resolves a
# relative path against the *calling* file's directory, so an absolute path keeps
# the lookup correct regardless of where pytest is invoked.
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


# --- helpers ----------------------------------------------------------------
def _spec_with_goal() -> PromptSpecification:
    """A representative spec with a goal set, for generation/revision timing."""
    return PromptSpecification(goal_confidence=0.9, goal="Write a product description")


# --- Req 2.1: interpretation completes within its budget --------------------
def test_interpretation_completes_within_budget() -> None:
    """Interpreting a message resolves within the budget and returns a result (Req 2.1)."""
    llm = MockLlmClient.with_responses(
        {
            "goal": "Write a product description",
            "confidence": 0.9,
            "parameters": {"audience": "developers", "tone": "friendly"},
        }
    )
    analyzer = IntentAnalyzer(llm)
    spec = PromptSpecification(goal_confidence=0.0)
    context: list[Message] = []

    start = time.perf_counter()
    result = analyzer.interpret("I want a product description", context, spec)
    elapsed = time.perf_counter() - start

    assert elapsed < BUDGET_SECONDS
    # A usable result is returned: the goal cleared the acceptance threshold and
    # the extracted parameters came through.
    assert result.goal == "Write a product description"
    assert result.confidence == pytest.approx(0.9)
    assert {p.name for p in result.extracted} == {"audience", "tone"}


# --- Req 4.3: generated prompt is ready to render promptly ------------------
def test_generated_prompt_ready_within_budget() -> None:
    """Generation produces a renderable GeneratedPrompt within the budget (Req 4.3).

    Req 4.3 is about the produced prompt rendering promptly after production. The
    thing that must be ready is the :class:`GeneratedPrompt`; once it exists the
    UI renders it synchronously. So we time production and assert the prompt is
    present (its text is ready to render) within the budget.
    """
    llm = MockLlmClient.echoing()
    generator = PromptGenerator(llm)
    spec = _spec_with_goal()

    start = time.perf_counter()
    result = generator.generate(spec)
    elapsed = time.perf_counter() - start

    assert elapsed < BUDGET_SECONDS
    assert isinstance(result.prompt, GeneratedPrompt)
    # The produced text is ready to render (non-empty: the echo includes the goal).
    assert result.prompt.text.strip() != ""


# --- Req 5.2: revision completes within its budget --------------------------
def test_revision_completes_within_budget() -> None:
    """Revising a prompt resolves within the budget and yields a new prompt (Req 5.2)."""
    llm = MockLlmClient.echoing()
    generator = PromptGenerator(llm)
    spec = _spec_with_goal()
    current = GeneratedPrompt(text="Original prompt text.", omitted_required=[])

    start = time.perf_counter()
    result = generator.revise(current, "Make it more concise", spec)
    elapsed = time.perf_counter() - start

    assert elapsed < BUDGET_SECONDS
    assert isinstance(result.prompt, GeneratedPrompt)
    assert result.prompt.text.strip() != ""


# --- Req 8.3: retry re-executes and resolves within budget ------------------
def test_retry_resolves_within_budget() -> None:
    """A retried operation that fails twice then succeeds resolves within budget (Req 8.3).

    The controller re-executes the failing operation and resolves to success on
    the third attempt. We assert it both resolves quickly and reports the
    expected outcome (success on attempt 3).
    """
    attempts = {"count": 0}

    def operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise MockLlmError("transient failure")
        return "done"

    controller: RetryController[str] = RetryController(retry_on=(MockLlmError,))

    start = time.perf_counter()
    result = controller.run(operation)
    elapsed = time.perf_counter() - start

    assert elapsed < BUDGET_SECONDS
    assert result.succeeded is True
    assert result.attempts == 3
    assert result.value == "done"


def test_retry_cap_resolves_within_budget() -> None:
    """An always-failing operation resolves (to the cap) within budget (Req 8.3).

    Even when every attempt fails, the controller must stop at the cap and report
    "could not be completed" promptly rather than looping or hanging.
    """

    def operation() -> str:
        raise MockLlmError("always fails")

    controller: RetryController[str] = RetryController(retry_on=(MockLlmError,))

    start = time.perf_counter()
    result = controller.run(operation)
    elapsed = time.perf_counter() - start

    assert elapsed < BUDGET_SECONDS
    assert result.succeeded is False
    assert result.attempts == controller.max_attempts
    assert result.message == COULD_NOT_COMPLETE_MESSAGE


# --- Req 7.5: new-session view clears within its budget ---------------------
def test_new_session_clears_within_budget() -> None:
    """Resetting the session clears messages and spec within the budget (Req 7.5)."""
    # Start from a populated session so "clears everything" is meaningful.
    session = conversation_manager.start_session("some earlier message")

    start = time.perf_counter()
    fresh = conversation_manager.reset_session()
    elapsed = time.perf_counter() - start

    assert elapsed < BUDGET_SECONDS
    assert fresh.messages == []
    # An empty spec: no goal and no collected parameters.
    assert fresh.spec is not None
    assert fresh.spec.goal is None
    assert fresh.spec.parameters == {}
    # The reset really produced a different, clean session.
    assert fresh.id != session.id


# --- AppTest-driven timing (Req 6.3 and Req 7.5 through the UI) -------------
#
# These two exercise the real Streamlit app end-to-end with AppTest, patching
# ``app._build_llm_client`` so no network is touched. AppTest runs the script to
# completion synchronously, so a run that returns within the budget means the
# in-progress spinner that wraps each LLM-backed step has been entered *and*
# exited (the spinner is a context manager; the script cannot finish with it
# still open). AppTest does not expose a live spinner element to introspect, so
# we assert the full ``.run()` resolves within budget and the resulting state is
# resolved (session updated, with either a produced prompt or a surfaced error).


def _mock_client_for_turn() -> MockLlmClient:
    """A mock client that drives a full turn: interpret, then generate.

    The app turn flow may call the client once (interpretation) and again
    (generation) within a single submission, so a ``default`` keeps later calls
    from raising. The first canned response identifies a high-confidence goal and
    fills both required parameters (audience, domain) so the flow proceeds
    straight to generation rather than stopping to elicit more.
    """
    return MockLlmClient(
        responses=[
            {
                "goal": "Write a product description",
                "confidence": 0.95,
                "parameters": {
                    "audience": "developers",
                    "domain": "software",
                    "tone": "friendly",
                },
            }
        ],
        default="A polished generated prompt for the product description.",
    )


def test_in_progress_indicator_clears_on_resolution() -> None:
    """Submitting a message resolves within budget with no spinner left open (Req 6.3).

    We submit a chat message and run the app. Because the in-progress indicator
    (``st.spinner``) wraps each processing step as a context manager, the only way
    ``.run()`` can return is for every spinner to have been entered and exited
    (cleared) on resolution. We assert the run completes within the budget, does
    not raise, and leaves a resolved state (a session with the submitted message
    recorded).
    """
    from streamlit.testing.v1 import AppTest

    with mock.patch("app._build_llm_client", return_value=_mock_client_for_turn()):
        app_test = AppTest.from_file(APP_PATH, default_timeout=BUDGET_SECONDS)
        app_test.run()

        # Submit a user message through the chat input and process it.
        app_test.chat_input[0].set_value("I want a product description").run()

        start = time.perf_counter()
        app_test.run()
        elapsed = time.perf_counter() - start

    # The run resolved within budget and did not raise (AppTest records uncaught
    # exceptions rather than propagating them).
    assert elapsed < BUDGET_SECONDS
    assert not app_test.exception
    # Resolution is observable: the submitted message is in the rendered history,
    # which only happens after the spinner-wrapped turn flow completed. Each
    # message is rendered as markdown inside a chat_message container, so the
    # submitted text appears among the app's markdown elements.
    markdown_texts = [block.value for block in app_test.markdown]
    assert any("I want a product description" in text for text in markdown_texts)


def test_new_session_button_clears_view_within_budget() -> None:
    """Clicking "New session" clears the conversation view within budget (Req 7.5).

    Drives the same end-to-end path a user takes: submit a message so there is
    something to clear, then click the New session button and assert the rendered
    conversation is empty afterwards, all within the budget.
    """
    from streamlit.testing.v1 import AppTest

    with mock.patch("app._build_llm_client", return_value=_mock_client_for_turn()):
        app_test = AppTest.from_file(APP_PATH, default_timeout=BUDGET_SECONDS)
        app_test.run()
        app_test.chat_input[0].set_value("I want a product description").run()

        # Sanity: there is a conversation to clear.
        assert len(app_test.chat_message) >= 1

        new_session_button = next(
            button for button in app_test.button if button.key == "new_session_button"
        )

        start = time.perf_counter()
        new_session_button.click().run()
        elapsed = time.perf_counter() - start

    assert elapsed < BUDGET_SECONDS
    assert not app_test.exception
    # The view cleared: no conversation messages are rendered after reset.
    assert len(app_test.chat_message) == 0
