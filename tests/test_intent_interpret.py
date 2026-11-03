"""Tests for the LLM-backed intent interpretation (tasks 7.2 and 7.3).

These cover :meth:`prompt_builder.intent_analyzer.IntentAnalyzer.interpret`, the
LLM-backed entry point that turns a user message (plus retained Session context)
into an :class:`~prompt_builder.intent_analyzer.InterpretationResult`. The LLM
is replaced with the deterministic, offline :class:`tests.mock_llm.MockLlmClient`
(design "Testing Strategy"), so everything here runs fast, offline, and with no
Streamlit server.

Two things are checked:

- **Task 7.2 / Property 6 (Req 2.5):** a Hypothesis property test asserting that
  when the analyzer's confidence is below the acceptance threshold, the goal
  stays unset (``None``) — which is exactly what drives the Elicitation_Engine to
  ask the user to describe the goal.
- **Task 7.3 (Req 7.3, 7.4):** example-based unit tests asserting that the
  retained Session messages are passed to the LLM as context (7.3), and that an
  unreadable context yields interpretation in isolation with
  ``context_unavailable=True`` (7.4).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_builder.intent_analyzer import (
    DEFAULT_ACCEPTANCE_THRESHOLD,
    IntentAnalyzer,
    InterpretationResult,
)
from prompt_builder.models import Message, PromptSpecification

from tests.mock_llm import MockLlmClient


def _goal_follow_up_is_produced(
    result: InterpretationResult, spec: PromptSpecification
) -> bool:
    """Model Property 6's "a goal-describing follow-up is produced" clause.

    The deterministic condition that drives the Elicitation_Engine to ask the
    user to describe the goal is: the interpreted goal is unset *and* the
    specification still has no confirmed goal. Applying the interpretation to the
    spec (as the per-turn flow does) leaves ``spec.goal`` unset, so a follow-up
    asking the user to describe the goal is warranted (Req 2.5). Returns whether
    that goal-asking follow-up would be produced.
    """
    # Mirror how the app folds an interpretation into the spec: a confirmed goal
    # is stored, otherwise the goal stays as it was (here, unset).
    if result.goal is not None:
        spec.goal = result.goal
    spec.goal_confidence = result.confidence
    # With no confirmed goal in the spec, the user must be asked to describe it.
    return spec.goal is None


# ---------------------------------------------------------------------------
# Task 7.2 - Property 6: Low confidence leaves the goal unset and asks for it
# ---------------------------------------------------------------------------

# Feature: conversational-prompt-builder, Property 6: Low confidence leaves the goal unset and asks for it
#
# Property: when the analyzer's confidence is strictly below the acceptance
# threshold, the identified goal stays unset (None) so the Elicitation_Engine
# asks the user to describe the goal, no matter what goal string the LLM
# proposed. The mock LLM returns a structured payload carrying a generated
# confidence strictly below the threshold together with some goal string; the
# analyzer must still leave the goal None.
#
# **Validates: Requirements 2.5**

# Confidence values strictly below the default acceptance threshold (0.6). We
# subtract a tiny epsilon from the threshold so the sampled upper bound itself
# stays below it (floats are exclusive-ish; keep it safe).
_below_threshold_confidence = st.floats(
    min_value=0.0,
    max_value=DEFAULT_ACCEPTANCE_THRESHOLD - 1e-6,
    allow_nan=False,
    allow_infinity=False,
)

# Some non-empty goal string the LLM claims to have identified. It must be
# non-whitespace so the analyzer would otherwise treat it as a real goal.
_goal_strings = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=60,
)


@settings(max_examples=200)
@given(confidence=_below_threshold_confidence, goal=_goal_strings)
def test_low_confidence_leaves_goal_unset(confidence: float, goal: str) -> None:
    """Property 6: a below-threshold confidence leaves the goal None (Req 2.5).

    The LLM proposes a real goal, but because its confidence is below the
    acceptance threshold the goal is not confirmed and stays unset, which is what
    warrants the goal-describing follow-up question. The reported confidence is
    the (below-threshold) value, so a goal-eliciting follow-up is warranted.
    """
    # A dict canned response becomes the mock response's ``structured`` payload,
    # which is what the Intent_Analyzer reads.
    llm = MockLlmClient.with_responses(
        {"goal": goal, "confidence": confidence, "parameters": {}}
    )
    analyzer = IntentAnalyzer(llm)

    spec = PromptSpecification(goal_confidence=0.0)
    result = analyzer.interpret("please help me", context=[], spec=spec)

    # Goal stays unset: low confidence means the goal is not confirmed, so the
    # Elicitation_Engine is left to ask the user to describe it (Req 2.5).
    assert result.goal is None
    # The confidence that drives that goal question is below the threshold.
    assert result.confidence < DEFAULT_ACCEPTANCE_THRESHOLD
    # Second clause of Property 6: a follow-up asking the user to describe the
    # goal is produced, because folding this interpretation into the spec leaves
    # the goal unset.
    assert _goal_follow_up_is_produced(result, spec) is True
    assert spec.goal is None


@settings(max_examples=200)
@given(confidence=_below_threshold_confidence)
def test_low_confidence_goal_unset_even_without_goal(confidence: float) -> None:
    """A below-threshold confidence with a null goal also leaves the goal unset.

    Covers the companion case where the LLM itself returned no goal: the result
    must still have ``goal is None`` and a below-threshold confidence.
    """
    llm = MockLlmClient.with_responses(
        {"goal": None, "confidence": confidence, "parameters": {}}
    )
    analyzer = IntentAnalyzer(llm)

    spec = PromptSpecification(goal_confidence=0.0)
    result = analyzer.interpret("something vague", context=[], spec=spec)

    assert result.goal is None
    assert result.confidence < DEFAULT_ACCEPTANCE_THRESHOLD
    # The goal-describing follow-up is produced here too (goal stays unset).
    assert _goal_follow_up_is_produced(result, spec) is True
    assert spec.goal is None


# ---------------------------------------------------------------------------
# Task 7.3 - Unit tests for context handling (Req 7.3, 7.4)
# ---------------------------------------------------------------------------


def _msg(sender: str, text: str, created_at: float) -> Message:
    """Build a Message with the real dataclass shape."""
    return Message(
        id=f"m-{created_at}", sender=sender, text=text, created_at=created_at
    )


def test_retained_messages_are_passed_as_context() -> None:
    """Req 7.3: the analyzer passes the retained session messages as context.

    Build a context list of Messages, interpret a new message, then inspect the
    mock's recorded ``calls`` (the ``LlmRequest`` objects it received) and assert
    the request prompt includes each retained message's text — the analyzer
    renders the retained Session content into the prompt it sends the LLM.
    """
    from prompt_builder.models import PromptSpecification

    context = [
        _msg("user", "I want to write a blog post", 1.0),
        _msg("system", "Who is the audience?", 2.0),
        _msg("user", "Software developers", 3.0),
    ]

    llm = MockLlmClient.with_responses(
        {"goal": "write a blog post", "confidence": 0.9, "parameters": {}}
    )
    analyzer = IntentAnalyzer(llm)

    analyzer.interpret(
        "Make it technical",
        context=context,
        spec=PromptSpecification(goal_confidence=0.0),
    )

    # The mock recorded exactly one request.
    assert len(llm.calls) == 1
    request = llm.calls[0]

    # The request content carries the rendered context. The LlmRequest exposes
    # the human-readable prompt on ``.prompt``.
    prompt = request.prompt
    for message in context:
        assert message.text in prompt, (
            f"retained message text {message.text!r} missing from prompt:\n{prompt}"
        )

    # The latest message is included too, so the LLM has both context and input.
    assert "Make it technical" in prompt


def test_unreadable_context_sets_flag_and_still_returns_result() -> None:
    """Req 7.4: unreadable context -> interpret in isolation with the flag set.

    Simulate retained context that cannot be read by passing an object that
    raises as soon as the analyzer tries to iterate it. ``_render_context`` then
    raises, interpret() falls back to interpreting the message on its own, sets
    ``context_unavailable=True``, and still returns a usable result.
    """
    from prompt_builder.models import PromptSpecification

    class UnreadableContext:
        """A stand-in for retained context that cannot be accessed."""

        def __iter__(self):
            raise RuntimeError("retained context is unreadable")

    llm = MockLlmClient.with_responses(
        {"goal": "do the thing", "confidence": 0.95, "parameters": {}}
    )
    analyzer = IntentAnalyzer(llm)

    result = analyzer.interpret(
        "help me",
        context=UnreadableContext(),  # type: ignore[arg-type]
        spec=PromptSpecification(goal_confidence=0.0),
    )

    # The context could not be read, so the message was interpreted in isolation
    # and the flag is set (Req 7.4).
    assert result.context_unavailable is True

    # Interpretation still produced a result: the LLM was still called and its
    # answer flowed through.
    assert llm.call_count == 1
    assert result.goal == "do the thing"
    assert result.confidence == 0.95

    # And because context was treated as empty, its (absent) text is not in the
    # prompt; the current message still is.
    request = llm.calls[0]
    assert "help me" in request.prompt


def test_malformed_message_in_context_sets_flag() -> None:
    """Req 7.4: a malformed entry in the context list also flags unavailability.

    A context list containing an object that lacks the Message shape makes
    ``_render_context`` raise while accessing ``.sender``/``.text``; interpret()
    must treat the whole context as unavailable rather than failing.
    """
    from prompt_builder.models import PromptSpecification

    class Malformed:
        """An entry that explodes when its attributes are read."""

        @property
        def sender(self):  # pragma: no cover - exact raising path may vary
            raise AttributeError("no sender")

        @property
        def text(self):  # pragma: no cover - exact raising path may vary
            raise AttributeError("no text")

    context = [
        _msg("user", "a real first message", 1.0),
        Malformed(),  # type: ignore[list-item]
    ]

    llm = MockLlmClient.with_responses(
        {"goal": "a goal", "confidence": 0.8, "parameters": {}}
    )
    analyzer = IntentAnalyzer(llm)

    result = analyzer.interpret(
        "second message",
        context=context,  # type: ignore[arg-type]
        spec=PromptSpecification(goal_confidence=0.0),
    )

    assert result.context_unavailable is True
    # Interpretation still returned a result.
    assert llm.call_count == 1
    assert result.goal == "a goal"
