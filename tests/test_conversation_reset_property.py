"""Property test for new-session clearing (task 4.4).

Covers Property 17 from the design: starting a new Session clears everything.
Regardless of any prior Session state — arbitrary messages, a populated
Prompt_Specification, collected parameters, an overwrite history, a generated
prompt, a non-zero follow-up count, or unavailable context — ``reset_session``
yields a fresh Session with no messages and an empty Prompt_Specification.

This test lives in its own file (separate from the other Conversation_Manager
property tests in tasks 4.2/4.3) to keep the properties cleanly isolated.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_builder.conversation_manager import reset_session
from prompt_builder.models import (
    GeneratedPrompt,
    Message,
    Overwrite,
    ParameterValue,
    PromptSpecification,
    Session,
)

# --- Generators that build arbitrary prior Session state ---------------------

senders = st.sampled_from(["user", "system"])

messages = st.builds(
    Message,
    id=st.text(min_size=1, max_size=12),
    sender=senders,
    text=st.text(min_size=0, max_size=50),
    created_at=st.floats(min_value=0.0, max_value=1e9, allow_nan=False),
)

parameter_values = st.builds(
    ParameterValue,
    name=st.text(min_size=1, max_size=12),
    value=st.text(min_size=0, max_size=20),
    status=st.sampled_from(["collected", "declined"]),
    attempts=st.integers(min_value=0, max_value=3),
)

overwrites = st.builds(
    Overwrite,
    parameter=st.text(min_size=1, max_size=12),
    previous=st.text(max_size=20),
    next=st.text(max_size=20),
)

specs = st.builds(
    PromptSpecification,
    goal_confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    goal=st.one_of(st.none(), st.text(min_size=1, max_size=40)),
    parameters=st.dictionaries(
        keys=st.text(min_size=1, max_size=12),
        values=parameter_values,
        max_size=5,
    ),
    overwrite_log=st.lists(overwrites, max_size=5),
)

generated_prompts = st.builds(
    GeneratedPrompt,
    text=st.text(max_size=50),
    omitted_required=st.lists(st.text(min_size=1, max_size=12), max_size=5),
)

prior_sessions = st.builds(
    Session,
    id=st.text(min_size=1, max_size=12),
    messages=st.lists(messages, max_size=10),
    spec=st.one_of(st.none(), specs),
    follow_up_count=st.integers(min_value=0, max_value=20),
    generated_prompt=st.one_of(st.none(), generated_prompts),
    context_available=st.booleans(),
)


# Feature: conversational-prompt-builder, Property 17: Starting a new session clears everything
# Validates: Requirements 6.5, 7.5
@settings(max_examples=200)
@given(prior=prior_sessions)
def test_starting_a_new_session_clears_everything(prior: Session) -> None:
    # A prior Session exists in whatever state. Starting a new Session must not
    # depend on it; reset_session() builds a fresh, empty Session.
    fresh = reset_session()

    # It is a Session with a fresh identifier distinct from nothing in
    # particular — it simply must be a valid, non-empty id.
    assert isinstance(fresh, Session)
    assert fresh.id

    # No messages carried over from any prior state (Req 7.5).
    assert fresh.messages == []

    # An empty Prompt_Specification: no goal, zero confidence, and no collected
    # parameters or overwrite history (Req 6.5).
    spec = fresh.spec
    assert isinstance(spec, PromptSpecification)
    assert spec.goal is None
    assert spec.goal_confidence == 0.0
    assert spec.parameters == {}
    assert spec.overwrite_log == []
