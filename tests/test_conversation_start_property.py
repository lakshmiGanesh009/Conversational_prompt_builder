"""Property test for new-session creation (task 4.2).

Covers Property 1 from the design: starting a Session from a non-empty first
message yields a Session that holds exactly that message (as a user message) and
whose Prompt_Specification is empty — no goal, confidence 0.0, no collected
parameters, and no overwrite history.

This test lives in its own file (separate from the other Conversation_Manager
property tests in tasks 4.3/4.4) to keep the properties cleanly isolated.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_builder.conversation_manager import start_session
from prompt_builder.models import PromptSpecification, Session


# Non-empty first messages: at least one non-whitespace character, within the
# 1-10,000 length bound that callers validate before starting a session.
first_messages = st.text(min_size=1, max_size=200).filter(lambda s: s.strip() != "")


# Feature: conversational-prompt-builder, Property 1: New session starts clean and holds the first message
# Validates: Requirements 1.2, 1.3
@settings(max_examples=200)
@given(first_message=first_messages)
def test_new_session_starts_clean_and_holds_first_message(first_message: str) -> None:
    session = start_session(first_message)

    # It is a Session with a fresh identifier.
    assert isinstance(session, Session)
    assert session.id

    # It contains exactly the first message, recorded as a user message.
    assert len(session.messages) == 1
    only_message = session.messages[0]
    assert only_message.sender == "user"
    assert only_message.text == first_message

    # Its Prompt_Specification is empty: no goal, zero confidence, and no
    # collected parameters or overwrite history (Req 1.3).
    spec = session.spec
    assert isinstance(spec, PromptSpecification)
    assert spec.goal is None
    assert spec.goal_confidence == 0.0
    assert spec.parameters == {}
    assert spec.overwrite_log == []
