"""Property test for message retention and capping (task 4.3).

# Feature: conversational-prompt-builder, Property 18: Messages are retained in order and capped at 200

This Hypothesis test exercises the Conversation_Manager's message-retention
rule: a Session keeps its messages in submission order, and when the count would
exceed the 200-message cap it retains exactly the 200 most recent, discarding the
oldest beyond the limit.

**Validates: Requirements 3.3, 7.1, 7.2**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_builder.conversation_manager import (
    MESSAGE_CAP,
    add_message,
    start_session,
)
from prompt_builder.models import Message, Session


def _message(index: int) -> Message:
    """Build a uniquely identifiable message for the given submission index.

    The id encodes the submission order so the test can assert both retention
    order and which messages survive capping without relying on timestamps.
    """
    return Message(
        id=f"m{index:04d}",
        sender="user" if index % 2 == 0 else "system",
        text=f"message {index}",
        created_at=float(index),
    )


# Cover sequences both under and over the 200-message cap (design test strategy).
@given(total=st.integers(min_value=0, max_value=600))
@settings(max_examples=150)
def test_messages_retained_in_order_and_capped_at_200(total: int) -> None:
    # Start from an empty Session so we control the exact sequence of adds.
    session = Session(id="cap-test")

    expected_ids = [f"m{i:04d}" for i in range(total)]
    for i in range(total):
        session = add_message(session, _message(i))

    retained_ids = [m.id for m in session.messages]

    # Never exceed the cap, regardless of how many were added.
    assert len(retained_ids) <= MESSAGE_CAP

    if total <= MESSAGE_CAP:
        # Under the cap: every message is retained, in submission order.
        assert retained_ids == expected_ids
    else:
        # Over the cap: exactly the 200 most recent remain, oldest discarded,
        # order preserved.
        assert len(retained_ids) == MESSAGE_CAP
        assert retained_ids == expected_ids[-MESSAGE_CAP:]

    # Order is monotonically increasing by submission index in all cases.
    created_order = [m.created_at for m in session.messages]
    assert created_order == sorted(created_order)


@given(extra=st.integers(min_value=1, max_value=400))
@settings(max_examples=150)
def test_oldest_discarded_when_exceeding_cap(extra: int) -> None:
    # Fill past the cap via start_session (first message) plus adds.
    session = start_session("first")
    total = MESSAGE_CAP + extra
    # One message already present from start_session; add the remainder.
    for i in range(1, total):
        session = add_message(session, _message(i))

    retained_ids = [m.id for m in session.messages]

    # Exactly the cap remains and the very first (oldest) message is gone.
    assert len(retained_ids) == MESSAGE_CAP
    assert session.messages[0].id != "first"
    # The most recent message added is still the last one retained.
    assert session.messages[-1].id == f"m{total - 1:04d}"
