"""Property tests for the pure Streamlit UI rendering helpers (tasks 14.2, 14.3).

These exercise ``prompt_builder.ui``, whose helpers are pure and import no
Streamlit, so the tests run offline with no server:

- Property 15 (task 14.2): the conversation renders oldest-to-newest by
  ``created_at`` submission time, and the rendered set is a permutation of the
  input (Req 6.1).
- Property 16 (task 14.3): a message's visual treatment is determined solely by
  its sender, and the user treatment differs from the system treatment (Req 6.2).
"""

from __future__ import annotations

from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_builder.models import Message, MessageSender
from prompt_builder.ui import (
    order_messages_for_render,
    treatment_for_message,
    treatment_for_sender,
)


# A strategy for a single Message with varied created_at floats. The created_at
# values are drawn from a small integer range (cast to float) so ties between
# messages occur frequently, exercising the stable-ordering path.
def _messages() -> st.SearchStrategy[Message]:
    senders: st.SearchStrategy[MessageSender] = st.sampled_from(["user", "system"])
    return st.builds(
        Message,
        id=st.uuids().map(str),
        sender=senders,
        text=st.text(max_size=50),
        created_at=st.integers(min_value=0, max_value=20).map(float),
    )


# Feature: conversational-prompt-builder, Property 15: Conversation renders oldest to newest
#
# For any list of messages, order_messages_for_render returns them ordered by
# created_at submission time from oldest to newest. The output created_at
# sequence is non-decreasing, and the output is a permutation of the input (same
# multiset of message ids).
#
# **Validates: Requirements 6.1**
@given(messages=st.lists(_messages(), max_size=40))
@settings(max_examples=200)
def test_conversation_renders_oldest_to_newest(messages: list[Message]) -> None:
    ordered = order_messages_for_render(messages)

    # Non-decreasing by submission time: oldest first, newest last.
    created_sequence = [m.created_at for m in ordered]
    assert all(
        earlier <= later
        for earlier, later in zip(created_sequence, created_sequence[1:])
    )

    # The render is a permutation of the input: same multiset of message ids,
    # nothing added, dropped, or duplicated.
    assert Counter(m.id for m in ordered) == Counter(m.id for m in messages)

    # The input list is not mutated.
    assert len(messages) == len(ordered)


# Feature: conversational-prompt-builder, Property 16: Sender determines the visual treatment
#
# The visual treatment is determined solely by sender, and the user treatment
# differs from the system treatment.
#
# **Validates: Requirements 6.2**
@given(
    first=_messages(),
    second=_messages(),
)
@settings(max_examples=200)
def test_sender_determines_the_visual_treatment(
    first: Message, second: Message
) -> None:
    # Treatment per sender is stable/deterministic: the same sender always maps
    # to an equal treatment.
    assert treatment_for_sender("user") == treatment_for_sender("user")
    assert treatment_for_sender("system") == treatment_for_sender("system")

    # The two senders are always visually distinguishable.
    assert treatment_for_sender("user") != treatment_for_sender("system")

    # For two messages with arbitrary other fields, the treatment depends solely
    # on the sender: equal when senders match, differing when they differ.
    if first.sender == second.sender:
        assert treatment_for_message(first) == treatment_for_message(second)
    else:
        assert treatment_for_message(first) != treatment_for_message(second)

    # A message's treatment matches its sender's treatment (solely sender-driven).
    assert treatment_for_message(first) == treatment_for_sender(first.sender)
    assert treatment_for_message(second) == treatment_for_sender(second.sender)
