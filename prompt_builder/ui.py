"""Pure rendering helpers for the Conversational Prompt Builder UI.

These functions hold the small amount of deterministic presentation logic the
Streamlit script needs, kept out of the UI script itself so they can be unit and
property tested offline, with no running Streamlit server. The actual drawing is
still done by Streamlit (for example ``st.chat_message``); these helpers only
decide *order* and *visual treatment*, which are the rules worth testing.

They encode two requirements:

- Conversation ordering (Req 6.1): messages are rendered in chronological order,
  oldest at the top and newest at the bottom, by their ``created_at`` submission
  time.
- Sender distinction (Req 6.2): each message's visual treatment is determined
  solely by its sender, and the treatment for user messages differs from the
  treatment for system messages, so the two are always visually distinguishable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Message, MessageSender

# Streamlit's ``st.chat_message`` accepts a role name and an optional avatar.
# We pin a distinct role/avatar per sender so user and system messages always
# get a different, consistent visual treatment (Req 6.2). Keeping these as data
# lets tests assert that the mapping is total and that the two differ.
_USER_TREATMENT_ROLE = "user"
_SYSTEM_TREATMENT_ROLE = "assistant"
_USER_AVATAR = "🧑"
_SYSTEM_AVATAR = "🤖"


@dataclass(frozen=True)
class VisualTreatment:
    """The persistent visual attributes applied to a message.

    The treatment is a function of the sender alone (Req 6.2). It is immutable so
    the two shared instances below can be reused and compared by value.

    Attributes:
        role: The ``st.chat_message`` role name that drives alignment and the
            default styling Streamlit applies to the bubble.
        avatar: A short, stable avatar marker shown beside the message.
        label: A human-readable sender label applied consistently to every
            message of the same type.
    """

    role: str
    avatar: str
    label: str


# One shared, immutable treatment per sender. Defined once so the mapping is
# total over ``MessageSender`` and the user/system treatments are guaranteed to
# differ.
_USER_TREATMENT = VisualTreatment(
    role=_USER_TREATMENT_ROLE,
    avatar=_USER_AVATAR,
    label="You",
)
_SYSTEM_TREATMENT = VisualTreatment(
    role=_SYSTEM_TREATMENT_ROLE,
    avatar=_SYSTEM_AVATAR,
    label="Assistant",
)


def order_messages_for_render(messages: list[Message]) -> list[Message]:
    """Return the messages ordered oldest-to-newest for rendering.

    Messages are ordered by their ``created_at`` submission time, oldest first
    (top) and newest last (bottom), as required for the conversation view
    (Req 6.1). The sort is stable, so messages sharing a ``created_at`` keep
    their existing relative order. The input list is not mutated; a new list is
    returned.

    Args:
        messages: The Session's messages, in any order.

    Returns:
        A new list of the same messages ordered from oldest to newest.
    """
    return sorted(messages, key=lambda message: message.created_at)


def treatment_for_sender(sender: MessageSender) -> VisualTreatment:
    """Return the visual treatment for a sender.

    The treatment is determined solely by the sender (Req 6.2): user messages get
    one treatment and system messages get another, and the two always differ. The
    returned instances are shared and immutable.

    Args:
        sender: The message sender, ``"user"`` or ``"system"``.

    Returns:
        The :class:`VisualTreatment` to apply to messages from that sender.
    """
    return _USER_TREATMENT if sender == "user" else _SYSTEM_TREATMENT


def treatment_for_message(message: Message) -> VisualTreatment:
    """Return the visual treatment for a single message.

    Convenience wrapper over :func:`treatment_for_sender` that reads the sender
    from the message, so callers can map a message straight to its treatment when
    rendering (Req 6.2).

    Args:
        message: The message to render.

    Returns:
        The :class:`VisualTreatment` for the message's sender.
    """
    return treatment_for_sender(message.sender)
