"""Conversation_Manager: owns the Session and its message state.

This module holds the deterministic rules for managing a single interaction
(a :class:`~prompt_builder.models.Session`): creating it from the first message,
recording messages while capping the history at the 200 most recent in
submission order, starting a fresh empty session, and recording a revision
request. It is kept out of the Streamlit UI script so the rules can be tested
offline, with no network and no running Streamlit server.

Requirements covered:

- Req 1.2, 1.3: starting a Session creates it from the first message and
  initializes an empty Prompt_Specification (no goal, no collected parameters).
- Req 3.3, 5.2: user responses and revision requests are recorded in the Session.
- Req 6.1, 7.1, 7.2: messages are retained in submission order, capped at the 200
  most recent (oldest discarded beyond the limit).
- Req 6.5, 7.5: starting a new Session yields an empty Session.

The functions are pure with respect to their inputs: each returns a new or
updated :class:`Session` rather than mutating global state. ``start_session`` and
``reset_session`` build brand-new sessions; ``add_message`` and
``record_revision`` operate on the session they are given.
"""

from __future__ import annotations

import time
import uuid

from prompt_builder.models import (
    Message,
    PromptSpecification,
    Session,
)

# The maximum number of messages a Session retains. Beyond this, the oldest
# messages are discarded so exactly the 200 most recent remain (Req 7.1, 7.2).
MESSAGE_CAP = 200


def _empty_specification() -> PromptSpecification:
    """Build an empty Prompt_Specification for a fresh Session.

    An empty specification has no goal (confidence 0.0, goal unset) and no
    collected parameters or overwrite history (Req 1.3).
    """
    return PromptSpecification(goal_confidence=0.0)


def _new_message(sender: str, text: str) -> Message:
    """Create a Message with a unique id and a submission timestamp.

    The ``created_at`` timestamp drives submission-order retention and rendering
    (Req 6.1, 7.1).
    """
    return Message(
        id=str(uuid.uuid4()),
        sender=sender,  # type: ignore[arg-type]  # MessageSender literal
        text=text,
        created_at=time.time(),
    )


def cap_messages(messages: list[Message], limit: int = MESSAGE_CAP) -> list[Message]:
    """Return the messages trimmed to the most recent ``limit``, order preserved.

    Messages are kept in submission order. When the count would exceed ``limit``,
    the oldest messages are discarded so exactly the most recent ``limit`` remain
    (Req 7.2). This is a pure function over the message list, exposed at module
    level so Property 18 can test the capping rule directly, independent of
    Session state.

    Args:
        messages: The message list to trim, in submission order.
        limit: The maximum number of messages to retain. Defaults to
            ``MESSAGE_CAP`` (200). A non-positive limit yields an empty list.

    Returns:
        A new list holding the most recent ``limit`` messages in submission order.
    """
    if limit <= 0:
        return []
    if len(messages) <= limit:
        return list(messages)
    return list(messages[-limit:])


def start_session(first_message: str) -> Session:
    """Create a new Session holding the first user message (Req 1.2, 1.3).

    The returned Session has a fresh id, contains exactly the first message (as a
    ``user`` message), and carries an empty Prompt_Specification with no goal and
    no collected parameters.

    Args:
        first_message: The text of the user's first submission. Callers validate
            the text (Req 1.4, 1.5) before starting the session.

    Returns:
        A new :class:`Session` seeded with the first message and an empty spec.
    """
    session = Session(
        id=str(uuid.uuid4()),
        messages=[_new_message("user", first_message)],
        spec=_empty_specification(),
    )
    return session


def add_message(session: Session, message: Message) -> Session:
    """Append a message to the Session, capping history at the 200 most recent.

    The message is added after the existing ones (submission order preserved),
    then the history is trimmed so at most ``MESSAGE_CAP`` messages remain,
    discarding the oldest beyond the limit (Req 3.3, 6.1, 7.1, 7.2).

    Args:
        session: The Session to record the message in.
        message: The message to append.

    Returns:
        The same :class:`Session` with its message list updated.
    """
    session.messages = cap_messages([*session.messages, message])
    return session


def reset_session() -> Session:
    """Start a fresh, empty Session (Req 6.5, 7.5).

    The returned Session has a new id, no messages, and an empty
    Prompt_Specification, so starting a new Session clears everything from the
    previous one.

    Returns:
        A new, empty :class:`Session`.
    """
    return Session(
        id=str(uuid.uuid4()),
        messages=[],
        spec=_empty_specification(),
    )


def record_revision(session: Session, revision_text: str) -> Session:
    """Record a revision request in the Session (Req 5.2).

    The revision request is stored as a ``user`` message so it is retained in the
    conversation history in submission order (and subject to the same 200-message
    cap). Callers validate the revision text (Req 5.3) before recording it.

    Args:
        session: The Session to record the revision request in.
        revision_text: The user's revision request text.

    Returns:
        The same :class:`Session` with the revision request appended.
    """
    return add_message(session, _new_message("user", revision_text))
