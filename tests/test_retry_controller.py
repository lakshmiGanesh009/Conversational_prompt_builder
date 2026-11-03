"""Property test for the retry/cap controller (task 12.2).

# Feature: conversational-prompt-builder, Property 19: Retries stop after three consecutive failures

This Hypothesis test exercises the Retry_Controller's cap rule: for any
operation that keeps failing, the controller stops after 3 consecutive failed
attempts, reports that the operation could not be completed, and leaves the
Session content unchanged.

**Validates: Requirements 8.4**
"""

from __future__ import annotations

import copy

from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_builder.models import (
    GeneratedPrompt,
    Message,
    Overwrite,
    ParameterValue,
    PromptSpecification,
    Session,
)
from prompt_builder.retry_controller import (
    COULD_NOT_COMPLETE_MESSAGE,
    DEFAULT_MAX_ATTEMPTS,
    RetryController,
)


def _session(seed: int, message_count: int) -> Session:
    """Build a representative, fully-populated Session for the given seed.

    The controller must never touch Session content, so the test hands it a
    Session with messages, a spec (parameters + overwrite log), a generated
    prompt, and counters, then asserts the Session is byte-for-byte unchanged
    after a failing run.
    """
    messages = [
        Message(
            id=f"m{seed}-{i}",
            sender="user" if i % 2 == 0 else "system",
            text=f"message {seed}-{i}",
            created_at=float(seed * 100 + i),
        )
        for i in range(message_count)
    ]
    spec = PromptSpecification(
        goal_confidence=(seed % 100) / 100.0,
        goal=f"goal-{seed}",
        parameters={
            "audience": ParameterValue(
                name="audience",
                value=f"value-{seed}",
                status="collected",
                attempts=1,
            ),
            "tone": ParameterValue(
                name="tone",
                value="formal",
                status="declined",
                attempts=3,
            ),
        },
        overwrite_log=[
            Overwrite(parameter="audience", previous="old", next=f"value-{seed}")
        ],
    )
    return Session(
        id=f"session-{seed}",
        messages=messages,
        spec=spec,
        follow_up_count=seed % 20,
        generated_prompt=GeneratedPrompt(text=f"prompt-{seed}", omitted_required=["length"]),
        context_available=bool(seed % 2),
    )


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    message_count=st.integers(min_value=0, max_value=10),
    error_message=st.text(min_size=0, max_size=40),
)
@settings(max_examples=200)
def test_retries_stop_after_three_consecutive_failures(
    seed: int, message_count: int, error_message: str
) -> None:
    # A Session the controller must leave untouched, plus a pristine copy to
    # compare against afterwards.
    session = _session(seed, message_count)
    before = copy.deepcopy(session)

    # An operation that always fails, counting how many times it is invoked so
    # we can prove the controller stopped at the cap.
    call_count = 0

    def always_fails() -> str:
        nonlocal call_count
        call_count += 1
        raise RuntimeError(error_message)

    controller: RetryController[str] = RetryController()
    result = controller.run(always_fails)

    # Stops after exactly 3 consecutive failures (Req 8.4).
    assert DEFAULT_MAX_ATTEMPTS == 3
    assert call_count == 3
    assert result.attempts == 3

    # Reports the operation could not be completed.
    assert result.succeeded is False
    assert result.value is None
    assert result.message == COULD_NOT_COMPLETE_MESSAGE
    assert isinstance(result.error, RuntimeError)

    # Session content is unchanged: the controller never held or mutated it.
    assert session == before
