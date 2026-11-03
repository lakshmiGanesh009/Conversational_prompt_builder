"""The deterministic retry/cap controller for the Conversational Prompt Builder.

This module is a plain, pure controller (design "Error Handling", Property 19,
Req 8.1-8.4). It wraps an operation that may fail and decides, after each
failure, whether another attempt is allowed. It stops after a fixed number of
consecutive failed attempts (the cap), reports that the operation could not be
completed, and never mutates any Session content itself.

Why it lives here, out of the UI
---------------------------------
The requirements describe a retry/cap behaviour (Req 8.1-8.4) that must be
testable without starting a Streamlit server and without any network access.
Keeping the attempt-counting and cap logic in a small, pure module lets the
property test (task 12.2, Property 19) exercise it directly with an operation
that always fails, and lets the UI layer (task 15.3) drive the same logic for a
Retry button without re-implementing it.

What it does *not* do
---------------------
- It does **not** touch the network. The only network-touching module is
  ``prompt_builder/llm_client.py``; this controller merely *calls* an operation
  the caller supplies and counts failures.
- It does **not** modify the Session, the Prompt_Specification, or any other
  state. On failure the caller's state is simply left as-is because the
  controller never had it (Req 8.1, 8.4: "retain the current Session content
  without modification").
- It does **not** sleep on a real clock by default. Backoff, if any, is injected
  as a callable so tests run instantly and deterministically.

Semantics (Req 8.1-8.4)
-----------------------
- An attempt runs the operation once.
- A *consecutive* failure count is tracked. The default cap is
  :data:`DEFAULT_MAX_ATTEMPTS` (3), matching "after 3 consecutive retry
  attempts" (Req 8.4).
- On the first success the controller stops and reports success, carrying the
  operation's value.
- When the cap is reached with every attempt having failed, the controller stops
  and reports that the operation could not be completed, carrying the last
  error.
- The controller is deterministic: given the same operation behaviour it always
  makes the same number of attempts and produces the same outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

# The maximum number of consecutive failed attempts before the controller gives
# up (Req 8.4: "after 3 consecutive retry attempts"). Kept here so the UI,
# tests, and controller all agree on the cap.
DEFAULT_MAX_ATTEMPTS = 3

# The value an operation produces on success.
T = TypeVar("T")

# A no-op backoff used by default so the controller never blocks on a real clock
# in tests. Receives the 1-based index of the attempt that just failed.
def _no_delay(_attempt: int) -> None:
    """Default backoff hook: do nothing (deterministic, non-blocking)."""
    return None


@dataclass(frozen=True)
class RetryResult(Generic[T]):
    """The outcome of running an operation under the retry controller.

    Immutable so callers cannot accidentally mutate a reported result. The
    controller returns exactly one of these; the UI reads it to decide whether to
    show the produced value or a "could not be completed" message (Req 8.4).

    Attributes:
        succeeded: ``True`` iff the operation eventually succeeded within the
            attempt cap.
        attempts: How many attempts were made (1-based). For a run that fails
            every time this equals the configured cap; for a run that succeeds it
            is the index of the successful attempt.
        value: The value returned by the operation on success, else ``None``.
        error: The exception raised by the final failed attempt, else ``None``.
            Present only when ``succeeded`` is ``False``.
        message: A human-readable summary. On failure this is the
            "could not be completed" message the UI surfaces (Req 8.4).
    """

    succeeded: bool
    attempts: int
    value: Optional[T] = None
    error: Optional[BaseException] = None
    message: str = ""


# The message surfaced when the operation never succeeds within the cap. Kept as
# a constant so the UI and tests can refer to the same wording (Req 8.4).
COULD_NOT_COMPLETE_MESSAGE = "The operation could not be completed. Please try again later."


class RetryController(Generic[T]):
    """A deterministic controller that re-runs a failing operation up to a cap.

    Construct it with the maximum number of consecutive attempts (default
    :data:`DEFAULT_MAX_ATTEMPTS`) and, optionally, which exception types count as
    a retryable failure and a backoff hook. Then call :meth:`run` with the
    operation.

    The controller is pure with respect to application state: it only calls the
    operation the caller hands it and counts outcomes. It never mutates Session
    content (Req 8.1, 8.4).
    """

    def __init__(
        self,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        *,
        retry_on: tuple[type[BaseException], ...] = (Exception,),
        on_backoff: Callable[[int], None] = _no_delay,
    ) -> None:
        """Create the controller.

        Args:
            max_attempts: The maximum number of consecutive attempts before the
                controller gives up. Must be at least 1. The default of 3 matches
                Req 8.4.
            retry_on: The exception types treated as a retryable failure. An
                exception not matching these propagates immediately (it is not a
                "the operation failed, try again" situation). Defaults to
                ``(Exception,)`` so any ordinary operation error is retried.
            on_backoff: A hook invoked after each *failed* attempt that will be
                followed by another attempt, receiving the 1-based index of the
                attempt that just failed. Defaults to a no-op so the controller
                never sleeps on a real clock; a caller that wants real backoff can
                inject a delay here. It is never called after the final attempt.

        Raises:
            ValueError: if ``max_attempts`` is less than 1.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._max_attempts = max_attempts
        self._retry_on = retry_on
        self._on_backoff = on_backoff

    @property
    def max_attempts(self) -> int:
        """The configured maximum number of consecutive attempts."""
        return self._max_attempts

    def should_retry(self, consecutive_failures: int) -> bool:
        """Decide whether another attempt is allowed after some failures.

        Pure helper exposed for callers (and tests) that drive attempts
        themselves, for example a UI Retry button that runs one attempt per
        click and needs to know whether to keep offering Retry.

        Args:
            consecutive_failures: How many consecutive attempts have failed so
                far (at least 0).

        Returns:
            ``True`` iff another attempt is permitted under the cap, i.e. the
            number of failures so far is below :attr:`max_attempts`.
        """
        return consecutive_failures < self._max_attempts

    def run(self, operation: Callable[[], T]) -> RetryResult[T]:
        """Run ``operation``, retrying on failure up to the attempt cap.

        Calls ``operation`` with no arguments. On success, returns immediately
        with a successful :class:`RetryResult`. On a retryable failure, records
        the error and tries again until the cap is reached, invoking the backoff
        hook between attempts. If every attempt fails, returns a failed
        :class:`RetryResult` carrying the last error and the
        :data:`COULD_NOT_COMPLETE_MESSAGE`.

        The controller never mutates application state; a failed run leaves the
        caller's Session content exactly as it was (Req 8.1, 8.4).

        Args:
            operation: A zero-argument callable performing the work (for example
                interpretation, generation, or revision). It should raise to
                signal failure.

        Returns:
            A :class:`RetryResult` describing the outcome.

        Raises:
            BaseException: Re-raises immediately if ``operation`` raises an
                exception whose type is not in ``retry_on`` (that is not a
                retryable operation failure).
        """
        last_error: Optional[BaseException] = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                value = operation()
            except self._retry_on as exc:
                last_error = exc
                # Only back off when another attempt will actually follow.
                if attempt < self._max_attempts:
                    self._on_backoff(attempt)
                continue
            else:
                return RetryResult(
                    succeeded=True,
                    attempts=attempt,
                    value=value,
                    error=None,
                    message="",
                )

        # Every attempt failed: report that the operation could not be completed
        # (Req 8.4). Session content is untouched because we never held it.
        return RetryResult(
            succeeded=False,
            attempts=self._max_attempts,
            value=None,
            error=last_error,
            message=COULD_NOT_COMPLETE_MESSAGE,
        )
