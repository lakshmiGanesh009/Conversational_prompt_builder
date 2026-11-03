"""Pure input-validation functions for the Conversational Prompt Builder.

These functions are deliberately kept out of the Streamlit UI script so they can
be tested offline, with no network and no running Streamlit server. They encode
the deterministic validation rules from the requirements:

- Message input (Req 1.4, 1.5): a submission is accepted if and only if it has at
  least one non-whitespace character AND its length is between 1 and 10,000
  characters inclusive. Whitespace-only input is rejected as "empty"; input longer
  than 10,000 characters is rejected as "too_long". These are distinct rule
  violations so the UI can show a rule-specific error.
- Revision request (Req 5.3): accepted if and only if it is non-empty and at most
  2000 characters. Empty is rejected as "empty"; over 2000 is rejected as
  "too_long".

Each function returns a :class:`ValidationResult` whose ``ok`` flag and ``rule``
name are both easy to read, so UI code and property tests (Property 2 and
Property 14) can assert on the accept/reject decision and the violated rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Rule limits, named so they read clearly and stay in one place.
MESSAGE_MIN_LENGTH = 1
MESSAGE_MAX_LENGTH = 10_000
REVISION_MAX_LENGTH = 2_000


class ValidationRule(str, Enum):
    """Stable identifiers for the validation rules that can be violated.

    Inherits from ``str`` so the value is directly usable as a plain, stable
    string (for example in UI lookups or test assertions) while still being an
    enum member.
    """

    EMPTY = "empty"
    TOO_LONG = "too_long"


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of validating an input string.

    Attributes:
        ok: ``True`` when the input is accepted, ``False`` when it is rejected.
        rule: The violated rule when ``ok`` is ``False`` (one of
            :class:`ValidationRule`), or ``None`` when the input is accepted.
    """

    ok: bool
    rule: Optional[ValidationRule] = None


# A single accepted result is immutable and carries no rule, so share one instance.
_ACCEPTED = ValidationResult(ok=True, rule=None)


def validate_message(text: str) -> ValidationResult:
    """Validate a user message submission.

    The message is accepted if and only if it contains at least one
    non-whitespace character AND its length is between ``MESSAGE_MIN_LENGTH`` and
    ``MESSAGE_MAX_LENGTH`` (inclusive).

    Rejection rules, checked so the reported rule is the most meaningful one:

    - ``ValidationRule.EMPTY``: the string is empty or contains only whitespace.
    - ``ValidationRule.TOO_LONG``: the string exceeds ``MESSAGE_MAX_LENGTH``
      characters.

    Args:
        text: The raw message text submitted by the user.

    Returns:
        A :class:`ValidationResult` describing whether the message is accepted
        and, if not, which rule it violated.
    """
    # Whitespace-only (and truly empty) input is classified as "empty" first.
    if not text.strip():
        return ValidationResult(ok=False, rule=ValidationRule.EMPTY)

    # Non-empty but over the maximum length is a distinct "too_long" violation.
    if len(text) > MESSAGE_MAX_LENGTH:
        return ValidationResult(ok=False, rule=ValidationRule.TOO_LONG)

    return _ACCEPTED


def validate_revision(text: str) -> ValidationResult:
    """Validate a revision request.

    The request is accepted if and only if it is non-empty and at most
    ``REVISION_MAX_LENGTH`` characters.

    Rejection rules:

    - ``ValidationRule.EMPTY``: the string is empty or contains only whitespace.
    - ``ValidationRule.TOO_LONG``: the string exceeds ``REVISION_MAX_LENGTH``
      characters.

    Args:
        text: The raw revision request text submitted by the user.

    Returns:
        A :class:`ValidationResult` describing whether the request is accepted
        and, if not, which rule it violated.
    """
    if not text.strip():
        return ValidationResult(ok=False, rule=ValidationRule.EMPTY)

    if len(text) > REVISION_MAX_LENGTH:
        return ValidationResult(ok=False, rule=ValidationRule.TOO_LONG)

    return _ACCEPTED
