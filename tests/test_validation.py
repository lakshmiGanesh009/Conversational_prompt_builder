"""Property-based tests for input validation (task 3.2).

Covers Property 2 (message input validation) from the design's Correctness
Properties. The validation functions under test are pure and offline, so these
tests need no network and no running Streamlit server.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_builder.validation import (
    MESSAGE_MAX_LENGTH,
    ValidationRule,
    validate_message,
)

# Whitespace characters used to build whitespace-only inputs, including the
# common unicode spaces that str.strip() treats as whitespace.
_WHITESPACE = " \t\n\r\f\v\u00a0\u2003"


def _accepts_reference(text: str) -> bool:
    """Independent reference for the accept decision (Req 1.4, 1.5).

    A submission is accepted iff it has at least one non-whitespace character
    AND its length is at most MESSAGE_MAX_LENGTH.
    """
    return bool(text.strip()) and len(text) <= MESSAGE_MAX_LENGTH


# Strategy mixing arbitrary text with inputs that land on the interesting
# regions of the input space: whitespace-only strings and strings whose length
# sits right around the 10,000-character boundary.
_whitespace_only = st.text(alphabet=_WHITESPACE, min_size=0, max_size=50)
_boundary_lengths = st.sampled_from(
    [
        MESSAGE_MAX_LENGTH - 1,
        MESSAGE_MAX_LENGTH,
        MESSAGE_MAX_LENGTH + 1,
        MESSAGE_MAX_LENGTH + 100,
    ]
)
_boundary_strings = _boundary_lengths.map(lambda n: "a" * n)
_messages = st.one_of(
    st.text(),
    _whitespace_only,
    _boundary_strings,
    # Non-whitespace content padded with whitespace, to exercise strip().
    st.tuples(_whitespace_only, st.text(min_size=1), _whitespace_only).map(
        "".join
    ),
)


# Feature: conversational-prompt-builder, Property 2: Message input validation
#
# For any input string, submission is accepted if and only if the string
# contains at least one non-whitespace character and its length is between 1
# and 10,000 characters; whitespace-only strings are rejected as empty, and
# strings longer than 10,000 characters are rejected for length.
#
# **Validates: Requirements 1.4, 1.5**
@settings(max_examples=200)
@given(text=_messages)
def test_property_message_input_validation(text: str) -> None:
    result = validate_message(text)

    # Accept iff there is a non-whitespace character AND length <= max.
    assert result.ok == _accepts_reference(text)

    if result.ok:
        # Accepted inputs carry no violated rule.
        assert result.rule is None
    elif not text.strip():
        # Whitespace-only (or empty) input is rejected as EMPTY, and EMPTY takes
        # precedence over length even when the string is also over the maximum.
        assert result.rule is ValidationRule.EMPTY
    else:
        # The only other rejection reason is exceeding the maximum length.
        assert result.rule is ValidationRule.TOO_LONG
        assert len(text) > MESSAGE_MAX_LENGTH
