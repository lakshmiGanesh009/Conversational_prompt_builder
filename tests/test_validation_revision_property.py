"""Property-based test for revision request validation (task 3.3).

# Feature: conversational-prompt-builder, Property 14: Revision request validation

Property 14 (design) / Requirement 5.3: a revision request string is accepted if
and only if it is non-empty (after stripping whitespace) and at most 2000
characters; otherwise it is rejected. Whitespace-only input is rejected as
``EMPTY`` and input over 2000 characters is rejected as ``TOO_LONG``.

The validation function is pure, so it runs fully offline with no network and no
Streamlit server. The generators below deliberately cover whitespace-only input
and the length boundary around ``REVISION_MAX_LENGTH`` (2000).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_builder.validation import (
    REVISION_MAX_LENGTH,
    ValidationRule,
    validate_revision,
)


# Text that may be empty, whitespace-only, or arbitrary content of any length.
_any_text = st.text()

# Whitespace-only strings (including the empty string) that must be rejected as EMPTY.
_whitespace_only = st.text(alphabet=" \t\n\r\f\v", min_size=0, max_size=50)

# Strings at or just below the 2000-char limit that contain real content.
_at_or_under_limit = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=REVISION_MAX_LENGTH,
)

# Strings strictly over the 2000-char limit that contain real content.
_over_limit = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=REVISION_MAX_LENGTH + 1,
    max_size=REVISION_MAX_LENGTH + 200,
)


def _expected_accept(text: str) -> bool:
    """Reference rule: accepted iff non-whitespace content and length <= 2000."""
    return bool(text.strip()) and len(text) <= REVISION_MAX_LENGTH


@settings(max_examples=200)
@given(text=st.one_of(_any_text, _whitespace_only, _at_or_under_limit, _over_limit))
def test_revision_accepted_iff_non_empty_and_within_limit(text: str) -> None:
    """Property 14: accept iff non-empty (non-whitespace) and length <= 2000."""
    result = validate_revision(text)

    assert result.ok is _expected_accept(text)

    if result.ok:
        # Accepted results carry no violated rule.
        assert result.rule is None
    else:
        # Rejections name a specific rule: EMPTY for whitespace-only (checked
        # first), TOO_LONG otherwise.
        if not text.strip():
            assert result.rule is ValidationRule.EMPTY
        else:
            assert len(text) > REVISION_MAX_LENGTH
            assert result.rule is ValidationRule.TOO_LONG


@settings(max_examples=200)
@given(text=_whitespace_only)
def test_whitespace_only_rejected_as_empty(text: str) -> None:
    """Empty and whitespace-only revision requests are rejected as EMPTY."""
    result = validate_revision(text)

    assert result.ok is False
    assert result.rule is ValidationRule.EMPTY


@settings(max_examples=200)
@given(text=_over_limit)
def test_over_limit_rejected_as_too_long(text: str) -> None:
    """Revision requests longer than 2000 characters are rejected as TOO_LONG."""
    result = validate_revision(text)

    assert result.ok is False
    assert result.rule is ValidationRule.TOO_LONG


def test_boundary_values_are_deterministic() -> None:
    """Explicit boundary checks around the 2000-character limit."""
    assert validate_revision("a" * REVISION_MAX_LENGTH).ok is True
    over = validate_revision("a" * (REVISION_MAX_LENGTH + 1))
    assert over.ok is False
    assert over.rule is ValidationRule.TOO_LONG
    assert validate_revision("").rule is ValidationRule.EMPTY
