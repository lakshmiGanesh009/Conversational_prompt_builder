"""Property-based tests for the Elicitation_Engine (tasks 8.2-8.7).

These tests exercise the pure, deterministic rules in
:mod:`prompt_builder.elicitation_engine`: how it chooses the next 1-3 missing
required Parameters to ask about, how it folds answers back into the
Prompt_Specification (collected / declined / retry), how it offers catalog
defaults as suggestions, and when it stops asking. The Engine performs no I/O,
so every test here runs fully offline with no network and no Streamlit server
and no LLM.

Each test is a single Hypothesis property with ``max_examples >= 100``. The
generators below mix collected, declined, and still-missing Parameters so the
selection rules are exercised across realistic specification states.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_builder import catalog
from prompt_builder.elicitation_engine import (
    MAX_ATTEMPTS,
    MAX_PARAMETERS_PER_QUESTION,
    QUESTION_LIMIT,
    FollowUpQuestion,
    handle_answer,
    next_question,
    question_limit_reached,
)
from prompt_builder.intent_analyzer import missing_parameters
from prompt_builder.models import (
    ParameterValue,
    PromptSpecification,
    Session,
)

# ---------------------------------------------------------------------------
# Catalog-derived helpers
# ---------------------------------------------------------------------------

# All catalog parameter names, in catalog order.
_ALL_NAMES = [definition.name for definition in catalog.CATALOG]

# The required parameter names (these are what the Engine asks about).
_REQUIRED_NAMES = catalog.required_parameter_names()

# Names that have a catalog default value, mapped to that default.
_DEFAULTS: dict[str, str] = {
    definition.name: definition.default_value
    for definition in catalog.CATALOG
    if definition.default_value is not None
}

# A non-empty, non-skip free-text value for use as a "usable" answer. Chosen to
# never collide with a SKIP_INDICATION once stripped/lower-cased.
_usable_text = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=40,
).filter(lambda s: s.strip() != "" and s.strip().lower() not in {
    "skip", "pass", "n/a", "none", "no",
})


def _empty_spec() -> PromptSpecification:
    """A fresh specification with no collected parameters."""
    return PromptSpecification(goal_confidence=0.0)


def _fresh_session() -> Session:
    """A fresh session that has presented no follow-up questions yet."""
    return Session(id="test-session")


# A per-parameter "state" the generators can assign:
#   - "missing":   no entry at all -> missing and askable
#   - "collected": a collected value -> satisfied, not missing
#   - "declined":  a declined entry at the attempt cap -> missing but not askable
_PARAM_STATES = st.sampled_from(["missing", "collected", "declined"])


@st.composite
def _specs_with_mixed_states(draw: st.DrawFn) -> PromptSpecification:
    """Build a spec assigning each catalog parameter a random state.

    Mixes collected, declined, and missing parameters (task requirement) so the
    selection logic is exercised across a wide range of specification states.
    """
    spec = _empty_spec()
    for name in _ALL_NAMES:
        state = draw(_PARAM_STATES)
        if state == "collected":
            spec.parameters[name] = ParameterValue(
                name=name,
                value=draw(_usable_text),
                status="collected",
                attempts=0,
            )
        elif state == "declined":
            spec.parameters[name] = ParameterValue(
                name=name,
                value="",
                status="declined",
                attempts=MAX_ATTEMPTS,
            )
        # "missing" -> leave absent
    return spec


# ===========================================================================
# 8.2 — Property 7
# ===========================================================================
# Feature: conversational-prompt-builder, Property 7: Follow-up questions request 1 to 3 missing parameters
@settings(max_examples=200)
@given(spec=_specs_with_mixed_states())
def test_follow_up_requests_one_to_three_missing_parameters(
    spec: PromptSpecification,
) -> None:
    """Property 7 / Req 3.1, 3.2.

    With at least one still-askable missing required Parameter and the Session
    under the question limit, :func:`next_question` returns a non-null question
    that references between 1 and 3 Parameters, all of which are currently
    missing (present in :func:`missing_parameters`).
    """
    session = _fresh_session()

    missing = missing_parameters(spec)
    # Only parameters that are missing AND still askable (not declined / not out
    # of attempts) are eligible. In this generator, declined params are capped
    # at MAX_ATTEMPTS so they are not askable.
    askable = [
        name
        for name in missing
        if spec.parameters.get(name) is None
        or spec.parameters[name].attempts < MAX_ATTEMPTS
    ]

    question = next_question(spec, session)

    if not askable:
        # Nothing eligible to ask -> no question.
        assert question is None
        return

    assert question is not None
    # Between 1 and 3 parameters (Req 3.2).
    assert 1 <= len(question.parameters) <= MAX_PARAMETERS_PER_QUESTION
    assert MAX_PARAMETERS_PER_QUESTION == 3
    # Every requested parameter is currently missing (Req 3.1).
    for name in question.parameters:
        assert name in missing
    # No duplicates in a single question.
    assert len(set(question.parameters)) == len(question.parameters)


# ===========================================================================
# 8.3 — Property 8
# ===========================================================================
# Feature: conversational-prompt-builder, Property 8: Usable answers mark parameters collected
@settings(max_examples=200)
@given(spec=_specs_with_mixed_states(), answer=_usable_text)
def test_usable_answer_marks_parameters_collected(
    spec: PromptSpecification, answer: str
) -> None:
    """Property 8 / Req 3.4.

    A usable answer (``usable=True`` with non-skip text) marks every requested
    Parameter "collected" with the provided value.
    """
    session = _fresh_session()
    question = next_question(spec, session)
    # Only meaningful when there is a question to answer.
    if question is None:
        return

    outcome = handle_answer(spec, question, answer, usable=True)

    assert outcome.kind == "collected"
    for name in question.parameters:
        stored = outcome.spec.parameters[name]
        assert stored.status == "collected"
        assert stored.value == answer
        assert stored.name == name
    # A collected parameter is no longer missing.
    still_missing = missing_parameters(outcome.spec)
    for name in question.parameters:
        assert name not in still_missing


# ===========================================================================
# 8.4 — Property 9
# ===========================================================================
# Feature: conversational-prompt-builder, Property 9: Defaults are offered as suggestions
@settings(max_examples=200)
@given(spec=_specs_with_mixed_states())
def test_defaults_are_offered_as_suggestions(
    spec: PromptSpecification,
) -> None:
    """Property 9 / Req 3.5.

    For a produced follow-up question, every requested Parameter that has a
    catalog default value appears in ``suggested_defaults`` with exactly that
    default, and no parameter without a default appears there.
    """
    session = _fresh_session()
    question = next_question(spec, session)
    if question is None:
        return

    for name in question.parameters:
        if name in _DEFAULTS:
            assert name in question.suggested_defaults
            assert question.suggested_defaults[name] == _DEFAULTS[name]
        else:
            assert name not in question.suggested_defaults

    # Suggested defaults never introduce parameters outside the question.
    for name in question.suggested_defaults:
        assert name in question.parameters


# ===========================================================================
# 8.5 — Property 10
# ===========================================================================
# Feature: conversational-prompt-builder, Property 10: Skipped parameters are declined and never re-asked
@settings(max_examples=200)
@given(
    spec=_specs_with_mixed_states(),
    skip_answer=st.sampled_from(["", "   ", "skip", "pass", "N/A", "none", "No", "\t\n"]),
)
def test_skipped_parameters_are_declined_and_never_reasked(
    spec: PromptSpecification, skip_answer: str
) -> None:
    """Property 10 / Req 3.6.

    An empty/skip answer marks each requested Parameter "declined", and no later
    :func:`next_question` in the same Session requests it again. Simulated by
    applying the skip then repeatedly calling ``next_question`` and asserting the
    declined parameters never reappear.
    """
    session = _fresh_session()
    question = next_question(spec, session)
    if question is None:
        return

    skipped = list(question.parameters)
    outcome = handle_answer(spec, question, skip_answer)

    assert outcome.kind == "declined"
    for name in skipped:
        stored = outcome.spec.parameters[name]
        assert stored.status == "declined"

    # Drive the conversation forward: keep asking and never-skip-answering the
    # remaining parameters. The skipped ones must never be re-presented.
    for _ in range(QUESTION_LIMIT):
        q = next_question(outcome.spec, session)
        if q is None:
            break
        for name in q.parameters:
            assert name not in skipped
        # Resolve this question usably so the loop makes progress.
        handle_answer(outcome.spec, q, "a usable value", usable=True)


# ===========================================================================
# 8.6 — Property 11
# ===========================================================================
# Feature: conversational-prompt-builder, Property 11: Unusable answers retry up to three attempts
@settings(max_examples=200)
@given(
    spec=_specs_with_mixed_states(),
    unusable=st.text(
        alphabet=st.characters(min_codepoint=33, max_codepoint=126),
        min_size=1,
        max_size=20,
    ).filter(
        lambda s: s.strip() != ""
        and s.strip().lower() not in {"skip", "pass", "n/a", "none", "no"}
    ),
)
def test_unusable_answers_retry_up_to_three_attempts(
    spec: PromptSpecification, unusable: str
) -> None:
    """Property 11 / Req 3.7.

    Repeated unusable (``usable=False``, non-skip) answers raise the attempt
    count up to :data:`MAX_ATTEMPTS` (3). While below the cap the Parameter is
    still re-presented; once the cap is reached it is no longer re-presented by
    :func:`next_question`.
    """
    session = _fresh_session()
    question = next_question(spec, session)
    if question is None:
        return

    targets = list(question.parameters)

    # First unusable attempt: attempt counts go to 1 (< cap), retry outcome.
    outcome = handle_answer(spec, question, unusable, usable=False)
    assert outcome.kind == "retry"
    for name in targets:
        stored = outcome.spec.parameters[name]
        assert stored.attempts == 1
        assert stored.status != "collected"

    # After one unusable answer the parameters are still askable, so the next
    # question re-presents (some of) them.
    q2 = next_question(outcome.spec, session)
    assert q2 is not None
    # At least one of the first question's targets is re-presented (they were the
    # first in catalog order and are still below the attempt cap).
    assert any(name in targets for name in q2.parameters)

    # Exhaust the attempts for the original targets by answering unusably on the
    # exact same set each time until each reaches MAX_ATTEMPTS.
    while True:
        remaining = [
            name
            for name in targets
            if outcome.spec.parameters[name].attempts < MAX_ATTEMPTS
        ]
        if not remaining:
            break
        q = FollowUpQuestion(parameters=remaining, text="retry", suggested_defaults={})
        outcome = handle_answer(outcome.spec, q, unusable, usable=False)

    # Every target is now at the cap...
    for name in targets:
        assert outcome.spec.parameters[name].attempts == MAX_ATTEMPTS
        assert outcome.spec.parameters[name].status != "collected"

    # ...and is therefore no longer re-presented by next_question in any later
    # question in this session.
    for _ in range(QUESTION_LIMIT):
        q = next_question(outcome.spec, session)
        if q is None:
            break
        for name in q.parameters:
            assert name not in targets
        handle_answer(outcome.spec, q, "a usable value", usable=True)


# ===========================================================================
# 8.7 — Property 12
# ===========================================================================
# Feature: conversational-prompt-builder, Property 12: Follow-up questions stop at twenty
@settings(max_examples=200)
@given(
    follow_up_count=st.integers(min_value=0, max_value=40),
    spec=_specs_with_mixed_states(),
)
def test_follow_up_questions_stop_at_twenty(
    follow_up_count: int, spec: PromptSpecification
) -> None:
    """Property 12 / Req 3.8.

    ``question_limit_reached`` is True iff ``follow_up_count >= 20``, and at or
    above the limit :func:`next_question` returns ``None`` regardless of how many
    required Parameters are still missing.
    """
    session = Session(id="limit-session", follow_up_count=follow_up_count)

    reached = question_limit_reached(session)
    assert reached == (follow_up_count >= QUESTION_LIMIT)
    assert QUESTION_LIMIT == 20

    question = next_question(spec, session)
    if reached:
        # At or above the limit nothing is asked, even with missing parameters.
        assert question is None
    else:
        # Below the limit, a question is produced exactly when something askable
        # is still missing.
        askable = [
            name
            for name in missing_parameters(spec)
            if spec.parameters.get(name) is None
            or spec.parameters[name].attempts < MAX_ATTEMPTS
        ]
        if askable:
            assert question is not None
        else:
            assert question is None
