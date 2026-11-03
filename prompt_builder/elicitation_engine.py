"""Elicitation_Engine: decide what to ask next and handle the answers.

The Elicitation_Engine turns the "which required parameters are still missing"
question (answered by :func:`~prompt_builder.intent_analyzer.missing_parameters`)
into concrete follow-up questions, and folds the user's answers back into the
Prompt_Specification. Everything here is pure deterministic Python with no
network access: the design keeps the fuzzy phrasing work for the LLM, but the
*rules* for choosing parameters, offering defaults, counting attempts, declining
skipped parameters, and stopping at the question limit are our own code (design:
"Everything deterministic stays in our own code").

Rules implemented (design Elicitation_Engine + Requirements 3.1-3.8):

- :func:`next_question` - pick 1-3 currently missing required parameters, attach
  their catalog default values as suggestions, and phrase a question. Returns
  ``None`` when nothing is missing or the Session has reached the question limit
  (Req 3.1, 3.2, 3.5, 3.8). Parameters that have been declined or have exhausted
  their 3 attempts are never re-presented (Req 3.6, 3.7).
- :func:`handle_answer` - classify the answer. A usable answer marks every
  requested parameter "collected" (Req 3.4); an empty / skip answer marks them
  "declined" so they are never asked again (Req 3.6); an unusable (non-skip)
  answer increments the attempt count and asks again, up to 3 attempts, after
  which the parameter is left unresolved (Req 3.7).
- :func:`question_limit_reached` - the Session limit is reached iff its follow-up
  count is 20 or more (Req 3.8).

The dataclasses :class:`FollowUpQuestion`, :class:`CollectedOutcome`,
:class:`DeclinedOutcome` and :class:`RetryOutcome` are defined here, matching the
design's Elicitation_Engine interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from prompt_builder import catalog
from prompt_builder.intent_analyzer import missing_parameters
from prompt_builder.models import (
    ParameterName,
    ParameterValue,
    PromptSpecification,
    Session,
)

__all__ = [
    "FollowUpQuestion",
    "CollectedOutcome",
    "DeclinedOutcome",
    "RetryOutcome",
    "AnswerOutcome",
    "QUESTION_LIMIT",
    "MAX_ATTEMPTS",
    "MAX_PARAMETERS_PER_QUESTION",
    "SKIP_INDICATIONS",
    "next_question",
    "handle_answer",
    "question_limit_reached",
]

# The Session stops presenting follow-up questions once this many have been
# presented (Req 3.8). The limit is reached iff ``follow_up_count >= 20``.
QUESTION_LIMIT = 20

# A parameter that keeps getting unusable (non-skip) answers is retried up to
# this many times; after the third attempt it is treated as unresolved and no
# longer re-presented (Req 3.7).
MAX_ATTEMPTS = 3

# A single follow-up question asks about at most this many parameters (Req 3.2).
MAX_PARAMETERS_PER_QUESTION = 3

# Answers that count as the user choosing to skip a parameter (Req 3.6). An empty
# or whitespace-only answer skips too; these are additional explicit words. They
# are compared case-insensitively against the stripped answer.
SKIP_INDICATIONS: frozenset[str] = frozenset({"skip", "pass", "n/a", "none", "no"})


@dataclass
class FollowUpQuestion:
    """A follow-up question asking the user for 1-3 missing parameters.

    Attributes:
        parameters: The parameter names this question asks about (1-3 of them,
            all currently missing). Req 3.1, 3.2.
        text: The human-readable question text.
        suggested_defaults: For each requested parameter that has a catalog
            default value, the default offered as a suggested answer (Req 3.5).
            Parameters without a default do not appear here.
    """

    parameters: list[ParameterName]
    text: str
    suggested_defaults: dict[ParameterName, str] = field(default_factory=dict)


@dataclass
class CollectedOutcome:
    """A usable answer: the requested parameters were marked "collected" (Req 3.4).

    Attributes:
        spec: The specification after the collected values were stored.
        kind: Discriminator, always ``"collected"``.
    """

    spec: PromptSpecification
    kind: Literal["collected"] = "collected"


@dataclass
class DeclinedOutcome:
    """A skip answer: the requested parameters were marked "declined" (Req 3.6).

    Declined parameters are never asked again in the Session.

    Attributes:
        spec: The specification after the parameters were marked declined.
        kind: Discriminator, always ``"declined"``.
    """

    spec: PromptSpecification
    kind: Literal["declined"] = "declined"


@dataclass
class RetryOutcome:
    """An unusable (non-skip) answer: ask again, up to 3 attempts (Req 3.7).

    Attributes:
        spec: The specification after the attempt count was incremented.
        attempts_left: How many attempts remain before the parameter is treated
            as unresolved. ``0`` means the third attempt has now been used and the
            parameter will no longer be re-presented.
        kind: Discriminator, always ``"retry"``.
    """

    spec: PromptSpecification
    attempts_left: int
    kind: Literal["retry"] = "retry"


# The three possible results of handling an answer (design: AnswerOutcome).
AnswerOutcome = Union[CollectedOutcome, DeclinedOutcome, RetryOutcome]


def question_limit_reached(session: Session) -> bool:
    """Return whether the Session has hit the 20-question limit (Req 3.8).

    The limit is reached if and only if the Session's ``follow_up_count`` is
    :data:`QUESTION_LIMIT` (20) or more. At or above the limit the Engine stops
    presenting follow-up questions.

    Args:
        session: The Session whose follow-up count is inspected.

    Returns:
        ``True`` iff ``session.follow_up_count >= QUESTION_LIMIT``.
    """
    return session.follow_up_count >= QUESTION_LIMIT


def _is_skip(answer: str) -> bool:
    """Return whether an answer is an empty or skip response (Req 3.6).

    An answer is a skip when it has no non-whitespace content, or when its
    stripped, lower-cased text is one of :data:`SKIP_INDICATIONS`.
    """
    stripped = answer.strip()
    if stripped == "":
        return True
    return stripped.lower() in SKIP_INDICATIONS


def _askable(spec: PromptSpecification, name: ParameterName) -> bool:
    """Return whether a missing parameter may still be presented.

    A missing required parameter is askable unless it is out of attempts. The
    :data:`attempts` counter is the single source of truth for "stop asking":

    - No stored value (``None``): askable; it has had no attempts yet.
    - ``attempts < MAX_ATTEMPTS``: askable. This covers a parameter that received
      one or two unusable answers and should be re-presented (Req 3.7).
    - ``attempts >= MAX_ATTEMPTS``: not askable. Both a skipped parameter
      (declined up front, Req 3.6) and one that exhausted its three retry attempts
      (Req 3.7) reach the cap, so neither is re-presented.

    A skip sets ``attempts`` to the cap so a declined parameter is never re-asked
    even though its status is the same ``"declined"`` the model allows; an
    in-progress retry keeps ``attempts`` below the cap so it is asked again. This
    keeps the two-value ``status`` model intact while honouring both rules.
    """
    existing = spec.parameters.get(name)
    if existing is None:
        return True
    return existing.attempts < MAX_ATTEMPTS


def _askable_missing(spec: PromptSpecification) -> list[ParameterName]:
    """Return the missing required parameters that may still be presented.

    Starts from :func:`missing_parameters` (required parameters with no collected
    value, in catalog order) and drops the ones that are declined or out of
    attempts, so the result is exactly what is eligible to be asked next.
    """
    return [name for name in missing_parameters(spec) if _askable(spec, name)]


def _phrase_question(
    names: list[ParameterName], suggested_defaults: dict[ParameterName, str]
) -> str:
    """Build deterministic question text for the chosen parameters.

    The LLM phrases questions in the running app; this deterministic phrasing
    keeps the engine pure and testable (no network) while still producing clear,
    human-readable text. Any suggested defaults are mentioned so the user knows a
    default is on offer (Req 3.5).
    """
    if len(names) == 1:
        subject = names[0]
    elif len(names) == 2:
        subject = f"{names[0]} and {names[1]}"
    else:
        subject = ", ".join(names[:-1]) + f", and {names[-1]}"

    text = f"Could you tell me about the {subject}?"
    if suggested_defaults:
        hints = ", ".join(
            f"{name} (default: {value})"
            for name, value in suggested_defaults.items()
        )
        text += f" If you're not sure, I can use these defaults: {hints}."
    return text


def next_question(
    spec: PromptSpecification, session: Session
) -> Optional[FollowUpQuestion]:
    """Build the next follow-up question, or ``None`` when there is nothing to ask.

    Returns ``None`` when the Session has reached the question limit (Req 3.8) or
    when no missing required parameter is eligible to be asked (all required
    parameters are collected, declined, or out of attempts). Otherwise picks the
    first 1-3 eligible missing parameters in catalog order (Req 3.1, 3.2), attaches
    the catalog default value of each (when it has one) as a suggested answer
    (Req 3.5), and phrases a question for them.

    This function does not mutate ``spec`` or ``session`` and performs no I/O; it
    simply reports what should be asked next. The caller increments the Session's
    follow-up count when it actually presents the question.

    Args:
        spec: The current specification, used to derive the missing parameters.
        session: The current Session, used for the question-limit check.

    Returns:
        A :class:`FollowUpQuestion` for 1-3 missing parameters, or ``None`` when
        nothing should be asked.
    """
    if question_limit_reached(session):
        return None

    eligible = _askable_missing(spec)
    if not eligible:
        return None

    chosen = eligible[:MAX_PARAMETERS_PER_QUESTION]

    suggested_defaults: dict[ParameterName, str] = {}
    for name in chosen:
        definition = catalog.get_definition(name)
        if definition is not None and definition.default_value is not None:
            suggested_defaults[name] = definition.default_value

    return FollowUpQuestion(
        parameters=list(chosen),
        text=_phrase_question(chosen, suggested_defaults),
        suggested_defaults=suggested_defaults,
    )


def handle_answer(
    spec: PromptSpecification,
    question: FollowUpQuestion,
    answer: str,
    *,
    usable: bool = True,
) -> AnswerOutcome:
    """Fold the user's answer to a follow-up question into the specification.

    Classifies the answer against the design's three outcomes and updates ``spec``
    in place accordingly:

    - **Skip / empty** (:func:`_is_skip`): the answer has no non-whitespace
      content or is an explicit skip word. Each requested parameter is marked
      ``"declined"`` and its attempt count is set to :data:`MAX_ATTEMPTS` so it is
      never asked again in the Session (Req 3.6). Returns a
      :class:`DeclinedOutcome`. A skip always wins, regardless of ``usable``.
    - **Usable** (``usable=True`` and not a skip): each requested parameter is
      marked ``"collected"`` with the provided answer as its value (Req 3.4).
      Returns a :class:`CollectedOutcome`.
    - **Unusable** (``usable=False`` and not a skip): the user replied but the
      reply does not provide a value the caller could use. Each requested
      parameter's attempt count is raised by one, up to :data:`MAX_ATTEMPTS`. While
      the count is below the cap the parameter stays missing and is re-presented on
      the next question; once it reaches the cap it is no longer re-presented
      (:func:`_askable` keys off the attempt count). The entry is stored with a
      ``"declined"`` status and no collected value, so it never counts as collected
      regardless of the attempt count. Returns a :class:`RetryOutcome` carrying the
      remaining attempts (Req 3.7).

    Whether a non-skip answer is *usable* is a judgment the caller makes (in the
    running app, with the LLM's help); the engine takes it as the ``usable`` flag
    so this function stays pure and deterministic. The default is ``True`` because
    a plain free-text answer normally provides the requested value.

    Args:
        spec: The specification to update (mutated in place and returned).
        question: The question being answered; its ``parameters`` are updated.
        answer: The user's free-text answer.
        usable: Whether a non-skip ``answer`` provides a usable value. ``False``
            drives the retry path (Req 3.7). Ignored for skip/empty answers.

    Returns:
        A :class:`CollectedOutcome`, :class:`DeclinedOutcome`, or
        :class:`RetryOutcome` describing what happened.
    """
    # Empty or explicit skip: decline every requested parameter and cap its
    # attempts so it is never re-presented (Req 3.6).
    if _is_skip(answer):
        for name in question.parameters:
            spec.parameters[name] = ParameterValue(
                name=name,
                value="",
                status="declined",
                attempts=MAX_ATTEMPTS,
            )
        return DeclinedOutcome(spec=spec)

    # Usable answer: mark every requested parameter collected (Req 3.4).
    if usable:
        for name in question.parameters:
            existing = spec.parameters.get(name)
            attempts = existing.attempts if existing is not None else 0
            spec.parameters[name] = ParameterValue(
                name=name,
                value=answer,
                status="collected",
                attempts=attempts,
            )
        return CollectedOutcome(spec=spec)

    # Unusable (non-skip) answer: raise the attempt count and ask again until the
    # count reaches MAX_ATTEMPTS, after which the parameter is no longer
    # re-presented (Req 3.7). The entry is never "collected", so a value-less
    # attempt can never satisfy the missing-parameter check.
    attempts_left = MAX_ATTEMPTS
    for name in question.parameters:
        existing = spec.parameters.get(name)
        attempts = min((existing.attempts if existing is not None else 0) + 1, MAX_ATTEMPTS)
        spec.parameters[name] = ParameterValue(
            name=name,
            value="",
            status="declined",
            attempts=attempts,
        )
        attempts_left = min(attempts_left, MAX_ATTEMPTS - attempts)
    return RetryOutcome(spec=spec, attempts_left=attempts_left)
