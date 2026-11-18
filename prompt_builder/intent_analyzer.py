"""Intent_Analyzer parameter tracking for the Conversational Prompt Builder.

The Intent_Analyzer uses the LLM for the fuzzy work of reading a user's intent,
but the rules for folding extracted parameters into the Prompt_Specification are
our own deterministic code. This module holds those pure, offline-testable
rules (design: "the merge/conflict rules are our own deterministic code"):

- :func:`merge_parameters` - fold a batch of extracted parameters into the spec.
  The most recently extracted value wins, and every value that overwrites a
  different existing value is recorded in the spec's overwrite log (Req 2.2,
  2.3).
- :func:`missing_parameters` - derive the required parameters that still have no
  value with status ``"collected"`` (Req 2.4). "Missing" is derived, never
  stored (design Model notes).

The LLM-backed :class:`IntentAnalyzer` (task 7.1) wraps an injected
:class:`~prompt_builder.llm_client.LlmClient` and turns a user message into an
:class:`InterpretationResult`. It applies the configured acceptance threshold
(leaving the goal unset when the LLM's confidence falls below it, Req 2.5) and
sets ``context_unavailable`` when the retained Session content cannot be read
(Req 7.4). The LLM does the fuzzy reading; the thresholding and context handling
are our own deterministic code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from prompt_builder import catalog
from prompt_builder.llm_client import LlmClient, LlmRequest
from prompt_builder.models import (
    Message,
    Overwrite,
    ParameterName,
    ParameterValue,
    PromptSpecification,
)

# Re-exported so callers can import the whole Intent_Analyzer interface from this
# module (the design lists Overwrite as part of this component). It is the same
# type stored in ``PromptSpecification.overwrite_log`` so the log stays
# type-consistent.
__all__ = [
    "ExtractedParameter",
    "InterpretationResult",
    "Overwrite",
    "MergeResult",
    "IntentAnalyzer",
    "DEFAULT_ACCEPTANCE_THRESHOLD",
    "merge_parameters",
    "missing_parameters",
]

# The minimum confidence the LLM must assign to the identified goal for that goal
# to be treated as confirmed (Req 2.5, design: "a single configured number (for
# example 0.6)"). A goal whose confidence is strictly below this stays unset so
# the Elicitation_Engine asks the user to describe the goal.
DEFAULT_ACCEPTANCE_THRESHOLD = 0.6


@dataclass
class ExtractedParameter:
    """A single parameter value the Intent_Analyzer read from a message.

    Attributes:
        name: The catalog parameter name this value maps to.
        value: The extracted value for that parameter.
    """

    name: ParameterName
    value: str


@dataclass
class InterpretationResult:
    """The outcome of interpreting a single user message (design: Intent_Analyzer).

    Populated by the LLM-backed ``interpret`` (task 7.1); defined here so the
    deterministic tracking functions and the interpreter share one shape.

    Attributes:
        confidence: The analyzer's confidence in the identified goal, 0..1.
        extracted: The parameters read from the message that map to the catalog.
        goal: The identified goal, or ``None`` when confidence is below the
            acceptance threshold (Req 2.5).
        context_unavailable: ``True`` when retained session context could not be
            read and the message was interpreted in isolation (Req 7.4).
    """

    confidence: float
    extracted: list[ExtractedParameter] = field(default_factory=list)
    goal: Optional[str] = None
    context_unavailable: bool = False


@dataclass
class MergeResult:
    """The outcome of merging extracted parameters into a specification.

    Attributes:
        spec: The specification after the merge (mutated in place and returned).
        overwrites: The overwrites performed by this merge, each recording a
            parameter whose existing value was replaced by a different new value
            (Req 2.3). Empty when no existing value was changed.
    """

    spec: PromptSpecification
    overwrites: list[Overwrite] = field(default_factory=list)


def merge_parameters(
    spec: PromptSpecification, extracted: list[ExtractedParameter]
) -> MergeResult:
    """Fold extracted parameters into the Prompt_Specification (Req 2.2, 2.3).

    For each extracted parameter that maps to a defined catalog parameter, store
    its value in the spec with status ``"collected"`` (Req 2.2). When the
    parameter already holds a *different* value, the most recently extracted
    value wins and an :class:`Overwrite` recording the previous value is appended
    to the spec's overwrite log (Req 2.3).

    Extracted parameters that do not map to a defined catalog parameter are
    ignored. Within a single call, later entries for the same parameter overwrite
    earlier ones, so processing is strictly in list order.

    The merge is deterministic and performs no I/O. The ``spec`` is mutated in
    place and also returned via the :class:`MergeResult`.

    Args:
        spec: The specification to merge into.
        extracted: The parameters read from the current message, in the order
            they were extracted.

    Returns:
        A :class:`MergeResult` holding the updated spec and the list of
        overwrites that occurred during this merge.
    """
    overwrites: list[Overwrite] = []

    for item in extracted:
        # Ignore anything that is not a defined catalog parameter (design:
        # "extracted parameters that map to defined Parameters").
        if catalog.get_definition(item.name) is None:
            continue

        existing = spec.parameters.get(item.name)
        # A conflict is an existing *collected* value that differs from the new
        # one; record it before overwriting (Req 2.3). Re-collecting the same
        # value, or replacing a previously declined parameter, is not a conflict.
        if (
            existing is not None
            and existing.status == "collected"
            and existing.value != item.value
        ):
            overwrite = Overwrite(
                parameter=item.name,
                previous=existing.value,
                next=item.value,
            )
            overwrites.append(overwrite)
            spec.overwrite_log.append(overwrite)

        # Last value wins: store (or replace) the collected value.
        spec.parameters[item.name] = ParameterValue(
            name=item.name,
            value=item.value,
            status="collected",
            attempts=existing.attempts if existing is not None else 0,
        )

    return MergeResult(spec=spec, overwrites=overwrites)


def missing_parameters(spec: PromptSpecification) -> list[ParameterName]:
    """Derive the still-missing required parameters (Req 2.4).

    A required parameter is missing when the spec has no value for it with status
    ``"collected"``. Declined parameters are not missing (the user chose to skip
    them), so only the ``"collected"`` status counts as satisfied (design Model
    notes). "Missing" is derived here, never stored.

    The result preserves catalog order so callers (such as the Elicitation_Engine)
    see a stable, predictable sequence.

    Args:
        spec: The specification whose collected values are inspected.

    Returns:
        The names of the required parameters that have no collected value, in
        catalog order.
    """
    missing: list[ParameterName] = []
    for name in catalog.required_parameter_names():
        value = spec.parameters.get(name)
        if value is None or value.status != "collected":
            missing.append(name)
    return missing


# The system instruction that frames the interpretation call. It tells the model
# to return a small, fixed JSON shape so the deterministic parsing below is
# predictable. The model does the fuzzy reading; our code does the thresholding.
_INTERPRET_SYSTEM = (
    "You interpret a user's message in a prompt-building assistant. "
    "Identify the user's goal, your confidence in that goal as a number "
    "between 0 and 1, and any parameters you can extract. "
    "Respond with a single JSON object with keys: "
    '"goal" (string or null), "confidence" (number 0..1), and '
    '"parameters" (an object mapping parameter name to its value). '
    "Only use these parameter names: "
    + ", ".join(sorted(d.name for d in catalog.CATALOG))
    + "."
)


def _render_context(context: list[Message]) -> str:
    """Render retained Session messages as plain text for the LLM (Req 7.3).

    Messages are rendered oldest-to-newest as ``sender: text`` lines. Reading the
    ``context`` list here is also what surfaces an unreadable context: if the
    supplied object cannot be iterated or its messages lack the expected shape,
    the caller treats the context as unavailable (Req 7.4).

    Args:
        context: The retained Session messages, in submission order.

    Returns:
        The rendered context, or an empty string when there are no messages.
    """
    lines: list[str] = []
    for msg in context:
        # Access the attributes the Message dataclass defines; a malformed entry
        # raises AttributeError, which interpret() turns into context_unavailable.
        lines.append(f"{msg.sender}: {msg.text}")
    return "\n".join(lines)


def _build_prompt(message: str, context_text: str) -> str:
    """Assemble the user-facing prompt text from the message and context."""
    if context_text:
        return (
            "Conversation so far:\n"
            f"{context_text}\n\n"
            "Latest user message:\n"
            f"{message}"
        )
    return f"User message:\n{message}"


def _structured_from_response(response: Any) -> dict[str, Any]:
    """Pull a structured dict out of an LLM response, however it is shaped.

    Prefers a ``structured`` payload when the response carries one (the mock and
    any structured-aware client expose this); otherwise parses the response
    ``text`` as JSON. Returns an empty dict when neither yields a usable object,
    so interpretation degrades to "no goal, nothing extracted" rather than
    raising.
    """
    structured = getattr(response, "structured", None)
    if isinstance(structured, dict):
        return structured

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _coerce_confidence(value: Any) -> float:
    """Coerce an LLM-supplied confidence into a float in ``[0.0, 1.0]``.

    Anything non-numeric or out of range is clamped; a missing/garbage value
    becomes ``0.0`` so a goal without a credible confidence is treated as
    below-threshold (Req 2.5).
    """
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _extracted_from(parameters: Any) -> list[ExtractedParameter]:
    """Turn the LLM's ``parameters`` payload into :class:`ExtractedParameter`s.

    Accepts a ``{name: value}`` mapping and keeps only entries whose value is a
    non-empty string. Parameters that do not map to the catalog are left in;
    :func:`merge_parameters` is the single place that filters to defined
    parameters, so this stays a faithful read of what the LLM returned.
    """
    extracted: list[ExtractedParameter] = []
    if isinstance(parameters, dict):
        for name, value in parameters.items():
            if not isinstance(name, str):
                continue
            if value is None:
                continue
            text = value if isinstance(value, str) else str(value)
            if text.strip() == "":
                continue
            extracted.append(ExtractedParameter(name=name, value=text))
    return extracted


class IntentAnalyzer:
    """LLM-backed interpreter of user messages (design: Intent_Analyzer).

    Wraps an injected :class:`~prompt_builder.llm_client.LlmClient` so the only
    network access stays behind that single client. :meth:`interpret` turns a
    message (plus retained Session context) into an :class:`InterpretationResult`,
    applying the acceptance threshold and the context-unavailable handling; the
    deterministic :func:`merge_parameters` / :func:`missing_parameters` functions
    in this module do the rest of the Intent_Analyzer's work.
    """

    def __init__(
        self,
        llm: LlmClient,
        *,
        acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
    ) -> None:
        """Create an analyzer.

        Args:
            llm: The LLM client used to interpret messages. Any object satisfying
                the :class:`LlmClient` protocol works, including the test mock.
            acceptance_threshold: The minimum goal confidence to accept a goal
                (Req 2.5). Defaults to :data:`DEFAULT_ACCEPTANCE_THRESHOLD`.
        """
        self._llm = llm
        self._acceptance_threshold = acceptance_threshold

    def interpret(
        self,
        message: str,
        context: list[Message],
        spec: PromptSpecification,
    ) -> InterpretationResult:
        """Interpret a user message against the Session context (Req 2.1-2.5, 7.3, 7.4).

        Renders the retained Session messages as context (Req 7.3), asks the LLM
        to identify the goal, its confidence, and any extracted parameters, then:

        - keeps the goal only when confidence is at or above the acceptance
          threshold; otherwise the goal stays ``None`` so the Elicitation_Engine
          asks the user to describe it (Req 2.5);
        - sets ``context_unavailable`` and interprets the message in isolation
          when the retained context cannot be read (Req 7.4).

        The ``spec`` is not mutated here; merging extracted parameters into the
        spec is :func:`merge_parameters`' job. On any LLM failure the underlying
        :class:`~prompt_builder.llm_client.LlmError` propagates so the caller can
        drive its retryable error handling (design "Error Handling").

        Args:
            message: The user message to interpret.
            context: The retained Session messages used as context.
            spec: The current specification (read for reference; not mutated).

        Returns:
            The :class:`InterpretationResult` for this message.

        Raises:
            LlmError: if the LLM request cannot be completed.
        """
        # Try to read the retained context. If that fails for any reason, fall
        # back to interpreting the current message on its own and flag it
        # (Req 7.4). Rendering context never calls the LLM, so this try/except
        # does not swallow LLM failures.
        context_unavailable = False
        try:
            context_text = _render_context(context)
        except Exception:  # noqa: BLE001 - any unreadable context degrades gracefully
            context_text = ""
            context_unavailable = True

        request = LlmRequest(
            prompt=_build_prompt(message, context_text),
            system=_INTERPRET_SYSTEM,
            operation="interpret",
        )

        # Let LlmError propagate: the caller leaves the Session unchanged and
        # offers a retry (Req 8.1).
        response = self._llm.complete(request)

        structured = _structured_from_response(response)
        confidence = _coerce_confidence(structured.get("confidence"))
        extracted = _extracted_from(structured.get("parameters"))

        raw_goal = structured.get("goal")
        goal = raw_goal if isinstance(raw_goal, str) and raw_goal.strip() else None
        # Below the threshold the goal is not confirmed, so leave it unset
        # regardless of what the LLM proposed (Req 2.5).
        if goal is not None and confidence < self._acceptance_threshold:
            goal = None

        return InterpretationResult(
            confidence=confidence,
            extracted=extracted,
            goal=goal,
            context_unavailable=context_unavailable,
        )
