"""Prompt_Generator for the Conversational Prompt Builder.

The Prompt_Generator turns a collected :class:`~prompt_builder.models.PromptSpecification`
into a :class:`~prompt_builder.models.GeneratedPrompt`. As the design puts it:
"The LLM does the writing; our code guarantees every collected parameter is
included and reports any omitted required ones."

Two responsibilities live here:

- :meth:`PromptGenerator.generate` - ask the LLM to write a polished prompt from
  the goal and the collected parameter values, then *deterministically* guarantee
  that every collected parameter value is literally present in the returned text
  (Req 4.1, 4.2). It also reports exactly which required parameters are not yet
  collected, so the caller can show them to the user (Req 4.4) - generating before
  every required parameter is collected is not an error (design "Early
  generation").
- :meth:`PromptGenerator.revise` - ask the LLM for an updated prompt that
  incorporates a free-text revision request, applying the same "every collected
  value is present" guarantee and the same omitted-required reporting (Req 5.2).

The "every collected value is present" guarantee is what Property 13 asserts and
is our own deterministic code, not something we trust the LLM to do: after the
model returns its text we check each collected value and, if any is missing,
append a structured ``Parameters:`` section listing each collected ``name: value``
so the value is literally present in the final text.

Only the LLM call touches the network, and it goes through the single
:class:`~prompt_builder.llm_client.LlmClient` (design "Architecture"). On an LLM
failure the client raises :class:`~prompt_builder.llm_client.LlmError`; we let it
propagate *without mutating the spec* so the caller (the retry controller / UI)
can leave the Session unchanged and offer a retry (Req 4.5, 5.4, design "Error
Handling"). Because :meth:`revise` returns a brand-new result and never mutates
the ``current`` prompt it was given, a failed revision naturally leaves the
previous prompt unchanged (Req 5.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from prompt_builder.intent_analyzer import missing_parameters
from prompt_builder.llm_client import LlmClient, LlmRequest
from prompt_builder.models import (
    GeneratedPrompt,
    ParameterName,
    ParameterValue,
    PromptSpecification,
)

__all__ = ["GenerationResult", "GenerationSettings", "PromptGenerator"]

# --- LLM sampling settings for prompt generation ----------------------------
# Writing a high-quality, long, well-structured prompt is a "focused but
# substantial" task, so we pick settings deliberately rather than relying on
# provider defaults (reference: promptingguide.ai "LLM Settings").
#
# - temperature low: we want reliable, well-organized prompts, not random
#   phrasing. The guide recommends low temperature for factual/structured work.
# - top_p left unset: the guide advises tuning temperature OR top_p, not both,
#   so we tune temperature only and let the provider default top_p apply.
# - max_tokens high: a detailed, multi-section prompt is long; a generous cap
#   avoids truncation while still bounding cost (the guide's "Max Length").
# - frequency_penalty small and positive: trims repeated phrasing across a long
#   prompt without distorting required terminology. We use frequency (not
#   presence) penalty and leave presence unset, per the guide's "tune one, not
#   both" advice.
_GEN_TEMPERATURE = 0.4
_GEN_MAX_TOKENS = 2048
_GEN_FREQUENCY_PENALTY = 0.2

# Revision is even more conservative: it should change only what the user asked
# for and otherwise preserve the prompt, so temperature is a touch lower.
_REVISE_TEMPERATURE = 0.3
_REVISE_MAX_TOKENS = 2048
_REVISE_FREQUENCY_PENALTY = 0.2


@dataclass
class GenerationSettings:
    """Inference settings for a generation / revision call (promptingguide.ai).

    A small, optional override object the caller (for example the UI's "Model
    settings" panel) can pass to :meth:`PromptGenerator.generate` and
    :meth:`PromptGenerator.revise` to control the LLM's sampling and length
    behavior at call time. Any field left ``None`` falls back to the module's
    task-tuned defaults for generation / revision, and any field the caller sets
    is forwarded to the LLM client (which in turn forwards only the set values to
    the provider, keeping provider defaults for the rest).

    The field set mirrors the Prompt Engineering Guide's LLM Settings page:
    temperature, top_p, max_tokens, frequency_penalty, presence_penalty, and
    stop. The guide advises tuning temperature OR top_p (and frequency OR
    presence penalty), not both; this object does not enforce that, leaving the
    choice to the caller/UI.

    Attributes:
        temperature: Sampling temperature (randomness). Low = focused/factual.
        max_tokens: Hard cap on generated tokens (prevents truncation/runaway).
        top_p: Nucleus-sampling cutoff (alternative to temperature).
        frequency_penalty: Penalty proportional to prior token frequency.
        presence_penalty: Flat penalty on any already-seen token.
        stop: Stop sequence(s) that halt generation.
    """

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str]] = None


@dataclass
class GenerationResult:
    """The outcome of generating or revising a prompt (design: Prompt_Generator).

    Attributes:
        prompt: The produced :class:`~prompt_builder.models.GeneratedPrompt`,
            whose ``text`` includes every collected parameter value and whose
            ``omitted_required`` lists the required parameters not collected.
        omitted_required: Exactly the required parameters with no collected value,
            in catalog order; empty when every required parameter is collected
            (Req 4.4). This mirrors ``prompt.omitted_required`` so callers can read
            it from either place.
    """

    prompt: GeneratedPrompt
    omitted_required: list[ParameterName]


# The system instruction framing the generation call. It tells the model to write
# a single, ready-to-use prompt from the supplied goal and parameters. The model
# does the writing; our deterministic code (below) enforces that every collected
# value actually appears.
_GENERATE_SYSTEM = (
    "You are an expert prompt engineer. You write a single, high-quality, "
    "ready-to-use prompt for a user. You are given the user's goal and a set of "
    "collected parameters (such as audience, domain, tone, format, and length). "
    "Write one polished, well-structured prompt that reflects the goal and "
    "incorporates every supplied parameter value.\n\n"
    "Organize the prompt into clearly labeled sections where helpful (for "
    "example: Role, Objective, Context, Audience, Tone & Voice, Output Format, "
    "Constraints, and Quality Bar). Be specific and give the model enough context "
    "to succeed.\n\n"
    "End the prompt with a section titled '# RECOMMENDED MODEL SETTINGS' that "
    "advises the reader which inference settings to use when running this prompt, "
    "chosen to fit the task (a factual/structured task should use a low "
    "temperature; a creative task a higher one). Include, each on its own line: "
    "Temperature, Top-p (noting to tune temperature OR top-p, not both), "
    "Max tokens (a concrete number sized to the expected output), Frequency "
    "penalty, Presence penalty (noting to tune frequency OR presence, not both), "
    "and any useful Stop sequences (or 'none'). Give a one-clause reason for the "
    "temperature choice.\n\n"
    "Return only the prompt text, with no preamble or explanation."
)

# The system instruction framing a revision call. It tells the model to rewrite an
# existing prompt to incorporate the user's revision request while still honouring
# the collected parameters.
_REVISE_SYSTEM = (
    "You revise an existing prompt for a user. You are given the current prompt, "
    "the user's requested change, the user's goal, and the collected parameters. "
    "Produce one updated prompt that incorporates the requested change while still "
    "reflecting the goal and every supplied parameter value. "
    "Return only the updated prompt text, with no preamble or explanation."
)


def _collected_parameters(spec: PromptSpecification) -> list[ParameterValue]:
    """Return the spec's collected parameters, in a deterministic order.

    Only values with status ``"collected"`` are included; declined parameters are
    deliberately left out of the prompt (the user chose to skip them). Ordering is
    by parameter name so the rendered output is stable across runs, which keeps the
    assembly behaviour deterministic.
    """
    collected = [
        value
        for value in spec.parameters.values()
        if value.status == "collected"
    ]
    collected.sort(key=lambda value: value.name)
    return collected


def _render_spec(spec: PromptSpecification) -> str:
    """Render the goal and collected parameters as plain text for the LLM.

    This is the human-readable content the model writes from. It is a faithful,
    deterministic rendering of what has been collected; it does not itself
    guarantee the values survive into the model's output (that guarantee is
    enforced afterwards by :func:`_ensure_values_present`).
    """
    lines: list[str] = []
    goal = spec.goal if spec.goal and spec.goal.strip() else "(not yet specified)"
    lines.append(f"Goal: {goal}")

    collected = _collected_parameters(spec)
    if collected:
        lines.append("Parameters:")
        for value in collected:
            lines.append(f"- {value.name}: {value.value}")
    else:
        lines.append("Parameters: (none collected)")
    return "\n".join(lines)


def _ensure_values_present(text: str, spec: PromptSpecification) -> str:
    """Guarantee every collected parameter value appears in ``text`` (Req 4.2).

    This is the deterministic backstop behind Property 13: we never rely on the
    LLM to include every value. Any collected value missing from the model's text
    is appended in a structured ``Parameters:`` section as ``name: value`` so the
    value is literally present in the final prompt.

    When the model already included every value (or there are no collected values),
    the text is returned unchanged.
    """
    collected = _collected_parameters(spec)
    missing_values = [value for value in collected if value.value not in text]
    if not missing_values:
        return text

    appended_lines = ["Parameters:"]
    for value in missing_values:
        appended_lines.append(f"{value.name}: {value.value}")
    appended = "\n".join(appended_lines)

    # Separate the appended section from the model's text with a blank line, but
    # only add leading separation when there is existing text to separate from.
    if text.strip():
        return f"{text.rstrip()}\n\n{appended}"
    return appended


def _resolve(settings: Optional[GenerationSettings], field: str, default: Any) -> Any:
    """Return ``settings.field`` when set, else the task-tuned ``default``.

    A field is considered "set" when ``settings`` is provided and the attribute
    is not ``None``. This lets a caller override only the settings it cares about
    (for example just temperature from a UI slider) while every other field keeps
    its generation/revision default.
    """
    if settings is None:
        return default
    value = getattr(settings, field, None)
    return value if value is not None else default


class PromptGenerator:
    """Builds and revises the Generated_Prompt via the LLM (design: Prompt_Generator).

    Wraps an injected :class:`~prompt_builder.llm_client.LlmClient` so the only
    network access stays behind that single client and tests can inject the mock.
    The LLM writes the prompt; this class guarantees every collected parameter is
    included (Req 4.2) and reports the required parameters still missing (Req 4.4).
    """

    def __init__(self, llm: LlmClient) -> None:
        """Create a generator.

        Args:
            llm: The LLM client used to write prompts. Any object satisfying the
                :class:`~prompt_builder.llm_client.LlmClient` protocol works,
                including the test mock.
        """
        self._llm = llm

    def generate(
        self,
        spec: PromptSpecification,
        settings: Optional[GenerationSettings] = None,
    ) -> GenerationResult:
        """Produce a Generated_Prompt from the specification (Req 4.1, 4.2, 4.4).

        Asks the LLM to write a polished prompt from the goal and collected
        parameter values, then deterministically ensures every collected value is
        present in the returned text (Req 4.2, Property 13). The omitted-required
        list is exactly the required parameters with no collected value, in catalog
        order, and is empty when all are collected (Req 4.4); generating before all
        required parameters are collected is intentionally not an error (design
        "Early generation").

        On an LLM failure the underlying
        :class:`~prompt_builder.llm_client.LlmError` propagates and ``spec`` is left
        unchanged, so the caller (retry controller / UI) can offer a retry (Req
        4.5, design "Error Handling"). The ``spec`` is never mutated here.

        Args:
            spec: The specification to generate from (read only; not mutated).

        Returns:
            A :class:`GenerationResult` whose prompt text includes every collected
            value and whose ``omitted_required`` lists the uncollected required
            parameters.

        Raises:
            LlmError: if the LLM request cannot be completed.
        """
        request = LlmRequest(
            prompt=_render_spec(spec),
            system=_GENERATE_SYSTEM,
            temperature=_resolve(settings, "temperature", _GEN_TEMPERATURE),
            max_tokens=_resolve(settings, "max_tokens", _GEN_MAX_TOKENS),
            frequency_penalty=_resolve(
                settings, "frequency_penalty", _GEN_FREQUENCY_PENALTY
            ),
            top_p=_resolve(settings, "top_p", None),
            presence_penalty=_resolve(settings, "presence_penalty", None),
            stop=_resolve(settings, "stop", None),
        )
        # Let LlmError propagate without touching the spec (Req 4.5).
        response = self._llm.complete(request)

        return self._build_result(response.text, spec)

    def revise(
        self,
        current: GeneratedPrompt,
        revision_text: str,
        spec: PromptSpecification,
        settings: Optional[GenerationSettings] = None,
    ) -> GenerationResult:
        """Produce an updated prompt incorporating a revision request (Req 5.2).

        Asks the LLM to rewrite ``current`` so it incorporates ``revision_text``
        while still reflecting the goal and collected parameters, then applies the
        same "every collected value present" guarantee and the same omitted-required
        computation as :meth:`generate`.

        On an LLM failure the underlying
        :class:`~prompt_builder.llm_client.LlmError` propagates. Neither ``current``
        nor ``spec`` is mutated, so a failed revision naturally leaves the previous
        prompt unchanged (Req 5.4, design "Error Handling"); the caller surfaces a
        retryable error.

        Args:
            current: The prompt being revised (read only; not mutated).
            revision_text: The user's requested change.
            spec: The specification backing the prompt (read only; not mutated).

        Returns:
            A :class:`GenerationResult` for the revised prompt.

        Raises:
            LlmError: if the LLM request cannot be completed.
        """
        request = LlmRequest(
            prompt=self._render_revision(current, revision_text, spec),
            system=_REVISE_SYSTEM,
            temperature=_resolve(settings, "temperature", _REVISE_TEMPERATURE),
            max_tokens=_resolve(settings, "max_tokens", _REVISE_MAX_TOKENS),
            frequency_penalty=_resolve(
                settings, "frequency_penalty", _REVISE_FREQUENCY_PENALTY
            ),
            top_p=_resolve(settings, "top_p", None),
            presence_penalty=_resolve(settings, "presence_penalty", None),
            stop=_resolve(settings, "stop", None),
        )
        # Let LlmError propagate; returning only on success leaves ``current``
        # unchanged on failure (Req 5.4).
        response = self._llm.complete(request)

        return self._build_result(response.text, spec)

    @staticmethod
    def _render_revision(
        current: GeneratedPrompt,
        revision_text: str,
        spec: PromptSpecification,
    ) -> str:
        """Assemble the user-facing content for a revision request."""
        return (
            "Current prompt:\n"
            f"{current.text}\n\n"
            "Requested change:\n"
            f"{revision_text}\n\n"
            f"{_render_spec(spec)}"
        )

    @staticmethod
    def _build_result(text: str, spec: PromptSpecification) -> GenerationResult:
        """Turn raw LLM text into a :class:`GenerationResult`.

        Applies the deterministic "every collected value present" guarantee
        (Req 4.2) and computes the omitted-required list via
        :func:`~prompt_builder.intent_analyzer.missing_parameters` (Req 4.4),
        reusing that single source of truth rather than recomputing it here.
        """
        final_text = _ensure_values_present(text, spec)
        omitted_required = missing_parameters(spec)
        prompt = GeneratedPrompt(text=final_text, omitted_required=omitted_required)
        return GenerationResult(prompt=prompt, omitted_required=omitted_required)
