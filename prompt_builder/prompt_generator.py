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

from prompt_builder.adaptive_planner import (
    TaskPlan,
    plan_prompt,
    relevant_missing_fields,
)
from prompt_builder import catalog
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
_BASE_GENERATE_SYSTEM = (
    "You are a senior prompt architect. Return one ready-to-paste prompt, not advice about writing one. "
    "Use the supplied task plan and brief faithfully. Include every captured value, but never force "
    "declined or irrelevant fields. Use {{UPPER_SNAKE_CASE}} placeholders only for inputs genuinely "
    "expected when the prompt is run; do not replace concrete details with placeholders. Omit empty "
    "sections and filler. Return only the final prompt text."
)

_DEPTH_INSTRUCTIONS = {
    "focused": (
        "Keep the prompt concise and execution-focused. Use only the relevant subset of labeled "
        "sections, usually Objective, Inputs, Output Contract, and Constraints. Do not add ceremony."
    ),
    "detailed": (
        "Produce a detailed prompt brief with the relevant labeled sections. Add explicit requirements, "
        "a clear output contract, factuality behavior, and a short quality check where useful."
    ),
    "comprehensive": (
        "Produce a comprehensive professional prompt brief. Use relevant labeled sections from Role, "
        "Objective, Context, Inputs, Audience, Requirements, Output Contract, Constraints, Examples, "
        "Edge Cases, Quality Checks, Definition of Done, Assumptions, and Recommended Model Settings. "
        "Include explicit do/do-not rules, measurable acceptance criteria, silent self-verification, "
        "and behavior for missing or conflicting information."
    ),
}

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


def recommended_model_settings(plan: TaskPlan) -> dict[str, Any]:
    """Return task/depth-specific guidance without tuning two samplers at once."""
    temperatures = {
        "software": 0.2,
        "summarize": 0.1,
        "rewrite": 0.3,
        "learning": 0.3,
        "video": 0.6,
        "marketing": 0.7,
        "general": 0.4,
    }
    caps = {"focused": 900, "detailed": 1800, "comprehensive": 3500}
    return {
        "temperature": temperatures[plan.task_type],
        "max_tokens": caps[plan.depth],
        "note": (
            "Tune temperature only; leave top-p at the provider default. "
            "max_tokens is a runtime truncation cap, not a request for quality or verbosity."
        ),
    }


def build_generation_instruction(spec: PromptSpecification, plan: Optional[TaskPlan] = None) -> str:
    """Build the deterministic adaptive instruction sent as the system message."""
    active = plan or plan_prompt(spec)
    return (
        f"{_BASE_GENERATE_SYSTEM}\n\nTASK TYPE: {active.task_type}\n"
        f"PROMPT DEPTH: {active.depth}\n{_DEPTH_INSTRUCTIONS[active.depth]}"
    )


def _render_spec(spec: PromptSpecification, plan: Optional[TaskPlan] = None) -> str:
    """Render a complete, deterministic prompt brief for the writing model."""
    active = plan or plan_prompt(spec)
    lines: list[str] = [
        "Prompt plan:",
        f"- Task type: {active.task_type}",
        f"- Effective depth: {active.depth}",
        f"- Goal / objective: {spec.goal if spec.goal and spec.goal.strip() else '(not yet specified)'}",
    ]
    collected = _collected_parameters(spec)
    if collected:
        lines.append("Captured brief fields:")
        for value in collected:
            lines.append(f"- {catalog.parameter_label(value.name)} [{value.name}]: {value.value}")
    else:
        lines.append("Captured brief fields: (none)")

    declined = sorted(name for name, value in spec.parameters.items() if value.status == "declined")
    if declined:
        lines.append("Declined fields (do not force or invent): " + ", ".join(declined))
    missing = relevant_missing_fields(spec, active)
    if missing:
        lines.append("Relevant information still missing:")
        for name in missing:
            lines.append(f"- {catalog.parameter_label(name)}")
        lines.append(
            "Use a {{VARIABLE}} only when that information is genuinely required at run time; "
            "otherwise state a conservative assumption inside the prompt."
        )

    settings = recommended_model_settings(active)
    lines.extend(
        [
            "Recommended settings to include only when useful:",
            f"- Temperature: {settings['temperature']}",
            f"- Max tokens runtime cap: {settings['max_tokens']}",
            f"- Guidance: {settings['note']}",
            "The target prompt must state the desired response length/detail semantically when relevant.",
        ]
    )
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


def _ensure_adaptive_structure(text: str, spec: PromptSpecification, plan: TaskPlan) -> str:
    """Add only critical missing sections for detailed/comprehensive prompts."""
    if plan.depth == "focused":
        return text

    lowered = text.casefold()
    additions: list[str] = []

    def missing_heading(*names: str) -> bool:
        return not any(f"# {name}" in lowered or f"## {name}" in lowered for name in names)

    if missing_heading("objective") and spec.goal:
        additions.append(f"# OBJECTIVE\n{spec.goal}")

    if plan.depth == "comprehensive" and missing_heading("requirements", "constraints"):
        rules = [
            "- DO follow every captured requirement and explicit output constraint.",
            "- DO distinguish supplied facts from assumptions.",
            "- DO NOT invent unavailable facts, sources, capabilities, or results.",
        ]
        additions.append("# REQUIREMENTS AND CONSTRAINTS\n" + "\n".join(rules))

    if missing_heading("quality checks", "self-check"):
        additions.append(
            "# QUALITY CHECKS\nBefore answering, silently verify that the response follows the output contract, "
            "covers every must-have item, respects exclusions, and identifies unsupported assumptions."
        )

    if plan.depth == "comprehensive" and missing_heading("definition of done", "acceptance criteria"):
        success = spec.parameters.get("success_criteria")
        done = success.value if success is not None and success.status == "collected" else (
            "The result is complete, internally consistent, directly usable, and each explicit requirement is verifiably satisfied."
        )
        additions.append(f"# DEFINITION OF DONE\n{done}")

    if plan.depth == "comprehensive" and missing_heading("if information is missing", "missing information"):
        additions.append(
            "# IF INFORMATION IS MISSING\nState the smallest necessary assumptions and continue when safe. "
            "If a missing fact would materially change or invalidate the result, ask a concise clarifying question instead of inventing it."
        )

    if plan.depth == "comprehensive" and missing_heading("recommended model settings", "model settings"):
        settings = recommended_model_settings(plan)
        additions.append(
            "# RECOMMENDED MODEL SETTINGS\n"
            f"- Temperature: {settings['temperature']}\n"
            f"- Max tokens: {settings['max_tokens']} (runtime cap only; follow the semantic length requirements above)\n"
            "- Top-p: provider default; tune temperature or top-p, not both"
        )

    if not additions:
        return text
    prefix = text.rstrip()
    return (prefix + "\n\n" if prefix else "") + "\n\n".join(additions)


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
        plan = plan_prompt(spec)
        request = LlmRequest(
            prompt=_render_spec(spec, plan),
            system=build_generation_instruction(spec, plan),
            temperature=_resolve(settings, "temperature", _GEN_TEMPERATURE),
            max_tokens=_resolve(settings, "max_tokens", _GEN_MAX_TOKENS),
            frequency_penalty=_resolve(
                settings, "frequency_penalty", _GEN_FREQUENCY_PENALTY
            ),
            top_p=_resolve(settings, "top_p", None),
            presence_penalty=_resolve(settings, "presence_penalty", None),
            stop=_resolve(settings, "stop", None),
            operation="generate",
        )
        # Let LlmError propagate without touching the spec (Req 4.5).
        response = self._llm.complete(request)

        return self._build_result(response.text, spec, plan)

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
        plan = plan_prompt(spec)
        request = LlmRequest(
            prompt=self._render_revision(current, revision_text, spec),
            system=_REVISE_SYSTEM + "\n\n" + build_generation_instruction(spec, plan),
            temperature=_resolve(settings, "temperature", _REVISE_TEMPERATURE),
            max_tokens=_resolve(settings, "max_tokens", _REVISE_MAX_TOKENS),
            frequency_penalty=_resolve(
                settings, "frequency_penalty", _REVISE_FREQUENCY_PENALTY
            ),
            top_p=_resolve(settings, "top_p", None),
            presence_penalty=_resolve(settings, "presence_penalty", None),
            stop=_resolve(settings, "stop", None),
            operation="revise",
        )
        # Let LlmError propagate; returning only on success leaves ``current``
        # unchanged on failure (Req 5.4).
        response = self._llm.complete(request)

        return self._build_result(response.text, spec, plan)

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
    def _build_result(
        text: str,
        spec: PromptSpecification,
        plan: Optional[TaskPlan] = None,
    ) -> GenerationResult:
        """Apply deterministic completeness, structure, and omission reporting."""
        active_plan = plan or plan_prompt(spec)
        final_text = _ensure_values_present(text, spec)
        final_text = _ensure_adaptive_structure(final_text, spec, active_plan)
        # Adaptive planning controls which questions are worth asking, but it
        # does not change the required-field contract. A focused prompt may stop
        # questioning early while still reporting required omissions.
        omitted_required = missing_parameters(spec)
        prompt = GeneratedPrompt(text=final_text, omitted_required=omitted_required)
        return GenerationResult(prompt=prompt, omitted_required=omitted_required)
