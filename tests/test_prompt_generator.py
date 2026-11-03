"""Tests for the Prompt_Generator (tasks 11.2 and 11.3).

This module covers the :class:`~prompt_builder.prompt_generator.PromptGenerator`:

- Task 11.2 is a Hypothesis property test (Property 13) asserting that a
  generated prompt includes every collected parameter's value and that its
  ``omitted_required`` list equals exactly the required catalog parameters with
  no collected value.
- Task 11.3 is a pair of example-based unit tests for the error paths: a
  generation failure propagates and leaves the spec unchanged (Req 4.5), and a
  revision failure keeps the previous prompt unchanged (Req 5.4).

All tests run fully offline using the mock LLM client (``tests/mock_llm.py``):
``MockLlmClient.echoing()`` echoes the request's rendered content back as the
response text, and ``MockLlmClient.failing()`` raises on every call.
"""

from __future__ import annotations

import copy

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_builder import catalog
from prompt_builder.models import (
    GeneratedPrompt,
    ParameterValue,
    PromptSpecification,
)
from prompt_builder.prompt_generator import PromptGenerator
from tests.mock_llm import MockLlmClient, MockLlmError

# All catalog parameter names and, specifically, the required ones. The property
# test draws from these so every generated spec uses real catalog names.
_ALL_NAMES = [definition.name for definition in catalog.CATALOG]
_REQUIRED_NAMES = set(catalog.required_parameter_names())

# A per-parameter disposition: either we collected a value, declined it, or it
# is simply absent from the spec (missing). The property test mixes all three.
_DISPOSITIONS = st.sampled_from(["collected", "declined", "missing"])

# Non-empty, printable values so that "value is a substring of the prompt text"
# assertions are meaningful. Restricting to printable ASCII keeps the echoed,
# rendered content easy to reason about.
_VALUES = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=40,
)


@st.composite
def _specs(draw: st.DrawFn) -> PromptSpecification:
    """Build a PromptSpecification mixing collected/declined/missing parameters.

    For each catalog parameter an independent disposition is drawn:

    - ``collected`` -> stored with status ``"collected"`` and a random value;
    - ``declined``  -> stored with status ``"declined"`` (a skipped parameter);
    - ``missing``   -> not present in the spec at all.

    The goal is optionally set. The result exercises the full input space the
    Prompt_Generator must handle: some collected, some declined, some missing,
    across both required and optional parameters.
    """
    spec = PromptSpecification(goal_confidence=draw(st.floats(0.0, 1.0)))

    maybe_goal = draw(st.one_of(st.none(), _VALUES))
    if maybe_goal is not None:
        spec.goal = maybe_goal

    for name in _ALL_NAMES:
        disposition = draw(_DISPOSITIONS)
        if disposition == "collected":
            spec.parameters[name] = ParameterValue(
                name=name,
                value=draw(_VALUES),
                status="collected",
                attempts=0,
            )
        elif disposition == "declined":
            spec.parameters[name] = ParameterValue(
                name=name,
                value=draw(_VALUES),
                status="declined",
                attempts=0,
            )
        # "missing" -> leave the parameter out of the spec entirely.

    return spec


# Feature: conversational-prompt-builder, Property 13: Generated prompt includes all collected parameters and reports omissions
@settings(max_examples=200)
@given(spec=_specs())
def test_generated_prompt_includes_collected_and_reports_omissions(
    spec: PromptSpecification,
) -> None:
    """Property 13: collected values are present; omissions are reported exactly.

    For any PromptSpecification, ``generate()`` produces a prompt whose text
    includes the value of every collected parameter (guaranteed by the
    generator's deterministic backstop regardless of what the LLM returns), and
    whose ``omitted_required`` list equals exactly the required catalog
    parameters that are not collected (empty when all required are collected).

    **Validates: Requirements 4.1, 4.2, 4.4**
    """
    generator = PromptGenerator(MockLlmClient.echoing())

    result = generator.generate(spec)

    # Reference model for what *should* be present / omitted.
    collected = {
        name: value
        for name, value in spec.parameters.items()
        if value.status == "collected"
    }
    expected_omitted = {
        name for name in _REQUIRED_NAMES if name not in collected
    }

    # Every collected parameter's value appears verbatim in the prompt text.
    for value in collected.values():
        assert value.value in result.prompt.text

    # The omitted-required list is exactly the uncollected required parameters.
    assert set(result.prompt.omitted_required) == expected_omitted
    assert set(result.omitted_required) == expected_omitted

    # Empty exactly when every required parameter was collected.
    all_required_collected = _REQUIRED_NAMES.issubset(collected.keys())
    assert (result.prompt.omitted_required == []) == all_required_collected


def _spec_with_collected() -> PromptSpecification:
    """A spec with a goal and a couple of collected parameters (for error paths)."""
    spec = PromptSpecification(goal_confidence=0.9, goal="write a blog post")
    spec.parameters["audience"] = ParameterValue(
        name="audience", value="beginners", status="collected", attempts=0
    )
    spec.parameters["tone"] = ParameterValue(
        name="tone", value="friendly", status="collected", attempts=1
    )
    return spec


def test_generate_failure_propagates_and_leaves_spec_unchanged() -> None:
    """Req 4.5: a generation failure raises and does not mutate the spec.

    With a failing LLM, ``generate()`` must let the error surface so the caller
    can offer a retry, and it must leave the collected parameters (and the whole
    spec) untouched.
    """
    spec = _spec_with_collected()
    snapshot = copy.deepcopy(spec)

    generator = PromptGenerator(MockLlmClient.failing())

    with pytest.raises(MockLlmError):
        generator.generate(spec)

    # The spec is unchanged: goal, collected parameters, and the overwrite log.
    assert spec.goal == snapshot.goal
    assert spec.goal_confidence == snapshot.goal_confidence
    assert spec.parameters == snapshot.parameters
    assert spec.overwrite_log == snapshot.overwrite_log


def test_revise_failure_keeps_previous_prompt_unchanged() -> None:
    """Req 5.4: a revision failure raises and leaves the previous prompt unchanged.

    With a failing LLM, ``revise()`` must raise, and the ``current`` prompt it
    was given must keep its original text and omitted-required list.
    """
    spec = _spec_with_collected()
    current = GeneratedPrompt(
        text="Write a friendly blog post for beginners.",
        omitted_required=["domain"],
    )
    original_text = current.text
    original_omitted = list(current.omitted_required)

    generator = PromptGenerator(MockLlmClient.failing())

    with pytest.raises(MockLlmError):
        generator.revise(current, "make it more formal", spec)

    # The previous prompt is untouched.
    assert current.text == original_text
    assert current.omitted_required == original_omitted


def test_generate_passes_recommended_sampling_settings() -> None:
    """Generation sends deliberate LLM settings (reference: promptingguide.ai).

    The generator should not rely on provider defaults: it sends a low
    temperature (reliable, structured output), a generous max_tokens (long,
    detailed prompts are not truncated), and a small frequency penalty (less
    repetition), per the LLM Settings guidance.
    """
    from prompt_builder.prompt_generator import (
        _GEN_FREQUENCY_PENALTY,
        _GEN_MAX_TOKENS,
        _GEN_TEMPERATURE,
    )

    client = MockLlmClient.echoing()
    generator = PromptGenerator(client)
    generator.generate(_spec_with_collected())

    assert len(client.calls) == 1
    request = client.calls[0]
    assert request.temperature == _GEN_TEMPERATURE
    assert request.max_tokens == _GEN_MAX_TOKENS
    assert request.frequency_penalty == _GEN_FREQUENCY_PENALTY
    # The guide advises tuning temperature OR top_p (and frequency OR presence),
    # not both; so top_p and presence_penalty are deliberately left unset.
    assert request.top_p is None
    assert request.presence_penalty is None


def test_revise_passes_recommended_sampling_settings() -> None:
    """Revision sends its own (more conservative) LLM settings."""
    from prompt_builder.prompt_generator import (
        _REVISE_FREQUENCY_PENALTY,
        _REVISE_MAX_TOKENS,
        _REVISE_TEMPERATURE,
    )

    client = MockLlmClient.echoing()
    generator = PromptGenerator(client)
    current = GeneratedPrompt(text="A draft prompt.", omitted_required=[])
    generator.revise(current, "make it more formal", _spec_with_collected())

    assert len(client.calls) == 1
    request = client.calls[0]
    assert request.temperature == _REVISE_TEMPERATURE
    assert request.max_tokens == _REVISE_MAX_TOKENS
    assert request.frequency_penalty == _REVISE_FREQUENCY_PENALTY
    assert request.top_p is None
    assert request.presence_penalty is None


def test_generate_applies_caller_supplied_settings_override() -> None:
    """A caller-supplied GenerationSettings overrides the generation defaults.

    The editable "Model settings" panel passes a GenerationSettings into
    generate(); any field it sets must reach the LLM request, while fields it
    leaves as None fall back to the task-tuned defaults.
    """
    from prompt_builder.prompt_generator import (
        _GEN_MAX_TOKENS,
        GenerationSettings,
    )

    client = MockLlmClient.echoing()
    generator = PromptGenerator(client)
    override = GenerationSettings(
        temperature=1.3,
        top_p=0.8,
        presence_penalty=0.5,
        stop=["###"],
    )
    generator.generate(_spec_with_collected(), override)

    request = client.calls[0]
    # Overridden fields reach the request.
    assert request.temperature == 1.3
    assert request.top_p == 0.8
    assert request.presence_penalty == 0.5
    assert request.stop == ["###"]
    # An unset field falls back to the generation default.
    assert request.max_tokens == _GEN_MAX_TOKENS
