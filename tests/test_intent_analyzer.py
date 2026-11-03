"""Property tests for the Intent_Analyzer's deterministic parameter tracking.

These tests exercise the pure, offline merge/derivation rules in
``prompt_builder.intent_analyzer`` (not the LLM-backed interpreter).
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from prompt_builder import catalog
from prompt_builder.intent_analyzer import (
    ExtractedParameter,
    merge_parameters,
    missing_parameters,
)
from prompt_builder.models import ParameterValue, PromptSpecification

# The parameter names the catalog defines; a merge only stores values for these.
_DEFINED_NAMES = sorted(d.name for d in catalog.CATALOG)


def _extracted_parameters() -> st.SearchStrategy[list[ExtractedParameter]]:
    """Generate lists of extracted parameters that map to defined Parameters.

    Names are drawn from the catalog so every generated entry is expected to be
    merged. Values are arbitrary non-empty text. Lists may repeat a name, which
    exercises the last-value-wins rule within a single merge.
    """
    names = st.sampled_from(_DEFINED_NAMES)
    values = st.text(min_size=1, max_size=40)
    return st.lists(
        st.builds(ExtractedParameter, name=names, value=values),
        max_size=12,
    )


# Feature: conversational-prompt-builder, Property 3: Extracted parameters are merged into the specification
#
# For any list of extracted parameters that map to defined Parameters, after
# merging, every such parameter has its value stored in the Prompt_Specification.
#
# Within a single merge later entries win, so the stored value is the LAST
# extracted value for each name.
#
# **Validates: Requirements 2.2**
@given(extracted=_extracted_parameters())
@settings(max_examples=200)
def test_extracted_parameters_are_merged_into_the_specification(
    extracted: list[ExtractedParameter],
) -> None:
    spec = PromptSpecification(goal_confidence=0.0)

    result = merge_parameters(spec, extracted)

    # The merge mutates in place and returns the same spec.
    assert result.spec is spec

    # Last-value-wins: the expected stored value for each name is its last
    # occurrence in the extracted list.
    expected_last: dict[str, str] = {}
    for item in extracted:
        expected_last[item.name] = item.value

    # Every extracted parameter that maps to a defined Parameter is stored, with
    # status "collected" and the most recently extracted value.
    for name, value in expected_last.items():
        assert name in spec.parameters
        stored = spec.parameters[name]
        assert stored.value == value
        assert stored.status == "collected"
        assert stored.name == name

    # Nothing beyond the extracted names appears in the spec.
    assert set(spec.parameters.keys()) == set(expected_last.keys())


# Feature: conversational-prompt-builder, Property 4: Conflicts keep the most recent value and are logged
#
# For any parameter that already has a value and a newly extracted different
# value, after merging the stored value equals the new value and the overwrite
# log records the previous value.
#
# **Validates: Requirements 2.3**
@given(
    name=st.sampled_from(_DEFINED_NAMES),
    previous=st.text(min_size=1, max_size=40),
    new=st.text(min_size=1, max_size=40),
)
@settings(max_examples=100)
def test_conflicts_keep_the_most_recent_value_and_are_logged(
    name: str,
    previous: str,
    new: str,
) -> None:
    # Only a *different* new value is a conflict; discard same-value pairs.
    assume(previous != new)

    spec = PromptSpecification(goal_confidence=0.0)

    # Seed an existing collected value for the parameter.
    merge_parameters(spec, [ExtractedParameter(name=name, value=previous)])
    assert spec.parameters[name].value == previous

    log_len_before = len(spec.overwrite_log)

    # Merge a different value for the same parameter: this is the conflict.
    result = merge_parameters(spec, [ExtractedParameter(name=name, value=new)])

    # The most recent value wins.
    assert spec.parameters[name].value == new
    assert spec.parameters[name].status == "collected"

    # The overwrite is reported by this merge and appended to the spec log,
    # recording the previous value and the new value for this parameter.
    assert len(result.overwrites) == 1
    overwrite = result.overwrites[0]
    assert overwrite.parameter == name
    assert overwrite.previous == previous
    assert overwrite.next == new

    assert len(spec.overwrite_log) == log_len_before + 1
    assert spec.overwrite_log[-1] == overwrite


# The required parameter names from the catalog; missing is derived only from
# these (optional parameters are never "missing" required ones).
_REQUIRED_NAMES = set(catalog.required_parameter_names())


def _parameter_values() -> st.SearchStrategy[dict[str, ParameterValue]]:
    """Generate a spec's ``parameters`` map mixing collected/declined/missing.

    For each defined catalog parameter we independently choose one of three
    states:

    - ``"collected"`` - a stored value with status "collected" (satisfies a
      required parameter),
    - ``"declined"`` - a stored value with status "declined" (the user skipped
      it; a required declined parameter is still *missing*),
    - ``None`` - no stored value at all (missing).

    This exercises the full cross-product of statuses across required and
    optional parameters, which is exactly what Property 5 ranges over.
    """
    states = st.sampled_from(["collected", "declined", "absent"])

    def _build(choices: dict[str, str]) -> dict[str, ParameterValue]:
        params: dict[str, ParameterValue] = {}
        for name, state in choices.items():
            if state == "absent":
                continue
            params[name] = ParameterValue(
                name=name,
                value="x",
                status=state,  # type: ignore[arg-type]
                attempts=0,
            )
        return params

    return st.fixed_dictionaries(
        {name: states for name in _DEFINED_NAMES}
    ).map(_build)


# Feature: conversational-prompt-builder, Property 5: Missing set is exactly the uncollected required parameters
#
# For any Prompt_Specification whose parameters mix collected, declined, and
# absent values across required and optional Parameters, the computed missing
# set equals exactly the required Parameters that have no value with status
# "collected". Declined and absent required parameters are missing; optional
# parameters are never part of the missing set.
#
# **Validates: Requirements 2.4**
@given(parameters=_parameter_values())
@settings(max_examples=100)
def test_missing_set_is_exactly_the_uncollected_required_parameters(
    parameters: dict[str, ParameterValue],
) -> None:
    spec = PromptSpecification(goal_confidence=0.0, parameters=parameters)

    missing = missing_parameters(spec)

    # The expected missing set: every required parameter that lacks a value with
    # status "collected" (absent or declined both count as missing).
    expected = {
        name
        for name in _REQUIRED_NAMES
        if parameters.get(name) is None or parameters[name].status != "collected"
    }

    # Exactly the uncollected required parameters, no more and no less.
    assert set(missing) == expected

    # Optional parameters are never reported as missing.
    assert set(missing) <= _REQUIRED_NAMES

    # The result is free of duplicates and preserves catalog order.
    assert len(missing) == len(set(missing))
    assert missing == [n for n in catalog.required_parameter_names() if n in expected]
