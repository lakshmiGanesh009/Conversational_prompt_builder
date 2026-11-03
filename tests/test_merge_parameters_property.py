"""Property-based test for parameter merging (task 6.2).

# Feature: conversational-prompt-builder, Property 3: Extracted parameters are merged into the specification

Property 3 (design) / Requirement 2.2: for any list of extracted parameters,
after :func:`prompt_builder.intent_analyzer.merge_parameters`, every extracted
parameter that maps to a defined catalog Parameter has its value stored in the
Prompt_Specification with status ``"collected"``. When the same parameter is
extracted more than once in the batch, the most recently extracted value wins
(last value wins). Extracted parameters that do not map to a defined catalog
Parameter are ignored and never appear in the spec.

``merge_parameters`` is a pure, deterministic function, so this test runs fully
offline with no network and no Streamlit server. The generators below mix
catalog-defined parameter names with unknown names, and produce repeated names
so the last-value-wins behaviour is exercised.

**Validates: Requirements 2.2**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_builder import catalog
from prompt_builder.intent_analyzer import ExtractedParameter, merge_parameters
from prompt_builder.models import PromptSpecification

# The names the catalog defines. Values extracted for these must be stored.
_DEFINED_NAMES = [definition.name for definition in catalog.CATALOG]

# Names that are deliberately NOT in the catalog; values for these are ignored.
_UNKNOWN_NAMES = ["colour", "mood", "zzz_unknown", "temperature", ""]

# A parameter name is either a known catalog name or an unknown one.
_names = st.sampled_from(_DEFINED_NAMES + _UNKNOWN_NAMES)

# Non-empty values so "stored value == extracted value" assertions are meaningful.
_values = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=40,
)

# A single extracted parameter: a (possibly unknown) name with a value.
_extracted_params = st.builds(
    ExtractedParameter, name=_names, value=_values
)

# A batch of extracted parameters in extraction order, including repeats.
_extracted_batches = st.lists(_extracted_params, min_size=0, max_size=12)


def _empty_spec() -> PromptSpecification:
    """A fresh, empty specification with no collected parameters."""
    return PromptSpecification(goal_confidence=0.0)


@settings(max_examples=200)
@given(extracted=_extracted_batches)
def test_defined_extracted_parameters_are_stored_collected(
    extracted: list[ExtractedParameter],
) -> None:
    """Property 3: every defined extracted parameter is stored as collected.

    Last value wins for repeated names; unknown names are ignored entirely.
    """
    spec = _empty_spec()

    result = merge_parameters(spec, extracted)

    # Reference model: walk the batch in order, keeping the last value seen for
    # each defined parameter name and dropping unknown names.
    expected_last: dict[str, str] = {}
    for item in extracted:
        if catalog.get_definition(item.name) is not None:
            expected_last[item.name] = item.value

    stored = result.spec.parameters

    # Only defined parameters are stored; unknown names never appear.
    assert set(stored.keys()) == set(expected_last.keys())

    for name, expected_value in expected_last.items():
        value = stored[name]
        # The stored value is the most recently extracted one (last value wins).
        assert value.value == expected_value
        # And it is marked collected (Req 2.2).
        assert value.status == "collected"
        assert value.name == name

    # No unknown name leaks into the spec.
    for unknown in _UNKNOWN_NAMES:
        if unknown not in expected_last:
            assert unknown not in stored


@settings(max_examples=200)
@given(name=st.sampled_from(_DEFINED_NAMES), values=st.lists(_values, min_size=1, max_size=6))
def test_last_value_wins_for_repeated_parameter(
    name: str, values: list[str]
) -> None:
    """Repeatedly extracting one parameter keeps exactly the final value."""
    spec = _empty_spec()
    extracted = [ExtractedParameter(name=name, value=v) for v in values]

    result = merge_parameters(spec, extracted)

    stored = result.spec.parameters[name]
    assert stored.value == values[-1]
    assert stored.status == "collected"


@settings(max_examples=200)
@given(
    unknowns=st.lists(
        st.builds(
            ExtractedParameter,
            name=st.sampled_from(_UNKNOWN_NAMES),
            value=_values,
        ),
        min_size=0,
        max_size=8,
    )
)
def test_unknown_parameters_are_ignored(
    unknowns: list[ExtractedParameter],
) -> None:
    """Extracted parameters with no catalog definition are never stored."""
    spec = _empty_spec()

    result = merge_parameters(spec, unknowns)

    assert result.spec.parameters == {}
    assert result.overwrites == []
