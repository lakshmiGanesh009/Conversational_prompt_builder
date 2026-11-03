"""Unit tests for the core data models and the parameter catalog (task 2.3).

These checks pin down the dataclass defaults the rest of the app relies on (an
empty specification has no goal and no collected parameters) and verify the
read-only catalog helpers expose required parameters and default values
correctly (Requirements 1.3 and 3.5).
"""

from __future__ import annotations

from prompt_builder import catalog
from prompt_builder.catalog import (
    CATALOG,
    get_definition,
    required_parameter_names,
)
from prompt_builder.models import (
    GeneratedPrompt,
    Message,
    Overwrite,
    ParameterDefinition,
    ParameterValue,
    PromptSpecification,
    Session,
)


# --- Dataclass defaults (Req 1.3, 7.1) --------------------------------------


def test_empty_spec_has_no_goal_and_no_parameters():
    spec = PromptSpecification(goal_confidence=0.0)

    assert spec.goal is None
    assert spec.parameters == {}
    assert spec.overwrite_log == []


def test_spec_mutable_defaults_are_not_shared_between_instances():
    first = PromptSpecification(goal_confidence=0.0)
    second = PromptSpecification(goal_confidence=0.0)

    first.parameters["audience"] = ParameterValue(
        name="audience", value="students", status="collected", attempts=1
    )
    first.overwrite_log.append(Overwrite(parameter="tone", previous="x", next="y"))

    # The second instance must keep its own empty containers.
    assert second.parameters == {}
    assert second.overwrite_log == []


def test_new_session_defaults_are_clean():
    session = Session(id="s1")

    assert session.messages == []
    assert session.spec is None
    assert session.follow_up_count == 0
    assert session.generated_prompt is None
    assert session.context_available is True


def test_session_mutable_defaults_are_not_shared_between_instances():
    first = Session(id="s1")
    second = Session(id="s2")

    first.messages.append(
        Message(id="m1", sender="user", text="hi", created_at=1.0)
    )

    assert second.messages == []


def test_parameter_definition_default_value_is_optional():
    required = ParameterDefinition(name="audience", required=True)
    optional = ParameterDefinition(
        name="tone", required=False, default_value="neutral"
    )

    assert required.default_value is None
    assert optional.default_value == "neutral"


def test_parameter_value_fields_round_trip():
    value = ParameterValue(
        name="tone", value="friendly", status="collected", attempts=2
    )

    assert value.name == "tone"
    assert value.value == "friendly"
    assert value.status == "collected"
    assert value.attempts == 2


def test_generated_prompt_tracks_omitted_required():
    prompt = GeneratedPrompt(text="do the thing", omitted_required=["domain"])

    assert prompt.text == "do the thing"
    assert prompt.omitted_required == ["domain"]


# --- Catalog helpers (Req 2.4, 3.5) -----------------------------------------


def test_required_parameter_names_match_required_definitions():
    expected = [d.name for d in CATALOG if d.required]

    assert required_parameter_names() == expected


def test_required_parameter_names_preserves_catalog_order():
    names = required_parameter_names()
    positions = [c.name for c in CATALOG]

    # Required names appear in the same relative order as in the catalog.
    assert names == sorted(names, key=positions.index)


def test_required_parameter_names_excludes_optional_parameters():
    optional_names = {d.name for d in CATALOG if not d.required}

    assert optional_names.isdisjoint(set(required_parameter_names()))


def test_get_definition_returns_matching_definition():
    for definition in CATALOG:
        assert get_definition(definition.name) is definition


def test_get_definition_returns_none_for_unknown_name():
    assert get_definition("nonexistent-parameter") is None


def test_required_parameters_have_no_default_value():
    for definition in CATALOG:
        if definition.required:
            assert definition.default_value is None


def test_optional_parameters_expose_their_defaults_via_get_definition():
    for definition in CATALOG:
        if not definition.required and definition.default_value is not None:
            fetched = get_definition(definition.name)
            assert fetched is not None
            assert fetched.default_value == definition.default_value


def test_catalog_names_are_unique():
    names = [d.name for d in catalog.CATALOG]

    assert len(names) == len(set(names))
