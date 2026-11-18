"""Focused tests for deterministic frontend view-model helpers."""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_builder.demo_examples import DEMO_EXAMPLES
from prompt_builder.frontend import (
    build_prompt_exports,
    derive_stepper,
    filter_templates,
    score_prompt_quality,
    template_categories,
)
from prompt_builder.models import GeneratedPrompt, Message, ParameterValue, PromptSpecification, Session


def _session(*, has_message=False, audience=False, domain=False, prompt=False):
    parameters = {}
    if audience:
        parameters["audience"] = ParameterValue("audience", "developers", "collected", 1)
    if domain:
        parameters["domain"] = ParameterValue("domain", "software", "collected", 1)
    return Session(
        id="session",
        messages=[Message("m", "user", "Build something", 1.0)] if has_message else [],
        spec=PromptSpecification(0.9, parameters=parameters),
        generated_prompt=GeneratedPrompt("Done", []) if prompt else None,
    )


def test_stepper_moves_from_idea_through_review() -> None:
    assert [s.status for s in derive_stepper(None)] == ["active", "upcoming", "upcoming", "upcoming"]
    assert [s.status for s in derive_stepper(_session(has_message=True))] == ["completed", "active", "upcoming", "upcoming"]
    assert [s.status for s in derive_stepper(_session(has_message=True, audience=True, domain=True))] == ["completed", "completed", "active", "upcoming"]
    review = derive_stepper(_session(has_message=True, prompt=True))
    assert [s.status for s in review] == ["completed", "completed", "completed", "active"]
    assert review[-1].state_label == "Current step"


@given(has_message=st.booleans(), audience=st.booleans(), domain=st.booleans(), prompt=st.booleans())
@settings(max_examples=100)
def test_stepper_always_has_exactly_one_accessibly_labelled_active_stage(
    has_message, audience, domain, prompt
) -> None:
    steps = derive_stepper(_session(
        has_message=has_message, audience=audience, domain=domain, prompt=prompt
    ))
    assert [step.status for step in steps].count("active") == 1
    assert all(step.state_label in {"Completed", "Current step", "Not started"} for step in steps)


EXAMPLES = [
    {"title": "Launch email", "description": "Announce a product", "request": "email", "goal": "launch", "category": "Marketing"},
    {"title": "Python plan", "description": "Learn Python", "request": "study", "goal": "learning", "category": "Learning"},
]


def test_demo_gallery_has_the_approved_categories_and_card_metadata() -> None:
    assert {example["category"] for example in DEMO_EXAMPLES} == {
        "Marketing",
        "Learning",
        "E-commerce",
        "Video",
        "Study Notes",
        "Software",
    }
    assert all(example["title"].strip() for example in DEMO_EXAMPLES)
    assert all(example["description"].strip() for example in DEMO_EXAMPLES)


def test_template_filtering_combines_search_and_category_and_preserves_order() -> None:
    assert template_categories(EXAMPLES) == ("All templates", "Learning", "Marketing")
    assert [item["title"] for item in filter_templates(EXAMPLES, "python")] == ["Python plan"]
    assert [item["title"] for item in filter_templates(EXAMPLES, category="Marketing")] == ["Launch email"]
    assert filter_templates(EXAMPLES, "python", "Marketing") == []


@given(query=st.text(max_size=30))
@settings(max_examples=100)
def test_template_search_never_adds_or_reorders_templates(query) -> None:
    result = filter_templates(EXAMPLES, query)
    indexes = [EXAMPLES.index(item) for item in result]
    assert indexes == sorted(indexes)
    assert all(item in EXAMPLES for item in result)


def test_quality_score_has_transparent_all_or_none_boundaries() -> None:
    empty = score_prompt_quality("")
    assert empty.overall == 0 and empty.passed == 0 and empty.total == 7
    assert all(not dimension.passed for dimension in empty.dimensions)

    complete = score_prompt_quality(
        "# ROLE\nExpert for {{AUDIENCE}}.\n# CONTEXT\nA specific 30-day project.\n"
        "# OUTPUT CONTRACT\n- First section\n- Second section\n# CONSTRAINTS\nDo not invent facts.\n"
        "# FEW-SHOT EXAMPLE\nFor example: concise output.\n# SELF-CHECK\nCheck every rule.\n"
        "# RECOMMENDED MODEL SETTINGS\nTemperature: 0.3; Max tokens: 1000."
    )
    assert complete.overall == 100 and complete.passed == complete.total == 7
    assert all(dimension.passed for dimension in complete.dimensions)


@given(text=st.text(max_size=1000))
@settings(max_examples=100)
def test_quality_score_is_always_bounded_and_dimensionally_consistent(text) -> None:
    score = score_prompt_quality(text)
    assert 0 <= score.overall <= 100
    assert score.passed == sum(dimension.passed for dimension in score.dimensions)
    assert score.total == len(score.dimensions) == 7


def test_exports_include_three_real_formats_and_structured_metadata() -> None:
    prompt = GeneratedPrompt("# Prompt\nWrite clearly.", ["audience"])
    exports = build_prompt_exports(
        prompt,
        settings={"temperature": 0.4, "stop": None},
        metadata={"source": "builder", "quality_score": 57},
        stem="My Prompt!",
    )
    assert [item.file_name for item in exports] == ["My-Prompt.md", "My-Prompt.txt", "My-Prompt.json"]
    assert exports[0].data == prompt.text == exports[1].data
    payload = json.loads(exports[2].data)
    assert payload["prompt"] == prompt.text
    assert payload["omitted_required"] == ["audience"]
    assert payload["settings"]["temperature"] == 0.4
    assert payload["metadata"]["quality_score"] == 57


def test_exports_remove_credential_fields_but_keep_model_token_settings() -> None:
    exports = build_prompt_exports(
        GeneratedPrompt("Safe prompt", []),
        settings={
            "api_key": "do-not-export",
            "max_tokens": 2048,
            "nested": {"access_token": "also-secret", "top_p": 0.9},
        },
        metadata={"source": "builder", "password": "secret"},
    )
    payload = json.loads(exports[2].data)

    assert payload["settings"] == {"max_tokens": 2048, "nested": {"top_p": 0.9}}
    assert payload["metadata"] == {"source": "builder"}
    assert "do-not-export" not in exports[2].data
    assert "also-secret" not in exports[2].data


@given(text=st.text(max_size=500), omitted=st.lists(st.text(max_size=30), max_size=5))
@settings(max_examples=100)
def test_json_export_round_trips_prompt_content(text, omitted) -> None:
    export = build_prompt_exports(GeneratedPrompt(text, omitted))[2]
    payload = json.loads(export.data)
    assert payload["prompt"] == text
    assert payload["omitted_required"] == omitted
