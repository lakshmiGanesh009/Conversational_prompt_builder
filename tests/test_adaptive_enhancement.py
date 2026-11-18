"""Focused unit and property coverage for adaptive prompt planning."""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st
from streamlit.testing.v1 import AppTest

from prompt_builder import catalog
from prompt_builder.adaptive_planner import (
    DEPTH_OPTIONS,
    classify_complexity,
    classify_task_type,
    coverage_items,
    plan_prompt,
    relevant_missing_fields,
)
from prompt_builder.demo_examples import DEMO_EXAMPLES
from prompt_builder.elicitation_engine import MAX_ATTEMPTS, next_question
from prompt_builder.frontend import score_prompt_quality
from prompt_builder.models import ParameterDefinition, ParameterValue, PromptSpecification, Session
from prompt_builder.prompt_generator import PromptGenerator
from tests.mock_llm import MockLlmClient

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _spec(goal: str, depth: str = "adaptive") -> PromptSpecification:
    return PromptSpecification(goal_confidence=0.95, goal=goal, prompt_depth=depth)


def test_expanded_catalog_preserves_original_constructor_and_required_contract() -> None:
    legacy = ParameterDefinition("custom", False, "default")
    assert legacy.name == "custom" and legacy.default_value == "default"
    assert catalog.required_parameter_names() == ["audience", "domain"]
    expected = {
        "objective", "role", "input_material", "output_schema", "platform",
        "must_include", "exclusions", "security_compliance", "examples",
        "success_criteria", "factuality", "edge_cases", "self_check",
        "tech_stack", "user_roles", "data_integrations", "learner_level",
        "assessment", "aspect_ratio", "captions_audio", "export_specs",
    }
    assert expected.issubset({definition.name for definition in catalog.CATALOG})
    assert all(not catalog.get_definition(name).required for name in expected)


def test_task_type_and_adaptive_complexity_classification_is_transparent() -> None:
    assert classify_task_type("Build a production web app with an API") == "software"
    assert classify_task_type("Create a launch campaign and CTA") == "marketing"
    assert classify_task_type("Make a 30-day study plan") == "learning"
    assert classify_task_type("Edit raw footage for TikTok") == "video"
    assert classify_task_type("Summarize this report") == "summarize"
    assert classify_task_type("Rewrite this paragraph") == "rewrite"
    assert classify_complexity("Summarize this report", "summarize") == "focused"
    assert classify_complexity("Build a production app with authentication", "software") == "comprehensive"


def test_depth_override_wins_over_adaptive_classification() -> None:
    assert plan_prompt(_spec("Summarize this note", "comprehensive")).depth == "comprehensive"
    assert plan_prompt(_spec("Build a production LMS", "focused")).depth == "focused"
    remembered = PromptSpecification(
        goal_confidence=0.0,
        prompt_depth="adaptive",
        task_type="software",
        resolved_depth="comprehensive",
    )
    assert plan_prompt(remembered).depth == "comprehensive"
    assert DEPTH_OPTIONS[0] == "adaptive"


def test_relevance_profiles_cover_each_supported_task_without_cross_domain_noise() -> None:
    cases = {
        "software": ("Build an LMS web app", "tech_stack"),
        "marketing": ("Write a product launch campaign", "offer"),
        "learning": ("Create a study plan for Python", "learner_level"),
        "video": ("Turn footage into a TikTok video", "aspect_ratio"),
        "summarize": ("Summarize the attached report", "input_material"),
    }
    for task_type, (goal, expected) in cases.items():
        plan = plan_prompt(_spec(goal, "comprehensive"))
        assert plan.task_type == task_type
        assert expected in plan.relevant_fields
    summary = plan_prompt(_spec("Summarize the attached report"))
    assert "tech_stack" not in summary.relevant_fields
    assert "user_roles" not in summary.relevant_fields


def test_adaptive_questions_are_task_specific_and_explain_why_details_matter() -> None:
    software_spec = _spec("Build a production LMS web app with authentication")
    software_plan = plan_prompt(software_spec)
    software_spec.task_type = software_plan.task_type
    software_q = next_question(software_spec, Session("s"), plan=software_plan, adaptive=True)
    assert software_q is not None
    assert software_q.parameters == ["audience", "platform", "tech_stack"]
    assert "determine" in software_q.text or "changes" in software_q.text

    summary_spec = _spec("Summarize this report")
    summary_plan = plan_prompt(summary_spec)
    summary_q = next_question(summary_spec, Session("m"), plan=summary_plan, adaptive=True)
    assert summary_q is not None
    assert summary_q.parameters == ["input_material", "audience", "tone"]
    assert "tech_stack" not in summary_q.parameters


# **Validates: Requirements 3.2, 3.6, 3.8**
@settings(max_examples=100)
@given(
    collected=st.sets(st.sampled_from(["audience", "platform", "tech_stack", "feature_scope"]), max_size=4),
    declined=st.sets(st.sampled_from(["audience", "platform", "tech_stack", "feature_scope"]), max_size=4),
)
def test_adaptive_question_never_reasks_collected_or_declined_fields(collected, declined) -> None:
    spec = _spec("Build a production web app with authentication", "comprehensive")
    declined = declined - collected
    for name in collected:
        spec.parameters[name] = ParameterValue(name, f"value-{name}", "collected", 0)
    for name in declined:
        spec.parameters[name] = ParameterValue(name, "", "declined", MAX_ATTEMPTS)
    plan = plan_prompt(spec)
    question = next_question(spec, Session("property"), plan=plan, adaptive=True)
    if question is not None:
        assert 1 <= len(question.parameters) <= 3
        assert not set(question.parameters).intersection(collected | declined)


def test_coverage_distinguishes_captured_missing_declined_and_optional() -> None:
    spec = _spec("Summarize this report", "focused")
    spec.parameters["audience"] = ParameterValue("audience", "executives", "collected", 0)
    spec.parameters["tone"] = ParameterValue("tone", "", "declined", MAX_ATTEMPTS)
    spec.parameters["platform"] = ParameterValue("platform", "Slack", "collected", 0)
    items = coverage_items(spec, plan_prompt(spec))
    statuses = {item.name: item.status for item in items}
    assert statuses["objective"] == "captured"
    assert statuses["input_material"] == "missing"
    assert statuses["audience"] == "captured"
    assert statuses["tone"] == "declined"
    assert statuses["platform"] == "optional"


def test_focused_and_comprehensive_generation_have_different_structure() -> None:
    focused = _spec("Summarize the supplied report", "focused")
    focused.task_type = "summarize"
    focused.parameters["input_material"] = ParameterValue("input_material", "{{SOURCE_MATERIAL}}", "collected", 0)
    focused_result = PromptGenerator(MockLlmClient.echoing()).generate(focused)
    assert "PROMPT DEPTH: focused" not in focused_result.prompt.text  # system instruction is not leaked
    assert "# DEFINITION OF DONE" not in focused_result.prompt.text

    comprehensive = _spec("Build a production LMS with authentication", "comprehensive")
    comprehensive.task_type = "software"
    comprehensive.parameters["tech_stack"] = ParameterValue("tech_stack", "Python and PostgreSQL", "collected", 0)
    comprehensive_result = PromptGenerator(MockLlmClient.echoing()).generate(comprehensive)
    assert "# REQUIREMENTS AND CONSTRAINTS" in comprehensive_result.prompt.text
    assert "# QUALITY CHECKS" in comprehensive_result.prompt.text
    assert "# DEFINITION OF DONE" in comprehensive_result.prompt.text
    assert "# RECOMMENDED MODEL SETTINGS" in comprehensive_result.prompt.text
    assert "runtime cap only" in comprehensive_result.prompt.text


def test_generation_includes_all_collected_values_without_forcing_declined_values() -> None:
    spec = _spec("Summarize the supplied report", "focused")
    spec.task_type = "summarize"
    spec.parameters["audience"] = ParameterValue("audience", "board members", "collected", 0)
    spec.parameters["input_material"] = ParameterValue("input_material", "quarterly report", "collected", 0)
    spec.parameters["tone"] = ParameterValue("tone", "SECRET_DECLINED_VALUE", "declined", MAX_ATTEMPTS)
    result = PromptGenerator(MockLlmClient.echoing()).generate(spec)
    assert "board members" in result.prompt.text
    assert "quarterly report" in result.prompt.text
    assert "SECRET_DECLINED_VALUE" not in result.prompt.text
    assert "tone" not in result.omitted_required
    # Focused depth may stop follow-up questions early, but required-field
    # omissions remain visible to the user and SLO telemetry.
    assert result.omitted_required == ["domain"]

    early = _spec("Summarize the supplied report", "focused")
    early.task_type = "summarize"
    early_result = PromptGenerator(MockLlmClient.echoing()).generate(early)
    assert set(early_result.omitted_required) == {"audience", "domain"}


def test_prompt_depth_ui_defaults_to_adaptive() -> None:
    at = AppTest.from_file(APP_PATH).run()
    assert not at.exception
    depth = next(widget for widget in at.selectbox if widget.key == "prompt_depth")
    assert depth.value == "Adaptive"


def test_depth_aware_quality_score_does_not_force_comprehensive_sections_on_focused_prompt() -> None:
    focused_text = (
        "# OBJECTIVE\nSummarize {{SOURCE_MATERIAL}} in at most 5 bullets.\n"
        "# OUTPUT FORMAT\n- Key point\n- Decision\n"
        "# CONSTRAINTS\nDo not invent facts.\n# QUALITY CHECK\nVerify each point against the source."
    )
    focused = score_prompt_quality(focused_text, depth="focused")
    comprehensive = score_prompt_quality(focused_text, depth="comprehensive")
    assert focused.overall > comprehensive.overall
    assert focused.total == 5
    assert comprehensive.total == 9


def test_all_existing_demos_keep_metadata_and_score_as_comprehensive_prompts() -> None:
    assert len(DEMO_EXAMPLES) >= 6
    for example in DEMO_EXAMPLES:
        assert example["category"] and example["params"] and example["prompt"]
        score = score_prompt_quality(example["prompt"], depth="comprehensive")
        assert score.overall >= 75, (example["title"], score)
