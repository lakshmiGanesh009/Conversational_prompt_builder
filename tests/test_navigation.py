"""Pure tests for the professional app shell and state-preserving actions."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_builder import adaptive_planner, elicitation_engine, navigation
from prompt_builder.demo_examples import DEMO_EXAMPLES
from prompt_builder.models import GeneratedPrompt, Message, ParameterValue, PromptSpecification, Session


def _session(*, prompt: bool = False) -> Session:
    return Session(
        id="keep-this-session",
        messages=[Message("m1", "user", "Build a useful app", 1.0)],
        spec=PromptSpecification(
            0.9,
            goal="Build a useful app",
            parameters={
                "audience": ParameterValue("audience", "teams", "collected", 0)
            },
            task_type="software",
            resolved_depth="focused",
        ),
        generated_prompt=GeneratedPrompt("Ready", []) if prompt else None,
    )


def test_page_definitions_are_unique_complete_and_home_is_safe_fallback() -> None:
    assert [page.key for page in navigation.PAGES] == [
        "Home", "Builder", "Templates", "Observability", "Guide"
    ]
    assert len({page.key for page in navigation.PAGES}) == len(navigation.PAGES)
    assert all(page.label and page.icon and page.heading and page.subheading for page in navigation.PAGES)
    assert navigation.resolve_page(None).key == "Home"
    assert navigation.resolve_page("not-a-page").key == "Home"
    assert navigation.resolve_page(" builder ").key == "Builder"


# Property: every arbitrary route value resolves to one of the declared pages.
# **Validates: Approved follow-up navigable-professional-app-shell A, H**
@given(value=st.one_of(st.none(), st.integers(), st.text(max_size=80)))
@settings(max_examples=100)
def test_route_resolution_always_returns_a_declared_page(value) -> None:
    assert navigation.resolve_page(value) in navigation.PAGES


def test_navigation_preserves_session_and_unrelated_state() -> None:
    session = _session()
    state = {"session": session, "draft": "keep", "route": "Home"}

    selected = navigation.navigate(state, "Templates")

    assert selected.key == "Templates"
    assert state["route"] == "Templates"
    assert state["session"] is session
    assert state["draft"] == "keep"


def test_home_resume_state_reports_step_details_and_prompt_presence() -> None:
    summary = navigation.resume_state(_session(prompt=True))
    assert summary.available
    assert summary.current_step == "Review"
    assert summary.captured_details == 2  # goal + collected audience
    assert summary.message_count == 1
    assert summary.has_prompt
    assert not navigation.resume_state(None).available


def test_template_preview_and_clear_filters_do_not_mutate_session() -> None:
    session = _session()
    state = {
        "session": session,
        "template_search": "python",
        "template_category": "Learning",
    }
    navigation.select_template_preview(state, DEMO_EXAMPLES[1]["title"])
    navigation.clear_template_filters(state)

    assert state["session"] is session
    assert state["template_preview"] == DEMO_EXAMPLES[1]["title"]
    assert state["template_search"] == ""
    assert state["template_category"] == "All templates"


def test_skip_pending_fields_declines_all_and_they_are_not_reasked() -> None:
    spec = PromptSpecification(
        0.9,
        goal="Write a launch plan",
        prompt_depth="detailed",
        task_type="marketing",
        resolved_depth="detailed",
    )
    skipped = navigation.skip_pending_fields(spec, ["audience", "tone", "audience"])

    assert skipped == ("audience", "tone")
    assert all(spec.parameters[name].status == "declined" for name in skipped)
    assert all(spec.parameters[name].attempts == elicitation_engine.MAX_ATTEMPTS for name in skipped)

    plan = adaptive_planner.plan_prompt(spec)
    next_question = elicitation_engine.next_question(spec, Session("s"), plan=plan, adaptive=True)
    assert next_question is not None
    assert not set(skipped).intersection(next_question.parameters)


def test_generate_now_is_available_only_for_active_unfinished_session() -> None:
    assert navigation.can_generate_now(_session())
    assert not navigation.can_generate_now(_session(prompt=True))
    assert not navigation.can_generate_now(None)
