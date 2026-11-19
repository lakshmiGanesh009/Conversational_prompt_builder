"""AppTest coverage for the five-page Streamlit product shell."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from prompt_builder import elicitation_engine
from prompt_builder.models import Message, ParameterValue, PromptSpecification, Session
from tests.mock_llm import MockLlmClient

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
CLIENT_SYMBOL = "prompt_builder.llm_client.OpenAiLlmClient"


def _active_session() -> Session:
    return Session(
        id="active-shell-session",
        messages=[Message("m1", "user", "Write a concise launch plan", 1.0)],
        spec=PromptSpecification(
            0.95,
            goal="Write a concise launch plan",
            prompt_depth="focused",
            task_type="general",
            resolved_depth="focused",
        ),
    )


def _button(at: AppTest, key: str):
    return next(item for item in at.button if item.key == key)


def _markdown(at: AppTest) -> str:
    return " ".join(item.value for item in at.markdown)


def test_home_has_primary_secondary_featured_and_direct_idea_actions() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not at.exception
    assert _button(at, "home_start").label == "Start building a prompt"
    assert _button(at, "home_templates").label == "Browse templates"
    assert any(button.key and button.key.startswith("demo_home_use_") for button in at.button)
    assert len(at.chat_input) == 1
    assert "Idea" in _markdown(at) and "Details" in _markdown(at) and "Generate" in _markdown(at)


def test_home_resume_preserves_and_reports_current_session() -> None:
    session = _active_session()
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["session"] = session
    at.session_state["view"] = "Home"
    at.run()

    assert not at.exception
    assert "Resume current prompt" in _markdown(at)
    _button(at, "home_resume").click().run()
    assert at.session_state["view"] == "Builder"
    assert at.session_state["session"].id == session.id


def test_templates_filter_preview_clear_and_use_template() -> None:
    session = _active_session()
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["session"] = session
    at.session_state["view"] = "Templates"
    at.run()

    search = at.text_input(key="template_search")
    search.set_value("python").run()
    assert any("1 of 6 templates shown" in caption.value for caption in at.caption)

    _button(at, "template_preview_1").click().run()
    preview_markup = _markdown(at)
    assert "30-day Python study plan" in preview_markup
    assert "cpb-preview-excerpt" in preview_markup
    assert "cpb-prompt-h" in preview_markup
    assert at.session_state["session"].id == session.id
    assert at.session_state["session"].generated_prompt is None

    _button(at, "template_clear_filters").click().run()
    assert at.session_state["template_search"] == ""
    assert at.session_state["template_category"] == "All templates"

    with patch(CLIENT_SYMBOL, return_value=MockLlmClient.with_responses()):
        _button(at, "demo_template_use_0").click().run()
    assert not at.exception
    assert at.session_state["view"] == "Builder"
    assert at.session_state["demo_mode"] is True
    assert at.session_state["session"].generated_prompt is not None


def test_generate_now_uses_existing_early_generation_and_reports_omissions() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["session"] = _active_session()
    at.session_state["view"] = "Builder"
    client = MockLlmClient(default="# OUTPUT\nA useful launch plan.")
    with patch(CLIENT_SYMBOL, return_value=client):
        at.run()
        _button(at, "builder_generate_now").click().run()

    assert not at.exception
    prompt = at.session_state["session"].generated_prompt
    assert prompt is not None
    assert prompt.omitted_required
    assert "Required details not included" in " ".join(info.value for info in at.info)


def test_generate_now_missing_key_keeps_session_and_shows_safe_retry_error() -> None:
    session = _active_session()
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["session"] = session
    at.session_state["view"] = "Builder"
    with patch("prompt_builder.llm_client._read_config", return_value=None):
        at.run()
        _button(at, "builder_generate_now").click().run()

    assert not at.exception
    assert at.session_state["session"].id == session.id
    assert at.session_state["session"].generated_prompt is None
    assert any("Prompt generation failed" in error.value for error in at.error)
    assert _button(at, "retry_button") is not None


def test_skip_pending_question_declines_fields_and_never_reasks_them() -> None:
    session = _active_session()
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["session"] = session
    at.session_state["view"] = "Builder"
    at.session_state["pending_question"] = ["audience"]
    with patch(CLIENT_SYMBOL, return_value=MockLlmClient(default="# OUTPUT\nDone")):
        at.run()
        _button(at, "builder_skip_question").click().run()

    value = at.session_state["session"].spec.parameters["audience"]
    assert value.status == "declined"
    assert value.attempts == elicitation_engine.MAX_ATTEMPTS
    assert "audience" not in at.session_state.get("pending_question", [])


def test_observability_and_guide_render_under_global_navigation() -> None:
    for route, expected in (("Observability", "How the objectives are defined"), ("Guide", "What makes a prompt strong")):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.session_state["view"] = route
        at.run()
        assert not at.exception
        assert at.radio(key="view").value == route
        assert len(at.chat_input) == 0
        assert expected in (" ".join(expander.label for expander in at.expander) + _markdown(at))


def test_shell_css_keeps_mobile_navigation_and_accessibility_guards() -> None:
    import app

    css = app._SHELL_CSS
    assert "@media (max-width: 700px)" in css
    assert "max-width: min(88vw, 22rem)" in css
    assert "prefers-reduced-motion: reduce" in css
    assert "min-height: 2.75rem" in css
    assert "--cpb-surface-elevated" in css
    assert "--cpb-danger" in css
    assert "--cpb-focus" in css
    assert "label:has(input:checked)" in css
    assert "cpb-session-utility" in css
    assert "grid-template-columns: 1fr 1fr" in css
    assert "overflow-x: hidden" in app._NAVY_CSS
    assert "material-symbols" in app._SAAS_CSS


def test_visual_tokens_have_one_canonical_definition_and_no_legacy_palette() -> None:
    """Named layers stay compatible while one token source controls the active palette."""
    import app

    all_css = app._CUSTOM_CSS + app._SAAS_CSS + app._NAVY_CSS + app._SHELL_CSS
    for token in (
        "--cpb-canvas:",
        "--cpb-surface:",
        "--cpb-surface-elevated:",
        "--cpb-border:",
        "--cpb-text:",
        "--cpb-muted:",
        "--cpb-primary:",
        "--cpb-cyan:",
        "--cpb-success:",
        "--cpb-warning:",
        "--cpb-danger:",
        "--cpb-focus:",
        "--cpb-shadow-sm:",
        "--cpb-radius-md:",
        "--cpb-space-1:",
    ):
        assert all_css.count(token) == 1

    lowered = all_css.lower()
    for legacy in ("#7c3aed", "#8b5cf6", "#a855f7", "#ec4899", "#db2777", "#f6f2ff", "#fdf2f8"):
        assert legacy not in lowered
