"""Unit tests for the pure rendering/helper functions in ``app.py``.

These cover two recently added, otherwise-untested code paths:

- :func:`app._render_prompt_html` - converts a prompt's light markdown-ish
  structure (``# HEADINGS``, ``**bold**``, `` `code` ``, ``>`` quotes, ordered
  and unordered lists) into a styled, wrapping HTML block, HTML-escaping all raw
  text first so untrusted prompt content cannot inject markup.
- :func:`app._current_settings` - builds a
  :class:`~prompt_builder.prompt_generator.GenerationSettings` from the sidebar
  "Model settings" widget values held in ``st.session_state`` (falling back to
  the module defaults, coercing ``max_tokens`` to ``int``, and splitting the
  comma-separated stop string into a list or ``None``).

Both functions are pure and importable without a browser, server, or API key, so
these run as plain unit tests. ``app._current_settings`` reads
``st.session_state``, which works in Streamlit's "bare mode" (no running script):
missing keys fall back to the defaults, and we can seed values by assigning into
``st.session_state`` directly.
"""

from __future__ import annotations

import streamlit as st

import app


# --- _render_prompt_html ----------------------------------------------------
def test_render_prompt_html_structures_headings_lists_and_quotes() -> None:
    """Headings, both list types, bold, inline code, and quotes become HTML."""
    text = (
        "# ROLE\n"
        "You are an **expert**.\n\n"
        "- one\n"
        "- two\n\n"
        "1. first\n"
        "2. second\n\n"
        "> a quote line\n\n"
        "Use `code` inline."
    )
    out = app._render_prompt_html(text)

    # Wrapping container with the styling class.
    assert out.startswith("<div class='cpb-prompt'>")
    assert out.endswith("</div>")
    # Heading rendered as the styled heading element (not a literal '#').
    assert "cpb-prompt-h'>ROLE<" in out
    assert "# ROLE" not in out
    # Inline markers.
    assert "<strong>expert</strong>" in out
    assert "<code>code</code>" in out
    # Unordered and ordered lists each present with their items.
    assert "<ul class='cpb-prompt-list'>" in out
    assert "<ol class='cpb-prompt-list'>" in out
    assert "<li>one</li>" in out
    assert "<li>first</li>" in out
    # Blockquote.
    assert "cpb-prompt-quote'>a quote line<" in out


def test_render_prompt_html_escapes_untrusted_markup() -> None:
    """Raw angle brackets / scripts are escaped, not emitted as live markup."""
    text = "Say <script>alert('x')</script> & <b>hi</b>"
    out = app._render_prompt_html(text)

    # The dangerous markup is escaped; no live tags leak into the output.
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "&lt;b&gt;" in out
    assert "&amp;" in out


def test_render_prompt_html_bold_inside_escaped_text_is_safe() -> None:
    """Our ``**bold**`` marker is applied, but surrounding HTML stays escaped."""
    text = "**Bold** and <tag> together"
    out = app._render_prompt_html(text)

    assert "<strong>Bold</strong>" in out
    assert "&lt;tag&gt;" in out


# --- _current_settings ------------------------------------------------------
def test_current_settings_returns_defaults_when_unset() -> None:
    """With no widget values set, the defaults are returned."""
    for key in (
        app.TEMPERATURE_KEY,
        app.MAX_TOKENS_KEY,
        app.TOP_P_KEY,
        app.FREQUENCY_PENALTY_KEY,
        app.PRESENCE_PENALTY_KEY,
        app.STOP_KEY,
    ):
        st.session_state.pop(key, None)

    settings = app._current_settings()

    assert settings.temperature == app._DEFAULT_TEMPERATURE
    assert settings.max_tokens == app._DEFAULT_MAX_TOKENS
    assert settings.top_p == app._DEFAULT_TOP_P
    assert settings.frequency_penalty == app._DEFAULT_FREQUENCY_PENALTY
    assert settings.presence_penalty == app._DEFAULT_PRESENCE_PENALTY
    # A blank stop string yields None (no stop sequences).
    assert settings.stop is None


def test_current_settings_reads_widget_values() -> None:
    """Set widget values flow through, with max_tokens coerced to int."""
    st.session_state[app.TEMPERATURE_KEY] = 1.2
    st.session_state[app.MAX_TOKENS_KEY] = 1024.0  # float from a slider
    st.session_state[app.TOP_P_KEY] = 0.85
    st.session_state[app.FREQUENCY_PENALTY_KEY] = 0.5
    st.session_state[app.PRESENCE_PENALTY_KEY] = 0.3
    st.session_state[app.STOP_KEY] = "###, END"

    settings = app._current_settings()

    assert settings.temperature == 1.2
    assert settings.max_tokens == 1024
    assert isinstance(settings.max_tokens, int)
    assert settings.top_p == 0.85
    assert settings.frequency_penalty == 0.5
    assert settings.presence_penalty == 0.3
    # Comma-separated stop string is split into trimmed, non-empty sequences.
    assert settings.stop == ["###", "END"]

    # Clean up so other tests / modules see a pristine session_state.
    for key in (
        app.TEMPERATURE_KEY,
        app.MAX_TOKENS_KEY,
        app.TOP_P_KEY,
        app.FREQUENCY_PENALTY_KEY,
        app.PRESENCE_PENALTY_KEY,
        app.STOP_KEY,
    ):
        st.session_state.pop(key, None)


def test_current_settings_blank_and_comma_only_stop_is_none() -> None:
    """A stop string of only blanks/commas yields None, not empty strings."""
    st.session_state[app.STOP_KEY] = "  , ,  "
    settings = app._current_settings()
    assert settings.stop is None
    st.session_state.pop(app.STOP_KEY, None)


# --- responsive SaaS CSS ----------------------------------------------------
def test_saas_css_contains_mobile_layout_and_safe_bottom_spacing() -> None:
    """The responsive shell stacks at phone width and clears the pinned input."""
    css = app._SAAS_CSS
    assert "@media (max-width: 700px)" in css
    assert "grid-template-columns: repeat(2" in css
    assert "padding: .7rem .75rem 9.5rem" in css
    assert "overflow" in app._CUSTOM_CSS or "overflow" in css


def test_frontend_css_does_not_depend_on_external_font_assets() -> None:
    """System font fallbacks work offline and icon ligature fonts are protected."""
    assert "@import url(" not in app._CUSTOM_CSS
    assert "material-symbols" in app._SAAS_CSS
