"""Streamlit entry point for the Conversational Prompt Builder.

Run with ``streamlit run app.py``.

This module is the thin UI/orchestration layer described in the design: it reads
and writes Streamlit's ``session_state``, renders the chat and the prompt panel,
and drives the deterministic ``prompt_builder`` modules plus the single LLM
client. All of the real rules (validation, conversation management, parameter
tracking, elicitation, prompt assembly, retry control) live in
``prompt_builder`` and are exercised here through their public APIs.

Importing this module performs no network access and needs no API key: the real
:class:`~prompt_builder.llm_client.OpenAiLlmClient` is constructed lazily, only
when an LLM-backed step actually runs, and any
:class:`~prompt_builder.llm_client.LlmError` is turned into an error banner with
a Retry control rather than a crash.

Tasks implemented here:

- 15.1 - render the free-text input (``st.chat_input``) and reject empty /
  whitespace / over-length submissions inline using ``validation.py`` without
  calling the LLM.
- 15.2 - render the conversation history via the ``ui.py`` helpers with
  ``st.chat_message`` and wire the per-turn flow (record -> interpret -> merge ->
  derive missing -> elicit or generate), writing the updated Session back to
  ``session_state``.
- 15.3 - the prompt panel (Generated_Prompt + omitted-required), Revise and Copy
  controls, in-progress indicator, error banners with a retry controller, the
  "prior context unavailable" note, and a new-session button.

The presentation layer (page config, theming CSS, header, sidebar, empty state,
parameter-progress tracker, and prompt panel chrome) is styled for a cleaner,
more product-like feel. None of the styling changes the behavior above; it only
reorganizes and decorates what the deterministic modules produce.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

import streamlit as st

from prompt_builder import (
    adaptive_planner,
    catalog,
    conversation_manager,
    dashboard,
    elicitation_engine,
    frontend,
    navigation,
    telemetry,
    validation,
)
from prompt_builder.intent_analyzer import IntentAnalyzer, merge_parameters, missing_parameters
from prompt_builder.llm_client import (
    LlmClient,
    LlmError,
    OpenAiLlmClient,
    provider_status_label,
)
from prompt_builder.models import (
    GeneratedPrompt,
    Message,
    ParameterValue,
    PromptSpecification,
    Session,
)
from prompt_builder.prompt_generator import GenerationSettings, PromptGenerator
from prompt_builder.retry_controller import COULD_NOT_COMPLETE_MESSAGE, RetryController
from prompt_builder.ui import order_messages_for_render, treatment_for_message
from prompt_builder.validation import ValidationRule

# --- session_state keys -----------------------------------------------------
# The single Session lives under this fixed key. Transient UI flags live under
# clearly-named keys so the turn flow and the renderer can share them.
SESSION_KEY = "session"
# A pending error from the last turn/generation, shown as a banner with Retry.
# Shape: {"stage": "turn"|"generate"|"revise", "message": str, "payload": ...}.
PENDING_ERROR_KEY = "pending_error"
# A one-shot "copied" confirmation flag, set by the Copy control.
COPY_CONFIRMED_KEY = "copy_confirmed"
# Local editing is client-session-only and never calls the LLM.
EDIT_MODE_KEY = "prompt_edit_mode"
EDIT_DRAFT_KEY = "prompt_edit_draft"
# Set when the last interpretation reported that prior context was unavailable.
CONTEXT_UNAVAILABLE_KEY = "context_unavailable"
# True while the current Session was seeded from a hardcoded demo example, so the
# prompt panel can show a "demo prompt (no AI)" ribbon.
DEMO_MODE_KEY = "demo_mode"
# The related brief fields asked in the most recent adaptive follow-up.
PENDING_QUESTION_KEY = "pending_question"
# Prompt depth is semantic output/planning depth, separate from max_tokens.
PROMPT_DEPTH_KEY = "prompt_depth"
PROMPT_DEPTH_OPTIONS = ("Adaptive", "Focused", "Detailed", "Comprehensive")

# --- Editable model settings (promptingguide.ai LLM Settings) ----------------
# The live inference settings the user can tune in the sidebar "Model settings"
# panel. These are passed into PromptGenerator.generate / .revise on every LLM
# call so the user can experiment with temperature, length, and sampling. The
# widget keys double as session_state keys (Streamlit stores widget values
# there), so the turn flow can read the current values without extra plumbing.
TEMPERATURE_KEY = "setting_temperature"
MAX_TOKENS_KEY = "setting_max_tokens"
TOP_P_KEY = "setting_top_p"
FREQUENCY_PENALTY_KEY = "setting_frequency_penalty"
PRESENCE_PENALTY_KEY = "setting_presence_penalty"
STOP_KEY = "setting_stop"

# Sensible starting points (match the generator's generation defaults so the
# panel reflects what would happen out of the box).
_DEFAULT_TEMPERATURE = 0.4
_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_TOP_P = 1.0
_DEFAULT_FREQUENCY_PENALTY = 0.2
_DEFAULT_PRESENCE_PENALTY = 0.0
_DEFAULT_STOP = ""
# The text of the most recently rejected submission (Req 1.4, 1.5). ``st.chat_input``
# clears its own value on submit, so to "retain the entered text in the input field"
# we stash the rejected text here and surface it back to the user for re-editing.
REJECTED_INPUT_KEY = "rejected_input"
# The rule-specific error for the most recently rejected submission (Req 1.4, 1.5).
# Stored so it survives the ``st.rerun()`` after handling the submission and is
# shown to the user on the next render rather than being discarded.
REJECTED_ERROR_KEY = "rejected_error"

# Top-level route state and compatibility aliases come from the navigation
# module, which is the single source of truth for page keys and labels. The
# visible radio retains its historical ``view`` key while route changes use a
# separate non-widget key, allowing buttons to navigate safely after rendering.
ROUTE_KEY = navigation.ROUTE_STATE_KEY
VIEW_KEY = "view"
HOME_VIEW = navigation.HOME_PAGE_KEY
BUILDER_VIEW = navigation.BUILDER_PAGE_KEY
TEMPLATES_VIEW = navigation.TEMPLATES_PAGE_KEY
OBSERVABILITY_VIEW = navigation.OBSERVABILITY_PAGE_KEY
GUIDE_VIEW = navigation.GUIDE_PAGE_KEY
TEMPLATE_PREVIEW_KEY = navigation.TEMPLATE_PREVIEW_KEY

PAGE_TITLE = "Conversational Prompt Builder"
PAGE_ICON = "💬"
PAGE_TAGLINE = "Turn a plain-language idea into a precise, high-quality prompt."


# --- Demo examples (no LLM) -------------------------------------------------
# The ready-made examples live in their own module because the content is large.
# Selecting one seeds a Session with a pre-written Generated_Prompt (no LLM call).
from prompt_builder.demo_examples import DEMO_EXAMPLES


# --- Theming ----------------------------------------------------------------
# Foundational component layer. Product tokens are defined once in
# ``_SHELL_CSS``; keeping the established named layers avoids breaking imports
# and tests while preventing competing token definitions.
_CUSTOM_CSS = """
<style>


/* Global type: a clean, modern sans across the whole app. Icon glyphs are
   excluded: Streamlit draws icons with a ligature font (the text "arrow_right"
   becomes an arrow), so overriding their font prints the icon's name instead. */
html, body, .stApp,
.stApp *:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]):not([class*="material-icons"]) {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

/* Tint the whole page with a restrained blue wash. */
.stApp {
    background:
        radial-gradient(1100px 520px at 12% -12%, #e4f2fa 0%, rgba(228,242,250,0) 58%),
        radial-gradient(900px 460px at 100% -6%, #e7f7fb 0%, rgba(231,247,251,0) 52%),
        var(--cpb-canvas);
    background-attachment: fixed;
}
.stApp, .stApp p, .stApp li, .stApp label { color: var(--cpb-ink); }

/* Comfortable, centered content with generous whitespace. */
.block-container {
    padding-top: 2.4rem;
    padding-bottom: 6rem;
    max-width: 940px;
}

/* Smooth scrolling and a tidy, themed scrollbar. */
* { scroll-behavior: smooth; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb {
    background: #a9c8da; border-radius: 999px; border: 2px solid transparent;
    background-clip: content-box;
}
::-webkit-scrollbar-thumb:hover { background: #7eacc4; background-clip: content-box; }

/* Code blocks use a cool-blue surface and wrap long lines. */
.stCode, [data-testid="stCode"] pre {
    background: #eaf4fb !important;
    border: 1px solid var(--cpb-border) !important;
    border-radius: 12px !important;
}
.stCode code, [data-testid="stCode"] code { color: #0b5278 !important; }
[data-testid="stCode"] pre, [data-testid="stCode"] code {
    white-space: pre-wrap !important;
    overflow-wrap: break-word !important;
    word-break: break-word !important;
}

/* --- App header ------------------------------------------------------- */
.cpb-header {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1.6rem;
    background: var(--cpb-surface);
    border: 1px solid var(--cpb-border);
    border-radius: 18px;
    box-shadow: var(--cpb-shadow);
    position: relative;
    overflow: hidden;
}
.cpb-header::before {
    content: "";
    position: absolute; inset: 0 auto 0 0;
    width: 5px;
    background: linear-gradient(180deg, var(--cpb-accent-2), var(--cpb-accent) 55%, var(--cpb-coral));
}
.cpb-header .cpb-logo {
    flex: none;
    width: 46px; height: 46px;
    display: grid; place-items: center;
    border-radius: 13px;
    font-size: 1.5rem;
    background: linear-gradient(135deg, var(--cpb-accent-2), var(--cpb-accent) 60%, var(--cpb-coral));
    box-shadow: 0 8px 18px -8px rgba(11, 99, 168, 0.7);
}
.cpb-header .cpb-head-text { display: flex; flex-direction: column; gap: 0.1rem; }
.cpb-header h1 {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
    padding: 0;
    line-height: 1.15;
    background: linear-gradient(90deg, var(--cpb-accent-2), var(--cpb-accent) 55%, var(--cpb-coral));
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    color: var(--cpb-accent);
}
.cpb-header p {
    margin: 0;
    font-size: 0.9rem;
    color: var(--cpb-muted);
}

/* --- Section label (small caps rule) ---------------------------------- */
.cpb-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--cpb-faint);
    margin: 0 0 0.6rem 0;
}

/* --- Empty / welcome state -------------------------------------------- */
.cpb-empty {
    border: 1px solid var(--cpb-border);
    border-radius: 18px;
    padding: 2rem 1.9rem;
    background:
        linear-gradient(135deg, #ffffff 0%, #eaf6fb 100%);
    color: var(--cpb-muted);
    box-shadow: var(--cpb-shadow);
}
.cpb-empty h3 {
    margin: 0 0 0.4rem 0;
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    color: var(--cpb-ink);
}
.cpb-empty p { margin: 0 0 1rem 0; font-size: 0.94rem; line-height: 1.55; }
.cpb-empty ol {
    margin: 0;
    padding-left: 0;
    list-style: none;
    counter-reset: step;
    color: var(--cpb-ink);
    font-size: 0.92rem;
}
.cpb-empty ol li {
    counter-increment: step;
    position: relative;
    padding: 0.1rem 0 0.1rem 2.1rem;
    margin-bottom: 0.55rem;
    line-height: 1.5;
}
.cpb-empty ol li::before {
    content: counter(step);
    position: absolute; left: 0; top: 0;
    width: 1.5rem; height: 1.5rem;
    display: grid; place-items: center;
    border-radius: 50%;
    font-size: 0.75rem; font-weight: 700;
    color: #fff;
    background: linear-gradient(135deg, var(--cpb-accent-2), var(--cpb-accent));
    box-shadow: 0 4px 10px -4px rgba(11, 99, 168, 0.7);
}

/* --- Example cards ---------------------------------------------------- */
.cpb-example-card {
    border: 1px solid var(--cpb-border);
    border-radius: var(--cpb-radius);
    padding: 1rem 1.05rem;
    margin-bottom: 0.55rem;
    height: 148px;                   /* equal heights keep every button aligned */
    box-sizing: border-box;
    overflow: hidden;
    background: var(--cpb-surface);
    box-shadow: var(--cpb-shadow-sm);
    transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
}
.cpb-example-card:hover {
    transform: translateY(-3px);
    border-color: var(--cpb-accent);
    box-shadow: 0 16px 34px -18px rgba(11, 99, 168, 0.55);
}
.cpb-example-title {
    font-weight: 700;
    font-size: 0.98rem;
    line-height: 1.3;
    color: var(--cpb-ink);
    margin-bottom: 0.4rem;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden;
}
.cpb-example-text {
    font-size: 0.85rem;
    color: var(--cpb-muted);
    line-height: 1.4;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
    overflow: hidden;
}
/* A small "demo" ribbon shown above the demo prompt panel. */
.cpb-demo-note {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    color: var(--cpb-accent-2);
    background: var(--cpb-accent-soft);
    border: 1px solid var(--cpb-border);
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    margin-bottom: 0.5rem;
}

/* --- Parameter rows (sidebar) ----------------------------------------- */
.cpb-chip {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    width: 100%;
    box-sizing: border-box;
    padding: 0.4rem 0.1rem;
    font-size: 0.85rem;
    color: var(--cpb-ink);
    border-bottom: 1px solid var(--cpb-border);
}
.cpb-chip:last-child { border-bottom: none; }
.cpb-chip .cpb-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--cpb-border); flex: none;
}
.cpb-chip.collected .cpb-dot { background: var(--cpb-ok); }
.cpb-chip.declined  .cpb-dot { background: var(--cpb-faint); }
.cpb-chip.missing   .cpb-dot { background: var(--cpb-pending); }
.cpb-chip .cpb-name { font-weight: 500; text-transform: capitalize; }
.cpb-chip .cpb-req  { margin-left: auto; font-size: 0.7rem; color: var(--cpb-faint); }
.cpb-chip.declined .cpb-name { color: var(--cpb-muted); text-decoration: line-through; }

/* --- Prompt panel ----------------------------------------------------- */
.cpb-panel-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--cpb-ink);
    margin: 0 0 0.6rem 0;
}

/* The reader-friendly generated-prompt card: wrapping text, comfortable
   measure and line-height, readable (non-monospace) font, no horizontal scroll. */
.cpb-prompt {
    background: var(--cpb-surface);
    border: 1px solid var(--cpb-border);
    border-left: 4px solid var(--cpb-accent);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    max-width: 72ch;                 /* comfortable reading measure */
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    font-size: 0.96rem;
    line-height: 1.7;
    color: #18324d;                  /* readable navy body text */
    overflow-wrap: break-word;       /* wrap long tokens, never h-scroll */
    word-break: break-word;
    box-shadow: 0 10px 28px -20px rgba(11, 69, 111, 0.55);
}
.cpb-prompt .cpb-prompt-h {
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--cpb-accent);
    margin: 1.15rem 0 0.4rem 0;
    padding-bottom: 0.25rem;
    border-bottom: 1px solid var(--cpb-border);
}
.cpb-prompt .cpb-prompt-h:first-child { margin-top: 0; }
.cpb-prompt .cpb-prompt-p {
    margin: 0 0 0.7rem 0;
    color: #18324d;
}
.cpb-prompt .cpb-prompt-list {
    margin: 0 0 0.8rem 0;
    padding-left: 1.3rem;
}
.cpb-prompt .cpb-prompt-list li {
    margin-bottom: 0.3rem;
    color: #24445f;
}
.cpb-prompt .cpb-prompt-quote {
    margin: 0.4rem 0 0.9rem 0;
    padding: 0.5rem 0.9rem;
    border-left: 3px solid var(--cpb-coral);
    background: var(--cpb-accent-soft);
    border-radius: 0 8px 8px 0;
    color: var(--cpb-muted);
    font-style: italic;
}
.cpb-prompt strong { color: var(--cpb-ink); font-weight: 700; }
.cpb-prompt code {
    background: var(--cpb-accent-soft);
    border: 1px solid var(--cpb-border);
    border-radius: 6px;
    padding: 0.05rem 0.35rem;
    font-size: 0.86em;
    color: #0b63a8;
    overflow-wrap: break-word;
}

/* Make the raw/copy code block wrap too, so even that view avoids h-scroll. */
.cpb-prompt-raw pre, .cpb-prompt-raw code {
    white-space: pre-wrap !important;
    word-break: break-word !important;
}

/* Chat bubbles: clear visual distinction between user and assistant turns. */
[data-testid="stChatMessage"] {
    background: var(--cpb-surface);
    border: 1px solid var(--cpb-border);
    border-radius: var(--cpb-radius);
    padding: 0.55rem 0.85rem;
    margin-bottom: 0.55rem;
    box-shadow: var(--cpb-shadow-sm);
}
/* The user's own messages get a soft cyan background to set them apart. */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #edf7fc;
    border-color: #b8dbea;
}

/* Secondary (default) buttons use a blue outline and quiet hover fill. */
.stButton > button {
    border-radius: 10px;
    border: 1px solid var(--cpb-border);
    background: var(--cpb-surface);
    color: var(--cpb-accent-2);
    font-weight: 600;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: var(--cpb-accent);
    color: var(--cpb-accent);
    background: var(--cpb-accent-soft);
}
/* Primary action buttons use the blue/cyan product accent. */
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, var(--cpb-accent-2), var(--cpb-accent) 55%, var(--cpb-coral));
    border: none;
    color: #fbfaff;
    box-shadow: 0 10px 24px -12px rgba(11, 99, 168, 0.7);
}
.stButton > button[kind="primary"]:hover {
    filter: brightness(1.06);
    color: #fbfaff;
    box-shadow: 0 12px 28px -10px rgba(7, 135, 173, 0.6);
}
/* Streamlit wraps button labels in <p>; the page-wide text colour would
   otherwise paint them navy on the primary fill. */
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span { color: #fbfaff !important; font-weight: 600; }
.stButton > button:not([kind="primary"]) p { color: var(--cpb-accent-2) !important; font-weight: 600; }
.stButton > button:focus-visible {
    outline: 3px solid var(--cpb-accent-soft) !important;
    outline-offset: 2px;
}

/* Progress bar in the accent color. */
[data-testid="stProgress"] [role="progressbar"] > div {
    background: linear-gradient(90deg, var(--cpb-accent-2), var(--cpb-accent));
}

/* Sidebar: a quiet blue panel partitioned from the main column. */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e8f2f8 0%, #f1f7fb 100%);
    border-right: 1px solid var(--cpb-border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
.cpb-sidebrand { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.1rem; }
.cpb-sidebrand-logo {
    width: 32px; height: 32px; display: grid; place-items: center;
    border-radius: 9px; font-size: 1.05rem;
    background: linear-gradient(135deg, var(--cpb-accent-2), var(--cpb-accent) 60%, var(--cpb-coral));
    box-shadow: 0 6px 14px -6px rgba(11, 99, 168, 0.7);
}
.cpb-sidebrand-name {
    font-size: 1.1rem; font-weight: 800; letter-spacing: -0.01em; color: var(--cpb-ink);
}

/* --- Form controls: inputs, text areas, chat input ------------------- */
.stTextInput input,
.stTextArea textarea,
[data-baseweb="input"] input {
    border-radius: 12px !important;
    border: 1px solid var(--cpb-border) !important;
    background: var(--cpb-surface) !important;
    color: var(--cpb-ink) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--cpb-accent) !important;
    box-shadow: 0 0 0 3px var(--cpb-accent-soft) !important;
    outline: none !important;
}
/* Top header and bottom chat bar blend into the tinted page instead of
   showing as flat bands of a different colour. */
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: transparent !important; }

/* The pinned chat input: ONE rounded, floating surface. The inner textarea
   drops its own border so there is no box-inside-a-box. */
[data-testid="stChatInput"] {
    border-radius: 16px !important;
    border: 1px solid var(--cpb-border) !important;
    background: var(--cpb-surface) !important;
    box-shadow: 0 14px 34px -18px rgba(11, 69, 111, 0.55) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--cpb-accent) !important;
    box-shadow: 0 0 0 3px var(--cpb-accent-soft), 0 14px 34px -18px rgba(11,69,111,.55) !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"] {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, var(--cpb-accent-2), var(--cpb-accent)) !important;
    color: #fbfaff !important;
    border-radius: 10px !important;
}

/* Sidebar stats: compact numbers instead of oversized digits. */
[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.4rem !important; }

/* Info boxes in the app palette instead of Streamlit's default blue. */
[data-testid="stAlertContentInfo"] { color: var(--cpb-ink) !important; }
div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {
    background: var(--cpb-accent-soft) !important;
}

/* --- Expanders: card-like, themed header ----------------------------- */
[data-testid="stExpander"] details {
    border: 1px solid var(--cpb-border) !important;
    border-radius: 12px !important;
    background: var(--cpb-surface) !important;
    box-shadow: var(--cpb-shadow-sm);
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-weight: 600;
    color: var(--cpb-accent-2);
    padding: 0.7rem 0.9rem !important;
}
[data-testid="stExpander"] summary:hover { color: var(--cpb-accent); }

/* --- Sliders in the accent color ------------------------------------- */
[data-testid="stSlider"] [role="slider"] {
    background: var(--cpb-accent) !important;
    border-color: var(--cpb-accent) !important;
    box-shadow: 0 2px 8px -2px rgba(7,135,173,0.6) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[style*="background"] {
    background: linear-gradient(90deg, var(--cpb-accent-2), var(--cpb-accent)) !important;
}

/* --- Alerts: softer, rounded, themed --------------------------------- */
[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid var(--cpb-border);
    box-shadow: var(--cpb-shadow-sm);
}

/* --- Metrics (sidebar session stats) --------------------------------- */
[data-testid="stMetric"] {
    background: var(--cpb-surface);
    border: 1px solid var(--cpb-border);
    border-radius: 12px;
    padding: 0.6rem 0.7rem;
    box-shadow: var(--cpb-shadow-sm);
}
[data-testid="stMetricValue"] { color: var(--cpb-accent-2); font-weight: 800; }

/* --- Spinner in the accent color ------------------------------------- */
[data-testid="stSpinner"] i, .stSpinner > div > div {
    border-top-color: var(--cpb-accent) !important;
}

/* --- Dividers: faint, not harsh -------------------------------------- */
hr { border-color: var(--cpb-border) !important; opacity: 0.8; }

/* Hide Streamlit's default chrome for a cleaner, app-like feel. */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { right: 1rem; }
</style>
"""

# Layout-normalization layer retained as a named compatibility boundary. It
# references the canonical tokens declared later instead of redefining them.
_SAAS_CSS = """
<style>
.stApp {
    background: var(--cpb-canvas);
    color-scheme: light;
}
.block-container { max-width: 1120px; padding-top: 1.5rem; padding-bottom: 8rem; }
.cpb-header {
    padding: .9rem 1rem; margin-bottom: 1rem; border-radius: var(--cpb-radius);
    box-shadow: var(--cpb-shadow-sm); overflow: visible;
}
.cpb-header::before { display: none; }
.cpb-header .cpb-logo, .cpb-sidebrand-logo {
    background: var(--cpb-accent-2); box-shadow: none;
}
.cpb-header h1 { font-size: 1.25rem; color: var(--cpb-ink); background: none; -webkit-text-fill-color: initial; }
.cpb-header p { font-size: .86rem; }
.cpb-empty { padding: 1.35rem; background: var(--cpb-surface); border-radius: var(--cpb-radius); box-shadow: var(--cpb-shadow-sm); }
.cpb-empty h3 { font-size: 1.1rem; }
.cpb-empty ol li::before { background: var(--cpb-accent-2); box-shadow: none; }
.cpb-template-card {
    min-height: 158px; box-sizing: border-box; display: flex; flex-direction: column;
    border: 1px solid var(--cpb-border); border-radius: var(--cpb-radius);
    padding: 1rem; margin: .25rem 0 .55rem; background: var(--cpb-surface);
    box-shadow: var(--cpb-shadow-sm);
}
.cpb-template-category {
    align-self: flex-start; font-size: .7rem; font-weight: 700; color: var(--cpb-accent-2);
    background: var(--cpb-accent-soft); border-radius: 999px; padding: .18rem .5rem;
    margin-bottom: .65rem; letter-spacing: .02em;
}
.cpb-template-title { font-size: .96rem; font-weight: 700; color: var(--cpb-ink); margin-bottom: .35rem; }
.cpb-template-description { font-size: .82rem; line-height: 1.45; color: var(--cpb-muted); }
.cpb-template-empty { padding: 1rem; border: 1px dashed var(--cpb-border); border-radius: 10px; color: var(--cpb-muted); }
.cpb-stepper {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .5rem;
    margin: 0 0 1rem; padding: .55rem; border: 1px solid var(--cpb-border);
    border-radius: var(--cpb-radius); background: var(--cpb-surface);
}
.cpb-step {
    min-width: 0; padding: .55rem .65rem; border-radius: 9px; border: 1px solid transparent;
}
.cpb-step.active { background: var(--cpb-accent-soft); border-color: #8cc9e2; }
.cpb-step.completed { background: #ecfdf5; border-color: #a7f3d0; }
.cpb-step.upcoming { background: var(--cpb-subtle); }
.cpb-step-kicker { display: block; font-size: .68rem; font-weight: 700; color: var(--cpb-muted); }
.cpb-step-label { display: block; font-size: .84rem; font-weight: 700; color: var(--cpb-ink); white-space: nowrap; }
.cpb-step-state { display: block; font-size: .68rem; color: var(--cpb-muted); }
.cpb-quality {
    display: flex; align-items: center; gap: .65rem; justify-content: flex-end;
    padding: .45rem .65rem; border: 1px solid var(--cpb-border); border-radius: 10px;
    background: var(--cpb-surface);
}
.cpb-quality-score { font-size: 1.15rem; line-height: 1; font-weight: 800; color: var(--cpb-accent-2); }
.cpb-quality-copy { font-size: .7rem; line-height: 1.25; color: var(--cpb-muted); }
.cpb-quality-row { display: flex; gap: .6rem; align-items: flex-start; padding: .55rem 0; border-bottom: 1px solid var(--cpb-border); }
.cpb-quality-row:last-child { border-bottom: none; }
.cpb-quality-mark { flex: none; width: 1.4rem; height: 1.4rem; display: grid; place-items: center; border-radius: 50%; font-weight: 800; }
.cpb-quality-mark.pass { color: #065f46; background: #d1fae5; }
.cpb-quality-mark.warn { color: #92400e; background: #fef3c7; }
.cpb-quality-label { font-size: .84rem; font-weight: 700; color: var(--cpb-ink); }
.cpb-quality-hint { font-size: .76rem; color: var(--cpb-muted); line-height: 1.4; }
.cpb-prompt { max-width: none; border-left: 3px solid var(--cpb-accent-2); box-shadow: var(--cpb-shadow-sm); }
[data-testid="stChatMessage"], [data-testid="stExpander"] details, [data-testid="stMetric"] { box-shadow: var(--cpb-shadow-sm); }
[data-testid="stSidebar"] { background: #eaf2f8; }
.stButton > button, .stDownloadButton > button { min-height: 2.65rem; border-radius: 9px; }
.stButton > button[kind="primary"] { background: var(--cpb-accent-2); box-shadow: none; }
.stButton > button[kind="primary"]:hover { background: var(--cpb-accent); box-shadow: none; filter: none; }
button:focus-visible, input:focus-visible, textarea:focus-visible, [role="radio"]:focus-visible,
[role="combobox"]:focus-visible, a:focus-visible {
    outline: 3px solid #59c7e3 !important; outline-offset: 2px !important;
}
/* Keep Streamlit's Material Symbols ligature font intact to avoid icon names
   appearing as text when the app-wide system font is applied. */
[data-testid="stIconMaterial"], [class*="material-symbols"], [class*="material-icons"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
}
@media (max-width: 700px) {
    .block-container { padding: .7rem .75rem 9.5rem; max-width: 100%; }
    .cpb-header { align-items: flex-start; margin-bottom: .75rem; }
    .cpb-header .cpb-logo { width: 38px; height: 38px; font-size: 1.15rem; }
    .cpb-header h1 { font-size: 1.05rem; }
    .cpb-header p { font-size: .78rem; line-height: 1.4; }
    .cpb-stepper { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .35rem; padding: .4rem; }
    .cpb-step { padding: .45rem .5rem; }
    .cpb-step-label { white-space: normal; }
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .5rem; }
    [data-testid="column"] { min-width: min(100%, 18rem) !important; flex: 1 1 100% !important; width: 100% !important; }
    .cpb-template-card { min-height: 0; }
    .cpb-prompt { padding: 1rem; font-size: .9rem; line-height: 1.6; }
    [data-testid="stBottom"] { padding-bottom: env(safe-area-inset-bottom); }
    [data-testid="stChatInput"] { border-radius: 12px !important; }
    .stButton > button, .stDownloadButton > button { width: 100%; min-height: 2.8rem; }
}
</style>
"""

# Deep-navy palette application layer retained under its established name. The
# canonical palette itself is declared once in ``_SHELL_CSS``.
_NAVY_CSS = """
<style>
html, body, .stApp { overflow-x: hidden; color-scheme: light; }
::-webkit-scrollbar-thumb { background: #a9c8da; background-clip: content-box; }
::-webkit-scrollbar-thumb:hover { background: #7eacc4; background-clip: content-box; }
.stApp { background: linear-gradient(180deg, #edf6fc 0, #f7fafc 18rem, #f4f8fc 100%); }
.block-container { min-width: 0; }
.cpb-header { background: #ffffff; border-color: #bfd3e3; box-shadow: var(--cpb-shadow); }
.cpb-header .cpb-logo, .cpb-sidebrand-logo {
    background: linear-gradient(145deg, #0b2d52, #145da0 68%, #0e9fba);
}
.cpb-header h1 { color: #0b2d52; }
.cpb-empty { background: linear-gradient(135deg, #ffffff, #edf8fc); border-color: #bfd3e3; }
.cpb-template-category, .cpb-demo-note { color: #0b5278; background: #dff4fb; border-color: #b8dfed; }
.cpb-template-card:hover { border-color: #4d9dc5; box-shadow: 0 8px 22px rgba(20,93,160,.12); }
.cpb-step.active { background: #e3f3fb; border-color: #8cc9e2; }
.cpb-step.completed { background: #e7f7f2; border-color: #90d4c4; }
.cpb-step.upcoming { background: #f1f5f9; }
.cpb-prompt { border-left-color: #0e91b8; color: #18324d; }
.cpb-prompt .cpb-prompt-h { color: #0b658d; border-color: #cbd8e6; }
.cpb-prompt .cpb-prompt-p, .cpb-prompt .cpb-prompt-list li { color: #18324d; }
.cpb-prompt .cpb-prompt-quote { border-left-color: #0e9fba; background: #e8f7fb; color: #344f69; }
.cpb-prompt code, .stCode, [data-testid="stCode"] pre { background: #eaf4fb !important; border-color: #bfd3e3 !important; color: #0b5278 !important; }
.stCode code, [data-testid="stCode"] code { color: #0b5278 !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: #edf7fc; border-color: #b8dbea; }
[data-testid="stSidebar"] { background: #eaf2f8; border-right-color: #b9cddd; }
.stButton > button, .stDownloadButton > button { min-height: 2.75rem; border-color: #aac5d8; color: #0b5278; }
.stButton > button:not([kind="primary"]) p { color: #0b5278 !important; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #1686ad; background: #e3f4fa; color: #08496b; }
.stButton > button[kind="primary"] { background: #145da0; color: #ffffff; }
.stButton > button[kind="primary"]:hover { background: #0b4c86; }
button:focus-visible, input:focus-visible, textarea:focus-visible, [role="radio"]:focus-visible,
[role="combobox"]:focus-visible, a:focus-visible { outline: 3px solid #59c7e3 !important; outline-offset: 2px !important; }
[data-testid="stChatInput"] { border-color: #9fc4d9 !important; box-shadow: 0 10px 26px rgba(11,45,82,.13) !important; }
[data-testid="stChatInput"]:focus-within { border-color: #0e91b8 !important; box-shadow: 0 0 0 3px #c7edf7, 0 10px 26px rgba(11,45,82,.13) !important; }
[data-testid="stChatInput"] button { background: #145da0 !important; }
[data-testid="stSlider"] [role="slider"] { background: #087ea4 !important; border-color: #087ea4 !important; }
[data-testid="stProgress"] [role="progressbar"] > div { background: linear-gradient(90deg, #145da0, #0e9fba); }
[data-testid="stMetricValue"], .cpb-quality-score { color: #0b5278; }
.cpb-chip { align-items: flex-start; }
.cpb-chip.captured .cpb-dot { background: var(--cpb-ok); }
.cpb-chip .cpb-name { overflow-wrap: anywhere; }
.cpb-chip .cpb-req { text-align: right; white-space: nowrap; }
@media (max-width: 700px) {
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { max-width: 100vw; overflow-x: hidden; }
    .cpb-header, .cpb-stepper, .cpb-empty, .cpb-prompt, .cpb-template-card { max-width: 100%; box-sizing: border-box; }
    .cpb-quality { justify-content: flex-start; }
}
</style>
"""

# Product-shell additions: route context, landing-page cards, accessible
# navigation, and responsive controls. This is kept separate from the existing
# component skin so the established Builder and dashboard styling remains intact.
_SHELL_CSS = """
<style>
.cpb-page-context { margin: 0 0 .9rem; }
.cpb-breadcrumb { margin: 0 0 .3rem; color: var(--cpb-muted); font-size: .78rem; }
.cpb-breadcrumb strong { color: var(--cpb-ink); }
.cpb-page-heading { margin: 0; color: #0b2d52; font-size: 1.55rem; line-height: 1.2; letter-spacing: -.025em; }
.cpb-page-subheading { margin: .35rem 0 0; max-width: 72ch; color: var(--cpb-muted); font-size: .93rem; line-height: 1.55; }
.cpb-hero {
    padding: clamp(1.4rem, 4vw, 2.6rem); border: 1px solid #b8d5e6; border-radius: 18px;
    background: linear-gradient(135deg, #fff 5%, #e9f7fc 62%, #dff4fb 100%);
    box-shadow: var(--cpb-shadow); margin-bottom: 1rem;
}
.cpb-hero-badge, .cpb-status-pill {
    display: inline-block; padding: .25rem .6rem; border-radius: 999px; border: 1px solid #acd9e8;
    background: #f7fdff; color: #0b5278; font-size: .72rem; font-weight: 750; letter-spacing: .025em;
}
.cpb-hero h2 { margin: .7rem 0 .55rem; max-width: 20ch; color: #082f56; font-size: clamp(1.75rem, 4vw, 2.55rem); line-height: 1.08; letter-spacing: -.04em; }
.cpb-hero p { margin: 0; max-width: 65ch; color: #344f69; font-size: 1rem; line-height: 1.65; }
.cpb-section-title { margin: 1.6rem 0 .25rem; color: #0b2d52; font-size: 1.12rem; }
.cpb-section-copy { margin: 0 0 .8rem; color: var(--cpb-muted); font-size: .87rem; }
.cpb-card, .cpb-resume, .cpb-preview {
    height: 100%; box-sizing: border-box; padding: 1rem; border: 1px solid var(--cpb-border);
    border-radius: 12px; background: #fff; box-shadow: var(--cpb-shadow-sm); overflow-wrap: anywhere;
}
.cpb-card h3, .cpb-resume h3, .cpb-preview h3 { margin: 0 0 .35rem; color: #0b2d52; font-size: .96rem; }
.cpb-card p, .cpb-resume p, .cpb-preview p { margin: 0; color: var(--cpb-muted); font-size: .82rem; line-height: 1.5; }
.cpb-flow { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .65rem; }
.cpb-flow-step { min-width: 0; padding: .9rem; border: 1px solid var(--cpb-border); border-radius: 11px; background: #fff; }
.cpb-flow-step b { display: block; color: #0b5278; font-size: .83rem; margin-bottom: .25rem; }
.cpb-flow-step span { color: var(--cpb-muted); font-size: .76rem; line-height: 1.4; }
.cpb-action-bar {
    display: flex; align-items: center; gap: .55rem; padding: .65rem; margin: 0 0 1rem;
    border: 1px solid var(--cpb-border); border-radius: 12px; background: rgba(255,255,255,.94);
}
.cpb-action-label { margin-right: auto; color: var(--cpb-muted); font-size: .78rem; font-weight: 700; }
.cpb-preview { margin: .6rem 0 1rem; border-left: 4px solid #0e91b8; }
.cpb-preview-meta { color: #0b5278; font-size: .76rem; font-weight: 700; margin-bottom: .5rem; }
.cpb-preview-excerpt { max-height: 16rem; overflow: auto; white-space: pre-wrap; padding: .8rem; border-radius: 9px; background: #edf6fb; color: #18324d; font-size: .8rem; line-height: 1.5; }
.cpb-privacy-note { padding: .8rem 1rem; border-left: 4px solid #087ea4; background: #e8f7fb; color: #24445f; border-radius: 0 10px 10px 0; font-size: .84rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { min-height: 2.75rem; padding: .35rem .45rem; border-radius: 9px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #dcecf5; }
[data-testid="stSidebar"] [data-testid="stRadio"] label p { font-weight: 700; color: #173b5c; }
[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapsedControl"] button { min-width: 2.75rem; min-height: 2.75rem; }
.stButton > button, .stDownloadButton > button, button { min-height: 2.75rem; }
[data-testid="stBaseButton-primary"] { background: #145da0 !important; color: #fff !important; }
[data-testid="stBaseButton-primary"]:hover { background: #0b4c86 !important; color: #fff !important; }
[data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-primary"] span { color: #fff !important; }
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; animation-iteration-count: 1 !important; }
}
@media (max-width: 700px) {
    [data-testid="stHeader"] { background: #edf6fc !important; min-height: 3rem; border-bottom: 1px solid rgba(185,205,221,.65); }
    [data-testid="stMain"] .block-container { padding-top: 3.25rem !important; }
    .cpb-page-heading { font-size: 1.16rem; line-height: 1.15; }
    .cpb-page-subheading { font-size: .84rem; }
    .cpb-hero { padding: 1rem; border-radius: 14px; }
    .cpb-hero h2 { font-size: 1.45rem; line-height: 1.08; margin: .55rem 0 .4rem; }
    .cpb-hero p { font-size: .9rem; line-height: 1.5; }
    .cpb-flow { grid-template-columns: 1fr 1fr; }
    .cpb-action-bar { align-items: stretch; flex-direction: column; }
    .cpb-action-label { margin: 0; }
    .cpb-preview-excerpt { max-height: 12rem; }
    [data-testid="stSidebar"] { max-width: min(88vw, 22rem); }
}
@media (max-width: 430px) {
    .cpb-flow { grid-template-columns: 1fr; }
}

/* Canonical product design system. Earlier named layers remain for compatibility;
   these tokens and component rules are the final source of visual truth. */
:root {
    /* Canvas and surface hierarchy. */
    --cpb-canvas: #f4f8fc;
    --cpb-surface: #ffffff;
    --cpb-surface-elevated: #ffffff;
    --cpb-surface-muted: #edf5fa;
    /* Borders and readable type. */
    --cpb-border: #bfd0de;
    --cpb-border-strong: #8fadc3;
    --cpb-text: #0a2540;
    --cpb-muted: #49657f;
    --cpb-faint: #536d84;
    /* Product and semantic colors. */
    --cpb-primary: #0b63ce;
    --cpb-primary-hover: #084ea6;
    --cpb-primary-soft: #e6f0fc;
    --cpb-cyan: #087b9f;
    --cpb-cyan-soft: #e3f5fb;
    --cpb-success: #087f68;
    --cpb-success-soft: #e7f7f1;
    --cpb-warning: #8a5700;
    --cpb-warning-soft: #fff5db;
    --cpb-danger: #b42332;
    --cpb-danger-soft: #feebee;
    --cpb-focus: #007c9f;
    /* Elevation, radii, and the 8px spacing scale. */
    --cpb-shadow-xs: 0 1px 2px rgba(10, 37, 64, .08);
    --cpb-shadow-sm: 0 1px 2px rgba(10, 37, 64, .07), 0 5px 14px rgba(10, 67, 105, .05);
    --cpb-shadow-md: 0 2px 5px rgba(10, 37, 64, .07), 0 14px 28px rgba(10, 67, 105, .07);
    --cpb-radius-sm: 8px;
    --cpb-radius-md: 12px;
    --cpb-radius-lg: 16px;
    --cpb-space-1: .5rem;
    --cpb-space-2: 1rem;
    --cpb-space-3: 1.5rem;
    --cpb-space-4: 2rem;
    /* Compatibility aliases used by established component rules. */
    --cpb-ink: var(--cpb-text);
    --cpb-subtle: var(--cpb-surface-muted);
    --cpb-accent: var(--cpb-cyan);
    --cpb-accent-2: var(--cpb-primary);
    --cpb-accent-soft: var(--cpb-cyan-soft);
    --cpb-coral: var(--cpb-cyan);
    --cpb-ok: var(--cpb-success);
    --cpb-pending: var(--cpb-warning);
    --cpb-radius: var(--cpb-radius-md);
    --cpb-shadow: var(--cpb-shadow-md);
}
html { background: var(--cpb-canvas); }
body, .stApp { color: var(--cpb-ink); line-height: 1.55; }
.stApp {
    background:
        linear-gradient(180deg, #eaf4fb 0, rgba(234,244,251,.82) 10rem, rgba(244,248,252,0) 24rem),
        var(--cpb-canvas);
}
.block-container { max-width: 1120px; padding: 1.35rem 2rem 11rem; }
.stApp h1, .stApp h2, .stApp h3 { color: var(--cpb-ink); text-wrap: balance; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--cpb-muted) !important; }
hr { margin-block: 1.5rem !important; border-color: var(--cpb-border) !important; }

/* Stable controls: 44px targets, clear hover/press/focus, no layout jumps. */
.stButton > button, .stDownloadButton > button, button,
[data-baseweb="select"] > div, [data-baseweb="input"] > div {
    min-height: 2.75rem;
}
.stButton > button, .stDownloadButton > button {
    border-radius: var(--cpb-radius-sm);
    border-color: var(--cpb-border-strong);
    box-shadow: none;
    transition: background-color .14s ease, border-color .14s ease, color .14s ease, box-shadow .14s ease, transform .08s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--cpb-cyan-soft);
    border-color: var(--cpb-cyan);
    color: #07577a;
    box-shadow: var(--cpb-shadow-xs);
}
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(1px); box-shadow: none; }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background: var(--cpb-primary) !important;
    border-color: var(--cpb-primary) !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    background: var(--cpb-primary-hover) !important;
    border-color: var(--cpb-primary-hover) !important;
}
button:focus-visible, input:focus-visible, textarea:focus-visible, summary:focus-visible,
[role="radio"]:focus-visible, [role="tab"]:focus-visible, [role="combobox"]:focus-visible,
a:focus-visible {
    outline: 3px solid var(--cpb-focus) !important;
    outline-offset: 2px !important;
}
[data-testid="stExpander"] details { border-color: var(--cpb-border) !important; border-radius: var(--cpb-radius-md) !important; }
[data-testid="stExpander"] summary { min-height: 2.75rem; align-items: center; }
[data-testid="stCode"] button { min-width: 2.75rem; min-height: 2.75rem; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-baseweb="select"] > div {
    border-color: var(--cpb-border) !important;
    border-radius: var(--cpb-radius-sm) !important;
}
[data-testid="stTextInput"] input:hover, [data-testid="stTextArea"] textarea:hover, [data-baseweb="select"] > div:hover {
    border-color: var(--cpb-border-strong) !important;
}

/* App shell and navigation. */
[data-testid="stSidebar"] { background: #e9f2f8; border-right: 1px solid var(--cpb-border); }
[data-testid="stSidebar"] .block-container { padding: 1.35rem 1rem 2rem; }
.cpb-sidebrand { margin-bottom: .2rem; }
.cpb-sidebrand-logo { border-radius: 10px; box-shadow: var(--cpb-shadow-xs); }
[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: .25rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    position: relative; min-height: 2.75rem; padding: .45rem .65rem; border: 1px solid transparent;
    border-radius: var(--cpb-radius-sm); transition: background-color .14s ease, border-color .14s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #dbeaf4; border-color: #bdd2e0; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #d8ebf8; border-color: #82b9d5; box-shadow: inset 3px 0 0 var(--cpb-primary);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p { color: #073c67; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:focus-visible) {
    outline: 3px solid var(--cpb-focus); outline-offset: 2px;
}
.cpb-session-utility {
    display: flex; align-items: center; gap: .65rem; margin: .75rem 0 1rem; padding: .65rem .7rem;
    border: 1px solid #b8cedd; border-radius: var(--cpb-radius-sm); background: rgba(255,255,255,.64);
}
.cpb-session-dot { width: .55rem; height: .55rem; flex: none; border-radius: 50%; background: var(--cpb-success); box-shadow: 0 0 0 3px var(--cpb-success-soft); }
.cpb-session-utility b, .cpb-session-utility small { display: block; }
.cpb-session-utility b { color: #173b57; font-size: .76rem; }
.cpb-session-utility small { color: var(--cpb-muted); font-size: .7rem; }

/* Page hierarchy. */
.cpb-page-context { margin: 0 0 1.15rem; }
.cpb-breadcrumb { margin-bottom: .4rem; font-size: .76rem; color: #567088; }
.cpb-page-heading { color: #082f56; font-size: clamp(1.65rem, 3vw, 2.25rem); letter-spacing: -.035em; }
.cpb-page-subheading { margin-top: .4rem; font-size: .94rem; color: #3f5c75; }
.cpb-section-title { margin-top: 2rem; padding-top: .25rem; font-size: 1.22rem; }
.cpb-section-copy { max-width: 78ch; margin-bottom: .9rem; line-height: 1.55; }

/* Home. */
.cpb-hero {
    position: relative; overflow: hidden; padding: clamp(1.5rem, 4vw, 2.75rem);
    border-color: #a9cfe3; border-radius: var(--cpb-radius-lg);
    background: linear-gradient(132deg, #fff 0 58%, #e5f5fb 100%); box-shadow: var(--cpb-shadow-md);
}
.cpb-hero::after {
    content: ""; position: absolute; right: -4rem; bottom: -6rem; width: 15rem; height: 15rem;
    border: 1.5rem solid rgba(7,135,173,.07); border-radius: 50%; pointer-events: none;
}
.cpb-hero-badge, .cpb-status-pill, .cpb-empty-badge {
    display: inline-flex; align-items: center; border: 1px solid #91cade; border-radius: 999px;
    background: #f7fdff; color: #07577a; font-size: .69rem; font-weight: 800; letter-spacing: .055em;
    padding: .25rem .6rem;
}
.cpb-hero h2 { position: relative; max-width: 18ch; color: #062f56; }
.cpb-hero p { position: relative; max-width: 60ch; }
.cpb-hero-meta { position: relative; display: flex; flex-wrap: wrap; gap: .45rem .75rem; margin-top: 1rem; }
.cpb-hero-meta span { padding: .25rem .55rem; border-radius: 999px; background: rgba(255,255,255,.75); color: #28526d; font-size: .74rem; font-weight: 700; }
.cpb-flow { gap: .75rem; }
.cpb-flow-step { position: relative; min-height: 7.2rem; padding: 1rem; border-color: var(--cpb-border); box-shadow: var(--cpb-shadow-xs); }
.cpb-flow-step b { color: #07577a; }
.cpb-card, .cpb-resume { border-color: var(--cpb-border); box-shadow: var(--cpb-shadow-sm); }
.cpb-card { min-height: 7.5rem; transition: border-color .14s ease, box-shadow .14s ease, transform .14s ease; }
.cpb-card:hover { border-color: #8ebbd1; box-shadow: var(--cpb-shadow-md); transform: translateY(-2px); }
.cpb-privacy-note { margin-top: 1rem; border-left-color: var(--cpb-cyan); background: var(--cpb-cyan-soft); }

/* Builder actions, progress and conversation. */
.cpb-action-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; margin: .2rem 0 .5rem; }
.cpb-action-heading span { color: #315b79; font-size: .73rem; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; }
.cpb-action-heading small { color: var(--cpb-faint); font-size: .72rem; }
div[data-testid="stHorizontalBlock"]:has(.st-key-builder_home) {
    gap: .55rem; padding: .6rem; margin-bottom: .85rem; border: 1px solid var(--cpb-border);
    border-radius: var(--cpb-radius-md); background: rgba(255,255,255,.72); box-shadow: var(--cpb-shadow-xs);
}
.st-key-builder_new_prompt button, .st-key-new_session_button button { color: var(--cpb-danger) !important; border-color: #dea7ad !important; }
.st-key-builder_new_prompt button p, .st-key-new_session_button button p { color: var(--cpb-danger) !important; }
.st-key-builder_new_prompt button:hover, .st-key-new_session_button button:hover { background: var(--cpb-danger-soft) !important; border-color: var(--cpb-danger) !important; }
.cpb-pending-hint { display: flex; align-items: center; gap: .7rem; margin: -.25rem 0 .5rem; padding: .6rem .75rem; border-radius: var(--cpb-radius-sm); background: var(--cpb-warning-soft); border: 1px solid #e6c87e; }
.cpb-pending-hint b { flex: none; color: #754900; font-size: .75rem; }
.cpb-pending-hint span { color: #6b5733; font-size: .75rem; }
.st-key-builder_skip_question button { color: #754900 !important; border-color: #d8b860 !important; background: #fffbef !important; }
.cpb-stepper { position: relative; gap: 1rem; padding: .65rem; border-color: var(--cpb-border); box-shadow: var(--cpb-shadow-xs); }
.cpb-step { position: relative; min-height: 4.35rem; padding: .65rem .75rem; border-color: transparent; }
.cpb-step:not(:last-child)::after { content: ""; position: absolute; z-index: 2; top: 50%; right: -1rem; width: 1rem; height: 2px; background: var(--cpb-border-strong); }
.cpb-step.active { background: #e4f3fb; border-color: #75bddb; box-shadow: inset 0 3px 0 var(--cpb-cyan); }
.cpb-step.completed { background: var(--cpb-success-soft); border-color: #8dd2c0; }
.cpb-step.upcoming { background: #f0f4f8; border-color: #e1e9ef; }
.cpb-step-kicker { margin-bottom: .15rem; color: #486a82; }
.cpb-step-state { margin-top: .1rem; }
.cpb-empty { padding: 1.5rem; border-color: #a9cfe3; border-radius: var(--cpb-radius-lg); background: linear-gradient(135deg, #fff, #eef8fc); }
.cpb-empty-badge { margin-bottom: .7rem; }
[data-testid="stChatMessage"] { padding: .75rem .9rem; margin-bottom: .75rem; border-color: var(--cpb-border); box-shadow: var(--cpb-shadow-sm); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { margin-left: min(6%, 3rem); background: #eaf5fb; border-color: #a9cede; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { margin-right: min(6%, 3rem); }
.cpb-message-label { display: block; margin-bottom: .25rem; color: #42667f; font-size: .68rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }
[data-testid="stChatInput"] { border-color: #83b5d0 !important; border-radius: var(--cpb-radius-md) !important; box-shadow: 0 8px 26px rgba(10, 67, 105, .16) !important; }
[data-testid="stChatInput"] button { min-width: 2.75rem; min-height: 2.75rem; background: var(--cpb-primary) !important; }

/* Template discovery and preview. */
.cpb-filter-intro { display: flex; align-items: baseline; gap: .65rem; margin: .2rem 0 .55rem; padding-bottom: .5rem; border-bottom: 1px solid var(--cpb-border); }
.cpb-filter-intro b { color: #173f5e; font-size: .84rem; }
.cpb-filter-intro span { color: var(--cpb-muted); font-size: .76rem; }
div[data-testid="stHorizontalBlock"]:has(.st-key-template_clear_filters) { align-items: end; padding: .75rem; border: 1px solid var(--cpb-border); border-radius: var(--cpb-radius-md); background: rgba(255,255,255,.68); }
.cpb-template-card { min-height: 10.4rem; margin-top: .45rem; padding: 1.05rem; border-color: var(--cpb-border); box-shadow: var(--cpb-shadow-sm); transition: border-color .14s ease, box-shadow .14s ease, transform .14s ease; }
.cpb-template-card:hover { transform: translateY(-2px); border-color: #77b5d1; box-shadow: var(--cpb-shadow-md); }
.cpb-template-category { margin-bottom: .75rem; color: #07577a; background: var(--cpb-cyan-soft); border: 1px solid #b6deeb; }
.cpb-template-title { font-size: 1rem; }
.cpb-template-description { max-width: 54ch; line-height: 1.5; }
.cpb-preview { padding: 1.1rem; border-left-color: var(--cpb-cyan); box-shadow: var(--cpb-shadow-sm); }
.cpb-preview-excerpt { max-height: 18rem; padding: 1rem; border: 1px solid #d4e3ed; background: #f0f7fb; line-height: 1.6; }
.cpb-preview-excerpt > .cpb-prompt {
    max-width: none; margin: 0; padding: 0; border: 0; border-radius: 0;
    background: transparent; box-shadow: none; font-size: .82rem; line-height: 1.55;
}
.cpb-preview-excerpt .cpb-prompt-h:first-child { margin-top: 0; }
.cpb-template-empty { padding: 1.25rem; border-color: #8fb6cc; background: #f7fbfd; text-align: center; }

/* Review workspace. */
.cpb-demo-note { padding: .25rem .6rem; border-color: #91cade; border-radius: 999px; color: #07577a; }
.cpb-panel-title { font-size: 1.15rem; font-weight: 800; color: #082f56; }
.cpb-quality { min-height: 3.3rem; border-color: #9fbed1; background: #f7fbfd; }
.cpb-quality-score { color: #07577a; }
.cpb-prompt { max-width: 82ch; margin-inline: auto; padding: 1.5rem clamp(1rem, 3vw, 2rem); border: 1px solid var(--cpb-border); border-left: 4px solid var(--cpb-cyan); border-radius: var(--cpb-radius-md); background: var(--cpb-surface-elevated); box-shadow: var(--cpb-shadow-sm); }
.cpb-prompt .cpb-prompt-h { margin-top: 1.35rem; color: #076489; }
.cpb-prompt .cpb-prompt-p, .cpb-prompt .cpb-prompt-list li { color: #183a55; }
.cpb-review-toolbar, .cpb-export-heading { display: flex; align-items: baseline; gap: .65rem; margin: 1rem 0 .45rem; }
.cpb-review-toolbar b, .cpb-export-heading b { color: #173f5e; font-size: .82rem; }
.cpb-review-toolbar span, .cpb-export-heading span { color: var(--cpb-muted); font-size: .75rem; }
div[data-testid="stHorizontalBlock"]:has(.st-key-copy_button) { padding: .6rem; border: 1px solid var(--cpb-border); border-radius: var(--cpb-radius-md); background: #f7fbfd; }
.st-key-cancel_prompt_edit button { color: var(--cpb-danger) !important; border-color: #dea7ad !important; }
.st-key-cancel_prompt_edit button p { color: var(--cpb-danger) !important; }
.cpb-quality-row { padding-block: .7rem; }
.cpb-quality-mark.pass { color: #075e50; background: var(--cpb-success-soft); }
.cpb-quality-mark.warn { color: #754900; background: var(--cpb-warning-soft); }

/* Guide. */
.cpb-guide-index { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; margin: .25rem 0 1.1rem; padding: .75rem; border: 1px solid var(--cpb-border); border-radius: var(--cpb-radius-md); background: rgba(255,255,255,.72); }
.cpb-guide-index span { color: #315b79; font-size: .72rem; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; }
.cpb-guide-index a { min-height: 2.75rem; display: inline-flex; align-items: center; padding: .35rem .65rem; border: 1px solid #a9c7d9; border-radius: 999px; color: #07577a; font-size: .76rem; font-weight: 700; text-decoration: none; transition: background-color .14s ease, border-color .14s ease, transform .08s ease; }
.cpb-guide-index a:hover { background: var(--cpb-cyan-soft); border-color: var(--cpb-cyan); }
.cpb-guide-index a:active { transform: translateY(1px); }
.cpb-guide-anchor { scroll-margin-top: 1rem; }
.cpb-guide-group { margin: 1.35rem 0 .55rem !important; color: #315b79 !important; font-size: .76rem !important; font-weight: 800 !important; letter-spacing: .09em !important; text-transform: uppercase; }

@media (prefers-reduced-motion: reduce) {
    .cpb-card:hover, .cpb-template-card:hover { transform: none; }
    .stButton > button:active, .stDownloadButton > button:active { transform: none; }
}
@media (max-width: 700px) {
    [data-testid="stHeader"] { background: #eaf4fb !important; }
    [data-testid="stMain"] .block-container { padding: 3.1rem .8rem 11.5rem !important; }
    .cpb-page-context { margin-bottom: .85rem; }
    .cpb-breadcrumb { font-size: .72rem; }
    .cpb-page-heading { font-size: 1.45rem; }
    .cpb-page-subheading { margin-top: .25rem; font-size: .84rem; line-height: 1.45; }
    .cpb-hero { padding: .9rem; }
    .cpb-hero::after { width: 8rem; height: 8rem; right: -4rem; bottom: -4rem; border-width: .8rem; }
    .cpb-hero h2 { max-width: 19ch; margin: .55rem 0 .4rem; font-size: 1.35rem; line-height: 1.08; }
    .cpb-hero p { font-size: .85rem; line-height: 1.45; }
    .cpb-hero-meta { gap: .25rem; margin-top: .7rem; }
    .cpb-hero-meta span { padding: .18rem .4rem; font-size: .66rem; }
    .cpb-flow { grid-template-columns: 1fr 1fr; }
    .cpb-flow-step { min-height: 6.6rem; }
    .cpb-action-heading { display: block; }
    .cpb-action-heading small { display: block; margin-top: .15rem; }
    div[data-testid="stHorizontalBlock"]:has(.st-key-builder_home) {
        display: grid !important; grid-template-columns: 1fr 1fr; gap: .45rem;
    }
    div[data-testid="stHorizontalBlock"]:has(.st-key-builder_home) > [data-testid="column"] {
        min-width: 0 !important; width: auto !important; flex: none !important;
    }
    .cpb-pending-hint { display: block; }
    .cpb-pending-hint span { display: block; margin-top: .2rem; }
    .cpb-stepper { grid-template-columns: 1fr 1fr; gap: .4rem; }
    .cpb-step { min-height: 4.15rem; }
    .cpb-step::after { display: none; }
    .cpb-empty { padding: 1.1rem; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { margin-inline: 0; }
    .cpb-filter-intro, .cpb-review-toolbar, .cpb-export-heading { display: block; }
    .cpb-filter-intro span, .cpb-review-toolbar span, .cpb-export-heading span { display: block; margin-top: .15rem; }
    div[data-testid="stHorizontalBlock"]:has(.st-key-template_clear_filters) { padding: .65rem; }
    .cpb-template-card { min-height: 0; }
    .cpb-preview-excerpt { max-height: 14rem; }
    .cpb-prompt { padding: 1rem; font-size: .91rem; line-height: 1.65; }
    div[data-testid="stHorizontalBlock"]:has(.st-key-copy_button) {
        display: grid !important; grid-template-columns: 1fr 1fr;
    }
    div[data-testid="stHorizontalBlock"]:has(.st-key-copy_button) > [data-testid="column"] {
        min-width: 0 !important; width: auto !important; flex: none !important;
    }
    .cpb-guide-index { align-items: stretch; }
    .cpb-guide-index span { flex: 0 0 100%; }
    .cpb-guide-index a { flex: 1 1 calc(50% - .5rem); justify-content: center; text-align: center; }
}
@media (max-width: 430px) {
    .cpb-flow { grid-template-columns: 1fr 1fr; }
    .cpb-flow-step { min-height: 6rem; padding: .75rem; }
    .cpb-flow-step span { font-size: .73rem; }
    .cpb-hero-meta span { font-size: .68rem; }
    .cpb-guide-index a { flex-basis: 100%; }
}
</style>
"""


# --- LLM client construction (lazy) -----------------------------------------
def _build_llm_client() -> LlmClient:
    """Construct the real LLM client on demand.

    Kept out of module import and out of the render path so importing ``app`` and
    rendering the page never needs an API key. Any failure constructing the
    client (for example a missing key) surfaces as
    :class:`~prompt_builder.llm_client.LlmError`, which the callers turn into an
    error banner with a Retry control.

    The client is wrapped in :class:`~prompt_builder.telemetry.InstrumentedLlmClient`
    so every agent call is measured (latency, outcome, tokens) in one place. A
    failure to even build the client (typically a missing API key) is recorded as
    a failed ``client_init`` call before the error propagates, so it counts
    against availability.
    """
    try:
        client = OpenAiLlmClient()
    except LlmError as exc:
        telemetry.record_failed_call("client_init", exc)
        raise
    return telemetry.InstrumentedLlmClient(client)


def _record_event(
    name: str,
    *,
    success: Optional[bool] = None,
    duration_s: Optional[float] = None,
) -> None:
    """Record a product-level event, never letting telemetry break the app."""
    try:
        telemetry.get_store().record_event(
            telemetry.AppEvent(
                ts=time.time(), name=name, success=success, duration_s=duration_s
            )
        )
    except Exception:  # noqa: BLE001 - observability is best-effort
        pass


# --- Session state helpers --------------------------------------------------
def _get_session() -> Optional[Session]:
    """Return the current Session from ``session_state`` (or ``None``)."""
    return st.session_state.get(SESSION_KEY)


def _set_session(session: Optional[Session]) -> None:
    """Store the current Session back into ``session_state``."""
    st.session_state[SESSION_KEY] = session


def _navigate(destination: str) -> None:
    """Change pages without touching the current in-memory Session."""
    navigation.navigate(st.session_state, destination, route_key=ROUTE_KEY)


def _on_navigation_change() -> None:
    """Copy the sidebar widget selection into the non-widget route state."""
    navigation.navigate(
        st.session_state,
        st.session_state.get(VIEW_KEY),
        route_key=ROUTE_KEY,
    )


def _start_new_prompt() -> None:
    """Reset only Builder-session state and navigate to a fresh Builder."""
    _set_session(conversation_manager.reset_session())
    for key in (
        PENDING_ERROR_KEY,
        CONTEXT_UNAVAILABLE_KEY,
        COPY_CONFIRMED_KEY,
        EDIT_MODE_KEY,
        EDIT_DRAFT_KEY,
        REJECTED_INPUT_KEY,
        REJECTED_ERROR_KEY,
        DEMO_MODE_KEY,
        PENDING_QUESTION_KEY,
    ):
        st.session_state.pop(key, None)
    _navigate(BUILDER_VIEW)


def _selected_depth() -> str:
    """Return the semantic prompt depth in normalized planner form."""
    return adaptive_planner.normalize_depth(st.session_state.get(PROMPT_DEPTH_KEY, "Adaptive"))


def _sync_prompt_plan(spec: PromptSpecification, request_text: Optional[str] = None) -> adaptive_planner.TaskPlan:
    """Apply the current depth choice and inferred task type to a specification."""
    spec.prompt_depth = _selected_depth()
    plan = adaptive_planner.plan_prompt(spec, request_text=request_text)
    spec.task_type = plan.task_type
    spec.resolved_depth = plan.depth
    return plan


def _current_settings() -> GenerationSettings:
    """Build a :class:`GenerationSettings` from the sidebar panel's current values.

    Reads the live widget values out of ``session_state`` (falling back to the
    defaults before the panel has been rendered) and returns them as a settings
    override passed into every generation / revision call. The stop field is a
    comma-separated string in the UI; it is split into a list of non-empty
    sequences (or left unset when blank).
    """
    stop_raw = st.session_state.get(STOP_KEY, _DEFAULT_STOP) or ""
    stop_list = [s.strip() for s in stop_raw.split(",") if s.strip()]
    return GenerationSettings(
        temperature=st.session_state.get(TEMPERATURE_KEY, _DEFAULT_TEMPERATURE),
        max_tokens=int(st.session_state.get(MAX_TOKENS_KEY, _DEFAULT_MAX_TOKENS)),
        top_p=st.session_state.get(TOP_P_KEY, _DEFAULT_TOP_P),
        frequency_penalty=st.session_state.get(
            FREQUENCY_PENALTY_KEY, _DEFAULT_FREQUENCY_PENALTY
        ),
        presence_penalty=st.session_state.get(
            PRESENCE_PENALTY_KEY, _DEFAULT_PRESENCE_PENALTY
        ),
        stop=stop_list or None,
    )


def _load_demo_example(example: dict) -> None:
    """Seed a Session from a hardcoded demo example, with NO LLM call.

    This is the "click to understand the output" path: it builds a Session as if
    the whole interview had already happened — the user's request as the first
    message, a short system acknowledgement, the collected parameters folded into
    the spec, and the pre-written :class:`~prompt_builder.models.GeneratedPrompt`
    ready in the prompt panel. Nothing here touches the network or needs an API
    key; it simply demonstrates what a finished turn looks like.

    The resulting Session flows through exactly the same renderers and the same
    Revise/Copy/new-session controls as a real session, so the demo behaves like
    the live app aside from skipping the LLM.
    """
    # Start a clean Session seeded with the user's plain-language request.
    session = conversation_manager.start_session(example["request"])

    # Build the spec from the hardcoded collected parameters (status "collected").
    spec = PromptSpecification(goal_confidence=1.0, goal=example["goal"])
    _sync_prompt_plan(spec, example["request"])
    for name, value in example["params"].items():
        spec.parameters[name] = ParameterValue(
            name=name,
            value=value,
            status="collected",
            attempts=1,
        )
    session.spec = spec

    # A short system acknowledgement so the conversation reads naturally.
    conversation_manager.add_message(
        session,
        _system_message(
            "Here's a ready-made example prompt built from your request. "
            "This one is pre-written (no AI call) so you can see the output. "
            "Type your own message anytime to build one for real."
        ),
    )

    # The hardcoded Generated_Prompt. ``omitted_required`` is empty because the
    # demo provides every required parameter.
    session.generated_prompt = GeneratedPrompt(
        text=example["prompt"],
        omitted_required=[],
    )

    # Clear any transient UI flags from a previous session and store it.
    _set_session(session)
    st.session_state.pop(PENDING_ERROR_KEY, None)
    st.session_state.pop(CONTEXT_UNAVAILABLE_KEY, None)
    st.session_state.pop(COPY_CONFIRMED_KEY, None)
    st.session_state.pop(EDIT_MODE_KEY, None)
    st.session_state.pop(EDIT_DRAFT_KEY, None)
    st.session_state.pop(REJECTED_INPUT_KEY, None)
    st.session_state.pop(REJECTED_ERROR_KEY, None)
    st.session_state.pop(PENDING_QUESTION_KEY, None)
    # Mark this Session as a demo so the prompt panel shows the "no AI" ribbon,
    # then move to Builder/Review. Route changes do not alter Session content.
    st.session_state[DEMO_MODE_KEY] = True
    _navigate(BUILDER_VIEW)
    _record_event("demo_loaded")


def _system_message(text: str) -> Message:
    """Build a ``system`` message with a unique id and submission timestamp.

    Mirrors the Conversation_Manager's own message shape so appended system
    messages (acknowledgements, follow-up questions) sort and render alongside
    user messages. Appended via ``conversation_manager.add_message`` so the same
    200-message cap applies.
    """
    return Message(
        id=str(uuid.uuid4()),
        sender="system",
        text=text,
        created_at=time.time(),
    )


# --- The per-turn flow (15.2) -----------------------------------------------
def _run_turn(session: Session) -> None:
    """Interpret the latest user message and advance the Session (Req 2.x, 3.x, 4.x).

    Calls the Intent_Analyzer with the retained messages as context (Req 7.3),
    merges extracted parameters into the spec (Req 2.2), derives the missing
    required parameters (Req 2.4), then either appends the next elicitation
    question as a system message (Req 3.1, 3.8) or invokes the Prompt_Generator
    and stores the Generated_Prompt (Req 4.1).

    On an LLM failure the Session is left as-is (the user message stays in
    history) and a pending error is recorded so the UI shows an error banner with
    a Retry control (Req 6.4, 8.1). The spec is always present on the Session
    (``start_session`` seeds an empty one).
    """
    spec = session.spec if session.spec is not None else PromptSpecification(goal_confidence=0.0)
    session.spec = spec

    # The latest user message is the one just appended; interpret it with the
    # rest of the retained conversation as context (Req 7.3).
    latest_text = session.messages[-1].text if session.messages else ""
    context = session.messages[:-1]

    try:
        llm = _build_llm_client()
        analyzer = IntentAnalyzer(llm)
        interpretation = analyzer.interpret(latest_text, context, spec)
    except LlmError as exc:
        # Keep the submitted message in history and offer a retry (Req 6.4, 8.1).
        st.session_state[PENDING_ERROR_KEY] = {
            "stage": "turn",
            "message": "The response could not be generated.",
            "detail": str(exc),
        }
        return

    # Clear any prior error now that this turn progressed.
    st.session_state.pop(PENDING_ERROR_KEY, None)

    # Record whether prior context was unavailable so the UI can note it (Req 7.4).
    session.context_available = not interpretation.context_unavailable
    st.session_state[CONTEXT_UNAVAILABLE_KEY] = interpretation.context_unavailable

    # Fold extracted parameters into the spec; keep the identified goal (Req 2.2).
    merge_parameters(spec, interpretation.extracted)
    if interpretation.goal is not None:
        spec.goal = interpretation.goal
    spec.goal_confidence = interpretation.confidence

    # Resolve the previous adaptive question. Explicit skip responses decline all
    # requested fields. A response from which none of those fields could be
    # extracted consumes one retry attempt without storing the same free text in
    # several unrelated fields.
    pending_names = st.session_state.pop(PENDING_QUESTION_KEY, None)
    if pending_names:
        pending = elicitation_engine.FollowUpQuestion(list(pending_names), "")
        normalized_answer = latest_text.strip().casefold()
        if not normalized_answer or normalized_answer in elicitation_engine.SKIP_INDICATIONS:
            elicitation_engine.handle_answer(spec, pending, latest_text)
        else:
            captured = {item.name for item in interpretation.extracted}
            objective_satisfied = "objective" in pending_names and bool(spec.goal)
            if not (captured.intersection(pending_names) or objective_satisfied):
                elicitation_engine.handle_answer(spec, pending, latest_text, usable=False)

    _advance_after_resolution(session, request_text=spec.goal or latest_text)


def _advance_after_resolution(session: Session, *, request_text: str = "") -> None:
    """Ask the next eligible question or generate when no question remains.

    This shared transition is used after a typed answer and after the explicit
    Skip control. It retains the existing max-question and retry rules and never
    re-asks fields whose elicitation status has been declined.
    """
    spec = session.spec if session.spec is not None else PromptSpecification(goal_confidence=0.0)
    session.spec = spec
    if spec.task_type is not None:
        plan = _sync_prompt_plan(spec, request_text or spec.goal)
        if (
            adaptive_planner.should_ask_follow_up(spec, plan)
            and not elicitation_engine.question_limit_reached(session)
        ):
            question = elicitation_engine.next_question(
                spec,
                session,
                plan=plan,
                adaptive=True,
                request_text=request_text or spec.goal,
            )
            if question is not None:
                session.follow_up_count += 1
                st.session_state[PENDING_QUESTION_KEY] = list(question.parameters)
                conversation_manager.add_message(session, _system_message(question.text))
                return
    else:
        missing = missing_parameters(spec)
        if missing and not elicitation_engine.question_limit_reached(session):
            question = elicitation_engine.next_question(spec, session)
            if question is not None:
                session.follow_up_count += 1
                st.session_state[PENDING_QUESTION_KEY] = list(question.parameters)
                conversation_manager.add_message(session, _system_message(question.text))
                return

    st.session_state.pop(PENDING_QUESTION_KEY, None)
    _generate_prompt(session)


def _skip_pending_question(session: Session) -> tuple[str, ...]:
    """Decline all fields in the current question and advance safely."""
    spec = session.spec if session.spec is not None else PromptSpecification(goal_confidence=0.0)
    session.spec = spec
    skipped = navigation.skip_pending_fields(
        spec, st.session_state.get(PENDING_QUESTION_KEY)
    )
    if not skipped:
        return ()
    st.session_state.pop(PENDING_QUESTION_KEY, None)
    labels = [catalog.get_definition(name).label if catalog.get_definition(name) else name for name in skipped]
    conversation_manager.add_message(
        session,
        _system_message("Skipped for this session: " + ", ".join(label or name for label, name in zip(labels, skipped)) + "."),
    )
    _advance_after_resolution(session, request_text=spec.goal or "")
    return skipped


def _generate_prompt(session: Session) -> None:
    """Invoke the Prompt_Generator and store the Generated_Prompt (Req 4.1, 4.4).

    On an LLM failure the collected parameters are retained unchanged and a
    pending generation error is recorded so the UI offers a Retry control
    (Req 4.5, 8.2).
    """
    spec = session.spec if session.spec is not None else PromptSpecification(goal_confidence=0.0)
    if spec.task_type is not None:
        _sync_prompt_plan(spec)
    try:
        llm = _build_llm_client()
        generator = PromptGenerator(llm)
        result = generator.generate(spec, _current_settings())
    except LlmError as exc:
        st.session_state[PENDING_ERROR_KEY] = {
            "stage": "generate",
            "message": "Prompt generation failed.",
            "detail": str(exc),
        }
        return

    st.session_state.pop(PENDING_ERROR_KEY, None)
    session.generated_prompt = result.prompt
    _record_event("prompt_generated", success=not result.prompt.omitted_required)


def _retry_generate(session: Session) -> None:
    """Retry generation under the retry controller, stopping after 3 tries (Req 8.4).

    Re-executes generation up to 3 consecutive times. On success the prompt is
    stored and the error cleared; on exhaustion the
    :data:`~prompt_builder.retry_controller.COULD_NOT_COMPLETE_MESSAGE` is shown
    and the Session content is left unchanged.
    """
    spec = session.spec if session.spec is not None else PromptSpecification(goal_confidence=0.0)

    settings = _current_settings()

    def operation() -> GeneratedPrompt:
        llm = _build_llm_client()
        return PromptGenerator(llm).generate(spec, settings).prompt

    controller: RetryController[GeneratedPrompt] = RetryController(retry_on=(LlmError,))
    result = controller.run(operation)

    if result.succeeded and result.value is not None:
        session.generated_prompt = result.value
        st.session_state.pop(PENDING_ERROR_KEY, None)
        _record_event("prompt_generated", success=not result.value.omitted_required)
    else:
        st.session_state[PENDING_ERROR_KEY] = {
            "stage": "generate",
            "message": COULD_NOT_COMPLETE_MESSAGE,
            "detail": "",
        }


def _submit_revision(
    session: Session, revision_text: str, *, is_retry: bool = False
) -> None:
    """Record and apply a revision request under the retry controller (Req 5.2).

    Records the revision request in the Session (Req 5.2), then drives
    ``PromptGenerator.revise`` through the retry controller (up to 3 attempts,
    Req 8.4). On success the updated prompt replaces the current one; on failure
    the previous prompt is left unchanged (Req 5.4) and an error banner is shown.
    """
    current = session.generated_prompt
    if current is None:
        return

    spec = session.spec if session.spec is not None else PromptSpecification(goal_confidence=0.0)

    # Record the revision request in the Session history (Req 5.2).
    conversation_manager.record_revision(session, revision_text)

    settings = _current_settings()

    def operation() -> GeneratedPrompt:
        llm = _build_llm_client()
        return PromptGenerator(llm).revise(current, revision_text, spec, settings).prompt

    controller: RetryController[GeneratedPrompt] = RetryController(retry_on=(LlmError,))
    result = controller.run(operation)

    # Feed the retry-recovery SLO: count this run when the user explicitly
    # retried, or when it needed more than one attempt, or when it failed.
    if is_retry or result.attempts > 1 or not result.succeeded:
        _record_event("retry_run", success=result.succeeded)

    if result.succeeded and result.value is not None:
        session.generated_prompt = result.value
        st.session_state.pop(PENDING_ERROR_KEY, None)
    else:
        # Revision failed: leave the previous prompt unchanged (Req 5.4). Stash the
        # revision text so the error banner can offer a Retry that re-runs it.
        st.session_state[PENDING_ERROR_KEY] = {
            "stage": "revise",
            "message": "The revision could not be completed.",
            "detail": result.message or "",
            "revision_text": revision_text,
        }


# --- Rendering --------------------------------------------------------------
def _render_header(page: navigation.Page) -> None:
    """Render lightweight breadcrumb and page-specific context."""
    import html

    st.markdown(
        '<div class="cpb-page-context">'
        '<p class="cpb-breadcrumb">Prompt Builder / '
        f'<strong>{html.escape(page.label)}</strong></p>'
        f'<h1 class="cpb-page-heading">{html.escape(page.heading)}</h1>'
        f'<p class="cpb-page-subheading">{html.escape(page.subheading)}</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_stepper(session: Optional[Session]) -> None:
    """Render the accessible four-stage session progress indicator."""
    steps = frontend.derive_stepper(session)
    parts = ['<div class="cpb-stepper" aria-label="Session progress">']
    for index, step in enumerate(steps, start=1):
        marker = "✓" if step.status == "completed" else str(index)
        parts.append(
            f'<div class="cpb-step {step.status}" aria-label="{step.label}: {step.state_label}">'
            f'<span class="cpb-step-kicker">{marker} · Step {index}</span>'
            f'<span class="cpb-step-label">{step.label}</span>'
            f'<span class="cpb-step-state">{step.state_label}</span></div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_template_preview(example: dict) -> None:
    """Render a non-mutating template preview with enough prompt context."""
    import html

    st.markdown(
        '<div class="cpb-preview">'
        f'<div class="cpb-preview-meta">{html.escape(example["category"])} · Demo template · No AI call</div>'
        f'<h3>{html.escape(example["title"])}</h3>'
        f'<p>{html.escape(example["description"])}</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    preview_html = _render_prompt_html(example["prompt"])
    st.markdown(
        f'<div class="cpb-preview-excerpt" aria-label="Template prompt preview">{preview_html}</div>',
        unsafe_allow_html=True,
    )
    preview_use, preview_close = st.columns([1, 1])
    with preview_use:
        if st.button(
            "Use this template",
            key=f"demo_preview_use_{DEMO_EXAMPLES.index(example)}",
            type="primary",
            width="stretch",
            help="Seeds a demo review session without calling an AI model.",
        ):
            _load_demo_example(example)
            st.rerun()
    with preview_close:
        if st.button("Close preview", key="template_preview_close", width="stretch"):
            st.session_state.pop(TEMPLATE_PREVIEW_KEY, None)
            st.rerun()


def _clear_template_filters() -> None:
    """Clear gallery controls before Streamlit recreates their widgets."""
    navigation.clear_template_filters(st.session_state)


def _render_template_gallery(*, featured: bool = False) -> None:
    """Render featured cards or the full searchable no-LLM template library."""
    import html

    if featured:
        matches = DEMO_EXAMPLES[:3]
    else:
        st.markdown(
            '<div class="cpb-filter-intro"><b>Find a starting point</b>'
            '<span>Search by outcome or narrow the library by category.</span></div>',
            unsafe_allow_html=True,
        )
        filter_row = st.columns([3, 2, 1])
        with filter_row[0]:
            query = st.text_input(
                "Search templates",
                key=navigation.TEMPLATE_SEARCH_KEY,
                placeholder="Search by goal, topic, or format",
                help="Searches template titles, descriptions, requests, goals, and categories.",
            )
        with filter_row[1]:
            category = st.selectbox(
                "Category",
                frontend.template_categories(DEMO_EXAMPLES),
                key=navigation.TEMPLATE_CATEGORY_KEY,
            )
        with filter_row[2]:
            st.markdown('<div style="height:1.78rem"></div>', unsafe_allow_html=True)
            st.button(
                "Clear filters",
                key="template_clear_filters",
                width="stretch",
                on_click=_clear_template_filters,
                help="Show all six templates.",
            )
        matches = frontend.filter_templates(DEMO_EXAMPLES, query, category)
        st.caption(f"{len(matches)} of {len(DEMO_EXAMPLES)} templates shown · All are pre-written demos; previewing never calls the AI.")

        selected_title = st.session_state.get(TEMPLATE_PREVIEW_KEY)
        selected = next((item for item in DEMO_EXAMPLES if item["title"] == selected_title), None)
        if selected is not None:
            _render_template_preview(selected)

    if not matches:
        st.markdown(
            '<div class="cpb-template-empty"><b>No templates match.</b> '
            "Try another search, choose a different category, or clear the filters.</div>",
            unsafe_allow_html=True,
        )
        return

    for start in range(0, len(matches), 2):
        cols = st.columns(2)
        for col, example in zip(cols, matches[start : start + 2]):
            index = DEMO_EXAMPLES.index(example)
            with col:
                st.markdown(
                    '<div class="cpb-template-card">'
                    f'<span class="cpb-template-category">{html.escape(example["category"])}</span>'
                    f'<div class="cpb-template-title">{html.escape(example["title"])}</div>'
                    f'<div class="cpb-template-description">{html.escape(example["description"])}</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
                if featured:
                    if st.button(
                        "Use template",
                        key=f"demo_home_use_{index}",
                        width="stretch",
                        type="primary",
                        help="Open a pre-written demo in Builder without an AI call.",
                    ):
                        _load_demo_example(example)
                        st.rerun()
                else:
                    preview_col, use_col = st.columns(2)
                    with preview_col:
                        if st.button(
                            "Preview",
                            key=f"template_preview_{index}",
                            width="stretch",
                            help="Inspect the prompt without replacing your current session.",
                        ):
                            navigation.select_template_preview(st.session_state, example["title"])
                            st.rerun()
                    with use_col:
                        if st.button(
                            "Use template",
                            key=f"demo_template_use_{index}",
                            width="stretch",
                            type="primary",
                            help="Seed a demo session and open Builder/Review. No AI call.",
                        ):
                            _load_demo_example(example)
                            st.rerun()


def _render_empty_state() -> None:
    """Render a focused Builder first-use state with a Templates route."""
    st.markdown(
        """
        <div class="cpb-empty">
            <span class="cpb-empty-badge">YOUR FIRST STEP</span>
            <h3>Start with one clear outcome</h3>
            <p>Use the chat input below to describe what you want. The builder will
            identify useful details, ask focused follow-ups, and create a prompt you can edit and export.</p>
            <ol>
                <li>Describe the result in plain language</li>
                <li>Answer or skip focused detail questions</li>
                <li>Generate, review, edit, and export</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    if st.button("Browse templates", key="builder_browse_templates", width="stretch"):
        _navigate(TEMPLATES_VIEW)
        st.rerun()


def _render_builder_action_bar(session: Optional[Session]) -> None:
    """Render explicit Builder navigation and state-changing actions."""
    st.markdown(
        '<div class="cpb-action-heading"><span>Workspace actions</span>'
        '<small>Session-safe navigation and generation controls</small></div>',
        unsafe_allow_html=True,
    )
    actions = st.columns(4)
    with actions[0]:
        if st.button("Home", key="builder_home", width="stretch", help="Return home without clearing this session."):
            _navigate(HOME_VIEW)
            st.rerun()
    with actions[1]:
        label = "Start over" if navigation.resume_state(session).available else "New prompt"
        if st.button(label, key="builder_new_prompt", width="stretch", help="Clear this prompt session and start a new one."):
            _start_new_prompt()
            st.rerun()
    with actions[2]:
        if st.button("Templates", key="builder_templates", width="stretch", help="Browse all no-AI demo templates."):
            _navigate(TEMPLATES_VIEW)
            st.rerun()
    with actions[3]:
        if navigation.can_generate_now(session):
            if st.button(
                "Generate now",
                key="builder_generate_now",
                type="primary",
                width="stretch",
                help="Generate from captured details now. Missing required details will be reported.",
            ):
                st.session_state.pop(PENDING_QUESTION_KEY, None)
                with st.spinner("Generating from the details captured so far..."):
                    _generate_prompt(session)
                _set_session(session)
                st.rerun()
        else:
            st.caption("Generate now appears after you share an idea.")

    pending = st.session_state.get(PENDING_QUESTION_KEY)
    if session is not None and pending and session.generated_prompt is None:
        st.markdown(
            '<div class="cpb-pending-hint"><b>Question controls</b>'
            '<span>Answer in the message box, skip this question, or generate from the details already captured.</span></div>',
            unsafe_allow_html=True,
        )
        if st.button(
            "Skip this question",
            key="builder_skip_question",
            width="stretch",
            help="Marks every detail in this question as declined so it will not be asked again.",
        ):
            with st.spinner("Skipping this question and checking what comes next..."):
                _skip_pending_question(session)
            _set_session(session)
            st.rerun()


def _render_home(session: Optional[Session]) -> None:
    """Render the useful landing page and its state-preserving actions."""
    st.markdown(
        """
        <section class="cpb-hero">
          <span class="cpb-hero-badge">GUIDED PROMPT WORKSPACE</span>
          <h2>From a rough idea to a prompt you can actually use</h2>
          <p>Describe the outcome in your own words. The Builder plans a focused interview,
          captures relevant constraints, then creates a detailed prompt you can edit, revise,
          copy, or export.</p>
          <div class="cpb-hero-meta" aria-label="Workspace benefits">
            <span>✓ Adaptive questions</span><span>✓ Local editing</span><span>✓ In-memory sessions</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    primary, secondary = st.columns(2)
    with primary:
        if st.button("Start building a prompt", key="home_start", type="primary", width="stretch"):
            _start_new_prompt()
            st.rerun()
    with secondary:
        if st.button("Browse templates", key="home_templates", width="stretch"):
            _navigate(TEMPLATES_VIEW)
            st.rerun()

    resume = navigation.resume_state(session)
    if resume.available:
        prompt_status = "Prompt ready for review" if resume.has_prompt else "Prompt not generated yet"
        st.markdown(
            '<div class="cpb-resume">'
            '<span class="cpb-status-pill">CURRENT IN-MEMORY SESSION</span>'
            '<h3>Resume current prompt</h3>'
            f'<p>Current step: <b>{resume.current_step}</b> · {resume.captured_details} details captured · '
            f'{resume.message_count} messages · {prompt_status}.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Resume current prompt", key="home_resume", type="primary", width="stretch"):
            _navigate(BUILDER_VIEW)
            st.rerun()

    st.markdown('<h2 class="cpb-section-title">How it works</h2>', unsafe_allow_html=True)
    st.markdown('<p class="cpb-section-copy">The same four stages stay visible while you build.</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cpb-flow">'
        '<div class="cpb-flow-step"><b>1 · Idea</b><span>State the outcome in plain language.</span></div>'
        '<div class="cpb-flow-step"><b>2 · Details</b><span>Answer relevant questions or skip what you do not know.</span></div>'
        '<div class="cpb-flow-step"><b>3 · Generate</b><span>Create now or wait until coverage is complete.</span></div>'
        '<div class="cpb-flow-step"><b>4 · Review</b><span>Edit, revise, copy, and export the result.</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<h2 class="cpb-section-title">Featured templates</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="cpb-section-copy"><b>Demo templates are pre-written and make no AI call.</b> '
        'Live generation begins only after you submit your own idea or choose Generate now.</p>',
        unsafe_allow_html=True,
    )
    _render_template_gallery(featured=True)

    st.markdown('<h2 class="cpb-section-title">Built for practical prompt work</h2>', unsafe_allow_html=True)
    capabilities = (
        ("Adaptive interview", "Question depth follows the task, so simple requests stay focused and complex work gets more coverage."),
        ("Ready-to-use output", "Generated prompts include structure, constraints, missing-information behavior, and a clear output contract."),
        ("Local editing and export", "Edit without an AI call, then export Markdown, plain text, or scrubbed JSON."),
        ("Model controls", "Tune runtime temperature, token cap, sampling, penalties, and stop sequences separately from prompt depth."),
        ("Private in-memory operation", "Sessions and metrics remain in memory. Telemetry stores timing and counts, never prompt or message text."),
    )
    for start in range(0, len(capabilities), 3):
        columns = st.columns(min(3, len(capabilities) - start))
        for column, (title, copy) in zip(columns, capabilities[start : start + 3]):
            with column:
                st.markdown(f'<div class="cpb-card"><h3>{title}</h3><p>{copy}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="cpb-privacy-note"><b>Start quickly:</b> type an idea in the message box below, or use the primary action above for a clean Builder session.</div>', unsafe_allow_html=True)


def _render_guide() -> None:
    """Render concise guidance in grouped, keyboard-accessible sections."""
    st.markdown(
        '<nav class="cpb-guide-index" aria-label="Guide sections">'
        '<span>Jump to</span>'
        '<a href="#guide-design">Prompt design</a>'
        '<a href="#guide-workspace">Using the workspace</a>'
        '<a href="#guide-recovery">Privacy & recovery</a>'
        '</nav>',
        unsafe_allow_html=True,
    )
    st.markdown('<div id="guide-design" class="cpb-guide-anchor"></div><h2 class="cpb-guide-group">Prompt design</h2>', unsafe_allow_html=True)
    with st.expander("What makes a prompt strong", expanded=True):
        st.markdown(
            "A strong prompt names the **objective**, relevant **context and inputs**, intended **audience**, "
            "important **constraints**, and the exact **output format**. Concrete limits and a definition of done "
            "reduce guesswork. If facts matter, say how the model should handle uncertainty and sources."
        )
    with st.expander("Prompt depth: Adaptive, Focused, Detailed, Comprehensive"):
        st.markdown(
            "- **Adaptive** selects depth from the task and is the recommended default.\n"
            "- **Focused** asks only for essentials and produces a concise prompt.\n"
            "- **Detailed** adds richer context, constraints, and quality checks.\n"
            "- **Comprehensive** collects the fullest brief and suits complex, reusable work."
        )
    with st.expander("Prompt depth versus runtime model settings"):
        st.markdown(
            "Prompt depth controls the *semantic detail and interview plan*. **Max tokens** is only a runtime cap "
            "on model output. Temperature and top-p affect sampling, while penalties affect repetition. "
            "Changing those controls does not automatically make the prompt itself more detailed."
        )

    st.markdown('<div id="guide-workspace" class="cpb-guide-anchor"></div><h2 class="cpb-guide-group">Using the workspace</h2>', unsafe_allow_html=True)
    with st.expander("Demo templates versus live AI generation"):
        st.markdown(
            "Templates are pre-written demonstrations. Previewing or using one makes **no model call**. "
            "A live call happens only after you submit your own message, choose **Generate now**, retry a failed call, "
            "or request an AI revision."
        )
    with st.expander("Edit, revise, copy, and export"):
        st.markdown(
            "Use **Edit locally** for changes that should not call AI. Use **Revise with AI** for a model-assisted rewrite. "
            "The raw prompt code block provides the browser's copy control. The server cannot verify whether a browser "
            "clipboard write succeeded. Exports are available as Markdown, plain text, and JSON."
        )

    st.markdown('<div id="guide-recovery" class="cpb-guide-anchor"></div><h2 class="cpb-guide-group">Privacy & recovery</h2>', unsafe_allow_html=True)
    with st.expander("Privacy and telemetry"):
        st.markdown(
            "Sessions and metrics are held in memory only and disappear when the process restarts. There is no account "
            "or database. Telemetry records operational timing, outcomes, token counts, and estimated cost; it never "
            "stores prompt text or message text. Export metadata is scrubbed of credential-labelled fields."
        )
    with st.expander("Troubleshooting API keys and failed calls"):
        st.markdown(
            "If no API key is configured, live actions show a safe error and keep your Session. Add the configured "
            "OpenAI-compatible API key to the environment, restart Streamlit, and choose **Retry**. For provider or "
            "network failures, wait briefly and retry. After repeated failures, your captured details and prompt remain available."
        )


def _render_history(session: Session) -> None:
    """Render the conversation oldest-to-newest with sender distinction (Req 6.1, 6.2)."""
    for message in order_messages_for_render(session.messages):
        treatment = treatment_for_message(message)
        with st.chat_message(treatment.role, avatar=treatment.avatar):
            speaker = "You" if message.sender == "user" else "Prompt Builder"
            st.markdown(
                f'<span class="cpb-message-label">{speaker}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(message.text)


def _render_context_note() -> None:
    """Show the "prior context unavailable" note when the last turn set it (Req 7.4)."""
    if st.session_state.get(CONTEXT_UNAVAILABLE_KEY):
        st.warning("Prior context was unavailable, so this message was interpreted on its own.")


def _render_retained_input() -> None:
    """Surface the text of a rejected submission so it isn't lost (Req 1.4, 1.5).

    ``st.chat_input`` clears its own value when a message is submitted, so when
    validation rejects a submission the entered text would otherwise disappear.
    The submission handler stashes the rejected text under
    :data:`REJECTED_INPUT_KEY`; here we render it back in an editable field so the
    user can fix and resubmit it without retyping. Editing or resubmitting the
    retained text clears it.
    """
    retained = st.session_state.get(REJECTED_INPUT_KEY)
    if not retained:
        return
    # The rule-specific error for the rejected submission (Req 1.4, 1.5).
    error_message = st.session_state.get(REJECTED_ERROR_KEY)
    if error_message:
        st.error(error_message)
    st.text_area(
        "Your message (not sent — please fix and resubmit)",
        value=retained,
        key="retained_input",
    )


def _render_error_banner(session: Session) -> None:
    """Show the pending error banner with a Retry control (Req 4.5, 6.4, 8.1-8.4)."""
    error = st.session_state.get(PENDING_ERROR_KEY)
    if not error:
        return

    st.error(error.get("message", "Something went wrong."))

    stage = error.get("stage")
    # The "could not be completed" terminal message has no further retry.
    if error.get("message") == COULD_NOT_COMPLETE_MESSAGE:
        return

    if stage in ("turn", "generate"):
        if st.button("Retry", key="retry_button"):
            with st.spinner("Retrying..."):
                _retry_generate(session) if stage == "generate" else _run_turn(session)
            # The retry recovered if no error is pending afterwards.
            _record_event(
                "retry_run", success=st.session_state.get(PENDING_ERROR_KEY) is None
            )
            _set_session(session)
            st.rerun()
    elif stage == "revise":
        # Re-run the revision the user asked for, using the stashed text.
        revision_text = error.get("revision_text", "")
        if revision_text and st.button("Retry", key="retry_button"):
            with st.spinner("Retrying the revision..."):
                _submit_revision(session, revision_text, is_retry=True)
            _set_session(session)
            st.rerun()


def _render_prompt_html(text: str) -> str:
    """Render prompt text as a styled, wrapping HTML block (accessible, no h-scroll).

    The generated prompts use a light markdown-ish structure (``# HEADINGS``,
    ``**bold**``, ``>`` quotes, ``-`` / numbered bullets). ``st.code`` would show
    this in a non-wrapping monospace block that overflows horizontally on long
    lines. Instead we convert the structure into clean, semantic HTML inside a
    styled container so the text wraps, keeps a comfortable reading measure, and
    uses a readable (non-monospace) font with generous line-height.

    All raw text is HTML-escaped first, then only our small, known set of inline
    markers (``**bold**`` and `` `code` ``) is re-introduced, so untrusted prompt
    text cannot inject markup.
    """
    import html
    import re

    def inline(segment: str) -> str:
        # Escape everything, then re-apply a tiny, safe subset of inline styling.
        escaped = html.escape(segment)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"`([^`]+?)`", r"<code>\1</code>", escaped)
        return escaped

    lines = text.split("\n")
    html_parts: list[str] = []
    list_buffer: list[str] = []
    list_type: Optional[str] = None  # "ul" or "ol"

    def flush_list() -> None:
        nonlocal list_type
        if list_buffer:
            tag = list_type or "ul"
            html_parts.append(
                f"<{tag} class='cpb-prompt-list'>" + "".join(list_buffer) + f"</{tag}>"
            )
            list_buffer.clear()
            list_type = None

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_list()
            continue

        # Headings: markdown '#'/'##' style.
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_list()
            html_parts.append(
                f"<div class='cpb-prompt-h'>{inline(heading.group(2))}</div>"
            )
            continue

        # Blockquote.
        if stripped.startswith(">"):
            flush_list()
            html_parts.append(
                f"<blockquote class='cpb-prompt-quote'>{inline(stripped[1:].strip())}</blockquote>"
            )
            continue

        # Ordered list item: "1. ", "2. ", ...
        ordered = re.match(r"^(\d+)[.)]\s+(.*)$", stripped)
        if ordered:
            if list_type not in (None, "ol"):
                flush_list()
            list_type = "ol"
            list_buffer.append(f"<li>{inline(ordered.group(2))}</li>")
            continue

        # Unordered list item: "- " or "* ".
        unordered = re.match(r"^[-*]\s+(.*)$", stripped)
        if unordered:
            if list_type not in (None, "ul"):
                flush_list()
            list_type = "ul"
            list_buffer.append(f"<li>{inline(unordered.group(1))}</li>")
            continue

        # Plain paragraph line.
        flush_list()
        html_parts.append(f"<p class='cpb-prompt-p'>{inline(stripped)}</p>")

    flush_list()
    return "<div class='cpb-prompt'>" + "".join(html_parts) + "</div>"


def _save_local_prompt_edit(session: Session, text: str) -> bool:
    """Save a non-empty local edit while preserving omitted-required metadata."""
    if session.generated_prompt is None or not text.strip():
        return False
    session.generated_prompt = GeneratedPrompt(
        text=text,
        omitted_required=list(session.generated_prompt.omitted_required),
    )
    return True


def _render_quality_breakdown(score: frontend.QualityScore) -> None:
    """Render the deterministic score dimensions with non-color status labels."""
    with st.expander("Quality score breakdown", expanded=False):
        st.caption(
            "This deterministic score checks prompt structure and coverage. "
            "It does not verify facts or guarantee model output quality."
        )
        for dimension in score.dimensions:
            mark = "✓" if dimension.passed else "!"
            css = "pass" if dimension.passed else "warn"
            detail = dimension.evidence if dimension.passed else dimension.improvement_hint
            st.markdown(
                '<div class="cpb-quality-row">'
                f'<span class="cpb-quality-mark {css}" aria-hidden="true">{mark}</span>'
                '<div>'
                f'<div class="cpb-quality-label">{dimension.label} · {dimension.status_label}</div>'
                f'<div class="cpb-quality-hint">{detail}</div>'
                "</div></div>",
                unsafe_allow_html=True,
            )


def _render_prompt_panel(session: Session) -> None:
    """Render prompt review, local editing, quality, copy, export, and revision."""
    prompt = session.generated_prompt
    if prompt is None:
        return

    st.divider()
    if st.session_state.get(DEMO_MODE_KEY):
        st.markdown(
            '<span class="cpb-demo-note">TEMPLATE SOURCE · NO AI CALL</span>',
            unsafe_allow_html=True,
        )

    quality_plan = adaptive_planner.plan_prompt(session.spec) if session.spec is not None else None
    quality = frontend.score_prompt_quality(
        prompt.text,
        depth=quality_plan.depth if quality_plan is not None else None,
    )
    title_col, score_col = st.columns([3, 1])
    with title_col:
        st.markdown('<div class="cpb-panel-title">Generated prompt</div>', unsafe_allow_html=True)
        st.caption("Review the prompt, make local edits, or export it for use elsewhere.")
    with score_col:
        st.markdown(
            '<div class="cpb-quality" aria-label="Prompt quality score '
            f'{quality.overall} out of 100">'
            f'<span class="cpb-quality-score">{quality.overall}</span>'
            '<span class="cpb-quality-copy">/ 100<br>structure score</span></div>',
            unsafe_allow_html=True,
        )

    if st.session_state.get(EDIT_MODE_KEY):
        st.info("Local edit mode does not call the AI. Save replaces only this session's prompt text.")
        if EDIT_DRAFT_KEY not in st.session_state:
            st.session_state[EDIT_DRAFT_KEY] = prompt.text
        edited_text = st.text_area(
            "Edit prompt text",
            key=EDIT_DRAFT_KEY,
            height=420,
            help="Edit the complete prompt. Save changes updates the current session; Cancel discards the draft.",
        )
        save_col, cancel_col, _ = st.columns([1, 1, 2])
        with save_col:
            save_clicked = st.button("Save changes", key="save_prompt_edit", type="primary", width="stretch")
        with cancel_col:
            cancel_clicked = st.button("Cancel", key="cancel_prompt_edit", width="stretch")
        if save_clicked:
            if _save_local_prompt_edit(session, edited_text):
                _set_session(session)
                st.session_state.pop(EDIT_MODE_KEY, None)
                st.session_state.pop(EDIT_DRAFT_KEY, None)
                st.rerun()
            else:
                st.error("The prompt cannot be empty. Add prompt text or choose Cancel.")
        if cancel_clicked:
            st.session_state.pop(EDIT_MODE_KEY, None)
            st.session_state.pop(EDIT_DRAFT_KEY, None)
            st.rerun()
    else:
        st.markdown(_render_prompt_html(prompt.text), unsafe_allow_html=True)

        st.markdown(
            '<div class="cpb-review-toolbar"><b>Prompt actions</b>'
            '<span>Edit locally without an AI call, or open the browser copy view.</span></div>',
            unsafe_allow_html=True,
        )
        action_copy, action_edit = st.columns(2)
        with action_copy:
            if st.button(
                "Copy prompt",
                key="copy_button",
                width="stretch",
                type="primary",
                help="Opens the raw prompt and its browser-provided copy control.",
            ):
                st.session_state[COPY_CONFIRMED_KEY] = True
        with action_edit:
            if st.button("Edit locally", key="edit_prompt_button", width="stretch"):
                st.session_state[EDIT_MODE_KEY] = True
                st.session_state[EDIT_DRAFT_KEY] = prompt.text
                st.rerun()

        copy_requested = bool(st.session_state.get(COPY_CONFIRMED_KEY))
        if copy_requested:
            st.info(
                "The browser copy view is open below. Use its copy icon; the server cannot verify clipboard success."
            )

        with st.expander("Raw prompt and browser copy", expanded=copy_requested):
            st.caption("Use the copy icon in the code block. Clipboard access happens in your browser; this server cannot verify success.")
            st.code(prompt.text, language="markdown")

        exports = frontend.build_prompt_exports(
            prompt,
            settings=frontend.settings_metadata(_current_settings()),
            metadata={
                "source": "template" if st.session_state.get(DEMO_MODE_KEY) else "builder",
                "quality_score": quality.overall,
                "quality_dimensions_passed": quality.passed,
            },
        )
        st.markdown(
            '<div class="cpb-export-heading"><b>Export</b><span>Download a portable copy of this prompt.</span></div>',
            unsafe_allow_html=True,
        )
        export_columns = st.columns(3)
        for column, export in zip(export_columns, exports):
            with column:
                st.download_button(
                    f"Export {export.label}",
                    export.data,
                    file_name=export.file_name,
                    mime=export.mime,
                    key=f"export_{export.file_name}",
                    width="stretch",
                )

        if copy_requested:
            st.session_state[COPY_CONFIRMED_KEY] = False

    _render_quality_breakdown(quality)

    if prompt.omitted_required:
        st.info(
            "Required details not included: " + ", ".join(prompt.omitted_required)
        )

    with st.expander("Revise with AI", expanded=False):
        if st.session_state.get(DEMO_MODE_KEY):
            st.info(
                "This template preview is pre-written and uses no AI. Start with your own message "
                "to generate a prompt that can be revised with AI. You can still use Edit locally above."
            )
        else:
            with st.form("revise_form", clear_on_submit=False):
                revision_text = st.text_area(
                    "Revision request",
                    key="revision_input",
                    placeholder="Describe the change you want...",
                )
                submitted = st.form_submit_button("Revise", type="primary", width="stretch")
            if submitted:
                result = validation.validate_revision(revision_text)
                if not result.ok:
                    if result.rule == ValidationRule.EMPTY:
                        st.error("A revision request can't be empty.")
                    elif result.rule == ValidationRule.TOO_LONG:
                        st.error(
                            f"A revision request can be at most {validation.REVISION_MAX_LENGTH} characters."
                        )
                else:
                    with st.spinner("Revising the prompt..."):
                        _submit_revision(session, revision_text)
                    _set_session(session)
                    st.rerun()


def _handle_submission(raw_text: str) -> None:
    """Validate and process a submitted message (Req 1.1, 1.4, 1.5, then the turn flow).

    Rejects empty / whitespace / over-length input inline without calling the LLM
    (Req 1.4, 1.5). On a valid message, records it (start a Session on the first
    message, otherwise append) and runs the turn flow (Req 2.x, 3.x, 4.x), writing
    the updated Session back to ``session_state``.
    """
    result = validation.validate_message(raw_text)
    if not result.ok:
        if result.rule == ValidationRule.EMPTY:
            message = "Please enter a message. Empty or whitespace-only input isn't accepted."
        else:  # ValidationRule.TOO_LONG
            message = (
                f"Your message is too long. The maximum is {validation.MESSAGE_MAX_LENGTH} characters."
            )
        # Retain the entered text and the rule-specific error so they survive the
        # rerun after this handler returns and are shown to the user (Req 1.4, 1.5).
        # ``st.chat_input`` clears itself on submit, so the renderer re-surfaces
        # the text for re-editing and the error below it.
        st.session_state[REJECTED_INPUT_KEY] = raw_text
        st.session_state[REJECTED_ERROR_KEY] = message
        _record_event("validation_rejected")
        # Do NOT call the LLM for invalid input.
        return

    # Accepted: drop any retained rejected text/error from a previous attempt.
    st.session_state.pop(REJECTED_INPUT_KEY, None)
    st.session_state.pop(REJECTED_ERROR_KEY, None)
    # Submitting an idea from Home is an explicit request to enter the Builder.
    _navigate(BUILDER_VIEW)
    # A real typed message leaves demo mode (this turn goes through the LLM).
    st.session_state.pop(DEMO_MODE_KEY, None)

    session = _get_session()
    is_fresh_session = session is None or not session.messages
    if session is None:
        session = conversation_manager.start_session(raw_text)
    else:
        session = conversation_manager.add_message(session, _user_message(raw_text))
    if is_fresh_session and session.spec is not None:
        _sync_prompt_plan(session.spec, raw_text)

    turn_started = time.perf_counter()
    with st.spinner("Thinking..."):
        _run_turn(session)
    # A turn succeeded if it finished without leaving an error banner pending.
    _record_event(
        "turn",
        success=st.session_state.get(PENDING_ERROR_KEY) is None,
        duration_s=time.perf_counter() - turn_started,
    )
    _set_session(session)


def _user_message(text: str) -> Message:
    """Build a ``user`` message for appending after the first turn.

    ``start_session`` seeds the first user message; subsequent user messages are
    appended through ``conversation_manager.add_message``, which needs a built
    :class:`~prompt_builder.models.Message`. This mirrors the manager's own
    message shape (unique id, submission timestamp) for a ``user`` message.
    """
    return Message(
        id=str(uuid.uuid4()),
        sender="user",
        text=text,
        created_at=time.time(),
    )


def _render_prompt_depth(session: Optional[Session]) -> None:
    """Render semantic depth control; Adaptive is the default."""
    st.markdown('<p class="cpb-eyebrow">Prompt depth</p>', unsafe_allow_html=True)
    selected = st.selectbox(
        "Prompt depth",
        PROMPT_DEPTH_OPTIONS,
        index=0,
        key=PROMPT_DEPTH_KEY,
        label_visibility="collapsed",
        help=(
            "Adaptive keeps simple requests focused and expands complex work. "
            "This controls prompt planning and detail, not the max_tokens runtime cap."
        ),
    )
    if session is not None and session.spec is not None:
        session.spec.prompt_depth = adaptive_planner.normalize_depth(selected)
        plan = adaptive_planner.plan_prompt(session.spec)
        st.caption(f"Effective depth: {plan.depth.title()} · {plan.task_type.title()} task")
    else:
        st.caption("Adaptive keeps simple tasks concise and expands complex tasks.")


def _render_parameter_tracker(session: Optional[Session]) -> None:
    """Render compact adaptive coverage without listing irrelevant catalog fields."""
    st.markdown('<p class="cpb-eyebrow">Brief coverage</p>', unsafe_allow_html=True)
    spec = session.spec if session is not None else None
    if spec is None:
        st.caption("Relevant details will appear after you describe the task.")
        return

    plan = adaptive_planner.plan_prompt(spec)
    items = adaptive_planner.coverage_items(spec, plan)
    relevant = [item for item in items if item.status != "optional"]
    captured = sum(item.status == "captured" for item in relevant)
    st.progress(
        captured / len(relevant) if relevant else 1.0,
        text=f"{captured} of {len(relevant)} relevant details captured",
    )
    st.caption(plan.rationale + " Skipped details stay skipped.")

    for item in relevant:
        symbol = {"captured": "✓", "missing": "?", "declined": "—"}[item.status]
        st.markdown(
            f'<div class="cpb-chip {item.status}">'
            f'<span class="cpb-dot" aria-hidden="true"></span>'
            f'<span class="cpb-name">{item.label}</span>'
            f'<span class="cpb-req">{symbol} {item.status}</span></div>',
            unsafe_allow_html=True,
        )

    optional = [item for item in items if item.status == "optional"]
    if optional:
        with st.expander(f"Other captured details ({len(optional)})", expanded=False):
            for item in optional:
                st.caption(f"✓ {item.label}")


def _render_model_settings() -> None:
    """Render the editable "Model settings" panel (promptingguide.ai LLM Settings).

    Sliders / inputs for temperature, max tokens, top-p, frequency and presence
    penalties, and stop sequences. Streamlit persists each widget's value in
    ``session_state`` under its key, so :func:`_current_settings` reads them back
    and passes them into every generation / revision call. A one-click reset
    restores the task-tuned defaults.
    """
    st.markdown('<p class="cpb-eyebrow">Model settings</p>', unsafe_allow_html=True)
    st.caption("Tune the LLM at call time. Guidance: adjust temperature OR top-p, not both.")

    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(st.session_state.get(TEMPERATURE_KEY, _DEFAULT_TEMPERATURE)),
        step=0.1,
        key=TEMPERATURE_KEY,
        help="Randomness. Low = focused/factual, high = creative/diverse.",
    )
    st.slider(
        "Max tokens",
        min_value=256,
        max_value=4096,
        value=int(st.session_state.get(MAX_TOKENS_KEY, _DEFAULT_MAX_TOKENS)),
        step=128,
        key=MAX_TOKENS_KEY,
        help="Hard cap on output length. Raise for long, detailed prompts.",
    )
    st.slider(
        "Top-p",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get(TOP_P_KEY, _DEFAULT_TOP_P)),
        step=0.05,
        key=TOP_P_KEY,
        help="Nucleus sampling. 1.0 = off. Tune this OR temperature, not both.",
    )
    st.slider(
        "Frequency penalty",
        min_value=-2.0,
        max_value=2.0,
        value=float(st.session_state.get(FREQUENCY_PENALTY_KEY, _DEFAULT_FREQUENCY_PENALTY)),
        step=0.1,
        key=FREQUENCY_PENALTY_KEY,
        help="Penalize tokens by how often they appear. Reduces repetition.",
    )
    st.slider(
        "Presence penalty",
        min_value=-2.0,
        max_value=2.0,
        value=float(st.session_state.get(PRESENCE_PENALTY_KEY, _DEFAULT_PRESENCE_PENALTY)),
        step=0.1,
        key=PRESENCE_PENALTY_KEY,
        help="Flat penalty on any seen token. Tune this OR frequency, not both.",
    )
    st.text_input(
        "Stop sequences (comma-separated)",
        value=st.session_state.get(STOP_KEY, _DEFAULT_STOP),
        key=STOP_KEY,
        placeholder="e.g. ###, END",
        help="Strings that halt generation. Leave blank for none.",
    )

    if st.button("Reset to defaults", key="reset_settings_button", width="stretch"):
        st.session_state[TEMPERATURE_KEY] = _DEFAULT_TEMPERATURE
        st.session_state[MAX_TOKENS_KEY] = _DEFAULT_MAX_TOKENS
        st.session_state[TOP_P_KEY] = _DEFAULT_TOP_P
        st.session_state[FREQUENCY_PENALTY_KEY] = _DEFAULT_FREQUENCY_PENALTY
        st.session_state[PRESENCE_PENALTY_KEY] = _DEFAULT_PRESENCE_PENALTY
        st.session_state[STOP_KEY] = _DEFAULT_STOP
        st.rerun()


def _render_sidebar(session: Optional[Session]) -> None:
    """Render persistent global navigation and page-relevant Builder controls."""
    with st.sidebar:
        st.markdown(
            f"""
            <div class="cpb-sidebrand">
                <span class="cpb-sidebrand-logo">{PAGE_ICON}</span>
                <span class="cpb-sidebrand-name">Prompt Builder</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Guided prompt design, review, and operational insight.")
        st.caption(provider_status_label())

        page_keys = [page.key for page in navigation.PAGES]
        # Synchronize the visible control before it is instantiated. Buttons
        # update ROUTE_KEY, while this widget callback updates ROUTE_KEY from a
        # keyboard/mouse navigation selection.
        st.session_state[VIEW_KEY] = navigation.resolve_page(
            st.session_state.get(ROUTE_KEY)
        ).key
        st.radio(
            "Navigation",
            page_keys,
            key=VIEW_KEY,
            format_func=lambda key: navigation.PAGE_BY_KEY[key].navigation_label,
            help="Use arrow keys to move between pages after focusing this control.",
            on_change=_on_navigation_change,
        )

        current_page = navigation.resolve_page(st.session_state.get(ROUTE_KEY))
        resume = navigation.resume_state(session)
        utility_state = (
            f"{resume.current_step} · {'prompt ready' if resume.has_prompt else 'in progress'}"
            if resume.available
            else "No active prompt"
        )
        st.markdown(
            '<div class="cpb-session-utility" aria-label="In-memory session status">'
            '<span class="cpb-session-dot" aria-hidden="true"></span>'
            '<div><b>In-memory session</b>'
            f'<small>{utility_state}</small></div></div>',
            unsafe_allow_html=True,
        )
        if current_page.key != BUILDER_VIEW:
            if st.button(
                "New prompt",
                key="nav_new_prompt",
                type="primary",
                width="stretch",
                help="Open Builder with a new empty in-memory session.",
            ):
                _start_new_prompt()
                st.rerun()
            if current_page.key == HOME_VIEW:
                st.divider()
                _render_prompt_depth(session)
                with st.expander("Advanced model settings", expanded=False):
                    _render_model_settings()
            return

        # Keep the established key for compatibility with original Session tests.
        if st.button("New session", key="new_session_button", width="stretch"):
            _start_new_prompt()
            st.rerun()

        st.divider()
        _render_prompt_depth(session)

        st.divider()
        _render_parameter_tracker(session)

        st.divider()
        with st.expander("Advanced model settings", expanded=False):
            _render_model_settings()

        st.divider()
        st.markdown('<p class="cpb-eyebrow">Session status</p>', unsafe_allow_html=True)
        message_count = len(session.messages) if session is not None else 0
        follow_ups = session.follow_up_count if session is not None else 0
        col_a, col_b = st.columns(2)
        col_a.metric("Messages", message_count)
        col_b.metric("Follow-ups", follow_ups)


def main() -> None:
    """Render the persistent shell and the selected product page."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        # Native Streamlit behavior opens the sidebar on desktop and keeps it
        # closed on narrow screens, where its built-in control remains visible.
        initial_sidebar_state="auto",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(_SAAS_CSS, unsafe_allow_html=True)
    st.markdown(_NAVY_CSS, unsafe_allow_html=True)
    st.markdown(_SHELL_CSS, unsafe_allow_html=True)

    session = _get_session()
    if ROUTE_KEY not in st.session_state:
        seeded_view = st.session_state.get(VIEW_KEY)
        st.session_state[ROUTE_KEY] = (
            navigation.resolve_page(seeded_view).key
            if seeded_view is not None
            else navigation.initial_page_key(session)
        )
    else:
        st.session_state[ROUTE_KEY] = navigation.resolve_page(
            st.session_state.get(ROUTE_KEY)
        ).key

    _render_sidebar(session)
    page = navigation.resolve_page(st.session_state.get(ROUTE_KEY))
    _render_header(page)

    if page.key == HOME_VIEW:
        _render_home(session)
    elif page.key == TEMPLATES_VIEW:
        _render_template_gallery()
    elif page.key == OBSERVABILITY_VIEW:
        dashboard.render_dashboard(show_header=False)
        return
    elif page.key == GUIDE_VIEW:
        _render_guide()
        return
    else:
        _render_builder_action_bar(session)
        has_messages = session is not None and bool(session.messages)
        _render_stepper(session)

        if not has_messages:
            _render_empty_state()
        else:
            _render_history(session)

        _render_context_note()
        if session is not None:
            _render_error_banner(session)

        _render_retained_input()
        session = _get_session()
        if session is not None:
            _render_prompt_panel(session)

    # Home keeps a direct idea-entry path to satisfy the original open-screen
    # input requirement. Templates, Observability, and Guide intentionally do
    # not show a chat box; their visible navigation provides Builder access.
    if page.key in {HOME_VIEW, BUILDER_VIEW}:
        placeholder = "Describe what you want to build..."
        raw_text = st.chat_input(placeholder)
        if raw_text is not None:
            _handle_submission(raw_text)
            st.rerun()


if __name__ == "__main__":
    main()
