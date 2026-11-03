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

from prompt_builder import catalog, conversation_manager, elicitation_engine, validation
from prompt_builder.intent_analyzer import IntentAnalyzer, merge_parameters, missing_parameters
from prompt_builder.llm_client import LlmClient, LlmError, OpenAiLlmClient
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
# Set when the last interpretation reported that prior context was unavailable.
CONTEXT_UNAVAILABLE_KEY = "context_unavailable"
# True while the current Session was seeded from a hardcoded demo example, so the
# prompt panel can show a "demo prompt (no AI)" ribbon.
DEMO_MODE_KEY = "demo_mode"

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

PAGE_TITLE = "Conversational Prompt Builder"
PAGE_ICON = "💬"
PAGE_TAGLINE = "Turn a plain-language idea into a precise, high-quality prompt."


# --- Demo examples (no LLM) -------------------------------------------------
# The ready-made examples live in their own module because the content is large.
# Selecting one seeds a Session with a pre-written Generated_Prompt (no LLM call).
from prompt_builder.demo_examples import DEMO_EXAMPLES


# --- Theming ----------------------------------------------------------------
# A small, self-contained stylesheet aiming for a minimal, professional look: a
# restrained slate/neutral palette, flat surfaces with hairline borders instead
# of bright fills or shadows, and clear partitioning between regions. No external
# assets; everything is scoped to this app's markup.
_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --cpb-ink: #2e1065;         /* deep indigo-plum - primary text (no black) */
    --cpb-muted: #6d5e9c;       /* muted violet - secondary text */
    --cpb-faint: #a99fce;       /* soft lilac - tertiary */
    --cpb-border: #e4def5;      /* pale lavender - hairline borders */
    --cpb-surface: #ffffff;     /* clean card surface */
    --cpb-subtle: #f3effd;      /* light lavender fill */
    --cpb-canvas: #f7f5fc;      /* page background wash */
    --cpb-accent: #7c3aed;      /* vivid violet - primary accent */
    --cpb-accent-2: #4f46e5;    /* indigo - gradient partner */
    --cpb-accent-soft: #ede9fe; /* violet-100 */
    --cpb-coral: #f43f8e;       /* coral-pink - lively secondary */
    --cpb-ok: #0ea5a4;          /* teal-cyan - collected */
    --cpb-pending: #f97316;     /* warm orange - missing */
    --cpb-radius: 14px;
    --cpb-shadow: 0 10px 30px -18px rgba(79, 70, 229, 0.45);
    --cpb-shadow-sm: 0 4px 14px -10px rgba(79, 70, 229, 0.5);
}

/* Global type: a clean, modern sans across the whole app. */
html, body, .stApp, .stApp * {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

/* Tint the whole page with soft, professional gradients. */
.stApp {
    background:
        radial-gradient(1100px 520px at 12% -12%, #ece4fe 0%, rgba(236,228,254,0) 58%),
        radial-gradient(900px 460px at 100% -6%, #fde4ef 0%, rgba(253,228,239,0) 52%),
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
    background: #d9cffb; border-radius: 999px; border: 2px solid transparent;
    background-clip: content-box;
}
::-webkit-scrollbar-thumb:hover { background: #c3b3f5; background-clip: content-box; }

/* Code block: a soft indigo surface rather than the default near-black, and
   wrap long lines so the raw view never forces horizontal scrolling. */
.stCode, [data-testid="stCode"] pre {
    background: #efeafe !important;
    border: 1px solid var(--cpb-border) !important;
    border-radius: 12px !important;
}
.stCode code, [data-testid="stCode"] code { color: #3b2b73 !important; }
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
    box-shadow: 0 8px 18px -8px rgba(124, 58, 237, 0.7);
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
        linear-gradient(135deg, #f3effe 0%, #fdeef6 100%);
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
    box-shadow: 0 4px 10px -4px rgba(124, 58, 237, 0.7);
}

/* --- Example cards ---------------------------------------------------- */
.cpb-example-card {
    border: 1px solid var(--cpb-border);
    border-radius: var(--cpb-radius);
    padding: 1rem 1.05rem;
    margin-bottom: 0.55rem;
    min-height: 120px;
    background: var(--cpb-surface);
    box-shadow: var(--cpb-shadow-sm);
    transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
}
.cpb-example-card:hover {
    transform: translateY(-3px);
    border-color: var(--cpb-accent);
    box-shadow: 0 16px 34px -18px rgba(124, 58, 237, 0.55);
}
.cpb-example-title {
    font-weight: 700;
    font-size: 0.98rem;
    color: var(--cpb-ink);
    margin-bottom: 0.4rem;
}
.cpb-example-text {
    font-size: 0.85rem;
    color: var(--cpb-muted);
    line-height: 1.4;
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
    color: #312360;                  /* soft dark indigo, not black */
    overflow-wrap: break-word;       /* wrap long tokens, never h-scroll */
    word-break: break-word;
    box-shadow: 0 10px 28px -20px rgba(79, 70, 229, 0.55);
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
    color: #312360;
}
.cpb-prompt .cpb-prompt-list {
    margin: 0 0 0.8rem 0;
    padding-left: 1.3rem;
}
.cpb-prompt .cpb-prompt-list li {
    margin-bottom: 0.3rem;
    color: #3b2b73;
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
    color: #4f46e5;
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
/* The user's own messages get a soft tinted background to set them apart. */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(180deg, #f6f2ff 0%, #fdf2f8 100%);
    border-color: #e7dcfb;
}

/* Secondary (default) buttons: lavender outline that lights up on hover. */
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
/* Primary action buttons: vivid violet->coral gradient, the main call to action. */
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, var(--cpb-accent-2), var(--cpb-accent) 55%, var(--cpb-coral));
    border: none;
    color: #fbfaff;
    box-shadow: 0 10px 24px -12px rgba(124, 58, 237, 0.7);
}
.stButton > button[kind="primary"]:hover {
    filter: brightness(1.06);
    color: #fbfaff;
    box-shadow: 0 12px 28px -10px rgba(244, 63, 142, 0.6);
}

/* Progress bar in the accent color. */
[data-testid="stProgress"] [role="progressbar"] > div {
    background: linear-gradient(90deg, var(--cpb-accent-2), var(--cpb-accent));
}

/* Sidebar: a soft lavender panel partitioned from the main column. */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f1ebfe 0%, #f9f0f8 100%);
    border-right: 1px solid var(--cpb-border);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
.cpb-sidebrand { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.1rem; }
.cpb-sidebrand-logo {
    width: 32px; height: 32px; display: grid; place-items: center;
    border-radius: 9px; font-size: 1.05rem;
    background: linear-gradient(135deg, var(--cpb-accent-2), var(--cpb-accent) 60%, var(--cpb-coral));
    box-shadow: 0 6px 14px -6px rgba(124, 58, 237, 0.7);
}
.cpb-sidebrand-name {
    font-size: 1.1rem; font-weight: 800; letter-spacing: -0.01em; color: var(--cpb-ink);
}

/* --- Form controls: inputs, text areas, chat input ------------------- */
.stTextInput input,
.stTextArea textarea,
[data-testid="stChatInput"] textarea,
[data-baseweb="input"] input {
    border-radius: 12px !important;
    border: 1px solid var(--cpb-border) !important;
    background: var(--cpb-surface) !important;
    color: var(--cpb-ink) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.stTextInput input:focus,
.stTextArea textarea:focus,
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--cpb-accent) !important;
    box-shadow: 0 0 0 3px var(--cpb-accent-soft) !important;
    outline: none !important;
}
/* The pinned chat input bar: a soft floating surface. */
[data-testid="stChatInput"] {
    border-radius: 14px;
    border: 1px solid var(--cpb-border);
    background: var(--cpb-surface);
    box-shadow: var(--cpb-shadow);
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
    box-shadow: 0 2px 8px -2px rgba(124,58,237,0.6) !important;
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


# --- LLM client construction (lazy) -----------------------------------------
def _build_llm_client() -> LlmClient:
    """Construct the real LLM client on demand.

    Kept out of module import and out of the render path so importing ``app`` and
    rendering the page never needs an API key. Any failure constructing the
    client (for example a missing key) surfaces as
    :class:`~prompt_builder.llm_client.LlmError`, which the callers turn into an
    error banner with a Retry control.
    """
    return OpenAiLlmClient()


# --- Session state helpers --------------------------------------------------
def _get_session() -> Optional[Session]:
    """Return the current Session from ``session_state`` (or ``None``)."""
    return st.session_state.get(SESSION_KEY)


def _set_session(session: Optional[Session]) -> None:
    """Store the current Session back into ``session_state``."""
    st.session_state[SESSION_KEY] = session


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
    st.session_state.pop(REJECTED_INPUT_KEY, None)
    st.session_state.pop(REJECTED_ERROR_KEY, None)
    # Mark this Session as a demo so the prompt panel shows the "no AI" ribbon.
    st.session_state[DEMO_MODE_KEY] = True


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

    missing = missing_parameters(spec)

    # If required parameters remain and we are under the question limit, ask the
    # next elicitation question (Req 3.1, 3.8); otherwise generate (Req 4.1).
    if missing and not elicitation_engine.question_limit_reached(session):
        question = elicitation_engine.next_question(spec, session)
        if question is not None:
            session.follow_up_count += 1
            conversation_manager.add_message(session, _system_message(question.text))
            return

    # No required parameters missing (or limit reached / nothing askable): produce
    # the Generated_Prompt. A failure here is handled as a generation error.
    _generate_prompt(session)


def _generate_prompt(session: Session) -> None:
    """Invoke the Prompt_Generator and store the Generated_Prompt (Req 4.1, 4.4).

    On an LLM failure the collected parameters are retained unchanged and a
    pending generation error is recorded so the UI offers a Retry control
    (Req 4.5, 8.2).
    """
    spec = session.spec if session.spec is not None else PromptSpecification(goal_confidence=0.0)
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
    else:
        st.session_state[PENDING_ERROR_KEY] = {
            "stage": "generate",
            "message": COULD_NOT_COMPLETE_MESSAGE,
            "detail": "",
        }


def _submit_revision(session: Session, revision_text: str) -> None:
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
def _render_header() -> None:
    """Render a minimal, bordered app header."""
    st.markdown(
        f"""
        <div class="cpb-header">
            <div class="cpb-logo">{PAGE_ICON}</div>
            <div class="cpb-head-text">
                <h1>{PAGE_TITLE}</h1>
                <p>{PAGE_TAGLINE}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_empty_state() -> None:
    """Render a friendly welcome/empty state before the first message."""
    st.markdown(
        """
        <div class="cpb-empty">
            <h3>Let's build your prompt</h3>
            <p>Describe what you want in plain language. I'll ask a few short
            follow-ups, then synthesize a polished prompt you can copy.</p>
            <ol>
                <li>Say what you're trying to create and who it's for</li>
                <li>Answer a couple of quick questions (or skip any)</li>
                <li>Review, revise, and copy the generated prompt</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        '<p class="cpb-eyebrow">Try an example (instant, no AI key needed)</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Pick one and press its button to see a ready-made prompt right away. "
        "Or just type your own message below to build one with AI."
    )
    # Lay the examples out in a tidy grid of up to 3 per row so adding more
    # examples never squashes the cards into unreadable narrow columns.
    per_row = 3
    for start in range(0, len(DEMO_EXAMPLES), per_row):
        row = DEMO_EXAMPLES[start : start + per_row]
        cols = st.columns(per_row)
        for col, example in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="cpb-example-card">
                        <div class="cpb-example-title">{example['title']}</div>
                        <div class="cpb-example-text">{example['request']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Show example prompt",
                    key=f"demo_{example['title']}",
                    use_container_width=True,
                    type="primary",
                ):
                    _load_demo_example(example)
                    st.rerun()


def _render_history(session: Session) -> None:
    """Render the conversation oldest-to-newest with sender distinction (Req 6.1, 6.2)."""
    for message in order_messages_for_render(session.messages):
        treatment = treatment_for_message(message)
        with st.chat_message(treatment.role, avatar=treatment.avatar):
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
            _set_session(session)
            st.rerun()
    elif stage == "revise":
        # Re-run the revision the user asked for, using the stashed text.
        revision_text = error.get("revision_text", "")
        if revision_text and st.button("Retry", key="retry_button"):
            with st.spinner("Retrying the revision..."):
                _submit_revision(session, revision_text)
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


def _render_prompt_panel(session: Session) -> None:
    """Render the Generated_Prompt panel with Revise and Copy controls (Req 4.3, 5.1-5.7)."""
    prompt = session.generated_prompt
    if prompt is None:
        return

    st.divider()
    # When the prompt came from a hardcoded demo example, make that explicit.
    if st.session_state.get(DEMO_MODE_KEY):
        st.markdown(
            '<span class="cpb-demo-note">EXAMPLE PROMPT · NO AI CALL</span>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="cpb-panel-title">Generated prompt</div>',
        unsafe_allow_html=True,
    )

    # Primary, reader-friendly display: the prompt rendered as wrapping, readable
    # prose (no horizontal scroll, comfortable line length and spacing, section
    # headings styled). This is the main view users read.
    st.markdown(_render_prompt_html(prompt.text), unsafe_allow_html=True)

    # The raw, selectable/copyable source lives in an expander so the built-in
    # ``st.code`` copy affordance (Req 5.5) is still available without forcing a
    # wide monospace block into the main reading flow.
    with st.expander("View raw / copy source", expanded=False):
        st.code(prompt.text, language="markdown")

    # A one-shot copy confirmation (Req 5.6). The explicit confirm button lets the
    # app surface a server-side confirmation indication in addition to the
    # built-in copy affordance.
    col_copy, _ = st.columns([1, 3])
    with col_copy:
        if st.button("Copy prompt", key="copy_button", use_container_width=True, type="primary"):
            st.session_state[COPY_CONFIRMED_KEY] = True
    if st.session_state.get(COPY_CONFIRMED_KEY):
        st.success("Prompt copied. Open 'View raw / copy source' and use the copy icon if needed.")
        st.session_state[COPY_CONFIRMED_KEY] = False

    # Report any required parameters that were omitted (Req 4.4).
    if prompt.omitted_required:
        st.info(
            "These required details were not provided and are omitted: "
            + ", ".join(prompt.omitted_required)
        )

    # Revise control: validate the request, then revise through the retry
    # controller (Req 5.1, 5.2, 5.3).
    with st.expander("Revise this prompt", expanded=False):
        # Revision is an LLM-backed operation. Demo prompts are hardcoded and were
        # produced with NO AI call, so revising them would need an API key and
        # otherwise fail with a confusing generic error. Make that explicit and
        # skip the LLM round-trip entirely in demo mode.
        if st.session_state.get(DEMO_MODE_KEY):
            st.info(
                "This is a pre-written example prompt (no AI was used), so it "
                "can't be revised here. Type your own message below to build a "
                "prompt with AI, then revise that one."
            )
        else:
            with st.form("revise_form", clear_on_submit=False):
                revision_text = st.text_area(
                    "Request a revision",
                    key="revision_input",
                    placeholder="Describe how you'd like the prompt changed...",
                )
                submitted = st.form_submit_button("Revise", use_container_width=True)
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
        # Do NOT call the LLM for invalid input.
        return

    # Accepted: drop any retained rejected text/error from a previous attempt.
    st.session_state.pop(REJECTED_INPUT_KEY, None)
    st.session_state.pop(REJECTED_ERROR_KEY, None)
    # A real typed message leaves demo mode (this turn goes through the LLM).
    st.session_state.pop(DEMO_MODE_KEY, None)

    session = _get_session()
    if session is None:
        session = conversation_manager.start_session(raw_text)
    else:
        session = conversation_manager.add_message(session, _user_message(raw_text))

    with st.spinner("Thinking..."):
        _run_turn(session)
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


def _render_parameter_tracker(session: Optional[Session]) -> None:
    """Render a sidebar progress tracker for the catalog parameters.

    Read-only view: for each catalog parameter it shows whether a value has been
    collected, declined, or is still missing, plus whether it's required. This is
    pure presentation derived from the spec; it does not change any behavior.
    """
    st.markdown('<p class="cpb-eyebrow">Prompt details</p>', unsafe_allow_html=True)
    spec = session.spec if session is not None else None
    params = spec.parameters if spec is not None else {}

    # Render the catalog parameters first (required/optional), then any additional
    # parameters captured on the spec that aren't in the catalog (the richer demo
    # examples carry extra, prompt-specific attributes). The union keeps the
    # catalog logic untouched while still surfacing everything collected.
    catalog_names = [definition.name for definition in catalog.CATALOG]
    extra_names = [name for name in params if name not in catalog_names]

    collected = sum(1 for v in params.values() if v.status == "collected")
    total = len(catalog_names) + len(extra_names)
    st.progress(collected / total if total else 0.0, text=f"{collected} of {total} captured")

    def _chip(name: str, required: bool) -> None:
        value = params.get(name)
        if value is not None and value.status == "collected":
            state = "collected"
        elif value is not None and value.status == "declined":
            state = "declined"
        else:
            state = "missing"
        req_label = "required" if required else "optional"
        display = name.replace("_", " ")
        st.markdown(
            f"""
            <div class="cpb-chip {state}">
                <span class="cpb-dot"></span>
                <span class="cpb-name">{display}</span>
                <span class="cpb-req">{req_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for definition in catalog.CATALOG:
        _chip(definition.name, definition.required)
    for name in extra_names:
        _chip(name, required=False)


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

    if st.button("Reset to defaults", key="reset_settings_button", use_container_width=True):
        st.session_state[TEMPERATURE_KEY] = _DEFAULT_TEMPERATURE
        st.session_state[MAX_TOKENS_KEY] = _DEFAULT_MAX_TOKENS
        st.session_state[TOP_P_KEY] = _DEFAULT_TOP_P
        st.session_state[FREQUENCY_PENALTY_KEY] = _DEFAULT_FREQUENCY_PENALTY
        st.session_state[PRESENCE_PENALTY_KEY] = _DEFAULT_PRESENCE_PENALTY
        st.session_state[STOP_KEY] = _DEFAULT_STOP
        st.rerun()


def _render_sidebar(session: Optional[Session]) -> None:
    """Render the sidebar: session info, parameter tracker, and new-session control (Req 6.5, 7.5)."""
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
        st.caption("A guided interview that writes your prompt for you.")

        # New-session control (Req 6.5, 7.5).
        if st.button("New session", key="new_session_button", use_container_width=True):
            # Start a fresh empty Session and clear transient UI flags so the view
            # clears the previous conversation and Prompt_Specification (Req 6.5, 7.5).
            _set_session(conversation_manager.reset_session())
            st.session_state.pop(PENDING_ERROR_KEY, None)
            st.session_state.pop(CONTEXT_UNAVAILABLE_KEY, None)
            st.session_state.pop(COPY_CONFIRMED_KEY, None)
            st.session_state.pop(REJECTED_INPUT_KEY, None)
            st.session_state.pop(REJECTED_ERROR_KEY, None)
            st.session_state.pop(DEMO_MODE_KEY, None)
            st.rerun()

        st.divider()
        _render_parameter_tracker(session)

        st.divider()
        with st.expander("⚙️ Model settings", expanded=False):
            _render_model_settings()

        # Lightweight session stats.
        st.divider()
        st.markdown('<p class="cpb-eyebrow">Session</p>', unsafe_allow_html=True)
        message_count = len(session.messages) if session is not None else 0
        follow_ups = session.follow_up_count if session is not None else 0
        col_a, col_b = st.columns(2)
        col_a.metric("Messages", message_count)
        col_b.metric("Follow-ups", follow_ups)


def main() -> None:
    """Render the chat UI and wire the conversation flow."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    session = _get_session()

    _render_sidebar(session)
    _render_header()

    # Center the conversation column for a more readable, app-like layout.
    _left, center, _right = st.columns([1, 6, 1])
    with center:
        has_messages = session is not None and bool(session.messages)

        # Empty / welcome state before the first message.
        if not has_messages:
            _render_empty_state()

        # Conversation history (oldest at top, newest at bottom) — Req 6.1, 6.2.
        if has_messages:
            _render_history(session)

        # Prior-context note and any pending error banner with Retry.
        _render_context_note()
        if session is not None:
            _render_error_banner(session)

        # Re-surface the text of a rejected submission so it isn't lost (Req 1.4, 1.5).
        _render_retained_input()

        # Prompt panel (shown once a prompt exists) — Req 4.3, 5.1-5.7.
        session = _get_session()
        if session is not None:
            _render_prompt_panel(session)

    # The free-text input (Req 1.1). Pinned to the bottom of the page by
    # ``st.chat_input``. Returns the submitted text on this rerun.
    raw_text = st.chat_input("Describe what you want to build...")
    if raw_text is not None:
        _handle_submission(raw_text)
        st.rerun()


if __name__ == "__main__":
    main()
