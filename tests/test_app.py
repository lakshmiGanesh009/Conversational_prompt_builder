"""UI presence and error-path tests for ``app.py`` (task 15.4).

These are example-based tests driven by Streamlit's ``AppTest`` harness
(``streamlit.testing.v1.AppTest``), which runs ``app.py`` in-process without a
browser or server. They exercise the thin UI/orchestration layer only; all the
deterministic rules are already covered by the property and unit tests next to
``prompt_builder``.

The app never touches the network here. ``app._build_llm_client`` constructs the
real :class:`~prompt_builder.llm_client.OpenAiLlmClient`; every test replaces
that construction with a deterministic, offline
:class:`tests.mock_llm.MockLlmClient`, so no API key is needed and the runs are
fully reproducible.

Injection point. ``AppTest.from_file`` runs ``app.py`` as a *fresh* module in its
own thread, so patching the already-imported ``app`` module object has no effect
on the script the harness actually executes. Instead the tests patch
``prompt_builder.llm_client.OpenAiLlmClient`` (with ``return_value`` set to the
mock) *before* calling ``.run()``: the freshly run ``app.py`` does
``from prompt_builder.llm_client import ... OpenAiLlmClient`` at run time and
therefore binds the patched class, so ``_build_llm_client()`` returns the mock.

Requirement coverage (design "Requirements", tasks.md 15.4):

- 1.1 - the free-text input is present on open.
- 5.1, 5.5 - Revise and Copy controls are present once a prompt is shown.
- 6.4, 8.1 - an interpretation (response) failure keeps the submitted message in
  history and surfaces an error banner.
- 8.2 - a generation (prompt) failure surfaces an error banner and retains the
  collected parameters / Session content.
- 5.6, 5.7 - the Copy control confirms on success; the copy-failure error path is
  documented where the app cannot produce a server-side failure.

Why the failure mocks raise :class:`~prompt_builder.llm_client.LlmError` rather
than ``MockLlmError``: the app catches ``LlmError`` specifically (both directly
in ``_run_turn`` / ``_generate_prompt`` and via the ``RetryController`` built
with ``retry_on=(LlmError,)``). ``MockLlmError`` is *not* an ``LlmError``
subclass, so only an actual ``LlmError`` flows through the app's error handling;
``MockLlmClient.failing(error=LlmError(...))`` gives us exactly that.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from prompt_builder.llm_client import LlmError
from prompt_builder.models import (
    GeneratedPrompt,
    Message,
    ParameterValue,
    PromptSpecification,
    Session,
)

from tests.mock_llm import MockLlmClient

# The app lives at the project root, the parent of this ``tests`` directory.
#
# ``AppTest.from_file`` resolves a *relative* path against the directory of the
# file that calls it (here, ``tests/``), so a bare ``"app.py"`` would resolve to
# the non-existent ``tests/app.py``. Pass an absolute path instead, computed from
# this test file's own location, so the harness finds the real root ``app.py``
# regardless of the working directory pytest is invoked from.
APP_PATH = str((Path(__file__).resolve().parent.parent / "app.py"))

# The symbol ``app._build_llm_client`` instantiates. Patching this (rather than
# ``app._build_llm_client`` itself) is what reaches the fresh module AppTest runs.
CLIENT_SYMBOL = "prompt_builder.llm_client.OpenAiLlmClient"


# --- helpers ----------------------------------------------------------------
def _user_message(text: str) -> Message:
    """Build a ``user`` message mirroring the app's own message shape."""
    return Message(id=str(uuid.uuid4()), sender="user", text=text, created_at=time.time())


def _collected(name: str, value: str) -> ParameterValue:
    """A collected :class:`ParameterValue` for pre-seeding a spec."""
    return ParameterValue(name=name, value=value, status="collected", attempts=0)


def _spec_all_required_collected() -> PromptSpecification:
    """A spec whose required parameters (audience, domain) are both collected.

    With both required parameters collected, ``missing_parameters`` is empty, so a
    turn flows straight to generation rather than asking a follow-up question.
    """
    return PromptSpecification(
        goal="write a blog post",
        goal_confidence=0.9,
        parameters={
            "audience": _collected("audience", "developers"),
            "domain": _collected("domain", "software"),
        },
    )


def _session_with_prompt() -> Session:
    """A Session that already carries a Generated_Prompt, ready to render."""
    return Session(
        id=str(uuid.uuid4()),
        messages=[_user_message("write a blog post for developers about software")],
        spec=_spec_all_required_collected(),
        generated_prompt=GeneratedPrompt(
            text="You are an expert. Write a blog post for developers about software.",
            omitted_required=[],
        ),
    )


def _run_with_client(
    client: MockLlmClient,
    *,
    session: Optional[Session] = None,
) -> AppTest:
    """Run ``app.py`` with the LLM client construction replaced by ``client``.

    Optionally pre-seeds ``session_state["session"]`` before the run so the test
    can start from a known Session (for example one that already has a prompt).
    """
    at = AppTest.from_file(APP_PATH)
    if session is not None:
        at.session_state["session"] = session
    with patch(CLIENT_SYMBOL, return_value=client):
        at.run()
    return at


def _button_by_key(at: AppTest, key: str):
    """Return the button with the given key, or ``None`` if not present."""
    for button in at.button:
        if button.key == key:
            return button
    return None


# --- 1.1: input present on open ---------------------------------------------
def test_chat_input_present_on_open() -> None:
    """Req 1.1: the free-text input is present when the app first renders.

    No interaction and no pre-seeded Session: a fresh open must still show the
    ``st.chat_input`` so the user can type their first message.
    """
    # A client that is never expected to be called on a bare open.
    at = _run_with_client(MockLlmClient.with_responses())

    assert not at.exception
    assert len(at.chat_input) == 1
    # The input carries the placeholder the app set, confirming it is our input.
    assert at.chat_input[0].placeholder == "Describe what you want to build..."


# --- 5.1 / 5.5: revise and copy controls present when a prompt is shown ------
def test_revise_and_copy_controls_present_when_prompt_shown() -> None:
    """Req 5.1, 5.5: once a prompt exists, the Revise and Copy controls render.

    Pre-seed a Session carrying a Generated_Prompt, then assert the prompt panel
    shows: the prompt text via ``st.code`` (whose built-in copy icon satisfies
    5.5), an explicit "Copy prompt" button, a revision text area, and a "Revise"
    submit button (5.1).
    """
    at = _run_with_client(
        MockLlmClient.with_responses(), session=_session_with_prompt()
    )

    assert not at.exception

    # The prompt itself is rendered in a code block (built-in copy affordance).
    assert len(at.code) >= 1
    assert "blog post" in at.code[0].value

    # The explicit server-side Copy control (5.5 / 5.6).
    copy_button = _button_by_key(at, "copy_button")
    assert copy_button is not None
    assert copy_button.label == "Copy prompt"

    # The Revise control: a text area to describe the change plus a submit button.
    # Form submit buttons are exposed through ``at.button`` in this Streamlit
    # version (there is no dedicated ``form_submit_button`` accessor).
    assert any(area.key == "revision_input" for area in at.text_area)
    assert "Revise" in [button.label for button in at.button]


# --- 6.4 / 8.1: interpretation (response) failure -----------------------------
def test_interpretation_failure_keeps_message_and_shows_error() -> None:
    """Req 6.4, 8.1: a response failure retains the message and shows an error.

    Submitting the first message triggers ``_run_turn`` -> Intent_Analyzer, which
    calls the (failing) LLM. The app must keep the submitted user text in the
    Session history and surface an error banner, leaving the Session otherwise
    unchanged so a Retry can be offered.
    """
    failing = MockLlmClient.failing(error=LlmError("interpretation down"))

    at = AppTest.from_file(APP_PATH)
    with patch(CLIENT_SYMBOL, return_value=failing):
        # An initial run renders the element tree (including the chat input);
        # the widget can only be driven after that first run.
        at.run()
        at.chat_input[0].set_value("write a blog post for developers").run()

    assert not at.exception

    # The submitted user message is retained in the Session history (6.4).
    session = at.session_state["session"]
    assert session is not None
    assert any(
        m.sender == "user" and m.text == "write a blog post for developers"
        for m in session.messages
    )
    # No prompt was produced because interpretation failed before generation.
    assert session.generated_prompt is None

    # An error banner is shown (8.1).
    assert len(at.error) >= 1
    assert any("could not be generated" in err.value.lower() for err in at.error)

    # A Retry control is offered so the user can re-run the failed turn.
    assert _button_by_key(at, "retry_button") is not None


# --- 8.2: generation (prompt) failure ----------------------------------------
def test_generation_failure_shows_error_and_retains_parameters() -> None:
    """Req 8.2: a prompt-generation failure shows an error and keeps parameters.

    Pre-seed a Session whose required parameters are already collected, so a new
    turn skips elicitation and goes straight to generation. The interpret call
    returns a benign structured payload (so interpretation succeeds), and the
    generate call fails. The app must surface a generation error and leave the
    collected parameters untouched (no prompt produced).
    """
    # First complete() = interpret (succeeds, extracts nothing new); second
    # complete() = generate (fails). The structured payload keeps confidence high
    # so the goal stays set and no low-confidence follow-up is asked.
    class _InterpretThenFail:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, request):  # noqa: ANN001 - mirrors the mock's shape
            self.calls += 1
            if self.calls == 1:
                from tests.mock_llm import MockLlmResponse

                return MockLlmResponse(
                    text="{}",
                    structured={"goal": "write a blog post", "confidence": 0.9, "parameters": {}},
                )
            raise LlmError("generation down")

    client = _InterpretThenFail()

    at = AppTest.from_file(APP_PATH)
    at.session_state["session"] = _session_with_prompt_cleared()
    with patch(CLIENT_SYMBOL, return_value=client):
        # Render first so the chat input exists, then submit a message.
        at.run()
        at.chat_input[0].set_value("make it about software").run()

    assert not at.exception

    session = at.session_state["session"]
    assert session is not None

    # The collected required parameters are retained unchanged (8.2).
    assert session.spec is not None
    assert session.spec.parameters["audience"].value == "developers"
    assert session.spec.parameters["domain"].value == "software"

    # A generation error banner is shown (8.2).
    assert len(at.error) >= 1
    assert any("prompt generation failed" in err.value.lower() for err in at.error)


def _session_with_prompt_cleared() -> Session:
    """A Session with required params collected but no prompt yet.

    Used by the generation-failure test: the turn should flow to generation
    (required params already collected) and then fail there, so we start with no
    Generated_Prompt.
    """
    return Session(
        id=str(uuid.uuid4()),
        messages=[_user_message("write a blog post for developers")],
        spec=_spec_all_required_collected(),
        generated_prompt=None,
    )


# --- 5.6 / 5.7: copy success confirmation (and copy-failure note) ------------
def test_copy_button_confirms_on_success() -> None:
    """Req 5.6: clicking Copy surfaces a success confirmation.

    With a prompt shown, clicking the server-side "Copy prompt" button sets the
    one-shot confirmation flag, which renders an ``st.success`` message on the
    following run.

    Req 5.7 (copy failure error): ``app.py`` performs the actual clipboard copy
    via ``st.code``'s built-in, client-side copy icon. A clipboard write failing
    is a browser-side event that the code block surfaces to the user directly; it
    is not reachable through the server-side ``AppTest`` harness, so there is no
    server-side failure path to assert here. This test therefore covers the
    success confirmation (5.6) and documents that the copy affordance (the
    ``st.code`` copy icon asserted in ``test_revise_and_copy_controls_present...``)
    is the mechanism whose client-side failure 5.7 refers to. No failure is
    fabricated that the app cannot actually produce.
    """
    client = MockLlmClient.with_responses()

    at = AppTest.from_file(APP_PATH)
    at.session_state["session"] = _session_with_prompt()
    with patch(CLIENT_SYMBOL, return_value=client):
        at.run()
        # Click the explicit Copy control, then rerun so the confirmation renders.
        copy_button = _button_by_key(at, "copy_button")
        assert copy_button is not None
        copy_button.click().run()

    assert not at.exception
    assert len(at.success) >= 1
    assert any("copied" in msg.value.lower() for msg in at.success)


# --- professional frontend: local prompt editing ---------------------------
def test_local_prompt_edit_save_updates_text_and_preserves_omissions() -> None:
    """Saving Edit mode updates only prompt text and keeps omission metadata."""
    session = _session_with_prompt()
    session.generated_prompt.omitted_required = ["tone"]
    at = AppTest.from_file(APP_PATH)
    at.session_state["session"] = session

    with patch(CLIENT_SYMBOL, return_value=MockLlmClient.with_responses()):
        at.run()
        edit = _button_by_key(at, "edit_prompt_button")
        assert edit is not None
        edit.click().run()
        draft = next(area for area in at.text_area if area.key == "prompt_edit_draft")
        draft.set_value("# Edited prompt\nUse the saved local text.").run()
        save = _button_by_key(at, "save_prompt_edit")
        assert save is not None
        save.click().run()

    saved = at.session_state["session"].generated_prompt
    assert saved.text == "# Edited prompt\nUse the saved local text."
    assert saved.omitted_required == ["tone"]
    assert _button_by_key(at, "edit_prompt_button") is not None


def test_local_prompt_edit_cancel_leaves_prompt_unchanged() -> None:
    """Cancel discards the local draft without replacing the current prompt."""
    session = _session_with_prompt()
    original = session.generated_prompt.text
    at = AppTest.from_file(APP_PATH)
    at.session_state["session"] = session

    with patch(CLIENT_SYMBOL, return_value=MockLlmClient.with_responses()):
        at.run()
        _button_by_key(at, "edit_prompt_button").click().run()
        next(area for area in at.text_area if area.key == "prompt_edit_draft").set_value(
            "This draft must be discarded."
        ).run()
        cancel = _button_by_key(at, "cancel_prompt_edit")
        assert cancel is not None
        cancel.click().run()

    assert at.session_state["session"].generated_prompt.text == original
    assert _button_by_key(at, "edit_prompt_button") is not None
