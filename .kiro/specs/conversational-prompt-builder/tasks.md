# Implementation Plan: Conversational Prompt Builder

## Overview

This plan builds the Conversational Prompt Builder as a small, local Streamlit web app written
entirely in Python (3.11+). One Streamlit script (`app.py`) renders the chat UI and runs with
`streamlit run app.py`. A conversation (a Session) lives entirely in Streamlit's in-memory
`session_state` and disappears when the user starts a new session or restarts the app. There is no
database and no server-side storage beyond Streamlit itself.

All the deterministic rules (message/revision validation, parameter merge/conflict,
missing-parameter derivation, elicitation rules, message capping at 200, prompt assembly, retry
controller) live in plain Python functions kept out of the UI script, so they can be tested without
the network and without starting a Streamlit server. A single Python LLM client module is the only
part that touches the network; the Intent_Analyzer and Prompt_Generator call it, and nothing else
does. Tests use a mock LLM client so the deterministic logic runs offline.

The build order follows the design's "test the rules without a server" principle: the first
deliverable is `README.md` (as the user requested), then the Python project scaffolding, then the
data models and the parameter catalog, then the pure deterministic logic (validation, conversation
management, parameter tracking, elicitation, prompt assembly, retry controller), then the single LLM
client module, then the four core components wired on top, then the Streamlit UI helpers and the
`app.py` wiring, and finally the tests for timing. Property-based tests (Hypothesis, `max_examples >=
100`) and unit tests (pytest) sit next to the code they check so errors surface early.

Each task builds on the previous ones and ends by wiring new code into what already exists, so there
is no orphaned code.

## Tasks

- [x] 1. Create the README and Python project scaffolding
  - [x] 1.1 Write README.md at the project root
    - Create `README.md` describing what the app does (a guided, multi-turn conversation that
      interviews the user in plain language and synthesizes a high-quality prompt) and the simple,
      no-jargon vision
    - Document setup and run steps: create a virtualenv (`python -m venv .venv` and activate it),
      install with `pip install -r requirements.txt`, run the app with `streamlit run app.py`, and
      run the tests with `pytest`
    - Document that the LLM API key/config is supplied via an environment variable (name the variable
      and note it points at a hosted, OpenAI-compatible API)
    - _Requirements: 1.1, 4.1, 5.1_

  - [x] 1.2 Create the Python package layout and tooling
    - Create the package directory `prompt_builder/` with `__init__.py`, a `tests/` directory with
      `__init__.py`, `requirements.txt` (streamlit, hypothesis, pytest, and the LLM SDK or
      `httpx`/`requests`), and pytest config (`pytest.ini` or `pyproject.toml`
      `[tool.pytest.ini_options]`)
    - Confirm `pytest` collects an empty suite without error
    - _Requirements: 1.1_

  - [x] 1.3 Create the placeholder Streamlit entry point
    - Create `app.py` at the project root that imports the `prompt_builder` package and renders a
      minimal Streamlit page, so `streamlit run app.py` starts cleanly
    - _Requirements: 1.1_

- [x] 2. Define data models and the parameter catalog
  - [x] 2.1 Implement the core dataclasses
    - In `prompt_builder/models.py`, define `ParameterDefinition`, `ParameterValue`, `Overwrite`,
      `PromptSpecification`, `Message`, `MessageSender`, `GeneratedPrompt`, and `Session` as
      `@dataclass`es with type hints exactly as in the design Data Models section
    - _Requirements: 1.3, 2.2, 4.4, 6.2, 7.1_

  - [x] 2.2 Create the parameter catalog
    - In `prompt_builder/catalog.py`, define the static catalog of `ParameterDefinition`s (name,
      required flag, optional default value) that the Intent_Analyzer and Elicitation_Engine both
      read, plus a helper to list required parameter names
    - _Requirements: 2.4, 3.5_

  - [x] 2.3 Write unit tests for models and catalog
    - In `tests/test_models.py`, test dataclass defaults (an empty spec has no goal and no
      parameters) and that the catalog exposes required parameters and defaults correctly
    - _Requirements: 1.3, 3.5_

- [x] 3. Implement input validation
  - [x] 3.1 Implement message and revision validation functions
    - In `prompt_builder/validation.py`, implement pure functions that validate a message (accept iff
      it has >= 1 non-whitespace character and length 1-10,000; classify whitespace-only as empty and
      over-length separately) and a revision request (accept iff non-empty and <= 2000 characters),
      each returning a result naming the violated rule
    - _Requirements: 1.4, 1.5, 5.3_

  - [x] 3.2 Write property test for message input validation
    - In `tests/test_validation.py`, add a Hypothesis test for **Property 2: Message input
      validation** — submission is accepted iff the string has at least one non-whitespace character
      and length 1-10,000; whitespace-only rejected as empty, over-10,000 rejected for length
    - Tag: `# Feature: conversational-prompt-builder, Property 2: Message input validation`
    - `max_examples >= 100`; generators cover whitespace-only and boundary-length strings
    - **Validates: Requirements 1.4, 1.5**

  - [x] 3.3 Write property test for revision request validation
    - In `tests/test_validation.py`, add a Hypothesis test for **Property 14: Revision request
      validation** — accepted iff non-empty and at most 2000 characters; otherwise rejected and the
      current Generated_Prompt is unchanged
    - Tag: `# Feature: conversational-prompt-builder, Property 14: Revision request validation`
    - `max_examples >= 100`; generators cover empty and boundary-length strings
    - **Validates: Requirements 5.3**

- [x] 4. Implement the Conversation_Manager and message capping
  - [x] 4.1 Implement session and message management
    - In `prompt_builder/conversation_manager.py`, implement `start_session`, `add_message` (append
      and cap to the 200 most recent in submission order via a pure helper over the message list),
      `reset_session`, and `record_revision`
    - _Requirements: 1.2, 1.3, 3.3, 5.2, 6.1, 6.5, 7.1, 7.2, 7.5_

  - [x] 4.2 Write property test for new-session creation
    - In `tests/test_conversation_manager.py`, add a Hypothesis test for **Property 1: New session
      starts clean and holds the first message** — for any non-empty first message, the Session
      contains that message and its spec has no goal and no collected parameters
    - Tag: `# Feature: conversational-prompt-builder, Property 1: New session starts clean and holds the first message`
    - `max_examples >= 100`
    - **Validates: Requirements 1.2, 1.3**

  - [x] 4.3 Write property test for message retention and capping
    - In `tests/test_conversation_manager.py`, add a Hypothesis test for **Property 18: Messages are
      retained in order and capped at 200** — the Session retains messages in submission order and
      keeps exactly the 200 most recent when the count would exceed 200
    - Tag: `# Feature: conversational-prompt-builder, Property 18: Messages are retained in order and capped at 200`
    - `max_examples >= 100`; generators cover sequences both under and over 200
    - **Validates: Requirements 3.3, 7.1, 7.2**

  - [x] 4.4 Write property test for new-session clearing
    - In `tests/test_conversation_manager.py`, add a Hypothesis test for **Property 17: Starting a new
      session clears everything** — starting a new Session yields a Session with no messages and an
      empty Prompt_Specification
    - Tag: `# Feature: conversational-prompt-builder, Property 17: Starting a new session clears everything`
    - `max_examples >= 100`
    - **Validates: Requirements 6.5, 7.5**

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement parameter tracking in the Intent_Analyzer
  - [x] 6.1 Implement parameter merge, conflict handling, and missing derivation
    - In `prompt_builder/intent_analyzer.py`, implement `merge_parameters` (last value wins, record
      each overwrite in the overwrite log) and `missing_parameters` (required parameters with no value
      of status "collected"); define the `InterpretationResult`, `Overwrite`, and `MergeResult`
      dataclasses here. Keep these as pure deterministic functions
    - _Requirements: 2.2, 2.3, 2.4_

  - [x] 6.2 Write property test for parameter merging
    - In `tests/test_intent_analyzer.py`, add a Hypothesis test for **Property 3: Extracted parameters
      are merged into the specification** — after merging, every extracted parameter that maps to a
      defined Parameter has its value stored in the spec
    - Tag: `# Feature: conversational-prompt-builder, Property 3: Extracted parameters are merged into the specification`
    - `max_examples >= 100`
    - **Validates: Requirements 2.2**

  - [x] 6.3 Write property test for conflict resolution
    - In `tests/test_intent_analyzer.py`, add a Hypothesis test for **Property 4: Conflicts keep the
      most recent value and are logged** — the stored value equals the new value and the overwrite log
      records the previous value
    - Tag: `# Feature: conversational-prompt-builder, Property 4: Conflicts keep the most recent value and are logged`
    - `max_examples >= 100`
    - **Validates: Requirements 2.3**

  - [x] 6.4 Write property test for the missing-parameter set
    - In `tests/test_intent_analyzer.py`, add a Hypothesis test for **Property 5: Missing set is
      exactly the uncollected required parameters** — the computed missing set equals exactly the
      required Parameters with no value of status "collected"
    - Tag: `# Feature: conversational-prompt-builder, Property 5: Missing set is exactly the uncollected required parameters`
    - `max_examples >= 100`; generators mix collected/declined/missing parameters
    - **Validates: Requirements 2.4**

- [x] 7. Implement intent interpretation (LLM-backed) and low-confidence handling
  - [x] 7.1 Implement interpret() with context and confidence threshold
    - In `prompt_builder/intent_analyzer.py`, implement `interpret(message, context, spec)` that calls
      the LLM client, returns `InterpretationResult` (confidence, extracted parameters, goal,
      context_unavailable), leaves the goal unset when confidence is below the configured acceptance
      threshold, and sets `context_unavailable` when retained context cannot be read
    - _Requirements: 2.1, 2.5, 7.3, 7.4_

  - [x] 7.2 Write property test for low-confidence goal handling
    - In `tests/test_intent_interpret.py`, add a Hypothesis test for **Property 6: Low confidence
      leaves the goal unset and asks for it** — when confidence is below the threshold, the goal stays
      unset and a follow-up question asking for the goal is produced
    - Tag: `# Feature: conversational-prompt-builder, Property 6: Low confidence leaves the goal unset and asks for it`
    - `max_examples >= 100`; use a mock LLM client
    - **Validates: Requirements 2.5**

  - [x] 7.3 Write unit tests for context handling
    - In `tests/test_intent_interpret.py`, with a mock LLM client, verify the analyzer receives the
      retained session messages as context (7.3) and that unavailable context yields interpretation in
      isolation with the indication flag set (7.4)
    - _Requirements: 7.3, 7.4_

- [x] 8. Implement the Elicitation_Engine
  - [x] 8.1 Implement question selection and answer handling
    - In `prompt_builder/elicitation_engine.py`, implement `next_question` (pick 1-3 missing
      parameters, attach suggested defaults, return null when none remain or the limit is reached),
      `handle_answer` (usable -> collected; empty/skip -> declined and never re-asked; unusable ->
      retry up to 3 attempts), and `question_limit_reached` (true iff follow-up count >= 20); define
      `FollowUpQuestion`, `CollectedOutcome`, `DeclinedOutcome`, `RetryOutcome` here
    - _Requirements: 3.1, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [x] 8.2 Write property test for follow-up question parameter count
    - In `tests/test_elicitation_engine.py`, add a Hypothesis test for **Property 7: Follow-up
      questions request 1 to 3 missing parameters** — with a missing required parameter and under the
      limit, the next question is non-null and references 1-3 currently missing parameters
    - Tag: `# Feature: conversational-prompt-builder, Property 7: Follow-up questions request 1 to 3 missing parameters`
    - `max_examples >= 100`; generators mix collected/declined/missing parameters
    - **Validates: Requirements 3.1, 3.2**

  - [x] 8.3 Write property test for collected answers
    - In `tests/test_elicitation_engine.py`, add a Hypothesis test for **Property 8: Usable answers
      mark parameters collected** — a usable answer marks the requested Parameter as "collected" with
      the provided value
    - Tag: `# Feature: conversational-prompt-builder, Property 8: Usable answers mark parameters collected`
    - `max_examples >= 100`
    - **Validates: Requirements 3.4**

  - [x] 8.4 Write property test for suggested defaults
    - In `tests/test_elicitation_engine.py`, add a Hypothesis test for **Property 9: Defaults are
      offered as suggestions** — every requested Parameter that has a default value appears in the
      question's suggested defaults
    - Tag: `# Feature: conversational-prompt-builder, Property 9: Defaults are offered as suggestions`
    - `max_examples >= 100`
    - **Validates: Requirements 3.5**

  - [x] 8.5 Write property test for skipped parameters
    - In `tests/test_elicitation_engine.py`, add a Hypothesis test for **Property 10: Skipped
      parameters are declined and never re-asked** — an empty/skip answer marks the Parameter
      "declined" and no later question in the Session requests it again
    - Tag: `# Feature: conversational-prompt-builder, Property 10: Skipped parameters are declined and never re-asked`
    - `max_examples >= 100`
    - **Validates: Requirements 3.6**

  - [x] 8.6 Write property test for retry attempts
    - In `tests/test_elicitation_engine.py`, add a Hypothesis test for **Property 11: Unusable answers
      retry up to three attempts** — unusable (non-skip) answers raise the attempt count up to 3;
      after the third attempt the Parameter is no longer re-presented
    - Tag: `# Feature: conversational-prompt-builder, Property 11: Unusable answers retry up to three attempts`
    - `max_examples >= 100`
    - **Validates: Requirements 3.7**

  - [x] 8.7 Write property test for the question limit
    - In `tests/test_elicitation_engine.py`, add a Hypothesis test for **Property 12: Follow-up
      questions stop at twenty** — the limit is reached iff the follow-up count is 20 or more, and at
      the limit the next follow-up question is null
    - Tag: `# Feature: conversational-prompt-builder, Property 12: Follow-up questions stop at twenty`
    - `max_examples >= 100`
    - **Validates: Requirements 3.8**

- [x] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement the single LLM client module
  - [x] 10.1 Implement the LLM client
    - In `prompt_builder/llm_client.py`, define `LlmClient` (Protocol), `LlmRequest`, `LlmResponse`,
      and a concrete client that calls the hosted OpenAI-compatible API (official SDK or
      `httpx`/`requests`), reading the API key/config from the environment variable and raising on
      failure. This is the only module that touches the network
    - _Requirements: 2.1, 4.1, 5.2, 8.1_

  - [x] 10.2 Add a mock LLM client for tests
    - In `tests/mock_llm.py`, add a mock LLM client that returns canned/structured responses, can echo
      structured content for assembly tests, and can be made to raise, for use across the property and
      unit tests
    - _Requirements: 2.1, 4.1_

- [x] 11. Implement the Prompt_Generator
  - [x] 11.1 Implement prompt assembly, early generation, and revision
    - In `prompt_builder/prompt_generator.py`, implement `generate(spec)` (LLM writes the prompt;
      guarantee every collected parameter is included; report required parameters that are not
      collected) and `revise(current, revision_text, spec)` (produce an updated prompt incorporating
      the revision); on LLM failure leave the spec unchanged and surface a retryable error; define
      `GenerationResult` here
    - _Requirements: 4.1, 4.2, 4.4, 4.5, 5.2, 5.4_

  - [x] 11.2 Write property test for prompt assembly
    - In `tests/test_prompt_generator.py`, add a Hypothesis test for **Property 13: Generated prompt
      includes all collected parameters and reports omissions** — the prompt includes every collected
      Parameter's value and the omitted-required list equals exactly the required Parameters that are
      not collected (empty when all collected)
    - Tag: `# Feature: conversational-prompt-builder, Property 13: Generated prompt includes all collected parameters and reports omissions`
    - `max_examples >= 100`; use the mock LLM that echoes structured content so every collected
      parameter can be asserted present; generators mix collected/declined/missing
    - **Validates: Requirements 4.1, 4.2, 4.4**

  - [x] 11.3 Write unit tests for generation and revision error paths
    - In `tests/test_prompt_generator.py`, with a mock LLM that raises: generation failure retains the
      collected parameters and offers retry (4.5); revision failure keeps the previous prompt
      unchanged (5.4)
    - _Requirements: 4.5, 5.4_

- [x] 12. Implement the retry controller
  - [x] 12.1 Implement the retry/cap controller
    - In `prompt_builder/retry_controller.py`, implement a plain controller that re-executes a failing
      operation, stops after 3 consecutive failed attempts, reports that the operation could not be
      completed, and leaves Session content unchanged
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [x] 12.2 Write property test for the retry controller
    - In `tests/test_retry_controller.py`, add a Hypothesis test for **Property 19: Retries stop after
      three consecutive failures** — the controller stops after 3 consecutive failures, reports the
      operation could not be completed, and leaves Session content unchanged
    - Tag: `# Feature: conversational-prompt-builder, Property 19: Retries stop after three consecutive failures`
    - `max_examples >= 100`
    - **Validates: Requirements 8.4**

- [x] 13. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Implement the Streamlit UI rendering helpers
  - [x] 14.1 Implement conversation rendering and sender distinction
    - In `prompt_builder/ui.py`, implement pure helper functions that order messages
      oldest-to-newest for rendering and map a message's sender to its visual treatment (user
      treatment differs from system treatment), ready to be used by `st.chat_message`
    - _Requirements: 6.1, 6.2_

  - [x] 14.2 Write property test for conversation ordering
    - In `tests/test_ui.py`, add a Hypothesis test for **Property 15: Conversation renders oldest to
      newest** — the view renders messages ordered by submission time from oldest to newest
    - Tag: `# Feature: conversational-prompt-builder, Property 15: Conversation renders oldest to newest`
    - `max_examples >= 100`
    - **Validates: Requirements 6.1**

  - [x] 14.3 Write property test for sender-driven visual treatment
    - In `tests/test_ui.py`, add a Hypothesis test for **Property 16: Sender determines the visual
      treatment** — the visual treatment is determined solely by sender, and the user treatment
      differs from the system treatment
    - Tag: `# Feature: conversational-prompt-builder, Property 16: Sender determines the visual treatment`
    - `max_examples >= 100`
    - **Validates: Requirements 6.2**

- [x] 15. Build the Streamlit app and wire everything together
  - [x] 15.1 Render the message input with inline validation
    - In `app.py`, render the free-text input (`st.chat_input`) and reject empty/whitespace and
      over-10,000-character submissions inline using `validation.py`, keeping the entered text and
      showing the rule-specific error without calling the LLM
    - _Requirements: 1.1, 1.4, 1.5_

  - [x] 15.2 Render the conversation history and wire the turn flow
    - In `app.py`, render the current Session's messages via the `ui.py` helpers using
      `st.chat_message` (oldest at top to newest at bottom, user and system visually distinguished),
      and on each submitted message record it via Conversation_Manager, call the Intent_Analyzer
      (passing retained messages as context), merge parameters and derive missing, then either produce
      the next Elicitation_Engine question or invoke the Prompt_Generator, writing the updated Session
      back to `session_state`
    - _Requirements: 2.1, 2.2, 2.4, 3.1, 3.8, 4.1, 6.1, 6.2, 7.3_

  - [x] 15.3 Add the prompt panel, status indicator, error banners, and new-session control
    - In `app.py`, display the Generated_Prompt (and any omitted required parameters) with a Revise
      control (validates and submits a revision request) and a Copy control (e.g. `st.code` copy
      button) that confirms on success within the budget and errors on failure; show an in-progress
      indicator (`st.spinner`) while processing and clear it on resolution; on failure keep the
      submitted message in history, show an error banner, and offer a Retry control driven by the
      retry controller; show the "prior context unavailable" note when set; and add a new-session
      button that resets the Session and clears the displayed conversation and Prompt_Specification
    - _Requirements: 4.3, 4.4, 5.1, 5.2, 5.3, 5.5, 5.6, 5.7, 6.3, 6.4, 6.5, 7.4, 7.5, 8.1, 8.2, 8.3, 8.4_

  - [x] 15.4 Write unit tests for UI presence and error paths with AppTest
    - In `tests/test_app.py`, using Streamlit `AppTest` and the mock LLM client: input field present
      on open (1.1); revise and copy controls present when a prompt is shown (5.1, 5.5); processing
      failure keeps the message and shows an error (6.4); response/prompt failure handling (8.1, 8.2);
      copy success confirmation and failure error (5.6, 5.7)
    - _Requirements: 1.1, 5.1, 5.5, 5.6, 5.7, 6.4, 8.1, 8.2_

- [x] 16. Add integration tests for timing criteria
  - [x] 16.1 Write integration tests for timing and resolution
    - In `tests/test_integration_timing.py`, with the mock LLM client and 1-3 representative examples
      each: interpretation completes within its budget (2.1); generated prompt renders promptly after
      production (4.3); revision completes within its budget (5.2); the in-progress indicator appears
      quickly and clears on resolution (6.3); retry re-executes and resolves within budget (8.3);
      new-session view clears within its budget (7.5)
    - _Requirements: 2.1, 4.3, 5.2, 6.3, 7.5, 8.3_

- [x] 17. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; they are the property,
  unit, and integration tests.
- Each task references specific requirements for traceability.
- Checkpoints ensure incremental validation.
- Property tests validate the 19 universal correctness properties from the design, each as a single
  Hypothesis test with `max_examples >= 100`, tagged
  `# Feature: conversational-prompt-builder, Property {number}: {property text}`, placed next to the
  code it checks, and using the mock LLM client where generation is involved.
- Unit tests validate specific examples, UI presence, and error paths (using Streamlit `AppTest`
  where helpful); integration tests cover the timing criteria.
- The LLM client (`prompt_builder/llm_client.py`) is the only module that touches the network; all
  deterministic logic stays in pure Python functions kept out of the Streamlit script.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "2.1", "2.2", "3.1"] },
    { "id": 2, "tasks": ["2.3", "3.2", "4.1", "6.1", "10.1", "10.2", "14.1"] },
    { "id": 3, "tasks": ["3.3", "4.2", "6.2", "7.1", "8.1", "12.1", "14.2"] },
    { "id": 4, "tasks": ["4.3", "6.3", "7.2", "8.2", "11.1", "12.2", "14.3"] },
    { "id": 5, "tasks": ["4.4", "6.4", "7.3", "8.3", "11.2"] },
    { "id": 6, "tasks": ["8.4", "11.3", "15.1"] },
    { "id": 7, "tasks": ["8.5", "15.2"] },
    { "id": 8, "tasks": ["8.6", "15.3"] },
    { "id": 9, "tasks": ["8.7", "15.4", "16.1"] }
  ]
}
```
