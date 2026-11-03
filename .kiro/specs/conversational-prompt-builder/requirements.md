# Requirements Document

## Introduction

The Conversational Prompt Builder is an application that helps users craft high-quality prompts through a guided, multi-turn conversation. Rather than requiring users to write a well-structured prompt themselves, the application interviews the user, progressively gathers the intent, context, constraints, and parameters relevant to what the user wants to produce, and then synthesizes a refined, high-quality prompt tailored to the user's goal. The application presents a simple, uncluttered interface focused on the conversation and the resulting prompt.

This document defines the requirements for the feature. Note: the first deliverable requested by the user is a `README.md` file describing the project; project implementation follows later. The README content is derived from these requirements.

## Glossary

- **System**: The Conversational Prompt Builder application as a whole.
- **Conversation_Manager**: The component that manages the multi-turn dialogue state and sequencing of exchanges between the user and the System.
- **Intent_Analyzer**: The component that interprets user messages to determine the user's goal and identify missing information.
- **Elicitation_Engine**: The component that generates follow-up questions to collect additional details and parameters from the user.
- **Prompt_Generator**: The component that synthesizes the final prompt from the collected information.
- **Session**: A single continuous interaction between a user and the System, from first message to prompt generation or termination.
- **Prompt_Specification**: The accumulated structured set of collected details, context, constraints, and parameters gathered during a Session.
- **Generated_Prompt**: The final prompt text produced by the Prompt_Generator.
- **User_Message**: A single input submitted by the user during a Session.
- **System message**: A single message produced by the System and displayed to the user during a Session, including acknowledgements, follow-up questions, and generated content.
- **Parameter**: A discrete attribute of the Prompt_Specification (for example, target audience, tone, output format, length, or domain).
- **Acceptance threshold**: The minimum confidence level the Intent_Analyzer must assign to an identified goal for that goal to be treated as confirmed.

## Requirements

### Requirement 1: Start a Conversation Session

**User Story:** As a user, I want to start a conversation with the application, so that I can begin describing what I want to build without preparing a formal prompt in advance.

#### Acceptance Criteria

1. WHEN a user opens the application, THE System SHALL display an interface containing a free-text input field that accepts a User_Message of 1 to 10,000 characters.
2. WHEN a user submits the first non-empty User_Message, THE Conversation_Manager SHALL create a new Session, associate the submitted User_Message with that Session, and display an acknowledgement that the Session has started.
3. WHEN a new Session is created, THE System SHALL initialize an empty Prompt_Specification containing no content for that Session.
4. IF a user submits a User_Message that is empty or contains only whitespace characters, THEN THE System SHALL reject the submission, retain any entered text in the input field, and display an error message indicating that non-empty input is required.
5. IF a user submits a User_Message exceeding 10,000 characters, THEN THE System SHALL reject the submission, retain the entered text in the input field, and display an error message indicating the maximum allowed character count.

### Requirement 2: Interpret User Intent

**User Story:** As a user, I want the application to understand what I am trying to create, so that it can ask relevant questions instead of generic ones.

#### Acceptance Criteria

1. WHEN a User_Message is submitted, THE Intent_Analyzer SHALL interpret the User_Message within 5 seconds to identify the user's stated goal.
2. WHEN the Intent_Analyzer interprets a User_Message, THE Intent_Analyzer SHALL update the Prompt_Specification with each piece of information extracted from the User_Message that maps to a defined Parameter.
3. IF the Intent_Analyzer extracts information that conflicts with a value already stored in the Prompt_Specification, THEN THE Intent_Analyzer SHALL retain the most recently extracted value and record that the earlier value was overwritten.
4. WHEN the Intent_Analyzer updates the Prompt_Specification, THE Intent_Analyzer SHALL produce the list of required Parameters that have no stored value as the set of still-missing Parameters.
5. IF the Intent_Analyzer assigns a confidence level below the acceptance threshold to the identified goal after interpreting all available User_Messages, THEN THE Elicitation_Engine SHALL ask the user to describe the goal and SHALL leave the Prompt_Specification goal field unset until a goal is confirmed.

### Requirement 3: Multi-Turn Elicitation of Details

**User Story:** As a user, I want the application to ask me follow-up questions, so that it can collect the details and parameters needed to produce a strong prompt.

#### Acceptance Criteria

1. WHILE one or more required Parameters are missing from the Prompt_Specification, THE Elicitation_Engine SHALL present a follow-up question to the user.
2. WHEN the Elicitation_Engine presents a follow-up question, THE Elicitation_Engine SHALL request information for a minimum of 1 and a maximum of 3 missing Parameters.
3. WHEN the user responds to a follow-up question, THE Conversation_Manager SHALL record the response in the Session and update the Prompt_Specification.
4. WHEN the user response to a follow-up question provides a value for a Parameter, THE System SHALL mark that Parameter as collected.
5. WHERE a Parameter has a default value, THE Elicitation_Engine SHALL offer the default value as a suggested answer in the follow-up question.
6. WHEN the user response to a follow-up question is empty or consists only of a skip indication, THE Elicitation_Engine SHALL treat the requested Parameter as declined and proceed to the next missing Parameter without presenting the same question again in the current Session.
7. IF the user response does not provide a usable value for any requested Parameter and is not a skip indication, THEN THE Elicitation_Engine SHALL re-present a follow-up question for the same Parameter together with an indication that the previous response could not be used, for a maximum of 3 attempts per Parameter.
8. WHILE the total number of follow-up questions presented in the current Session is less than 20, THE Elicitation_Engine SHALL continue presenting follow-up questions for missing required Parameters, and upon reaching 20 THE Elicitation_Engine SHALL stop presenting further follow-up questions.

### Requirement 4: Generate the High-Quality Prompt

**User Story:** As a user, I want the application to generate a high-quality prompt from my answers, so that the prompt produces the result I want.

#### Acceptance Criteria

1. WHEN all required Parameters in the Prompt_Specification are collected, THE Prompt_Generator SHALL produce a Generated_Prompt from the Prompt_Specification.
2. WHEN the Prompt_Generator produces a Generated_Prompt, THE Prompt_Generator SHALL include every collected Parameter from the Prompt_Specification in the Generated_Prompt, with no collected Parameter omitted.
3. WHEN a Generated_Prompt is produced, THE System SHALL display the complete Generated_Prompt to the user within 2 seconds of production.
4. WHEN a user requests prompt generation before all required Parameters are collected, THE Prompt_Generator SHALL produce a Generated_Prompt using the collected Parameters and SHALL display to the user a list identifying each required Parameter that was omitted.
5. IF the Prompt_Generator fails to produce a Generated_Prompt from the Prompt_Specification, THEN THE System SHALL display an error message indicating that generation failed, SHALL retain the collected Parameters in the Prompt_Specification, and SHALL allow the user to retry generation.

### Requirement 5: Review and Refine the Generated Prompt

**User Story:** As a user, I want to review and adjust the generated prompt, so that I can correct or refine it before using it.

#### Acceptance Criteria

1. WHEN a Generated_Prompt is displayed, THE System SHALL provide a control that allows the user to request a revision.
2. WHEN a user submits a revision request, THE Conversation_Manager SHALL record the revision request in the Session and THE Prompt_Generator SHALL produce an updated Generated_Prompt that incorporates the revision request within 10 seconds.
3. IF a user submits a revision request that is empty or exceeds 2000 characters, THEN THE System SHALL reject the request, display an error message indicating the allowed length, and retain the current Generated_Prompt unchanged.
4. IF the Prompt_Generator fails to produce an updated Generated_Prompt, THEN THE System SHALL display an error message indicating the revision could not be completed and retain the previous Generated_Prompt unchanged.
5. WHEN a Generated_Prompt is displayed, THE System SHALL provide a control that copies the Generated_Prompt text to the user's clipboard.
6. WHEN a user copies the Generated_Prompt, THE System SHALL display a confirmation indication within 1 second of the copy action completing.
7. IF the copy action fails, THEN THE System SHALL display an error message indicating the copy did not complete.

### Requirement 6: Simple User Interface

**User Story:** As a user, I want a simple and clear interface, so that I can focus on the conversation without distraction.

#### Acceptance Criteria

1. WHEN a Session is active, THE System SHALL display the conversation history for the current Session in chronological order from oldest message at the top to newest message at the bottom.
2. THE System SHALL visually distinguish User_Messages from System messages in the conversation history using at least one persistent visual attribute (such as alignment, label, or background treatment) applied consistently to every message of the same type.
3. WHEN the System begins processing a User_Message, THE System SHALL display an in-progress indicator within 500 milliseconds and SHALL continue displaying it until the System response is rendered or an error is shown.
4. IF processing a User_Message fails, THEN THE System SHALL remove the in-progress indicator and display an error indication stating that the response could not be generated, while retaining the submitted User_Message in the conversation history.
5. WHEN the user activates the new-Session control, THE System SHALL start a new empty Session and clear the displayed conversation history of the previous Session.

### Requirement 7: Session Persistence Within an Interaction

**User Story:** As a user, I want my conversation context to be retained throughout the session, so that I do not have to repeat information I already provided.

#### Acceptance Criteria

1. WHILE a Session is active, THE Conversation_Manager SHALL retain all User_Messages and System messages for that Session in submission order, up to a maximum of 200 messages.
2. IF the number of messages in an active Session reaches 200, THEN THE Conversation_Manager SHALL retain the 200 most recent messages and discard the oldest messages beyond that limit.
3. WHEN a User_Message is submitted during an active Session, THE Intent_Analyzer SHALL use the retained Session content as context for interpretation.
4. IF the retained Session content cannot be accessed during interpretation, THEN THE Intent_Analyzer SHALL interpret the current User_Message in isolation and THE System SHALL display an indication that prior context was unavailable.
5. WHEN a user starts a new Session, THE System SHALL clear the Prompt_Specification and conversation history of the previous Session from the active view within 2 seconds.

### Requirement 8: Error Handling

**User Story:** As a user, I want clear feedback when something goes wrong, so that I know how to proceed.

#### Acceptance Criteria

1. IF the System fails to generate a response to a User_Message, THEN THE System SHALL display an error message indicating that response generation failed, SHALL retain the current Session content without modification, and SHALL provide a control to retry the failed operation.
2. IF the Prompt_Generator fails to produce a Generated_Prompt, THEN THE System SHALL display an error message indicating that prompt generation failed, SHALL retain the current Session content without modification, and SHALL provide a control that allows the user to retry generation.
3. WHEN the user activates the retry control for a failed operation, THE System SHALL re-execute the failed operation and SHALL display a result or error message within 30 seconds.
4. IF a failed operation is retried and continues to fail after 3 consecutive retry attempts, THEN THE System SHALL display an error message indicating that the operation could not be completed and SHALL retain the current Session content without modification.
