"""Core data models for the Conversational Prompt Builder.

These dataclasses live in Streamlit's ``session_state`` for the duration of a
single interaction; there is no database or server-side persistence. They match
the design's Data Models section exactly.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

# A named attribute of the prompt (audience, tone, format, length, domain, ...)
ParameterName = str


@dataclass
class ParameterDefinition:
    name: ParameterName
    required: bool
    default_value: Optional[str] = None
    # Optional presentation metadata. Appended with defaults so existing callers
    # and positional construction remain backward compatible.
    label: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None


@dataclass
class ParameterValue:
    name: ParameterName
    value: str
    status: Literal["collected", "declined"]
    attempts: int


@dataclass
class Overwrite:
    parameter: ParameterName
    previous: str
    next: str


# The accumulated details for a session (Req 1.3, 2.2)
@dataclass
class PromptSpecification:
    goal_confidence: float
    goal: Optional[str] = None
    parameters: dict[ParameterName, ParameterValue] = field(default_factory=dict)
    overwrite_log: list[Overwrite] = field(default_factory=list)
    # Adaptive prompt controls are optional and defaulted for compatibility with
    # sessions and tests created before the richer brief model existed.
    prompt_depth: str = "adaptive"
    task_type: Optional[str] = None
    resolved_depth: Optional[str] = None


MessageSender = Literal["user", "system"]


@dataclass
class Message:
    id: str
    sender: MessageSender
    text: str
    created_at: float


@dataclass
class GeneratedPrompt:
    text: str
    omitted_required: list[ParameterName]


# A single interaction, held in session_state only (Req 7)
@dataclass
class Session:
    id: str
    messages: list[Message] = field(default_factory=list)
    spec: Optional[PromptSpecification] = None
    follow_up_count: int = 0
    generated_prompt: Optional[GeneratedPrompt] = None
    context_available: bool = True
