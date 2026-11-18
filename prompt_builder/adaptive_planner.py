"""Deterministic task classification, depth planning, and brief coverage.

The planner performs no I/O and uses transparent keyword/length rules. It keeps
question selection predictable while allowing the LLM to do fuzzy extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from prompt_builder import catalog
from prompt_builder.models import PromptSpecification

TaskType = Literal["software", "marketing", "learning", "video", "summarize", "rewrite", "general"]
ResolvedDepth = Literal["focused", "detailed", "comprehensive"]
RequestedDepth = Literal["adaptive", "focused", "detailed", "comprehensive"]

DEPTH_OPTIONS: tuple[str, ...] = ("adaptive", "focused", "detailed", "comprehensive")

_PROFILES: dict[TaskType, tuple[str, ...]] = {
    "software": (
        "objective", "audience", "platform", "tech_stack", "feature_scope",
        "user_roles", "data_integrations", "format", "constraints",
        "security_compliance", "accessibility", "nonfunctional", "deployment",
        "success_criteria", "edge_cases", "self_check", "model_settings",
    ),
    "marketing": (
        "objective", "product", "audience", "offer", "channel", "tone",
        "voice", "call_to_action", "proof_points", "input_material", "format",
        "length", "exclusions", "security_compliance", "success_criteria",
        "factuality", "examples", "self_check",
    ),
    "learning": (
        "objective", "learner_level", "audience", "input_material", "duration",
        "format", "assessment", "background_context", "language", "reading_level",
        "factuality", "citation_expectations", "success_criteria", "examples", "self_check",
    ),
    "video": (
        "objective", "platform", "audience", "duration", "input_material",
        "aspect_ratio", "pacing_style", "captions_audio", "call_to_action",
        "export_specs", "format", "constraints", "success_criteria", "examples", "self_check",
    ),
    "summarize": (
        "objective", "input_material", "audience", "tone", "length", "format",
        "factuality", "citation_expectations",
    ),
    "rewrite": (
        "objective", "input_material", "audience", "tone", "voice", "length",
        "format", "must_include", "exclusions", "factuality",
    ),
    "general": (
        "objective", "audience", "background_context", "role", "input_material",
        "deliverable", "format", "tone", "length", "platform", "must_include",
        "exclusions", "constraints", "examples", "success_criteria", "factuality",
        "citation_expectations", "edge_cases", "self_check", "model_settings",
    ),
}

_DEPTH_BUDGET = {"focused": 5, "detailed": 10, "comprehensive": 100}

_TASK_KEYWORDS: tuple[tuple[TaskType, tuple[str, ...]], ...] = (
    ("summarize", ("summarize", "summarise", "summary", "condense", "key points", "tl;dr")),
    ("rewrite", ("rewrite", "rephrase", "proofread", "edit this", "improve this copy")),
    ("software", ("software", "website", "web app", "mobile app", "api", "code", "database", "lms", "saas", "build an app", "implement")),
    ("marketing", ("marketing", "campaign", "launch email", "ad copy", "sales page", "product description", "conversion", "cta", "brand")),
    ("learning", ("study plan", "lesson", "curriculum", "learn", "teaching", "study notes", "course outline", "quiz")),
    ("video", ("video", "youtube", "reels", "tiktok", "shorts", "footage", "aspect ratio", "captions")),
)

_COMPLEX_MARKERS = (
    "production", "comprehensive", "complete", "end-to-end", "architecture",
    "authentication", "permissions", "database", "integration", "security",
    "accessibility", "deployment", "acceptance criteria", "compliance", "schema",
)
_DETAILED_MARKERS = (
    "detailed", "plan", "strategy", "multiple", "step-by-step", "table",
    "examples", "sources", "citations", "timeline", "assessment",
)

_FIELD_GUIDANCE: dict[str, tuple[str, str]] = {
    "audience": ("Who is the audience, and what is their expertise level?", "Audience and expertise determine terminology, examples, and assumed knowledge."),
    "platform": ("Where will this be used or shipped?", "The target platform changes format, constraints, and implementation choices."),
    "tech_stack": ("Do you have a preferred stack or tools?", "Stack choices make the result implementable instead of generic."),
    "feature_scope": ("What must be in the first version, and what is out of scope?", "A scope boundary prevents overbuilding and exposes the core workflow."),
    "user_roles": ("Which user roles and permissions are needed?", "Roles determine authorization rules and user journeys."),
    "data_integrations": ("What data, entities, APIs, or integrations are involved?", "Data boundaries shape architecture, validation, and failure handling."),
    "product": ("What product or service are you promoting?", "Product specifics anchor benefits and prevent generic copy."),
    "offer": ("What is the offer or value proposition?", "The offer determines the message hierarchy and persuasion strategy."),
    "channel": ("Which channel and placement is this for?", "Channel constraints affect length, format, and calls to action."),
    "learner_level": ("What does the learner already know?", "Starting level controls pacing, vocabulary, and prerequisite assumptions."),
    "input_material": ("What source material or input should the model use?", "Source material defines the facts to preserve and can be supplied later as {{SOURCE_MATERIAL}}."),
    "duration": ("What duration or time budget should the result fit?", "A time target keeps scope and pacing realistic."),
    "assessment": ("How should understanding or success be assessed?", "An assessment makes the learning outcome observable."),
    "aspect_ratio": ("What aspect ratio and dimensions are required?", "Canvas constraints affect framing, text safety, and export settings."),
    "pacing_style": ("What pacing and visual style should the edit use?", "Pacing and style determine shot cadence and retention choices."),
    "captions_audio": ("What are the caption, voice, music, and audio requirements?", "Audio and captions affect accessibility and platform readiness."),
    "export_specs": ("What final export specifications are required?", "Export settings prevent an otherwise good edit from being unusable."),
    "tone": ("What tone should the response use?", "Tone aligns the response with the audience and context."),
    "length": ("How concise or detailed should the target response be?", "Semantic length guidance controls useful detail; max_tokens is only a runtime cap."),
    "format": ("What output format or schema do you need?", "A clear output contract makes the response ready to use."),
    "factuality": ("How should uncertain facts, sources, or citations be handled?", "Accuracy rules prevent invented details and define uncertainty behavior."),
    "success_criteria": ("What measurable conditions define a successful result?", "Acceptance criteria let the model verify completeness before answering."),
}


@dataclass(frozen=True)
class TaskPlan:
    task_type: TaskType
    requested_depth: RequestedDepth
    depth: ResolvedDepth
    relevant_fields: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class CoverageItem:
    name: str
    label: str
    status: Literal["captured", "missing", "declined", "optional"]


def normalize_depth(value: Optional[str]) -> RequestedDepth:
    normalized = (value or "adaptive").strip().casefold()
    return normalized if normalized in DEPTH_OPTIONS else "adaptive"  # type: ignore[return-value]


def classify_task_type(text: str) -> TaskType:
    """Classify a request using ordered, visible keyword rules."""
    normalized = " ".join((text or "").casefold().split())
    for task_type, keywords in _TASK_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return task_type
    return "general"


def classify_complexity(text: str, task_type: Optional[TaskType] = None) -> ResolvedDepth:
    """Resolve adaptive depth from task type, explicit scope markers, and length."""
    normalized = " ".join((text or "").casefold().split())
    kind = task_type or classify_task_type(normalized)
    if any(word in normalized for word in ("brief", "quick", "simple", "concise")) and not any(
        word in normalized for word in _COMPLEX_MARKERS
    ):
        return "focused"
    complex_hits = sum(marker in normalized for marker in _COMPLEX_MARKERS)
    detailed_hits = sum(marker in normalized for marker in _DETAILED_MARKERS)
    if kind == "software" and (complex_hits >= 1 or len(normalized.split()) >= 12):
        return "comprehensive"
    if complex_hits >= 2 or len(normalized.split()) >= 45:
        return "comprehensive"
    if kind in {"summarize", "rewrite"} and complex_hits == 0 and detailed_hits <= 1:
        return "focused"
    if len(normalized.split()) <= 7 and complex_hits == 0 and detailed_hits == 0:
        return "focused"
    if detailed_hits or kind in {"marketing", "learning", "video"} or len(normalized.split()) >= 18:
        return "detailed"
    return "focused"


def plan_prompt(
    spec: PromptSpecification,
    *,
    request_text: Optional[str] = None,
    requested_depth: Optional[str] = None,
) -> TaskPlan:
    """Build the relevance/depth plan for a specification without mutating it."""
    source = request_text or spec.goal or ""
    inferred_type = classify_task_type(source)
    task_type: TaskType = (
        spec.task_type if spec.task_type in _PROFILES and request_text is None else inferred_type
    )  # type: ignore[assignment]
    requested = normalize_depth(requested_depth or spec.prompt_depth)
    if requested == "adaptive":
        if source.strip():
            depth: ResolvedDepth = classify_complexity(source, task_type)
        elif spec.resolved_depth in {"focused", "detailed", "comprehensive"}:
            depth = spec.resolved_depth  # type: ignore[assignment]
        else:
            depth = classify_complexity(source, task_type)
    else:
        depth = requested  # type: ignore[assignment]
    profile = _PROFILES[task_type]
    relevant = profile[: _DEPTH_BUDGET[depth]]
    return TaskPlan(
        task_type=task_type,
        requested_depth=requested,
        depth=depth,
        relevant_fields=tuple(relevant),
        rationale=f"{task_type.title()} task; {depth} coverage selected from request scope.",
    )


def relevant_missing_fields(spec: PromptSpecification, plan: TaskPlan) -> list[str]:
    """Return relevant fields not collected or explicitly declined, in plan order."""
    missing: list[str] = []
    for name in plan.relevant_fields:
        if name == "objective" and spec.goal and spec.goal.strip():
            continue
        value = spec.parameters.get(name)
        if value is None:
            missing.append(name)
        elif value.status != "collected" and value.attempts < 3:
            missing.append(name)
    return missing


def should_ask_follow_up(spec: PromptSpecification, plan: TaskPlan) -> bool:
    """Return whether more questioning materially improves the selected depth.

    Focused prompts stop once the objective plus one relevant detail are known;
    remaining details can become run-time placeholders. Detailed and
    comprehensive prompts continue through their selected relevance profile.
    """
    missing = relevant_missing_fields(spec, plan)
    if not missing:
        return False
    if plan.depth != "focused":
        return True
    resolved = 1 if spec.goal and spec.goal.strip() else 0
    for name in plan.relevant_fields:
        if name == "objective":
            continue
        value = spec.parameters.get(name)
        if value is not None and (value.status == "collected" or value.attempts >= 3):
            resolved += 1
    return resolved < 2


def coverage_items(spec: PromptSpecification, plan: TaskPlan) -> tuple[CoverageItem, ...]:
    """Describe captured, missing, and declined relevant brief fields."""
    items: list[CoverageItem] = []
    for name in plan.relevant_fields:
        value = spec.parameters.get(name)
        if name == "objective" and spec.goal and spec.goal.strip():
            status = "captured"
        elif value is None:
            status = "missing"
        elif value.status == "collected":
            status = "captured"
        else:
            status = "declined"
        items.append(CoverageItem(name, catalog.parameter_label(name), status))
    # Catalog fields outside the task plan remain available but are not forced.
    for name, value in spec.parameters.items():
        if name not in plan.relevant_fields and value.status == "collected":
            items.append(CoverageItem(name, catalog.parameter_label(name), "optional"))
    return tuple(items)


def field_question(name: str) -> tuple[str, str]:
    """Return a plain-language question and why the field matters."""
    if name in _FIELD_GUIDANCE:
        return _FIELD_GUIDANCE[name]
    definition = catalog.get_definition(name)
    label = catalog.parameter_label(name)
    description = definition.description if definition and definition.description else "It improves the final prompt."
    return f"What should I know about {label.casefold()}?", description
