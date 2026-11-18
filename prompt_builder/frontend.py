"""Pure frontend helpers for the professional Builder experience.

This module contains deterministic view-model logic only. It does not import
Streamlit, touch session state, call the network, or persist user content.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from prompt_builder import catalog
from prompt_builder.models import GeneratedPrompt, Session


@dataclass(frozen=True)
class StepState:
    """One accessible stage in the four-step Builder workflow."""

    key: str
    label: str
    status: str  # "completed", "active", or "upcoming"
    state_label: str


@dataclass(frozen=True)
class QualityDimension:
    """A transparent, binary quality check and its improvement guidance."""

    key: str
    label: str
    passed: bool
    evidence: str
    improvement_hint: str

    @property
    def status_label(self) -> str:
        return "Included" if self.passed else "Needs attention"


@dataclass(frozen=True)
class QualityScore:
    """Deterministic quality coverage score for generated prompt text."""

    overall: int
    passed: int
    total: int
    dimensions: tuple[QualityDimension, ...]


@dataclass(frozen=True)
class ExportFile:
    """One downloadable representation of a generated prompt."""

    label: str
    file_name: str
    mime: str
    data: str


def _required_collected(session: Session) -> bool:
    spec = session.spec
    if spec is None:
        return False
    for name in catalog.required_parameter_names():
        value = spec.parameters.get(name)
        if value is None or value.status != "collected":
            return False
    return True


def derive_stepper(session: Optional[Session]) -> tuple[StepState, ...]:
    """Derive Idea → Details → Generate → Review from current Session data.

    A declined required parameter still leaves the Details stage active because
    the deterministic business rule considers only collected required values
    complete. If a prompt is generated early, all prior stages are shown as
    complete and Review becomes active because that is the user's current task.
    """

    has_idea = bool(session and session.messages)
    has_prompt = bool(session and session.generated_prompt)
    details_complete = bool(session and _required_collected(session))

    if has_prompt:
        active_index = 3
    elif not has_idea:
        active_index = 0
    elif not details_complete:
        active_index = 1
    else:
        active_index = 2

    definitions = (
        ("idea", "Idea"),
        ("details", "Details"),
        ("generate", "Generate"),
        ("review", "Review"),
    )
    states: list[StepState] = []
    for index, (key, label) in enumerate(definitions):
        status = "completed" if index < active_index else "active" if index == active_index else "upcoming"
        state_label = {"completed": "Completed", "active": "Current step", "upcoming": "Not started"}[status]
        states.append(StepState(key, label, status, state_label))
    return tuple(states)


def template_categories(examples: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return stable, alphabetized category options with ``All templates`` first."""

    categories = sorted({str(example.get("category", "Other")).strip() or "Other" for example in examples})
    return ("All templates", *categories)


def filter_templates(
    examples: Sequence[Mapping[str, Any]],
    query: str = "",
    category: str = "All templates",
) -> list[Mapping[str, Any]]:
    """Filter templates by category and case-insensitive text search.

    Search covers title, compact description, request, category, and goal. Input
    order is preserved so the curated gallery order remains stable.
    """

    normalized_query = " ".join(query.casefold().split())
    normalized_category = category.casefold().strip()
    matches: list[Mapping[str, Any]] = []
    for example in examples:
        example_category = str(example.get("category", "Other")).strip() or "Other"
        if normalized_category not in ("", "all templates") and example_category.casefold() != normalized_category:
            continue
        haystack = " ".join(
            str(example.get(field, ""))
            for field in ("title", "description", "request", "category", "goal")
        ).casefold()
        if normalized_query and normalized_query not in " ".join(haystack.split()):
            continue
        matches.append(example)
    return matches


def _has_heading(text: str, words: Iterable[str]) -> bool:
    lowered = text.casefold()
    return any(
        re.search(rf"(?m)^\s*#{{0,6}}\s*{re.escape(word.casefold())}\b", lowered)
        for word in words
    )


def score_prompt_quality(text: str, *, depth: Optional[str] = None) -> QualityScore:
    """Score structural coverage; this heuristic does not verify factual truth.

    Legacy callers receive the original seven dimensions. When a resolved prompt
    depth is supplied, focused prompts are not penalized for comprehensive-only
    ceremony, while detailed/comprehensive prompts receive richer checks.
    """

    source = text or ""
    lowered = source.casefold()
    heading_count = len(re.findall(r"(?m)^\s*#{1,6}\s+\S", source))
    list_count = len(re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+\S", source))

    checks = (
        QualityDimension(
            "context",
            "Context",
            _has_heading(source, ("context", "background", "role", "audience", "objective", "inputs")),
            "A Context, Background, Role, Audience, Objective, or Inputs section is present.",
            "Add the objective, source inputs, situation, audience, or model role.",
        ),
        QualityDimension(
            "specificity",
            "Specificity",
            bool(re.search(r"\{\{[^{}]+\}\}|\b\d+(?:[-–]\d+)?\b|\b(?:must|exactly|at least|at most)\b", lowered)),
            "The prompt includes a variable, measurable limit, or explicit requirement.",
            "Add concrete inputs, measurable limits, or reusable {{VARIABLES}}.",
        ),
        QualityDimension(
            "constraints",
            "Constraints",
            _has_heading(source, ("constraints", "rules", "requirements")) or "do not" in lowered,
            "A Constraints/Rules section or an explicit 'do not' rule is present.",
            "State the important boundaries and what the response must avoid.",
        ),
        QualityDimension(
            "structure",
            "Output structure",
            _has_heading(source, ("output", "output contract", "format", "deliverable")) and (heading_count >= 2 or list_count >= 2),
            "An output/format section and a multi-part structure are present.",
            "Define the output format and list the sections in the required order.",
        ),
        QualityDimension(
            "examples",
            "Examples",
            _has_heading(source, ("example", "examples", "few-shot", "worked example")) or "for example" in lowered,
            "An example or few-shot section is present.",
            "Include one short example that demonstrates the expected level and style.",
        ),
        QualityDimension(
            "quality_check",
            "Quality check",
            _has_heading(source, ("self-check", "quality check", "definition of done", "acceptance criteria")),
            "A self-check, definition of done, or acceptance criteria section is present.",
            "Add a short self-check or definition of done before the final response.",
        ),
        QualityDimension(
            "model_settings",
            "Model settings",
            _has_heading(source, ("recommended model settings", "model settings")) or (
                "temperature" in lowered and ("max tokens" in lowered or "top-p" in lowered)
            ),
            "Recommended inference settings are present.",
            "Add optional temperature and max-token guidance when it helps reuse.",
        ),
    )
    if depth is not None:
        normalized_depth = depth.casefold()
        richer = (
            QualityDimension(
                "factuality",
                "Factuality behavior",
                bool(re.search(r"\b(?:do not invent|uncertain|verify|cite|source|factual)\b", lowered)),
                "The prompt defines how facts, uncertainty, or sources are handled.",
                "Tell the model not to invent facts and how to handle uncertainty or citations.",
            ),
            QualityDimension(
                "missing_information",
                "Missing information",
                _has_heading(source, ("if information is missing", "missing information", "assumptions")),
                "The prompt defines assumptions or missing-information behavior.",
                "State when to make a safe assumption and when to ask a clarifying question.",
            ),
        )
        if normalized_depth == "focused":
            wanted = {"context", "specificity", "constraints", "structure", "quality_check"}
            checks = tuple(check for check in checks if check.key in wanted)
        elif normalized_depth == "detailed":
            checks = tuple(check for check in checks if check.key != "model_settings") + richer[:1]
        elif normalized_depth == "comprehensive":
            checks = checks + richer

    passed = sum(dimension.passed for dimension in checks)
    overall = round((passed / len(checks)) * 100) if checks else 0
    return QualityScore(overall, passed, len(checks), checks)


def _is_sensitive_export_key(key: Any) -> bool:
    """Return whether a metadata key commonly identifies a credential.

    Export callers should pass only display-safe metadata. This final guard keeps
    accidentally supplied credentials out of JSON without treating benign model
    controls such as ``max_tokens`` as secrets.
    """

    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
    return normalized in {
        "api_key",
        "apikey",
        "authorization",
        "client_secret",
        "password",
        "refresh_token",
        "access_token",
    }


def _without_secrets(value: Any) -> Any:
    """Recursively remove credential-labelled fields from export metadata."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_secrets(item)
            for key, item in value.items()
            if not _is_sensitive_export_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_without_secrets(item) for item in value]
    return value


def prompt_export_payload(
    prompt: GeneratedPrompt,
    *,
    settings: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the structured, privacy-conscious JSON export payload.

    Prompt text is exported intentionally. Metadata and settings are scrubbed of
    credential-labelled fields as a defense in depth measure; the app itself
    passes only the six public generation controls and non-secret source/quality
    information.
    """

    return {
        "prompt": prompt.text,
        "omitted_required": list(prompt.omitted_required),
        "metadata": _without_secrets(dict(metadata or {})),
        "settings": _without_secrets(dict(settings or {})),
    }


def build_prompt_exports(
    prompt: GeneratedPrompt,
    *,
    settings: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    stem: str = "generated-prompt",
) -> tuple[ExportFile, ...]:
    """Build Markdown, plain-text, and structured JSON downloads."""

    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", stem).strip("-") or "generated-prompt"
    payload = prompt_export_payload(prompt, settings=settings, metadata=metadata)
    return (
        ExportFile("Markdown", f"{safe_stem}.md", "text/markdown", prompt.text),
        ExportFile("Plain text", f"{safe_stem}.txt", "text/plain", prompt.text),
        ExportFile(
            "JSON",
            f"{safe_stem}.json",
            "application/json",
            json.dumps(payload, indent=2, ensure_ascii=False),
        ),
    )


def settings_metadata(settings: Any) -> dict[str, Any]:
    """Convert the GenerationSettings dataclass to JSON-safe export metadata."""

    return asdict(settings)
