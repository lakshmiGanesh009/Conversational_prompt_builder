"""Pure route and shell-action helpers for the Streamlit product UI.

The app reruns from top to bottom after every interaction.  These helpers keep
route selection deterministic and ensure navigation changes only shell state,
never the in-memory conversation Session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from prompt_builder import elicitation_engine, frontend
from prompt_builder.models import PromptSpecification, Session

ROUTE_STATE_KEY = "route"
TEMPLATE_PREVIEW_KEY = "template_preview"
TEMPLATE_SEARCH_KEY = "template_search"
TEMPLATE_CATEGORY_KEY = "template_category"


@dataclass(frozen=True)
class Page:
    """Stable metadata for one top-level product page."""

    key: str
    label: str
    icon: str
    heading: str
    subheading: str

    @property
    def navigation_label(self) -> str:
        """Return the visible icon-plus-text label used in sidebar navigation."""

        return f"{self.icon}  {self.label}"


PAGES: tuple[Page, ...] = (
    Page(
        "Home",
        "Home",
        "🏠",
        "Prompt workspace",
        "Start with an idea, add the details that matter, and leave with a prompt ready to use.",
    ),
    Page(
        "Builder",
        "Builder",
        "✍️",
        "Prompt Builder",
        "Describe the outcome you need, answer focused questions, and review the generated prompt.",
    ),
    Page(
        "Templates",
        "Templates",
        "🗂️",
        "Prompt templates",
        "Preview six complete, no-AI examples or use one as the current review session.",
    ),
    Page(
        "Observability",
        "Observability",
        "📊",
        "Observability",
        "Inspect in-memory model-call metrics, service objectives, exports, and sample traffic.",
    ),
    Page(
        "Guide",
        "Guide",
        "📖",
        "Practical guide",
        "Learn what strengthens a prompt, how controls differ, and how to recover from common errors.",
    ),
)

PAGE_BY_KEY: Mapping[str, Page] = {page.key: page for page in PAGES}
DEFAULT_PAGE_KEY = "Home"
HOME_PAGE_KEY = "Home"
BUILDER_PAGE_KEY = "Builder"
TEMPLATES_PAGE_KEY = "Templates"
OBSERVABILITY_PAGE_KEY = "Observability"
GUIDE_PAGE_KEY = "Guide"


@dataclass(frozen=True)
class ResumeState:
    """Compact, content-free summary of an in-memory Builder Session."""

    available: bool
    current_step: str
    captured_details: int
    has_prompt: bool
    message_count: int


def resolve_page(value: Any) -> Page:
    """Resolve a route key or label, falling back safely to Home.

    Values are matched case-insensitively so older session-state values remain
    useful after a rerun or upgrade. Unknown, empty, and non-string values never
    escape into the renderer.
    """

    if isinstance(value, Page):
        return PAGE_BY_KEY.get(value.key, PAGE_BY_KEY[DEFAULT_PAGE_KEY])
    if isinstance(value, str):
        normalized = value.strip().casefold()
        for page in PAGES:
            if normalized in {page.key.casefold(), page.label.casefold()}:
                return page
    return PAGE_BY_KEY[DEFAULT_PAGE_KEY]


def initial_page_key(session: Optional[Session]) -> str:
    """Open a meaningful existing Session in Builder; otherwise land on Home."""

    if session is not None and (
        bool(session.messages)
        or session.generated_prompt is not None
        or bool(session.spec and (session.spec.goal or session.spec.parameters))
    ):
        return BUILDER_PAGE_KEY
    return DEFAULT_PAGE_KEY


def navigate(
    state: MutableMapping[str, Any],
    destination: Any,
    *,
    route_key: str = ROUTE_STATE_KEY,
) -> Page:
    """Update only the current route, preserving Session and all other state."""

    page = resolve_page(destination)
    state[route_key] = page.key
    return page


def resume_state(session: Optional[Session]) -> ResumeState:
    """Describe useful progress without exposing message or prompt content."""

    if session is None:
        return ResumeState(False, "Idea", 0, False, 0)
    captured = 0
    if session.spec is not None:
        captured = int(bool(session.spec.goal)) + sum(
            value.status == "collected" and bool(value.value.strip())
            for value in session.spec.parameters.values()
        )
    available = bool(
        session.messages or captured or session.generated_prompt is not None
    )
    steps = frontend.derive_stepper(session)
    current = next((step.label for step in steps if step.status == "active"), "Idea")
    return ResumeState(
        available=available,
        current_step=current,
        captured_details=captured,
        has_prompt=session.generated_prompt is not None,
        message_count=len(session.messages),
    )


def can_generate_now(session: Optional[Session]) -> bool:
    """Return whether an explicit early-generation action is meaningful."""

    return bool(
        session is not None
        and session.messages
        and session.spec is not None
        and session.generated_prompt is None
    )


def skip_pending_fields(
    spec: PromptSpecification,
    pending_fields: Optional[Sequence[str]],
) -> tuple[str, ...]:
    """Decline every field in the pending question through elicitation rules.

    The existing engine assigns the maximum attempt count, which makes these
    fields ineligible for future questions while preserving the question-limit
    and retry semantics used by the rest of the app.
    """

    names = tuple(dict.fromkeys(str(name) for name in (pending_fields or ()) if str(name)))
    if not names:
        return ()
    question = elicitation_engine.FollowUpQuestion(list(names), "")
    elicitation_engine.handle_answer(spec, question, "skip")
    return names


def select_template_preview(
    state: MutableMapping[str, Any],
    template_title: str,
    *,
    preview_key: str = TEMPLATE_PREVIEW_KEY,
) -> None:
    """Select a preview without replacing or modifying the active Session."""

    state[preview_key] = template_title


def clear_template_filters(
    state: MutableMapping[str, Any],
    *,
    search_key: str = TEMPLATE_SEARCH_KEY,
    category_key: str = TEMPLATE_CATEGORY_KEY,
) -> None:
    """Restore the template gallery's deterministic default filters."""

    state[search_key] = ""
    state[category_key] = "All templates"
