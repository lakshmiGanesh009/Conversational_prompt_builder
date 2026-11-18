"""Service Level Objectives (SLOs) for the Conversational Prompt Builder.

Vocabulary, briefly:

- **SLI** (indicator): a measured number, such as "the share of agent calls that
  succeeded" or "the 95th percentile generation latency".
- **SLO** (objective): the target for that number, such as "at least 99%" or
  "at most 30 seconds".
- **Error budget**: how much failure the objective allows. A 99% objective over
  1,000 calls allows 10 bad calls. The budget remaining tells you how close you
  are to breaching.
- **Burn rate**: how fast the budget is being spent. 1.0 means failing at exactly
  the allowed rate; 2.0 means twice as fast.

This module is pure and deterministic. It defines every SLO in one list
(:data:`DEFAULT_SLOS`) and evaluates them against the records collected by
:mod:`prompt_builder.telemetry`. The dashboard and the README both read from this
list, so the objectives are stated in exactly one place.

Two kinds of objective
----------------------
- ``ratio``: "good events / total events must be at least ``target``" (for
  example availability). These have an error budget and a burn rate.
- ``threshold``: "the ``percentile`` of a measured value must be at most
  ``target``" (for example p95 latency). These report *headroom* instead of an
  error budget.

Statuses
--------
``met``, ``at_risk`` (still meeting the objective but close to the line),
``breached``, and ``no_data`` (fewer than ``min_samples`` measurements, so any
verdict would be noise).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence, Union

from prompt_builder.telemetry import AppEvent, LlmCall, percentile

__all__ = [
    "AT_RISK_BUDGET",
    "AT_RISK_HEADROOM",
    "DEFAULT_SLOS",
    "MIN_SAMPLES",
    "SloDefinition",
    "SloResult",
    "Status",
    "evaluate_slo",
    "evaluate_slos",
    "overall_status",
]

Status = Literal["met", "at_risk", "breached", "no_data"]

# Below this many measurements an SLO reports ``no_data`` rather than a verdict.
# One failure in three calls is a 33% failure rate, but it says little about the
# service, so we wait for a handful of samples before judging.
MIN_SAMPLES = 5

# A ratio SLO that is still met turns ``at_risk`` once less than this fraction of
# its error budget is left.
AT_RISK_BUDGET = 0.25

# A threshold SLO that is still met turns ``at_risk`` once it is within this
# fraction of its limit (for example a 30 s limit warns above 24 s).
AT_RISK_HEADROOM = 0.20

_EPSILON = 1e-9

# What a measure function returns: (good, total) for ratio SLOs, or a list of
# measured values for threshold SLOs.
RatioMeasure = Callable[[Sequence[LlmCall], Sequence[AppEvent]], tuple[int, int]]
ValuesMeasure = Callable[[Sequence[LlmCall], Sequence[AppEvent]], list[float]]


@dataclass(frozen=True)
class SloDefinition:
    """One objective, stated as data plus a function that measures it.

    Attributes:
        id: Stable identifier such as ``"SLO-01"``.
        name: Short human name.
        description: One sentence on what is measured and why it matters.
        category: Grouping for the dashboard: ``"Agent calls"``,
            ``"User journey"``, or ``"Cost"``.
        kind: ``"ratio"`` or ``"threshold"`` (see the module docstring).
        target: For ``ratio``, the minimum good fraction (0 to 1). For
            ``threshold``, the maximum allowed value of the percentile.
        measure: Extracts the measurements from the raw records.
        unit: Unit label for threshold values, such as ``"s"`` or ``"tokens"``.
            Unused for ratio SLOs.
        percentile: For ``threshold`` SLOs, which percentile is compared with
            ``target`` (for example 95).
    """

    id: str
    name: str
    description: str
    category: str
    kind: Literal["ratio", "threshold"]
    target: float
    measure: Union[RatioMeasure, ValuesMeasure]
    unit: str = ""
    percentile: Optional[float] = None

    def format_value(self, value: Optional[float]) -> str:
        """Format a measured value (an SLI) for display."""
        if value is None:
            return "n/a"
        if self.kind == "ratio":
            return f"{value * 100:.1f}%"
        if self.unit == "s":
            return f"{value:.1f} s"
        return f"{value:,.0f} {self.unit}".strip()

    @property
    def target_label(self) -> str:
        """The objective as a short phrase, for example ``"at least 99.0%"``."""
        if self.kind == "ratio":
            return f"at least {self.target * 100:.1f}%"
        return f"p{self.percentile:g} at most {self.format_value(self.target)}"


@dataclass(frozen=True)
class SloResult:
    """The evaluation of one SLO over a set of records.

    Attributes:
        definition: The objective that was evaluated.
        status: ``met``, ``at_risk``, ``breached`` or ``no_data``.
        sli: The measured value: a fraction for ratio SLOs, the percentile value
            for threshold SLOs. ``None`` when there are no measurements.
        sample_size: How many measurements the verdict rests on.
        budget_remaining: For ratio SLOs, the fraction of the error budget left
            (1.0 untouched, 0.0 spent, negative overspent). For threshold SLOs,
            the headroom below the limit as a fraction of the limit (negative
            when over). ``None`` when there is no verdict.
        burn_rate: For ratio SLOs, how many times faster than allowed the budget
            is being spent. ``None`` for threshold SLOs, when no verdict exists,
            or when the target is 100% (no budget).
        bad: For ratio SLOs, how many measurements were bad. ``None`` otherwise.
    """

    definition: SloDefinition
    status: Status
    sli: Optional[float]
    sample_size: int
    budget_remaining: Optional[float] = None
    burn_rate: Optional[float] = None
    bad: Optional[int] = None


# --------------------------------------------------------------------------
# Measurement helpers (small, named, and easy to test)
# --------------------------------------------------------------------------
def _ok_calls(calls: Sequence[LlmCall], *operations: str) -> list[LlmCall]:
    """Successful calls, optionally limited to the given operations."""
    return [
        c for c in calls if c.success and (not operations or c.operation in operations)
    ]


def _availability(calls: Sequence[LlmCall], _events: Sequence[AppEvent]) -> tuple[int, int]:
    return sum(1 for c in calls if c.success), len(calls)


def _latency_of(operation: str) -> ValuesMeasure:
    def measure(calls: Sequence[LlmCall], _events: Sequence[AppEvent]) -> list[float]:
        return [c.latency_s for c in _ok_calls(calls, operation)]

    return measure


def _not_truncated(calls: Sequence[LlmCall], _events: Sequence[AppEvent]) -> tuple[int, int]:
    long_outputs = _ok_calls(calls, "generate", "revise")
    return sum(1 for c in long_outputs if not c.truncated), len(long_outputs)


def _tokens_per_call(calls: Sequence[LlmCall], _events: Sequence[AppEvent]) -> list[float]:
    return [float(c.total_tokens) for c in _ok_calls(calls) if c.total_tokens > 0]


def _event_ratio(name: str) -> RatioMeasure:
    def measure(_calls: Sequence[LlmCall], events: Sequence[AppEvent]) -> tuple[int, int]:
        matching = [e for e in events if e.name == name and e.success is not None]
        return sum(1 for e in matching if e.success), len(matching)

    return measure


def _turn_durations(_calls: Sequence[LlmCall], events: Sequence[AppEvent]) -> list[float]:
    return [
        e.duration_s
        for e in events
        if e.name == "turn" and e.success and e.duration_s is not None
    ]


# --------------------------------------------------------------------------
# The objectives
# --------------------------------------------------------------------------
DEFAULT_SLOS: list[SloDefinition] = [
    # ---- Agent calls: is the model reachable, fast, and complete? ----------
    SloDefinition(
        id="SLO-01",
        name="Agent call availability",
        description=(
            "Share of model calls that return a response. Includes calls that "
            "failed before sending, such as a missing API key."
        ),
        category="Agent calls",
        kind="ratio",
        target=0.99,
        measure=_availability,
    ),
    SloDefinition(
        id="SLO-02",
        name="Interpretation latency",
        description=(
            "p95 time for the agent to understand a user message. Keeps the "
            "conversation feeling responsive."
        ),
        category="Agent calls",
        kind="threshold",
        target=6.0,
        unit="s",
        percentile=95,
        measure=_latency_of("interpret"),
    ),
    SloDefinition(
        id="SLO-03",
        name="Prompt generation latency",
        description="p95 time to write a full prompt, the longest step users wait on.",
        category="Agent calls",
        kind="threshold",
        target=30.0,
        unit="s",
        percentile=95,
        measure=_latency_of("generate"),
    ),
    SloDefinition(
        id="SLO-04",
        name="Revision latency",
        description="p95 time to apply a user's revision request to a prompt.",
        category="Agent calls",
        kind="threshold",
        target=30.0,
        unit="s",
        percentile=95,
        measure=_latency_of("revise"),
    ),
    SloDefinition(
        id="SLO-05",
        name="Complete outputs (no truncation)",
        description=(
            "Share of generated and revised prompts that finished naturally "
            "instead of being cut off at the max-tokens limit."
        ),
        category="Agent calls",
        kind="ratio",
        target=0.99,
        measure=_not_truncated,
    ),
    # ---- User journey: does the product work for people? --------------------
    SloDefinition(
        id="SLO-06",
        name="Turn success rate",
        description=(
            "Share of user messages processed end to end without showing an "
            "error banner."
        ),
        category="User journey",
        kind="ratio",
        target=0.98,
        measure=_event_ratio("turn"),
    ),
    SloDefinition(
        id="SLO-07",
        name="Turn latency",
        description=(
            "p95 total time for a successful turn, including every model call it "
            "made (interpret, then possibly generate)."
        ),
        category="User journey",
        kind="threshold",
        target=40.0,
        unit="s",
        percentile=95,
        measure=_turn_durations,
    ),
    SloDefinition(
        id="SLO-08",
        name="Retry recovery",
        description=(
            "Share of retry runs that recover within the 3-attempt cap, instead "
            "of ending in 'could not be completed'."
        ),
        category="User journey",
        kind="ratio",
        target=0.90,
        measure=_event_ratio("retry_run"),
    ),
    SloDefinition(
        id="SLO-09",
        name="Prompt completeness",
        description=(
            "Share of generated prompts that include every required parameter "
            "(audience and domain), rather than omitting some."
        ),
        category="User journey",
        kind="ratio",
        target=0.90,
        measure=_event_ratio("prompt_generated"),
    ),
    # ---- Cost: is usage staying within budget? ------------------------------
    SloDefinition(
        id="SLO-10",
        name="Token budget per call",
        description=(
            "p95 total tokens (input plus output) per call. Guards against "
            "runaway context or output that inflates cost."
        ),
        category="Cost",
        kind="threshold",
        target=6000.0,
        unit="tokens",
        percentile=95,
        measure=_tokens_per_call,
    ),
]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_slo(
    definition: SloDefinition,
    calls: Sequence[LlmCall],
    events: Sequence[AppEvent],
    *,
    min_samples: int = MIN_SAMPLES,
) -> SloResult:
    """Evaluate one objective against the given records."""
    if definition.kind == "ratio":
        good, total = definition.measure(calls, events)  # type: ignore[misc]
        return _evaluate_ratio(definition, good, total, min_samples)
    values = definition.measure(calls, events)  # type: ignore[assignment]
    return _evaluate_threshold(definition, values, min_samples)


def _evaluate_ratio(
    definition: SloDefinition, good: int, total: int, min_samples: int
) -> SloResult:
    if total < min_samples:
        return SloResult(
            definition=definition,
            status="no_data",
            sli=(good / total) if total else None,
            sample_size=total,
            bad=total - good,
        )

    sli = good / total
    bad = total - good
    allowed_error = 1.0 - definition.target

    if allowed_error > _EPSILON:
        budget_remaining: Optional[float] = 1.0 - bad / (allowed_error * total)
        burn_rate: Optional[float] = (bad / total) / allowed_error
    else:
        # A 100% target has no budget to spend: any bad event is an overspend.
        budget_remaining = 1.0 if bad == 0 else -1.0
        burn_rate = None

    meets = sli >= definition.target - _EPSILON
    if not meets:
        status: Status = "breached"
    elif budget_remaining is not None and budget_remaining < AT_RISK_BUDGET:
        status = "at_risk"
    else:
        status = "met"

    return SloResult(
        definition=definition,
        status=status,
        sli=sli,
        sample_size=total,
        budget_remaining=budget_remaining,
        burn_rate=burn_rate,
        bad=bad,
    )


def _evaluate_threshold(
    definition: SloDefinition, values: list[float], min_samples: int
) -> SloResult:
    if len(values) < min_samples:
        return SloResult(
            definition=definition,
            status="no_data",
            sli=percentile(values, definition.percentile or 95),
            sample_size=len(values),
        )

    sli = percentile(values, definition.percentile or 95)
    assert sli is not None  # values is non-empty here
    headroom = (definition.target - sli) / definition.target

    if sli > definition.target + _EPSILON:
        status: Status = "breached"
    elif headroom < AT_RISK_HEADROOM:
        status = "at_risk"
    else:
        status = "met"

    return SloResult(
        definition=definition,
        status=status,
        sli=sli,
        sample_size=len(values),
        budget_remaining=headroom,
    )


def evaluate_slos(
    calls: Sequence[LlmCall],
    events: Sequence[AppEvent],
    definitions: Optional[Sequence[SloDefinition]] = None,
    *,
    min_samples: int = MIN_SAMPLES,
) -> list[SloResult]:
    """Evaluate every objective (default: :data:`DEFAULT_SLOS`)."""
    return [
        evaluate_slo(d, calls, events, min_samples=min_samples)
        for d in (DEFAULT_SLOS if definitions is None else definitions)
    ]


def overall_status(results: Sequence[SloResult]) -> Status:
    """Roll results up to one status: the worst verdict among those with data.

    ``breached`` beats ``at_risk`` beats ``met``. Objectives with ``no_data`` are
    ignored; if every objective has no data the answer is ``no_data``.
    """
    statuses = {r.status for r in results}
    for status in ("breached", "at_risk", "met"):
        if status in statuses:
            return status  # type: ignore[return-value]
    return "no_data"
