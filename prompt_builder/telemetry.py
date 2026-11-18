"""Telemetry for the Conversational Prompt Builder: measure every agent call.

This module is the measurement half of the app's observability story (the other
half, :mod:`prompt_builder.slo`, turns these measurements into SLO results, and
:mod:`prompt_builder.dashboard` draws them).

Design notes
------------
- **The network boundary stays single.** ``llm_client.py`` is still the only
  module that talks to the model. :class:`InstrumentedLlmClient` is a thin
  wrapper around *any* ``LlmClient`` (the real one or a test mock), so every
  agent call is measured in exactly one place and the components that make the
  calls (Intent_Analyzer, Prompt_Generator) never know they are being observed.
- **Two kinds of records.** An :class:`LlmCall` is one request to the model
  (latency, success, token usage, finish reason). An :class:`AppEvent` is a
  higher-level product signal (a user turn finished, a retry run recovered, a
  prompt came back complete). SLOs mix both.
- **In memory, bounded, thread-safe.** :class:`MetricsStore` keeps the most
  recent :data:`MAX_RECORDS` of each kind behind a lock. A Streamlit server
  process serves many sessions, so one process-wide store (see
  :func:`get_store`) gives a view across all of them. Nothing is written to
  disk; a restart starts fresh. The dashboard offers CSV/JSON export.
- **No prompt content is recorded.** Only timings, counts, token numbers, model
  names, and a short, key-redacted error string. User text and generated prompts
  are never stored here.

Everything in this module is pure Python with no Streamlit import, so it is fast
to test and safe to import anywhere.
"""

from __future__ import annotations

import math
import random
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

__all__ = [
    "AppEvent",
    "CallSummary",
    "InstrumentedLlmClient",
    "LlmCall",
    "MAX_RECORDS",
    "MetricsStore",
    "PRICING_PER_M_TOKENS",
    "call_to_dict",
    "estimate_cost",
    "event_to_dict",
    "get_store",
    "percentile",
    "record_failed_call",
    "seed_sample_traffic",
    "summarize_by_operation",
    "summarize_calls",
]

# How many records of each kind the store keeps. Old records fall off the front.
MAX_RECORDS = 5000

# Longest error text kept per failed call. Provider errors can be verbose.
_MAX_ERROR_CHARS = 300

# Approximate list prices in USD per 1,000,000 tokens as (input, output).
#
# These are ESTIMATES used only to give the dashboard a rough cost figure. They
# are not read from the provider, they change over time, and a model that is not
# listed simply shows no cost. Check your provider's pricing page and edit this
# table to match your contract.
PRICING_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
}

# Provider errors may echo credentials. Mask common provider prefixes plus opaque
# token-shaped values before any error text enters the process-wide store.
_PREFIXED_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(?:sk-|hf_|github_pat_|gh[pousr]_|xox[baprs]-|AIza)"
    r"[A-Za-z0-9_\-*]{6,}\b"
)
_OPAQUE_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?=[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-]))"
    r"(?=[A-Za-z0-9_-]*[0-9_-])"
    r"[A-Za-z0-9_-]+"
)
_BEARER_PATTERN = re.compile(r"(?i)(\bBearer\s+)[^\s,;]{6,}")


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LlmCall:
    """One request to the language model (one "agent call").

    Attributes:
        ts: Wall-clock time the call finished, seconds since the epoch.
        operation: Which agent step made the call: ``"interpret"``,
            ``"generate"``, ``"revise"``, ``"client_init"`` (the client could not
            even be built, for example no API key), or ``"unknown"``.
        success: ``True`` if the model returned a response.
        latency_s: Seconds from sending the request to getting a result or an
            error.
        model: The model that served the request, when known.
        prompt_tokens: Input tokens billed, when the provider reported them.
        completion_tokens: Output tokens generated, when reported.
        finish_reason: Why generation stopped. ``"length"`` means the output was
            cut off at ``max_tokens``.
        error_type: Exception class name for a failed call.
        error_message: Short, key-redacted error text for a failed call.
        synthetic: ``True`` for generated sample data, never for real traffic.
    """

    ts: float
    operation: str
    success: bool
    latency_s: float
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    synthetic: bool = False

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens (missing counts are treated as zero)."""
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    @property
    def truncated(self) -> bool:
        """``True`` when the model stopped because it hit ``max_tokens``."""
        return self.finish_reason == "length"


@dataclass(frozen=True)
class AppEvent:
    """A product-level signal that is not itself a model call.

    Names in use:

    - ``"turn"``: a user message was processed end to end. ``success`` says
      whether it finished without an error banner; ``duration_s`` is the total
      time, so it includes every model call the turn made.
    - ``"retry_run"``: a retry controller run (after a failure). ``success`` says
      whether it recovered within the attempt cap.
    - ``"prompt_generated"``: a prompt was produced. ``success`` is ``True`` when
      it had no omitted required parameters (a "complete" prompt).
    - ``"demo_loaded"``: a hardcoded example was opened.
    - ``"validation_rejected"``: an input was rejected before any model call.

    Attributes:
        ts: Wall-clock time of the event, seconds since the epoch.
        name: One of the names above.
        success: Outcome where that makes sense, else ``None``.
        duration_s: Duration where that makes sense, else ``None``.
        synthetic: ``True`` for generated sample data.
    """

    ts: float
    name: str
    success: Optional[bool] = None
    duration_s: Optional[float] = None
    synthetic: bool = False


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------
class MetricsStore:
    """A thread-safe, bounded, in-memory store of calls and events."""

    def __init__(self, max_records: int = MAX_RECORDS) -> None:
        self._lock = threading.Lock()
        self._calls: deque[LlmCall] = deque(maxlen=max_records)
        self._events: deque[AppEvent] = deque(maxlen=max_records)

    def record_call(self, call: LlmCall) -> None:
        """Append one model-call record."""
        with self._lock:
            self._calls.append(call)

    def record_event(self, event: AppEvent) -> None:
        """Append one product-event record."""
        with self._lock:
            self._events.append(event)

    def calls(self, since: Optional[float] = None) -> list[LlmCall]:
        """Return a copy of the call records, optionally only those at/after ``since``."""
        with self._lock:
            items = list(self._calls)
        return items if since is None else [c for c in items if c.ts >= since]

    def events(self, since: Optional[float] = None) -> list[AppEvent]:
        """Return a copy of the event records, optionally only those at/after ``since``."""
        with self._lock:
            items = list(self._events)
        return items if since is None else [e for e in items if e.ts >= since]

    def has_synthetic(self) -> bool:
        """``True`` if any stored record is generated sample data."""
        with self._lock:
            return any(c.synthetic for c in self._calls) or any(
                e.synthetic for e in self._events
            )

    def clear(self) -> None:
        """Remove every record."""
        with self._lock:
            self._calls.clear()
            self._events.clear()


# One store per server process, shared by every browser session.
_DEFAULT_STORE = MetricsStore()


def get_store() -> MetricsStore:
    """Return the process-wide store the app records into."""
    return _DEFAULT_STORE


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------
def _redact(text: str) -> str:
    """Trim an error string and mask provider-independent credential shapes."""
    text = _PREFIXED_CREDENTIAL_PATTERN.sub("***REDACTED***", text)
    text = _BEARER_PATTERN.sub(r"\1***REDACTED***", text)
    text = _OPAQUE_TOKEN_PATTERN.sub("***REDACTED***", text)
    return text if len(text) <= _MAX_ERROR_CHARS else text[:_MAX_ERROR_CHARS] + "..."


def record_failed_call(
    operation: str,
    exc: BaseException,
    *,
    latency_s: float = 0.0,
    model: Optional[str] = None,
    store: Optional[MetricsStore] = None,
    now: Optional[float] = None,
) -> None:
    """Record a failed call.

    Used by :class:`InstrumentedLlmClient` and directly by the app for failures
    that happen *before* a request exists, such as a missing API key when the
    client is constructed (operation ``"client_init"``).
    """
    _safe_record_call(
        store or get_store(),
        lambda: LlmCall(
            ts=time.time() if now is None else now,
            operation=operation,
            success=False,
            latency_s=latency_s,
            model=model,
            error_type=type(exc).__name__,
            error_message=_redact(str(exc)),
        ),
    )


def _safe_record_call(store: MetricsStore, build: Callable[[], LlmCall]) -> None:
    """Record a call, swallowing any recording error.

    Observability must never break the product: if building or storing the
    record fails for any reason, the model call's real result (or its real
    exception) must still reach the caller untouched.
    """
    try:
        store.record_call(build())
    except Exception:  # noqa: BLE001 - telemetry is best-effort by design
        pass


class InstrumentedLlmClient:
    """Wrap any ``LlmClient`` and record one :class:`LlmCall` per ``complete``.

    The wrapper is transparent: it passes the request through unchanged, returns
    the inner client's response unchanged, and re-raises the inner client's
    exception unchanged. It only observes. Missing usage fields on a response
    (common with test doubles) are recorded as ``None``.
    """

    def __init__(
        self,
        inner: Any,
        *,
        store: Optional[MetricsStore] = None,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], float] = time.time,
    ) -> None:
        """Create the wrapper.

        Args:
            inner: The client to observe (anything with ``complete(request)``).
            store: Where to record. Defaults to the process-wide store.
            clock: Monotonic timer used for latency (injectable for tests).
            now: Wall-clock source used for timestamps (injectable for tests).
        """
        self._inner = inner
        self._store = store
        self._clock = clock
        self._now = now

    def complete(self, request: Any) -> Any:
        """Forward to the inner client, recording latency and outcome."""
        store = self._store or get_store()
        operation = getattr(request, "operation", None) or "unknown"
        requested_model = getattr(request, "model", None)

        started = self._clock()
        try:
            response = self._inner.complete(request)
        except Exception as exc:  # noqa: BLE001 - observe every failure, then re-raise
            record_failed_call(
                operation,
                exc,
                latency_s=self._clock() - started,
                model=requested_model,
                store=store,
                now=self._now(),
            )
            raise

        latency = self._clock() - started
        _safe_record_call(
            store,
            lambda: LlmCall(
                ts=self._now(),
                operation=operation,
                success=True,
                latency_s=latency,
                model=getattr(response, "model", None) or requested_model,
                prompt_tokens=getattr(response, "prompt_tokens", None),
                completion_tokens=getattr(response, "completion_tokens", None),
                finish_reason=getattr(response, "finish_reason", None),
            ),
        )
        return response


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def percentile(values: list[float], p: float) -> Optional[float]:
    """Return the ``p``-th percentile (0-100) using linear interpolation.

    This matches the default of ``numpy.percentile``. Returns ``None`` for an
    empty list. ``p`` is clamped to the 0-100 range.
    """
    if not values:
        return None
    ordered = sorted(values)
    p = min(100.0, max(0.0, p))
    rank = (len(ordered) - 1) * p / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    fraction = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def estimate_cost(
    model: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> Optional[float]:
    """Estimate USD cost for one call, or ``None`` if the model is not priced.

    The model name is matched by longest prefix so a dated variant such as
    ``gpt-4o-mini-2024-07-18`` uses the ``gpt-4o-mini`` price. See
    :data:`PRICING_PER_M_TOKENS` for the caveat that prices are estimates.
    """
    if not model:
        return None
    matches = [name for name in PRICING_PER_M_TOKENS if model.startswith(name)]
    if not matches:
        return None
    input_price, output_price = PRICING_PER_M_TOKENS[max(matches, key=len)]
    return (
        (prompt_tokens or 0) * input_price + (completion_tokens or 0) * output_price
    ) / 1_000_000


@dataclass(frozen=True)
class CallSummary:
    """Aggregate figures for a set of calls.

    Latency figures use *successful* calls only, because a call that fails
    instantly (for example a missing API key) would otherwise make the service
    look faster than it is.

    Attributes:
        total: Number of calls.
        successes: Calls that returned a response.
        failures: Calls that raised.
        success_rate: ``successes / total``, or ``None`` when there are no calls.
        avg_latency_s: Mean latency of successful calls.
        p50_latency_s: Median latency of successful calls.
        p95_latency_s: 95th percentile latency of successful calls.
        p99_latency_s: 99th percentile latency of successful calls.
        prompt_tokens: Total input tokens.
        completion_tokens: Total output tokens.
        total_tokens: Input plus output tokens.
        truncated: Successful calls cut off at ``max_tokens``.
        est_cost_usd: Estimated cost of the priced calls, or ``None`` if none
            could be priced.
    """

    total: int
    successes: int
    failures: int
    success_rate: Optional[float]
    avg_latency_s: Optional[float]
    p50_latency_s: Optional[float]
    p95_latency_s: Optional[float]
    p99_latency_s: Optional[float]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    truncated: int
    est_cost_usd: Optional[float]


def summarize_calls(calls: list[LlmCall]) -> CallSummary:
    """Aggregate a list of calls into a :class:`CallSummary`."""
    ok = [c for c in calls if c.success]
    latencies = [c.latency_s for c in ok]

    costs = [estimate_cost(c.model, c.prompt_tokens, c.completion_tokens) for c in calls]
    priced = [cost for cost in costs if cost is not None]

    prompt_tokens = sum(c.prompt_tokens or 0 for c in calls)
    completion_tokens = sum(c.completion_tokens or 0 for c in calls)
    return CallSummary(
        total=len(calls),
        successes=len(ok),
        failures=len(calls) - len(ok),
        success_rate=(len(ok) / len(calls)) if calls else None,
        avg_latency_s=(sum(latencies) / len(latencies)) if latencies else None,
        p50_latency_s=percentile(latencies, 50),
        p95_latency_s=percentile(latencies, 95),
        p99_latency_s=percentile(latencies, 99),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        truncated=sum(1 for c in ok if c.truncated),
        est_cost_usd=sum(priced) if priced else None,
    )


def summarize_by_operation(calls: list[LlmCall]) -> dict[str, CallSummary]:
    """Summarize calls grouped by operation, in first-seen order."""
    grouped: dict[str, list[LlmCall]] = {}
    for call in calls:
        grouped.setdefault(call.operation, []).append(call)
    return {operation: summarize_calls(items) for operation, items in grouped.items()}


# --------------------------------------------------------------------------
# Sample data
# --------------------------------------------------------------------------
def seed_sample_traffic(
    store: MetricsStore,
    *,
    turns: int = 40,
    seed: int = 7,
    now: Optional[float] = None,
    span_s: float = 3600.0,
) -> None:
    """Fill ``store`` with realistic-looking SYNTHETIC traffic.

    Lets the dashboard be explored without an API key or real users. Every
    record is flagged ``synthetic=True`` so the dashboard can say so. The data is
    deterministic for a given ``seed``.

    Shape of the fake traffic: each turn makes an interpret call; about 55% then
    ask a follow-up question and the rest generate a prompt (some of those get a
    revision). Roughly 2% of calls fail, 1.5% of long outputs are truncated, and
    about 8% of generated prompts miss a required parameter. Latencies are
    log-normal around plausible medians with a few slow outliers.
    """
    rng = random.Random(seed)
    end = time.time() if now is None else now
    start = end - span_s
    model = "gpt-4o-mini"

    def latency(median: float) -> float:
        value = rng.lognormvariate(math.log(median), 0.4)
        return value * (3.0 if rng.random() < 0.03 else 1.0)

    def make_call(ts: float, operation: str, median: float, p_in: tuple[int, int],
                  p_out: tuple[int, int], can_truncate: bool) -> LlmCall:
        lat = latency(median)
        if rng.random() < 0.02:
            return LlmCall(
                ts=ts, operation=operation, success=False,
                latency_s=min(lat, 5.0), model=model,
                error_type="LlmError",
                error_message="LLM request failed: simulated upstream timeout",
                synthetic=True,
            )
        truncated = can_truncate and rng.random() < 0.015
        return LlmCall(
            ts=ts, operation=operation, success=True, latency_s=lat, model=model,
            prompt_tokens=rng.randint(*p_in),
            completion_tokens=rng.randint(*p_out),
            finish_reason="length" if truncated else "stop",
            synthetic=True,
        )

    for i in range(turns):
        ts = start + (i + 1) * span_s / (turns + 1)
        calls: list[LlmCall] = [
            make_call(ts, "interpret", 1.8, (600, 900), (80, 200), False)
        ]
        generated = rng.random() >= 0.55
        if generated:
            calls.append(make_call(ts + 2, "generate", 12.0, (500, 900), (900, 1800), True))
        for call in calls:
            store.record_call(call)

        failed = any(not c.success for c in calls)
        retry_ok = False
        if failed:
            retry_ok = rng.random() < 0.7
            store.record_event(AppEvent(ts=ts + 3, name="retry_run", success=retry_ok, synthetic=True))

        turn_ok = (not failed) or retry_ok
        duration = sum(c.latency_s for c in calls) + rng.uniform(0.05, 0.4)
        store.record_event(
            AppEvent(ts=ts + 4, name="turn", success=turn_ok, duration_s=duration, synthetic=True)
        )

        if generated and not failed:
            complete = rng.random() >= 0.08
            store.record_event(
                AppEvent(ts=ts + 4, name="prompt_generated", success=complete, synthetic=True)
            )
            if rng.random() < 0.15:
                revise = make_call(ts + 30, "revise", 9.0, (1500, 2500), (900, 1600), True)
                store.record_call(revise)
                if not revise.success:
                    store.record_event(
                        AppEvent(ts=ts + 31, name="retry_run", success=rng.random() < 0.7, synthetic=True)
                    )

    for k in range(max(1, turns // 8)):
        store.record_event(
            AppEvent(ts=start + (k + 1) * span_s / (turns // 8 + 2), name="demo_loaded", synthetic=True)
        )
    for k in range(max(1, turns // 10)):
        store.record_event(
            AppEvent(ts=start + (k + 1) * span_s / (turns // 10 + 2) + 5, name="validation_rejected", synthetic=True)
        )


def call_to_dict(call: LlmCall) -> dict[str, Any]:
    """Plain-dict form of a call, for export and tables."""
    return asdict(call)


def event_to_dict(event: AppEvent) -> dict[str, Any]:
    """Plain-dict form of an event, for export and tables."""
    return asdict(event)
