"""The Observability dashboard for the Conversational Prompt Builder.

Shows how the agent is behaving: every model call, the SLOs measured from them,
and the product-level signals around them. It reads from the process-wide
:class:`~prompt_builder.telemetry.MetricsStore` and evaluates the objectives in
:mod:`prompt_builder.slo`.

Layout, top to bottom:

1. Controls: time window and refresh.
2. Key numbers: calls, success rate, latency, tokens, estimated cost.
3. SLO scorecards: one card per objective with its status, value, target, and
   error budget (or headroom).
4. Charts: latency, traffic, token usage, and errors.
5. Breakdown by agent step, and a filterable table of recent calls.
6. Exports (CSV/JSON), the SLO definitions, and maintenance actions.

The data-shaping helpers (:func:`calls_frame`, :func:`bucket_frequency`,
:func:`traffic_frame`) are plain functions so they can be tested without a
browser. Only the ``render_*`` functions touch Streamlit.

The dashboard shows aggregate numbers and short, key-redacted error text. It
never shows prompt or message content, because telemetry never stores it.
"""

from __future__ import annotations

import html
import json
import time
from collections import Counter
from typing import Optional

import pandas as pd
import streamlit as st

from prompt_builder import telemetry
from prompt_builder.slo import (
    DEFAULT_SLOS,
    MIN_SAMPLES,
    SloResult,
    Status,
    evaluate_slos,
    overall_status,
)

__all__ = [
    "WINDOWS",
    "bucket_frequency",
    "calls_frame",
    "render_dashboard",
    "traffic_frame",
]

# Selectable time windows: label -> seconds back from now (None = everything).
WINDOWS: dict[str, Optional[int]] = {
    "Last 15 minutes": 15 * 60,
    "Last hour": 60 * 60,
    "Last 24 hours": 24 * 60 * 60,
    "All recorded": None,
}

# Chart colors, kept in step with the app palette.
_TEAL = "#087f75"
_CORAL = "#c2413b"  # semantic failure red, not a decorative accent
_INDIGO = "#145da0"
_VIOLET = "#0e9fba"

# (pandas frequency, bucket size in seconds), smallest first.
_BUCKETS = [
    ("1min", 60),
    ("5min", 300),
    ("15min", 900),
    ("1h", 3600),
    ("6h", 21600),
    ("1D", 86400),
]

# Aim for at most this many bars on a time chart.
_MAX_BUCKETS = 40

# How many recent calls the table shows.
_RECENT_LIMIT = 200

# Status -> (label, CSS suffix). The label carries a symbol so the status is
# readable without relying on color alone.
_STATUS_STYLE: dict[str, tuple[str, str]] = {
    "met": ("\u2713 Met", "met"),
    "at_risk": ("! At risk", "risk"),
    "breached": ("\u2715 Breached", "breached"),
    "no_data": ("\u2013 Not enough data", "nodata"),
}

_CSS = """
<style>
.obs-title { font-size: 1.35rem; font-weight: 800; letter-spacing: -0.015em;
    color: var(--cpb-ink); margin: 0 0 0.15rem 0; }
.obs-sub { color: var(--cpb-muted); font-size: 0.92rem; margin: 0 0 1.1rem 0; }
.obs-section { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--cpb-faint); margin: 1.6rem 0 0.6rem 0; }

.obs-overall { border-radius: 12px; padding: 0.7rem 1rem; font-weight: 600;
    font-size: 0.95rem; border: 1px solid var(--cpb-border); margin-bottom: 0.9rem; }
.obs-overall-met { background: #e7f8f7; color: #0b5f5e; border-color: #b7e8e6; }
.obs-overall-at_risk { background: #fff3e6; color: #9a4a06; border-color: #fcd9b0; }
.obs-overall-breached { background: #ffe8f1; color: #a3134d; border-color: #fbb9d3; }
.obs-overall-no_data { background: var(--cpb-subtle); color: var(--cpb-muted); }

.obs-slo { background: var(--cpb-surface); border: 1px solid var(--cpb-border);
    border-radius: 14px; padding: 0.9rem 1rem; margin-bottom: 0.7rem;
    box-shadow: var(--cpb-shadow-sm); border-top: 4px solid var(--cpb-border); }
.obs-slo-met { border-top-color: #0ea5a4; }
.obs-slo-risk { border-top-color: #f97316; }
.obs-slo-breached { border-top-color: #e11d48; }
.obs-slo-top { display: flex; justify-content: space-between; align-items: center; }
.obs-slo-id { font-size: 0.72rem; font-weight: 700; color: var(--cpb-faint);
    letter-spacing: 0.06em; }
.obs-pill { font-size: 0.72rem; font-weight: 700; padding: 0.15rem 0.55rem;
    border-radius: 999px; }
.obs-pill-met { background: #e7f8f7; color: #0b5f5e; }
.obs-pill-risk { background: #fff3e6; color: #9a4a06; }
.obs-pill-breached { background: #ffe8f1; color: #a3134d; }
.obs-pill-nodata { background: var(--cpb-subtle); color: var(--cpb-muted); }
.obs-slo-name { font-weight: 700; color: var(--cpb-ink); margin: 0.35rem 0 0.1rem 0; }
.obs-slo-sli { font-size: 1.55rem; font-weight: 800; color: var(--cpb-ink);
    line-height: 1.2; }
.obs-slo-target { display: block; font-size: 0.78rem; font-weight: 500;
    color: var(--cpb-muted); }
.obs-bar { height: 7px; background: var(--cpb-subtle); border-radius: 999px;
    overflow: hidden; margin: 0.6rem 0 0.35rem 0; }
.obs-bar-fill { height: 100%; border-radius: 999px; }
.obs-bar-met { background: #0ea5a4; }
.obs-bar-risk { background: #f97316; }
.obs-bar-breached { background: #e11d48; }
.obs-slo-meta { font-size: 0.78rem; color: var(--cpb-muted); }
.obs-slo-desc { font-size: 0.78rem; color: var(--cpb-muted); margin-top: 0.35rem; }

.obs-empty { border: 1px dashed var(--cpb-border); border-radius: 14px;
    background: var(--cpb-subtle); padding: 1.4rem 1.5rem; color: var(--cpb-muted); }
.obs-empty b { color: var(--cpb-ink); }
.obs-synth { display: inline-block; font-size: 0.78rem; font-weight: 600;
    color: #9a4a06; background: #fff3e6; border: 1px solid #fcd9b0;
    border-radius: 10px; padding: 0.35rem 0.7rem; margin-bottom: 0.8rem; }
</style>
"""


# --------------------------------------------------------------------------
# Data shaping (pure; tested without a browser)
# --------------------------------------------------------------------------
_FRAME_COLUMNS = [
    "time",
    "operation",
    "status",
    "success",
    "latency_s",
    "model",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "finish_reason",
    "error_type",
    "error_message",
    "synthetic",
]


def calls_frame(calls: list[telemetry.LlmCall]) -> pd.DataFrame:
    """Turn call records into a DataFrame, oldest first.

    Always returns the same columns, even for no calls, so callers can filter
    without special cases. ``time`` is a UTC datetime; ``status`` is ``"ok"`` or
    ``"failed"``.
    """
    if not calls:
        return pd.DataFrame(columns=_FRAME_COLUMNS)

    rows = [
        {
            "time": pd.to_datetime(c.ts, unit="s"),
            "operation": c.operation,
            "status": "ok" if c.success else "failed",
            "success": c.success,
            "latency_s": c.latency_s,
            "model": c.model,
            "prompt_tokens": c.prompt_tokens,
            "completion_tokens": c.completion_tokens,
            "total_tokens": c.total_tokens,
            "finish_reason": c.finish_reason,
            "error_type": c.error_type,
            "error_message": c.error_message,
            "synthetic": c.synthetic,
        }
        for c in calls
    ]
    return pd.DataFrame(rows, columns=_FRAME_COLUMNS).sort_values("time").reset_index(drop=True)


def bucket_frequency(span_seconds: float) -> str:
    """Pick a pandas resample frequency giving a readable number of bars."""
    for freq, size in _BUCKETS:
        if span_seconds / size <= _MAX_BUCKETS:
            return freq
    return _BUCKETS[-1][0]


def traffic_frame(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Count succeeded and failed calls per time bucket (empty buckets are zero)."""
    if frame.empty:
        return pd.DataFrame(columns=["Succeeded", "Failed"])
    counts = pd.DataFrame(
        {
            "Succeeded": frame["success"].astype(int).to_numpy(),
            "Failed": (~frame["success"].astype(bool)).astype(int).to_numpy(),
        },
        index=pd.DatetimeIndex(frame["time"]),
    )
    return counts.resample(freq).sum()


def _tokens_frame(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Sum input and output tokens per time bucket."""
    ok = frame[frame["success"].astype(bool)] if not frame.empty else frame
    if ok.empty:
        return pd.DataFrame(columns=["Input tokens", "Output tokens"])
    tokens = pd.DataFrame(
        {
            "Input tokens": ok["prompt_tokens"].fillna(0).astype(float).to_numpy(),
            "Output tokens": ok["completion_tokens"].fillna(0).astype(float).to_numpy(),
        },
        index=pd.DatetimeIndex(ok["time"]),
    )
    return tokens.resample(freq).sum()


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def _fmt_seconds(value: Optional[float]) -> str:
    return "\u2014" if value is None else f"{value:.2f} s"


def _fmt_percent(value: Optional[float]) -> str:
    return "\u2014" if value is None else f"{value * 100:.1f}%"


def _fmt_cost(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"${value:.4f}" if value < 1 else f"${value:,.2f}"


def _slo_meta(result: SloResult) -> str:
    """The small line under an SLO's bar: budget/headroom, burn, sample size."""
    n = result.sample_size
    if result.status == "no_data":
        return f"Needs {MIN_SAMPLES}+ samples (have {n})"

    parts: list[str] = []
    left = result.budget_remaining
    if result.definition.kind == "ratio":
        parts.append(f"{result.bad} of {n} bad")
        if left is not None:
            parts.append(
                f"{max(left, 0) * 100:.0f}% of error budget left"
                if left >= 0
                else "error budget overspent"
            )
        if result.burn_rate is not None:
            parts.append(f"burn {result.burn_rate:.1f}x")
    else:
        if left is not None:
            parts.append(
                f"{left * 100:.0f}% headroom"
                if left >= 0
                else f"{abs(left) * 100:.0f}% over the limit"
            )
        parts.append(f"n={n}")
    return " \u00b7 ".join(parts)


def _slo_card_html(result: SloResult) -> str:
    """One SLO scorecard as HTML. Only static, escaped text is interpolated."""
    d = result.definition
    label, css = _STATUS_STYLE[result.status]

    bar = ""
    if result.budget_remaining is not None and result.status != "no_data":
        fill = min(1.0, max(0.0, result.budget_remaining)) * 100
        what = "Error budget remaining" if d.kind == "ratio" else "Headroom below the limit"
        bar = (
            f'<div class="obs-bar" role="img" aria-label="{what} {fill:.0f} percent">'
            f'<div class="obs-bar-fill obs-bar-{css}" style="width:{fill:.0f}%"></div></div>'
        )
    else:
        bar = '<div class="obs-bar"></div>'

    return (
        f'<div class="obs-slo obs-slo-{css}">'
        f'<div class="obs-slo-top"><span class="obs-slo-id">{html.escape(d.id)}</span>'
        f'<span class="obs-pill obs-pill-{css}">{html.escape(label)}</span></div>'
        f'<div class="obs-slo-name">{html.escape(d.name)}</div>'
        f'<div class="obs-slo-sli">{html.escape(d.format_value(result.sli))}'
        f'<span class="obs-slo-target">Objective: {html.escape(d.target_label)}</span></div>'
        f"{bar}"
        f'<div class="obs-slo-meta">{html.escape(_slo_meta(result))}</div>'
        f'<div class="obs-slo-desc">{html.escape(d.description)}</div>'
        f"</div>"
    )


def _overall_banner_html(results: list[SloResult]) -> str:
    status: Status = overall_status(results)
    counts = Counter(r.status for r in results)
    headline = {
        "met": "All measured objectives are being met",
        "at_risk": "Objectives are being met, but some are close to the line",
        "breached": "One or more objectives are breached",
        "no_data": "Not enough traffic yet to judge the objectives",
    }[status]
    detail = " \u00b7 ".join(
        f"{counts[key]} {name}"
        for key, name in (
            ("breached", "breached"),
            ("at_risk", "at risk"),
            ("met", "met"),
            ("no_data", "not enough data"),
        )
        if counts[key]
    )
    return (
        f'<div class="obs-overall obs-overall-{status}">{html.escape(headline)}'
        f' <span style="font-weight:500">({html.escape(detail)})</span></div>'
    )


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _render_controls() -> Optional[int]:
    """Window picker + refresh. Returns the window in seconds (None = all)."""
    col_window, col_refresh = st.columns([3, 1])
    with col_window:
        label = st.selectbox(
            "Time window",
            list(WINDOWS),
            index=2,
            key="obs_window",
            help="Only calls and events inside this window feed the numbers below.",
        )
    with col_refresh:
        st.write("")  # nudge the button down to align with the select box
        st.button("Refresh", key="obs_refresh", width="stretch")
    return WINDOWS[label]


def _render_kpis(summary: telemetry.CallSummary) -> None:
    st.markdown('<p class="obs-section">Agent calls</p>', unsafe_allow_html=True)
    row1 = st.columns(3)
    row1[0].metric("Calls", f"{summary.total:,}", help="Model requests, including ones that failed.")
    row1[1].metric(
        "Success rate",
        _fmt_percent(summary.success_rate),
        help=f"{summary.failures} failed of {summary.total}.",
    )
    row1[2].metric(
        "Truncated outputs",
        f"{summary.truncated:,}",
        help="Successful calls that stopped at the max-tokens limit.",
    )
    row2 = st.columns(3)
    row2[0].metric("Latency p50", _fmt_seconds(summary.p50_latency_s), help="Successful calls only.")
    row2[1].metric("Latency p95", _fmt_seconds(summary.p95_latency_s), help="Successful calls only.")
    row2[2].metric("Latency p99", _fmt_seconds(summary.p99_latency_s), help="Successful calls only.")
    row3 = st.columns(3)
    row3[0].metric("Input tokens", f"{summary.prompt_tokens:,}")
    row3[1].metric("Output tokens", f"{summary.completion_tokens:,}")
    row3[2].metric(
        "Estimated cost",
        _fmt_cost(summary.est_cost_usd),
        help="Rough estimate from list prices in telemetry.PRICING_PER_M_TOKENS. "
        "Check your provider's pricing; models not in the table show n/a.",
    )


def _render_slos(calls: list[telemetry.LlmCall], events: list[telemetry.AppEvent]) -> None:
    st.markdown('<p class="obs-section">Service level objectives</p>', unsafe_allow_html=True)
    results = evaluate_slos(calls, events)
    st.markdown(_overall_banner_html(results), unsafe_allow_html=True)

    categories: list[str] = []
    for definition in DEFAULT_SLOS:
        if definition.category not in categories:
            categories.append(definition.category)

    for category in categories:
        st.markdown(f"**{html.escape(category)}**")
        in_category = [r for r in results if r.definition.category == category]
        for start in range(0, len(in_category), 2):
            cols = st.columns(2)
            for col, result in zip(cols, in_category[start : start + 2]):
                col.markdown(_slo_card_html(result), unsafe_allow_html=True)


def _render_charts(frame: pd.DataFrame, span_seconds: float) -> None:
    st.markdown('<p class="obs-section">Trends</p>', unsafe_allow_html=True)
    freq = bucket_frequency(span_seconds)
    tab_latency, tab_traffic, tab_tokens, tab_errors = st.tabs(
        ["Latency", "Traffic", "Tokens", "Errors"]
    )

    with tab_latency:
        ok = frame[frame["success"].astype(bool)] if not frame.empty else frame
        if ok.empty:
            st.info("No successful calls in this window yet.")
        else:
            st.caption("Each dot is one successful call. Times are UTC.")
            st.scatter_chart(
                ok,
                x="time",
                y="latency_s",
                color="operation",
                x_label="Time (UTC)",
                y_label="Latency (seconds)",
                height=300,
            )

    with tab_traffic:
        traffic = traffic_frame(frame, freq)
        if traffic.empty:
            st.info("No calls in this window yet.")
        else:
            st.caption(f"Calls per bucket ({freq}). Times are UTC.")
            st.bar_chart(
                traffic,
                color=[_TEAL, _CORAL],
                x_label="Time (UTC)",
                y_label="Calls",
                height=300,
            )

    with tab_tokens:
        tokens = _tokens_frame(frame, freq)
        if tokens.empty or float(tokens.to_numpy().sum()) == 0.0:
            st.info("No token usage reported in this window yet.")
        else:
            st.caption(f"Tokens per bucket ({freq}). Times are UTC.")
            st.bar_chart(
                tokens,
                color=[_INDIGO, _VIOLET],
                x_label="Time (UTC)",
                y_label="Tokens",
                height=300,
            )

    with tab_errors:
        failed = frame[~frame["success"].astype(bool)] if not frame.empty else frame
        if failed.empty:
            st.success("No failed calls in this window.")
        else:
            by_type = (
                failed.groupby(["operation", "error_type"], dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            st.caption("Failures by agent step and error type.")
            st.dataframe(by_type, hide_index=True, width="stretch")
            st.caption("Most recent errors (API keys are redacted).")
            st.dataframe(
                failed.sort_values("time", ascending=False)
                .head(20)[["time", "operation", "error_type", "error_message"]],
                hide_index=True,
                width="stretch",
            )


def _render_breakdown(calls: list[telemetry.LlmCall]) -> None:
    st.markdown('<p class="obs-section">By agent step</p>', unsafe_allow_html=True)
    by_op = telemetry.summarize_by_operation(calls)
    if not by_op:
        st.info("No calls in this window yet.")
        return
    rows = [
        {
            "Step": operation,
            "Calls": s.total,
            "Success rate": _fmt_percent(s.success_rate),
            "p50 (s)": None if s.p50_latency_s is None else round(s.p50_latency_s, 2),
            "p95 (s)": None if s.p95_latency_s is None else round(s.p95_latency_s, 2),
            "Tokens": s.total_tokens,
            "Truncated": s.truncated,
            "Est. cost (USD)": None if s.est_cost_usd is None else round(s.est_cost_usd, 4),
        }
        for operation, s in by_op.items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_recent(frame: pd.DataFrame) -> None:
    st.markdown('<p class="obs-section">Recent calls</p>', unsafe_allow_html=True)
    if frame.empty:
        st.info("No calls in this window yet.")
        return

    col_ops, col_status = st.columns([3, 1])
    operations = sorted(frame["operation"].unique())
    chosen = col_ops.multiselect("Agent step", operations, default=operations, key="obs_ops")
    status = col_status.selectbox("Outcome", ["All", "Succeeded", "Failed"], key="obs_status")

    shown = frame[frame["operation"].isin(chosen)]
    if status == "Succeeded":
        shown = shown[shown["success"].astype(bool)]
    elif status == "Failed":
        shown = shown[~shown["success"].astype(bool)]

    st.caption(f"Showing the newest {min(len(shown), _RECENT_LIMIT)} of {len(shown)} matching calls.")
    columns = [
        "time", "operation", "status", "latency_s", "model", "prompt_tokens",
        "completion_tokens", "finish_reason", "error_type", "error_message", "synthetic",
    ]
    st.dataframe(
        shown.sort_values("time", ascending=False).head(_RECENT_LIMIT)[columns],
        hide_index=True,
        width="stretch",
    )


def _render_exports(calls: list[telemetry.LlmCall], events: list[telemetry.AppEvent]) -> None:
    st.markdown('<p class="obs-section">Export</p>', unsafe_allow_html=True)
    calls_rows = [telemetry.call_to_dict(c) for c in calls]
    events_rows = [telemetry.event_to_dict(e) for e in events]
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())

    col_a, col_b, col_c = st.columns(3)
    col_a.download_button(
        "Calls (CSV)",
        pd.DataFrame(calls_rows).to_csv(index=False).encode("utf-8"),
        file_name=f"agent-calls-{stamp}.csv",
        mime="text/csv",
        key="obs_dl_calls",
        width="stretch",
        disabled=not calls_rows,
    )
    col_b.download_button(
        "Events (CSV)",
        pd.DataFrame(events_rows).to_csv(index=False).encode("utf-8"),
        file_name=f"app-events-{stamp}.csv",
        mime="text/csv",
        key="obs_dl_events",
        width="stretch",
        disabled=not events_rows,
    )
    col_c.download_button(
        "Everything (JSON)",
        json.dumps({"calls": calls_rows, "events": events_rows}, indent=2).encode("utf-8"),
        file_name=f"observability-{stamp}.json",
        mime="application/json",
        key="obs_dl_json",
        width="stretch",
        disabled=not (calls_rows or events_rows),
    )


def _render_definitions() -> None:
    with st.expander("How the objectives are defined", expanded=False):
        st.caption(
            "An SLO is a target for a measured number. The error budget is how much "
            "failure the target allows; burn rate is how fast it is being spent "
            "(1.0x = exactly the allowed rate). An objective shows 'not enough data' "
            f"until it has at least {MIN_SAMPLES} measurements."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "ID": d.id,
                        "Category": d.category,
                        "Objective": d.name,
                        "Target": d.target_label,
                        "What it measures": d.description,
                    }
                    for d in DEFAULT_SLOS
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def _render_maintenance(store: telemetry.MetricsStore) -> None:
    with st.expander("Maintenance", expanded=False):
        st.caption(
            "These act on the shared metrics store for this whole server, not just "
            "your session. Metrics live in memory and reset when the server restarts."
        )
        col_seed, col_clear = st.columns(2)
        if col_seed.button(
            "Add sample data",
            key="obs_add_sample",
            width="stretch",
            help="Adds clearly-labelled SYNTHETIC traffic so you can explore the dashboard.",
        ):
            telemetry.seed_sample_traffic(store, turns=40, seed=int(time.time()) % 10_000)
            st.rerun()
        if col_clear.button("Clear all metrics", key="obs_clear", width="stretch"):
            store.clear()
            st.rerun()


def render_dashboard(*, show_header: bool = True) -> None:
    """Render the whole Observability view within either standalone or shell context."""
    st.markdown(_CSS, unsafe_allow_html=True)
    if show_header:
        st.markdown('<p class="obs-title">Observability</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="obs-sub">How the agent is performing: every model call, '
            "the objectives we hold ourselves to, and the budget left to spend.</p>",
            unsafe_allow_html=True,
        )

    store = telemetry.get_store()
    window = _render_controls()
    now = time.time()
    since = None if window is None else now - window

    calls = store.calls(since=since)
    events = store.events(since=since)

    if store.has_synthetic():
        st.markdown(
            '<span class="obs-synth">Includes synthetic sample data. '
            "Use Maintenance \u2192 Clear all metrics to remove it.</span>",
            unsafe_allow_html=True,
        )

    if not calls and not events:
        st.markdown(
            '<div class="obs-empty"><b>No telemetry in this window yet.</b><br>'
            "Use the Builder and every model call will appear here. Or open "
            "<i>Maintenance</i> below and add sample data to explore the dashboard "
            "without an API key.</div>",
            unsafe_allow_html=True,
        )
    else:
        frame = calls_frame(calls)
        span = (
            (frame["time"].max() - frame["time"].min()).total_seconds()
            if len(frame) > 1
            else 0.0
        )
        _render_kpis(telemetry.summarize_calls(calls))
        _render_slos(calls, events)
        _render_charts(frame, span)
        _render_breakdown(calls)
        _render_recent(frame)
        _render_exports(calls, events)

    _render_definitions()
    _render_maintenance(store)
