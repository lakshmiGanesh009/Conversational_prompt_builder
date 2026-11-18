"""Tests for the Observability dashboard.

Two kinds of test:

- Pure helpers (``calls_frame``, ``bucket_frequency``, ``traffic_frame``) and the
  scorecard HTML builder, tested directly with no browser.
- AppTest-driven checks that switch the real app into the Observability view and
  confirm it renders, filters, seeds, and clears without raising.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from prompt_builder import dashboard
from prompt_builder.slo import DEFAULT_SLOS, SloDefinition, SloResult, evaluate_slo
from prompt_builder.telemetry import LlmCall, get_store, seed_sample_traffic

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _call(ts, ok=True, op="interpret", latency=1.0):
    return LlmCall(ts=ts, operation=op, success=ok, latency_s=latency)


# --- calls_frame ------------------------------------------------------------------
def test_calls_frame_for_no_calls_still_has_every_column() -> None:
    frame = dashboard.calls_frame([])
    assert frame.empty
    assert "latency_s" in frame.columns and "status" in frame.columns


def test_calls_frame_is_sorted_oldest_first_and_labels_status() -> None:
    frame = dashboard.calls_frame([_call(200.0, ok=False), _call(100.0, ok=True)])
    assert list(frame["status"]) == ["ok", "failed"]
    assert list(frame["success"]) == [True, False]
    assert frame["time"].is_monotonic_increasing


# --- bucket_frequency ---------------------------------------------------------------
@pytest.mark.parametrize(
    "span_seconds, expected",
    [
        (0, "1min"),
        (40 * 60, "1min"),  # exactly 40 one-minute bars
        (40 * 60 + 1, "5min"),  # one second more needs the next size up
        (3 * 3600, "5min"),
        (24 * 3600, "1h"),
        (10 * 86400, "6h"),  # exactly 40 six-hour bars
        (10 * 86400 + 1, "1D"),  # one second more tips over to daily bars
        (10_000 * 86400, "1D"),  # never runs out of sizes
    ],
)
def test_bucket_frequency_keeps_charts_to_a_readable_bar_count(span_seconds, expected) -> None:
    assert dashboard.bucket_frequency(span_seconds) == expected


# --- traffic_frame ----------------------------------------------------------------------
def test_traffic_frame_counts_outcomes_and_zero_fills_quiet_buckets() -> None:
    # Two calls in minute 0 (one failed), nothing in minutes 1-2, one call in minute 3.
    frame = dashboard.calls_frame(
        [_call(0.0), _call(10.0, ok=False), _call(180.0)]
    )
    traffic = dashboard.traffic_frame(frame, "1min")

    assert list(traffic.columns) == ["Succeeded", "Failed"]
    assert list(traffic["Succeeded"]) == [1, 0, 0, 1]
    assert list(traffic["Failed"]) == [1, 0, 0, 0]


def test_traffic_frame_for_no_calls_is_empty() -> None:
    assert dashboard.traffic_frame(dashboard.calls_frame([]), "1min").empty


# --- scorecard HTML -------------------------------------------------------------------------
def _result_for(definition_id, calls):
    definition = next(d for d in DEFAULT_SLOS if d.id == definition_id)
    return evaluate_slo(definition, calls, [])


def test_scorecard_shows_status_value_target_and_budget() -> None:
    calls = [_call(1.0) for _ in range(990)] + [_call(1.0, ok=False) for _ in range(10)]
    card = dashboard._slo_card_html(_result_for("SLO-01", calls))

    assert "SLO-01" in card and "Agent call availability" in card
    assert "At risk" in card  # status is spelled out, not only colored
    assert "99.0%" in card and "at least 99.0%" in card
    assert "10 of 1000 bad" in card
    assert "burn 1.0x" in card


def test_scorecard_for_a_breach_says_the_budget_is_overspent() -> None:
    calls = [_call(1.0) for _ in range(980)] + [_call(1.0, ok=False) for _ in range(20)]
    card = dashboard._slo_card_html(_result_for("SLO-01", calls))
    assert "Breached" in card and "error budget overspent" in card


def test_scorecard_without_enough_data_says_how_much_is_needed() -> None:
    card = dashboard._slo_card_html(_result_for("SLO-01", [_call(1.0)]))
    assert "Not enough data" in card
    assert "Needs 5+ samples (have 1)" in card


def test_scorecard_text_is_html_escaped() -> None:
    hostile = SloDefinition(
        id="X",
        name="<script>alert(1)</script>",
        description="fish & chips",
        category="c",
        kind="ratio",
        target=0.9,
        measure=lambda calls, events: (0, 0),
    )
    card = dashboard._slo_card_html(SloResult(hostile, "no_data", None, 0))
    assert "<script>" not in card
    assert "&lt;script&gt;" in card and "fish &amp; chips" in card


# --- the real view, driven through AppTest ---------------------------------------------------
def _open_observability() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    at.radio(key="view").set_value("Observability").run()
    return at


def _markdown(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def test_observability_view_replaces_the_chat_page() -> None:
    at = _open_observability()
    assert not at.exception
    assert len(at.chat_input) == 0  # the chat box belongs to the Builder only
    assert not any(b.key == "new_session_button" for b in at.button)


def test_empty_dashboard_explains_itself_and_still_shows_definitions() -> None:
    at = _open_observability()
    assert not at.exception
    assert "No telemetry in this window yet" in _markdown(at)
    assert any("How the objectives are defined" in e.label for e in at.expander)


def test_dashboard_renders_every_section_for_seeded_traffic() -> None:
    seed_sample_traffic(get_store(), turns=40)
    at = _open_observability()

    assert not at.exception
    markdown = _markdown(at)
    assert markdown.count('class="obs-slo ') == len(DEFAULT_SLOS)
    assert "obs-overall" in markdown
    assert "synthetic sample data" in markdown  # sample data is always labelled
    assert len(at.tabs) == 4
    assert {m.label for m in at.metric} >= {"Calls", "Success rate", "Latency p95", "Estimated cost"}


def test_time_window_and_outcome_filters_apply() -> None:
    seed_sample_traffic(get_store(), turns=40)
    at = _open_observability()
    all_calls = int(next(m for m in at.metric if m.label == "Calls").value.replace(",", ""))

    at.selectbox(key="obs_window").select("Last 15 minutes").run()
    assert not at.exception
    recent = int(next(m for m in at.metric if m.label == "Calls").value.replace(",", ""))
    assert 0 < recent < all_calls  # the window really narrows what is counted

    at.selectbox(key="obs_status").select("Failed").run()
    assert not at.exception


def test_add_sample_data_and_clear_buttons_work() -> None:
    at = _open_observability()
    assert get_store().calls() == []

    next(b for b in at.button if b.key == "obs_add_sample").click().run()
    assert not at.exception
    assert get_store().calls() and get_store().has_synthetic()

    next(b for b in at.button if b.key == "obs_clear").click().run()
    assert not at.exception
    assert get_store().calls() == []
    assert "No telemetry in this window yet" in _markdown(at)


def test_switching_back_to_the_builder_restores_the_chat_input() -> None:
    at = _open_observability()
    at.radio(key="view").set_value("Builder").run()
    assert not at.exception
    assert len(at.chat_input) == 1


def test_calls_frame_time_column_is_datetime() -> None:
    frame = dashboard.calls_frame([_call(1_700_000_000.0)])
    assert pd.api.types.is_datetime64_any_dtype(frame["time"])
