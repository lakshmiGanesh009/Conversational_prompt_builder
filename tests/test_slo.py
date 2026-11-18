"""Tests for ``prompt_builder.slo``: objectives, error budgets, and statuses.

Three layers: the budget/burn-rate arithmetic with hand-computed numbers, which
records feed each of the ten default objectives, and a Hypothesis property tying
the verdict to the underlying ratio.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from prompt_builder.slo import (
    DEFAULT_SLOS,
    MIN_SAMPLES,
    SloDefinition,
    SloResult,
    evaluate_slo,
    evaluate_slos,
    overall_status,
)
from prompt_builder.telemetry import AppEvent, LlmCall

SLO = {d.id: d for d in DEFAULT_SLOS}


def _call(op="interpret", ok=True, latency=1.0, **kwargs):
    return LlmCall(ts=1.0, operation=op, success=ok, latency_s=latency, **kwargs)


def _calls(n_ok, n_bad, **kwargs):
    return [_call(ok=True, **kwargs) for _ in range(n_ok)] + [
        _call(ok=False, **kwargs) for _ in range(n_bad)
    ]


# --- error budget arithmetic (SLO-01: availability >= 99%) ------------------------
def test_exactly_on_target_is_at_risk_with_an_empty_budget() -> None:
    result = evaluate_slo(SLO["SLO-01"], _calls(990, 10), [])
    assert result.status == "at_risk"  # meeting the target, but nothing left to spare
    assert result.sli == pytest.approx(0.99)
    assert result.budget_remaining == pytest.approx(0.0)
    assert result.burn_rate == pytest.approx(1.0)
    assert result.bad == 10 and result.sample_size == 1000


def test_half_the_budget_spent_is_met() -> None:
    result = evaluate_slo(SLO["SLO-01"], _calls(995, 5), [])
    assert result.status == "met"
    assert result.budget_remaining == pytest.approx(0.5)
    assert result.burn_rate == pytest.approx(0.5)


def test_double_the_allowed_failures_is_breached_with_burn_rate_two() -> None:
    result = evaluate_slo(SLO["SLO-01"], _calls(980, 20), [])
    assert result.status == "breached"
    assert result.budget_remaining == pytest.approx(-1.0)
    assert result.burn_rate == pytest.approx(2.0)


def test_no_failures_leaves_the_whole_budget() -> None:
    result = evaluate_slo(SLO["SLO-01"], _calls(100, 0), [])
    assert result.status == "met"
    assert result.budget_remaining == pytest.approx(1.0)
    assert result.burn_rate == pytest.approx(0.0)


def test_a_hundred_percent_target_has_no_budget() -> None:
    strict = SloDefinition(
        id="X", name="x", description="", category="c", kind="ratio",
        target=1.0, measure=lambda calls, events: (9, 10),
    )
    result = evaluate_slo(strict, [], [])
    assert result.status == "breached"
    assert result.burn_rate is None  # undefined when nothing is allowed
    assert result.budget_remaining == -1.0


# --- threshold arithmetic (SLO-03: generate p95 <= 30 s) ---------------------------
@pytest.mark.parametrize(
    "latency, status, headroom",
    [(10.0, "met", 2 / 3), (26.0, "at_risk", 4 / 30), (40.0, "breached", -1 / 3)],
)
def test_threshold_status_and_headroom(latency, status, headroom) -> None:
    calls = [_call(op="generate", latency=latency) for _ in range(10)]
    result = evaluate_slo(SLO["SLO-03"], calls, [])
    assert result.status == status
    assert result.sli == pytest.approx(latency)
    assert result.budget_remaining == pytest.approx(headroom)
    assert result.burn_rate is None


# --- not enough data --------------------------------------------------------------------
def test_too_few_samples_gives_no_verdict() -> None:
    result = evaluate_slo(SLO["SLO-01"], _calls(MIN_SAMPLES - 1, 0), [])
    assert result.status == "no_data"
    assert result.sample_size == MIN_SAMPLES - 1
    assert result.budget_remaining is None


def test_min_samples_is_configurable() -> None:
    assert evaluate_slo(SLO["SLO-01"], _calls(3, 0), [], min_samples=3).status == "met"


# --- which records feed which objective ------------------------------------------------------
def test_availability_counts_client_init_failures_against_every_operation() -> None:
    calls = _calls(9, 0) + [_call(op="client_init", ok=False)]
    result = evaluate_slo(SLO["SLO-01"], calls, [])
    assert result.sample_size == 10
    assert result.sli == pytest.approx(0.9)
    assert result.status == "breached"


def test_each_latency_slo_uses_only_successful_calls_of_its_own_operation() -> None:
    calls = (
        [_call(op="generate", latency=10.0) for _ in range(5)]
        + [_call(op="interpret", latency=100.0) for _ in range(5)]  # other op: ignored
        + [_call(op="generate", ok=False, latency=999.0)]  # failure: ignored
    )
    result = evaluate_slo(SLO["SLO-03"], calls, [])
    assert result.sample_size == 5
    assert result.sli == pytest.approx(10.0)
    assert result.status == "met"


def test_truncation_slo_looks_only_at_generate_and_revise() -> None:
    calls = (
        [_call(op="generate", finish_reason="stop") for _ in range(9)]
        + [_call(op="generate", finish_reason="length")]
        + [_call(op="interpret", finish_reason="length") for _ in range(20)]  # ignored
    )
    result = evaluate_slo(SLO["SLO-05"], calls, [])
    assert result.sample_size == 10
    assert result.sli == pytest.approx(0.9)
    assert result.status == "breached"


def test_turn_slos_use_turn_events() -> None:
    events = [AppEvent(ts=1.0, name="turn", success=True, duration_s=5.0) for _ in range(9)]
    events.append(AppEvent(ts=1.0, name="turn", success=False, duration_s=99.0))
    events.append(AppEvent(ts=1.0, name="demo_loaded"))  # not a turn: ignored

    success = evaluate_slo(SLO["SLO-06"], [], events)
    assert success.sample_size == 10 and success.sli == pytest.approx(0.9)
    assert success.status == "breached"  # 90% < 98%

    latency = evaluate_slo(SLO["SLO-07"], [], events)
    assert latency.sample_size == 9  # the failed turn's duration is excluded
    assert latency.sli == pytest.approx(5.0)


def test_events_without_an_outcome_are_not_counted_in_ratios() -> None:
    events = [AppEvent(ts=1.0, name="turn", success=None) for _ in range(10)]
    assert evaluate_slo(SLO["SLO-06"], [], events).status == "no_data"


def test_retry_and_completeness_slos_use_their_own_events() -> None:
    retries = [AppEvent(ts=1.0, name="retry_run", success=True) for _ in range(5)]
    prompts = [AppEvent(ts=1.0, name="prompt_generated", success=False) for _ in range(5)]
    events = retries + prompts
    assert evaluate_slo(SLO["SLO-08"], [], events).status == "met"
    assert evaluate_slo(SLO["SLO-09"], [], events).status == "breached"


def test_token_budget_uses_successful_calls_with_reported_tokens() -> None:
    calls = [_call(prompt_tokens=3000, completion_tokens=500) for _ in range(5)]
    calls += [_call() for _ in range(10)]  # no usage reported: ignored, not zeros
    result = evaluate_slo(SLO["SLO-10"], calls, [])
    assert result.sample_size == 5
    assert result.sli == pytest.approx(3500.0)
    assert result.status == "met"


# --- the default catalogue ------------------------------------------------------------------
def test_default_slos_are_well_formed() -> None:
    assert [d.id for d in DEFAULT_SLOS] == [f"SLO-{n:02d}" for n in range(1, 11)]
    assert {d.category for d in DEFAULT_SLOS} == {"Agent calls", "User journey", "Cost"}
    for d in DEFAULT_SLOS:
        assert d.name and d.description
        assert d.kind in ("ratio", "threshold")
        if d.kind == "ratio":
            assert 0.0 < d.target <= 1.0
        else:
            assert d.target > 0 and d.percentile is not None


def test_every_default_slo_has_no_data_on_empty_input() -> None:
    results = evaluate_slos([], [])
    assert len(results) == len(DEFAULT_SLOS)
    assert {r.status for r in results} == {"no_data"}
    assert overall_status(results) == "no_data"


def test_formatting_helpers() -> None:
    assert SLO["SLO-01"].format_value(0.9894) == "98.9%"
    assert SLO["SLO-01"].target_label == "at least 99.0%"
    assert SLO["SLO-03"].format_value(12.34) == "12.3 s"
    assert SLO["SLO-03"].target_label == "p95 at most 30.0 s"
    assert SLO["SLO-10"].format_value(6000) == "6,000 tokens"
    assert SLO["SLO-01"].format_value(None) == "n/a"


# --- overall status ---------------------------------------------------------------------------
def _result(status):
    return SloResult(definition=DEFAULT_SLOS[0], status=status, sli=None, sample_size=0)


@pytest.mark.parametrize(
    "statuses, expected",
    [
        (["met", "met"], "met"),
        (["met", "at_risk"], "at_risk"),
        (["met", "at_risk", "breached"], "breached"),
        (["no_data", "met"], "met"),  # objectives without data do not hide good news
        (["no_data", "no_data"], "no_data"),
        ([], "no_data"),
    ],
)
def test_overall_status_is_the_worst_verdict_with_data(statuses, expected) -> None:
    assert overall_status([_result(s) for s in statuses]) == expected


# --- property: the verdict follows the ratio -------------------------------------------------
@settings(max_examples=100)
@given(
    total=st.integers(min_value=MIN_SAMPLES, max_value=500),
    bad_fraction=st.floats(min_value=0.0, max_value=1.0),
    target=st.floats(min_value=0.5, max_value=0.999),
)
def test_ratio_verdict_is_breached_exactly_when_below_target(total, bad_fraction, target) -> None:
    """Breached iff good/total < target; the error budget is negative iff breached."""
    bad = round(total * bad_fraction)
    good = total - bad
    assume(abs(good / total - target) > 1e-6)  # stay clear of float-equality edge

    definition = SloDefinition(
        id="P", name="p", description="", category="c", kind="ratio",
        target=target, measure=lambda calls, events: (good, total),
    )
    result = evaluate_slo(definition, [], [])

    below_target = good / total < target
    assert (result.status == "breached") is below_target
    assert (result.budget_remaining < 0) is below_target
    assert result.sample_size == total and result.bad == bad
