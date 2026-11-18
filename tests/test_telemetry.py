"""Tests for ``prompt_builder.telemetry``: the measurement half of observability.

Covers the instrumenting wrapper (what it records, that it is transparent, that it
can never break a model call), the bounded store, the statistics helpers, the
operation tags the agent components put on their requests, and the synthetic
sample-traffic seeder. Everything runs offline against fakes.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_builder.intent_analyzer import IntentAnalyzer
from prompt_builder.llm_client import LlmError, LlmRequest, LlmResponse
from prompt_builder.models import GeneratedPrompt, ParameterValue, PromptSpecification
from prompt_builder.prompt_generator import PromptGenerator
from prompt_builder.telemetry import (
    AppEvent,
    InstrumentedLlmClient,
    LlmCall,
    MetricsStore,
    estimate_cost,
    percentile,
    record_failed_call,
    seed_sample_traffic,
    summarize_by_operation,
    summarize_calls,
)
from tests.mock_llm import MockLlmClient


class _FakeInner:
    """A minimal client: returns a fixed response or raises a fixed error."""

    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return self._response


def _ticking_clock(*values):
    """A clock that returns the given readings in order (start, then end)."""
    readings = iter(values)
    return lambda: next(readings)


# --- InstrumentedLlmClient ---------------------------------------------------
def test_wrapper_records_a_successful_call_with_usage() -> None:
    store = MetricsStore()
    response = LlmResponse(
        text="hello",
        model="gpt-4o-mini",
        prompt_tokens=120,
        completion_tokens=30,
        finish_reason="stop",
    )
    client = InstrumentedLlmClient(
        _FakeInner(response=response),
        store=store,
        clock=_ticking_clock(100.0, 102.5),
        now=lambda: 5000.0,
    )

    result = client.complete(LlmRequest(prompt="hi", operation="generate"))

    assert result is response  # transparent: same object back
    [call] = store.calls()
    assert call.operation == "generate"
    assert call.success is True
    assert call.latency_s == pytest.approx(2.5)
    assert call.ts == 5000.0
    assert (call.model, call.prompt_tokens, call.completion_tokens) == ("gpt-4o-mini", 120, 30)
    assert call.finish_reason == "stop"
    assert call.total_tokens == 150
    assert call.truncated is False


def test_wrapper_flags_truncation_from_finish_reason() -> None:
    store = MetricsStore()
    client = InstrumentedLlmClient(
        _FakeInner(response=LlmResponse(text="x", finish_reason="length")), store=store
    )
    client.complete(LlmRequest(prompt="hi", operation="generate"))
    assert store.calls()[0].truncated is True


def test_wrapper_records_failure_and_reraises_the_same_exception() -> None:
    store = MetricsStore()
    boom = LlmError("upstream down")
    client = InstrumentedLlmClient(
        _FakeInner(error=boom), store=store, clock=_ticking_clock(10.0, 11.25)
    )

    with pytest.raises(LlmError) as raised:
        client.complete(LlmRequest(prompt="hi", operation="interpret"))

    assert raised.value is boom  # not wrapped, not replaced
    [call] = store.calls()
    assert call.success is False
    assert call.operation == "interpret"
    assert call.latency_s == pytest.approx(1.25)
    assert call.error_type == "LlmError"
    assert call.error_message == "upstream down"


def test_wrapper_tolerates_responses_without_usage_fields() -> None:
    """Test doubles only supply text; usage fields must simply be recorded as None."""
    store = MetricsStore()
    client = InstrumentedLlmClient(MockLlmClient.with_responses("plain text"), store=store)

    client.complete(LlmRequest(prompt="hi", operation="revise"))

    [call] = store.calls()
    assert call.success is True
    assert call.prompt_tokens is None and call.completion_tokens is None
    assert call.finish_reason is None
    assert call.total_tokens == 0


def test_wrapper_defaults_operation_to_unknown() -> None:
    store = MetricsStore()
    client = InstrumentedLlmClient(_FakeInner(response=LlmResponse(text="x")), store=store)
    client.complete(object())  # a request with no .operation attribute at all
    assert store.calls()[0].operation == "unknown"


def test_error_messages_are_redacted_and_truncated() -> None:
    store = MetricsStore()
    record_failed_call(
        "interpret",
        LlmError("Incorrect API key provided: sk-abcdef1234567890XYZ. See docs."),
        store=store,
    )
    record_failed_call(
        "generate",
        LlmError("Compatible provider rejected hf_testcredential123456789."),
        store=store,
    )
    record_failed_call(
        "revise",
        LlmError("Authorization: Bearer opaque-provider-token-1234567890"),
        store=store,
    )
    record_failed_call("generate", LlmError("x" * 1000), store=store)

    openai_error, compatible_error, bearer_error, truncated = store.calls()
    assert "abcdef1234567890XYZ" not in openai_error.error_message
    assert "testcredential123456789" not in compatible_error.error_message
    assert "opaque-provider-token-1234567890" not in bearer_error.error_message
    assert "REDACTED" in openai_error.error_message
    assert "REDACTED" in compatible_error.error_message
    assert "Bearer ***REDACTED***" in bearer_error.error_message
    assert len(truncated.error_message) <= 303  # 300 chars + "..."


def test_recording_failure_never_breaks_a_successful_call() -> None:
    class BrokenStore(MetricsStore):
        def record_call(self, call):
            raise RuntimeError("disk full")

    response = LlmResponse(text="fine")
    client = InstrumentedLlmClient(_FakeInner(response=response), store=BrokenStore())
    assert client.complete(LlmRequest(prompt="hi")) is response


def test_recording_failure_never_masks_the_real_exception() -> None:
    class BrokenStore(MetricsStore):
        def record_call(self, call):
            raise RuntimeError("disk full")

    real = LlmError("the real problem")
    client = InstrumentedLlmClient(_FakeInner(error=real), store=BrokenStore())
    with pytest.raises(LlmError) as raised:
        client.complete(LlmRequest(prompt="hi"))
    assert raised.value is real


# --- operation tags set by the agent components ---------------------------------
def test_agent_components_tag_their_requests_with_an_operation() -> None:
    spec = PromptSpecification(goal_confidence=0.9, goal="write a post")
    spec.parameters["audience"] = ParameterValue("audience", "devs", "collected", 1)

    generate_mock = MockLlmClient.echoing()
    PromptGenerator(generate_mock).generate(spec)
    assert generate_mock.calls[0].operation == "generate"

    revise_mock = MockLlmClient.echoing()
    PromptGenerator(revise_mock).revise(
        GeneratedPrompt(text="draft", omitted_required=[]), "shorter", spec
    )
    assert revise_mock.calls[0].operation == "revise"

    interpret_mock = MockLlmClient.with_responses(
        {"goal": "write a post", "confidence": 0.9, "parameters": {}}
    )
    IntentAnalyzer(interpret_mock).interpret("write a post", [], PromptSpecification(goal_confidence=0.0))
    assert interpret_mock.calls[0].operation == "interpret"


# --- MetricsStore -------------------------------------------------------------------
def _call(ts, ok=True, latency=1.0, op="interpret", **kwargs):
    return LlmCall(ts=ts, operation=op, success=ok, latency_s=latency, **kwargs)


def test_store_is_bounded_and_keeps_the_newest_records() -> None:
    store = MetricsStore(max_records=3)
    for ts in range(1, 6):
        store.record_call(_call(float(ts)))
        store.record_event(AppEvent(ts=float(ts), name="turn", success=True))
    assert [c.ts for c in store.calls()] == [3.0, 4.0, 5.0]
    assert [e.ts for e in store.events()] == [3.0, 4.0, 5.0]


def test_store_since_filter_and_clear() -> None:
    store = MetricsStore()
    for ts in (1.0, 2.0, 3.0):
        store.record_call(_call(ts))
    assert [c.ts for c in store.calls(since=2.0)] == [2.0, 3.0]
    store.clear()
    assert store.calls() == [] and store.events() == []


def test_store_returns_copies_not_internal_lists() -> None:
    store = MetricsStore()
    store.record_call(_call(1.0))
    store.calls().clear()
    assert len(store.calls()) == 1


def test_store_reports_synthetic_data() -> None:
    store = MetricsStore()
    assert store.has_synthetic() is False
    store.record_call(_call(1.0, synthetic=True))
    assert store.has_synthetic() is True


# --- percentile ---------------------------------------------------------------------
def test_percentile_known_values() -> None:
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    assert percentile([10.0], 99) == 10.0
    assert percentile([], 50) is None


@settings(max_examples=100)
@given(
    values=st.lists(
        st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=60,
    ),
    p1=st.floats(min_value=0, max_value=100),
    p2=st.floats(min_value=0, max_value=100),
)
def test_percentile_stays_within_bounds_and_is_monotonic(values, p1, p2) -> None:
    """Any percentile lies between min and max, and a higher p never gives less."""
    low_p, high_p = sorted([p1, p2])
    eps = 1e-9 * (1 + max(values))  # tolerate a float rounding step
    low = percentile(values, low_p)
    high = percentile(values, high_p)

    assert min(values) - eps <= low <= high + eps
    assert high <= max(values) + eps
    assert percentile(values, 0) == min(values)
    assert percentile(values, 100) == max(values)


# --- cost & summaries -------------------------------------------------------------------
def test_estimate_cost_uses_longest_prefix_and_ignores_unpriced_models() -> None:
    # 1000 in * $0.15/M + 1000 out * $0.60/M = $0.00075
    assert estimate_cost("gpt-4o-mini", 1000, 1000) == pytest.approx(0.00075)
    assert estimate_cost("gpt-4o-mini-2024-07-18", 1000, 1000) == pytest.approx(0.00075)
    assert estimate_cost("some-other-model", 1000, 1000) is None
    assert estimate_cost(None, 1000, 1000) is None


def test_summary_latency_ignores_failed_calls() -> None:
    calls = [
        _call(1.0, latency=1.0),
        _call(2.0, latency=2.0),
        _call(3.0, latency=3.0),
        _call(4.0, ok=False, latency=0.0),  # fails instantly; must not drag stats down
    ]
    summary = summarize_calls(calls)
    assert (summary.total, summary.successes, summary.failures) == (4, 3, 1)
    assert summary.success_rate == pytest.approx(0.75)
    assert summary.p50_latency_s == 2.0
    assert summary.avg_latency_s == pytest.approx(2.0)


def test_summary_counts_tokens_truncation_and_cost() -> None:
    calls = [
        _call(1.0, model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=500, finish_reason="stop"),
        _call(2.0, model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=500, finish_reason="length"),
    ]
    summary = summarize_calls(calls)
    assert summary.prompt_tokens == 2000
    assert summary.completion_tokens == 1000
    assert summary.total_tokens == 3000
    assert summary.truncated == 1
    assert summary.est_cost_usd == pytest.approx(2 * (1000 * 0.15 + 500 * 0.60) / 1_000_000)


def test_empty_summary_is_all_zero_and_none() -> None:
    summary = summarize_calls([])
    assert summary.total == 0 and summary.success_rate is None
    assert summary.p95_latency_s is None and summary.est_cost_usd is None


def test_summarize_by_operation_groups_calls() -> None:
    calls = [_call(1.0, op="interpret"), _call(2.0, op="generate"), _call(3.0, op="interpret")]
    grouped = summarize_by_operation(calls)
    assert {name: s.total for name, s in grouped.items()} == {"interpret": 2, "generate": 1}


# --- sample traffic -------------------------------------------------------------------
def test_sample_traffic_is_synthetic_and_deterministic() -> None:
    first, second = MetricsStore(), MetricsStore()
    seed_sample_traffic(first, turns=30, seed=3, now=1_000_000.0)
    seed_sample_traffic(second, turns=30, seed=3, now=1_000_000.0)

    assert first.calls() == second.calls()
    assert first.events() == second.events()
    assert first.calls() and first.events()
    assert all(c.synthetic for c in first.calls())
    assert all(e.synthetic for e in first.events())
    assert first.has_synthetic()
