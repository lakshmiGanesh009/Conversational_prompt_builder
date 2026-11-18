"""Shared pytest fixtures.

The telemetry store is process-wide (see ``prompt_builder.telemetry.get_store``)
and the Streamlit ``AppTest`` runs share the test process, so records written by
one test would otherwise be visible to the next. This fixture empties the store
before and after every test so each one starts from a known-clean state.
"""

from __future__ import annotations

import pytest

from prompt_builder.telemetry import get_store


@pytest.fixture(autouse=True)
def _clean_metrics_store():
    get_store().clear()
    yield
    get_store().clear()
