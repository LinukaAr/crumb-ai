"""
tests/test_tools.py – tests for crumb.tools module.

Tests argument validation, error returns, and basic tool dispatch.
"""
from __future__ import annotations

import textwrap

from crumb.tools import ToolRegistry


def _make_registry() -> ToolRegistry:
    """Create a registry with minimal test data."""
    from crumb.data import load_csv

    data = textwrap.dedent("""
    date,item,units_sold,unit_price_lkr
    2026-01-05,Croissant,10,280
    2026-01-06,Croissant,12,280
    2026-01-07,Croissant,15,280
    2026-01-05,Espresso,20,450
    2026-01-06,Espresso,22,450
    """).strip().encode()

    df, _ = load_csv(data)
    registry = ToolRegistry()
    registry.set_data(df, "test_hash")
    return registry


# ---------------------------------------------------------------------------
# list_items
# ---------------------------------------------------------------------------
def test_list_items():
    reg = _make_registry()
    result = reg.call("list_items", {})
    assert "items" in result
    assert "Croissant" in result["items"]
    assert "Espresso" in result["items"]


# ---------------------------------------------------------------------------
# find_anomalies
# ---------------------------------------------------------------------------
def test_find_anomalies_unknown_item():
    reg = _make_registry()
    result = reg.call("find_anomalies", {"item": "DoesNotExist"})
    assert "error" in result


def test_find_anomalies_valid_item():
    reg = _make_registry()
    result = reg.call("find_anomalies", {"item": "Croissant"})
    assert "anomalies" in result
    assert isinstance(result["anomalies"], list)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_valid():
    reg = _make_registry()
    result = reg.call("summarize", {"item": "Croissant", "period": "last_30_days"})
    assert "total" in result or "error" in result  # may be empty period


def test_summarize_invalid_period():
    reg = _make_registry()
    result = reg.call("summarize", {"item": "Croissant", "period": "bad_period"})
    assert "error" in result


def test_summarize_unknown_item():
    reg = _make_registry()
    result = reg.call("summarize", {"item": "UnknownItem", "period": "last_7_days"})
    assert "error" in result


# ---------------------------------------------------------------------------
# top_items
# ---------------------------------------------------------------------------
def test_top_items_units():
    reg = _make_registry()
    result = reg.call("top_items", {"period": "last_30_days", "metric": "units"})
    assert "rankings" in result or "error" in result


def test_top_items_invalid_metric():
    reg = _make_registry()
    result = reg.call("top_items", {"period": "last_7_days", "metric": "bad_metric"})
    assert "error" in result


# ---------------------------------------------------------------------------
# forecast – argument validation
# ---------------------------------------------------------------------------
def test_forecast_bad_dates():
    reg = _make_registry()
    result = reg.call("forecast", {"item": "Croissant", "start_date": "not-a-date", "end_date": "also-bad"})
    assert "error" in result


def test_forecast_end_before_start():
    reg = _make_registry()
    result = reg.call("forecast", {
        "item": "Croissant",
        "start_date": "2026-03-10",
        "end_date": "2026-03-05",
    })
    assert "error" in result


def test_forecast_unknown_item():
    reg = _make_registry()
    result = reg.call("forecast", {
        "item": "GhostPastry",
        "start_date": "2026-01-20",
        "end_date": "2026-01-25",
    })
    assert "error" in result


def test_forecast_resolves_plural_item_and_future_date():
    reg = _make_registry()
    result = reg.call("forecast", {
        "item": "croissants",
        "start_date": "2026-01-10",
        "end_date": "2026-01-10",
    })
    assert result.get("item") == "Croissant"
    assert result.get("dates") == ["2026-01-10"]


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------
def test_unknown_tool():
    reg = _make_registry()
    result = reg.call("explode_everything", {})
    assert "error" in result


# ---------------------------------------------------------------------------
# No data loaded
# ---------------------------------------------------------------------------
def test_no_data_loaded():
    reg = ToolRegistry()
    result = reg.call("list_items", {})
    assert "error" in result
