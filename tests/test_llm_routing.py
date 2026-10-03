"""
tests/test_llm_routing.py – tests for crumb.llm module.

All Ollama HTTP calls are mocked. Tests cover:
- Successful routing to each of the 5 tools
- Invalid JSON fallback to keyword router
- Post-check number verification (rejects fabricated numbers)
- Ollama unavailable handling
"""
from __future__ import annotations

import json
from unittest.mock import patch

from crumb.llm import (
    OllamaUnavailableError,
    _extract_numbers,
    _keyword_route,
    _verify_numbers,
    chat,
)
from crumb.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_registry() -> ToolRegistry:
    import textwrap

    from crumb.data import load_csv

    data = textwrap.dedent("""
    date,item,units_sold,unit_price_lkr
    2026-01-05,Croissant,10,280
    2026-01-06,Croissant,12,280
    2026-01-07,Croissant,15,280
    2026-01-05,Espresso,20,450
    """).strip().encode()

    df, _ = load_csv(data)
    registry = ToolRegistry()
    registry.set_data(df, "test_hash")
    return registry


# ---------------------------------------------------------------------------
# _extract_numbers
# ---------------------------------------------------------------------------
def test_extract_numbers_basic():
    nums = _extract_numbers("I recommend baking 42 croissants (range: 30–55).")
    assert 42.0 in nums
    assert 30.0 in nums
    assert 55.0 in nums


def test_extract_numbers_empty():
    assert _extract_numbers("No numbers here!") == []


# ---------------------------------------------------------------------------
# _verify_numbers – post-check
# ---------------------------------------------------------------------------
def test_verify_numbers_passes_with_matching():
    tool_result = {"median": [42], "lower_p10": [30], "upper_p90": [55]}
    answer = "You should bake 42 units (range 30–55)."
    assert _verify_numbers(answer, tool_result) is True


def test_verify_numbers_rejects_fabricated():
    """A clearly fabricated number not in tool_result should fail."""
    tool_result = {"median": [42], "lower_p10": [30], "upper_p90": [55]}
    answer = "You should bake 9999 croissants tomorrow."
    assert _verify_numbers(answer, tool_result) is False


def test_verify_numbers_empty_answer():
    """An answer with no numbers should always pass."""
    assert _verify_numbers("I can't tell from this data.", {"total": 5}) is True


def test_verify_numbers_allows_rounding():
    """Rounding within 5% should be accepted."""
    tool_result = {"median": [100]}
    answer = "Bake about 102 units."  # within 5% of 100
    assert _verify_numbers(answer, tool_result) is True


# ---------------------------------------------------------------------------
# Keyword router
# ---------------------------------------------------------------------------
def test_keyword_router_forecast():
    reg = _make_registry()
    tool, args = _keyword_route("How many croissants should I bake?", reg)
    assert tool == "forecast"
    assert "start_date" in args
    assert "end_date" in args


def test_keyword_router_anomalies():
    reg = _make_registry()
    tool, _args = _keyword_route("Anything odd happen last month?", reg)
    assert tool == "find_anomalies"


def test_keyword_router_top_items():
    reg = _make_registry()
    tool, _args = _keyword_route("What are the top selling items?", reg)
    assert tool == "top_items"


def test_keyword_router_list_items():
    reg = _make_registry()
    tool, _args = _keyword_route("List all available items", reg)
    assert tool == "list_items"


# ---------------------------------------------------------------------------
# Routing with mocked Ollama – successful JSON
# ---------------------------------------------------------------------------
@patch("crumb.llm._ollama_generate")
@patch("crumb.llm._ollama_chat")
def test_route_forecast_tool(mock_chat, mock_generate):
    reg = _make_registry()
    mock_generate.return_value = json.dumps({
        "tool": "forecast",
        "args": {"item": "Croissant", "start_date": "2026-01-20", "end_date": "2026-01-26"},
    })
    mock_chat.return_value = "You should bake about 10 units tomorrow (range 8–12)."

    result = chat("How many croissants should I bake next week?", reg)
    assert result["tool"] == "forecast"
    assert "answer" in result


@patch("crumb.llm._ollama_generate")
@patch("crumb.llm._ollama_chat")
def test_route_list_items(mock_chat, mock_generate):
    reg = _make_registry()
    mock_generate.return_value = json.dumps({"tool": "list_items", "args": {}})
    mock_chat.return_value = "You have Croissant and Espresso."

    result = chat("What items do you have?", reg)
    assert result["tool"] == "list_items"


@patch("crumb.llm._ollama_generate")
@patch("crumb.llm._ollama_chat")
def test_route_find_anomalies(mock_chat, mock_generate):
    reg = _make_registry()
    mock_generate.return_value = json.dumps({
        "tool": "find_anomalies",
        "args": {"item": "Croissant"},
    })
    mock_chat.return_value = "No unusual days found."

    result = chat("Did anything odd happen with Croissant?", reg)
    assert result["tool"] == "find_anomalies"


@patch("crumb.llm._ollama_generate")
@patch("crumb.llm._ollama_chat")
def test_route_top_items(mock_chat, mock_generate):
    reg = _make_registry()
    mock_generate.return_value = json.dumps({
        "tool": "top_items",
        "args": {"period": "last_30_days", "metric": "units"},
    })
    mock_chat.return_value = "Espresso is the top item."

    result = chat("What's my best-selling item?", reg)
    assert result["tool"] == "top_items"


# ---------------------------------------------------------------------------
# Invalid JSON fallback → keyword router
# ---------------------------------------------------------------------------
@patch("crumb.llm._ollama_generate")
@patch("crumb.llm._ollama_chat")
def test_invalid_json_falls_back_to_keyword_router(mock_chat, mock_generate):
    reg = _make_registry()
    mock_generate.return_value = "this is not json at all"
    mock_chat.return_value = "I found no unusual days."

    # Should not raise; should fall back to keyword router
    result = chat("How many croissants to bake Saturday?", reg)
    assert result["tool"] is not None
    assert "answer" in result


# ---------------------------------------------------------------------------
# Ollama unavailable
# ---------------------------------------------------------------------------
@patch("crumb.llm._ollama_generate", side_effect=OllamaUnavailableError("Ollama is down"))
def test_ollama_unavailable(mock_generate):
    reg = _make_registry()
    result = chat("How many croissants?", reg)
    assert "Ollama" in result["answer"] or "ollama" in result["answer"].lower()
    assert result["tool"] is None


# ---------------------------------------------------------------------------
# Post-check fallback to template
# ---------------------------------------------------------------------------
@patch("crumb.llm._ollama_generate")
@patch("crumb.llm._ollama_chat")
def test_postcheck_falls_back_to_template_on_fabricated_number(mock_chat, mock_generate):
    reg = _make_registry()
    mock_generate.return_value = json.dumps({
        "tool": "list_items",
        "args": {},
    })
    # Both phrase attempts return a fabricated number
    mock_chat.return_value = "You have 99999 items available."

    result = chat("What items do you have?", reg)
    # Should fall back to template (which won't contain 99999)
    assert "99999" not in result["answer"]
