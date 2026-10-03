"""
crumb/tools.py – tool registry with JSON schemas and pure-Python implementations.

Five tools:
1. forecast(item, start_date, end_date)
2. find_anomalies(item, start_date, end_date)
3. summarize(item, period)
4. top_items(period, metric)
5. list_items()

All tools validate their arguments and return JSON-serialisable dicts.
Errors are returned as {"error": "..."} instead of raising exceptions.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI-style JSON Schema)
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "forecast",
        "description": (
            "Forecast sales for a specific item over a date range. "
            "Returns median prediction, lower/upper bounds, method, and backtest metrics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "Item name to forecast."},
                "start_date": {
                    "type": "string",
                    "description": "Start date of the forecast window (YYYY-MM-DD).",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date of the forecast window (YYYY-MM-DD).",
                },
            },
            "required": ["item", "start_date", "end_date"],
        },
    },
    {
        "name": "find_anomalies",
        "description": (
            "Find unusual sales days for an item within a date range. "
            "Flags days where sales deviate significantly from expected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "Item name to analyse."},
                "start_date": {
                    "type": "string",
                    "description": "Start date (YYYY-MM-DD). Optional.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date (YYYY-MM-DD). Optional.",
                },
            },
            "required": ["item"],
        },
    },
    {
        "name": "summarize",
        "description": (
            "Summarise sales for an item over a period. "
            "Returns total, average per day, best/worst weekday, and vs-prior change."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "Item name. Use '__all__' for all items."},
                "period": {
                    "type": "string",
                    "enum": ["last_7_days", "last_30_days"],
                    "description": (
                        "Period to summarise: 'last_7_days', 'last_30_days', "
                        "or 'month:YYYY-MM'."
                    ),
                },
            },
            "required": ["item", "period"],
        },
    },
    {
        "name": "top_items",
        "description": "Rank items by units sold or revenue over a period.",
        "parameters": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Period: 'last_7_days', 'last_30_days', or 'month:YYYY-MM'.",
                },
                "metric": {
                    "type": "string",
                    "enum": ["units", "revenue"],
                    "description": "Ranking metric.",
                },
            },
            "required": ["period", "metric"],
        },
    },
    {
        "name": "list_items",
        "description": "List all item names available in the loaded sales data.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]

# name → schema lookup
TOOL_SCHEMA_MAP: dict[str, dict] = {s["name"]: s for s in TOOL_SCHEMAS}


# ---------------------------------------------------------------------------
# Tool registry class
# ---------------------------------------------------------------------------
class ToolRegistry:
    """
    Holds a reference to the loaded DataFrame and dispatches tool calls.

    Usage::

        registry = ToolRegistry()
        registry.set_data(df, data_hash)
        result = registry.call("forecast", {"item": "Croissant", "start_date": "...", ...})
    """

    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None
        self._data_hash: str = ""

    def set_data(self, df: pd.DataFrame, data_hash: str) -> None:
        """Register the current loaded DataFrame."""
        self._df = df
        self._data_hash = data_hash

    def has_data(self) -> bool:
        return self._df is not None and not self._df.empty

    # ---- Dispatch ---------------------------------------------------------

    def call(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """
        Validate *args* and call the named tool.

        Returns a JSON-serialisable dict. Never raises; errors come back as
        ``{"error": "..."}``.
        """
        if not self.has_data():
            return {"error": "No data loaded. Please upload a CSV file first."}

        dispatch = {
            "forecast": self._forecast,
            "find_anomalies": self._find_anomalies,
            "summarize": self._summarize,
            "top_items": self._top_items,
            "list_items": self._list_items,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return {"error": f"Unknown tool '{tool_name}'."}
        try:
            return fn(**args)
        except TypeError as exc:
            return {"error": f"Invalid arguments for '{tool_name}': {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    # ---- Tool implementations --------------------------------------------

    def _list_items(self) -> dict:
        items = sorted(self._df["item"].unique().tolist())
        return {"items": items}

    def _forecast(
        self,
        item: str,
        start_date: str,
        end_date: str,
    ) -> dict:
        from crumb.forecast import forecast as _forecast_fn

        if not _validate_item(item, self._df):
            return {"error": f"Item '{item}' not found. Use list_items to see available items."}

        try:
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
        except (TypeError, ValueError):
            return {"error": "Invalid date format. Use YYYY-MM-DD."}

        horizon = (end - start).days + 1
        if horizon <= 0:
            return {"error": "end_date must be after start_date."}
        if horizon > 90:
            return {"error": "Forecast horizon cannot exceed 90 days."}

        result = _forecast_fn(self._df, self._data_hash, item, horizon_days=horizon)
        # Filter to requested window
        dates_in_window = [
            (d, m, lo, hi)
            for d, m, lo, hi in zip(
                result.dates, result.median, result.lower, result.upper
            )
            if start_date <= d <= end_date
        ]
        if not dates_in_window:
            return {"error": "No forecast dates in the requested window."}

        return {
            "item": item,
            "dates": [r[0] for r in dates_in_window],
            "median": [r[1] for r in dates_in_window],
            "lower_p10": [r[2] for r in dates_in_window],
            "upper_p90": [r[3] for r in dates_in_window],
            "method": result.method,
            "interval_method": result.interval_method,
            "warning": result.warning,
            "backtest_metrics": result.backtest_metrics,
        }

    def _find_anomalies(
        self,
        item: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        from crumb.anomalies import find_anomalies, find_possible_closures

        if not _validate_item(item, self._df):
            return {"error": f"Item '{item}' not found. Use list_items to see available items."}

        anomalies = find_anomalies(self._df, item, start=start_date, end=end_date)
        closures = find_possible_closures(self._df)

        return {
            "item": item,
            "anomalies": [asdict(a) for a in anomalies],
            "possible_closure_dates": closures,
        }

    def _summarize(self, item: str, period: str) -> dict:
        from crumb.summary import summarize

        _validate_period(period)
        if item != "__all__" and not _validate_item(item, self._df):
            return {"error": f"Item '{item}' not found. Use list_items to see available items."}

        return summarize(self._df, item, period)

    def _top_items(self, period: str, metric: str) -> dict:
        from crumb.summary import top_items

        if metric not in ("units", "revenue"):
            return {"error": "metric must be 'units' or 'revenue'."}
        _validate_period(period)
        rows = top_items(self._df, period, metric)  # type: ignore[arg-type]
        return {"period": period, "metric": metric, "rankings": rows}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _validate_item(item: str, df: pd.DataFrame) -> bool:
    return item in df["item"].unique()


def _validate_period(period: str) -> None:
    valid = {"last_7_days", "last_30_days"}
    if period in valid:
        return
    if period.startswith("month:"):
        ym = period[len("month:"):]
        parts = ym.split("-")
        if len(parts) == 2:
            try:
                _year, month = int(parts[0]), int(parts[1])
                if 1 <= month <= 12:
                    return
            except ValueError:
                pass
    raise ValueError(
        f"Invalid period '{period}'. "
        "Use 'last_7_days', 'last_30_days', or 'month:YYYY-MM'."
    )
