"""Tests for responsive forecast request handling."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from crumb.app import (
    _FORECAST_CACHE,
    _FORECAST_CACHE_LOCK,
    _FORECAST_IN_FLIGHT,
    _cached_forecast,
)
from crumb.forecast import ForecastResult


def test_concurrent_forecast_requests_share_one_fit():
    """Repeated requests for one item must not fit TabPFN twice."""
    with _FORECAST_CACHE_LOCK:
        _FORECAST_CACHE.clear()
        _FORECAST_IN_FLIGHT.clear()

    result = ForecastResult(
        item="Croissant",
        dates=["2026-10-01"],
        median=[10],
        lower=[8],
        upper=[12],
        method="baseline_fallback",
        interval_method="heuristic_25pct",
    )

    def slow_forecast(*_args, **_kwargs):
        time.sleep(0.05)
        return result

    with (
        patch("crumb.app.forecast", side_effect=slow_forecast) as mocked,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        futures = [
            pool.submit(_cached_forecast, object(), "hash", "Croissant", 30)
            for _ in range(2)
        ]
        assert [future.result() for future in futures] == [result, result]

    assert mocked.call_count == 1
