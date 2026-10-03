"""
tests/test_forecast.py – tests for crumb.forecast module.

These tests mock TabPFN so they don't require the model weights.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from crumb.forecast import clear_cache, forecast


def _make_df(n_days: int = 80, item: str = "Croissant") -> pd.DataFrame:
    """Build a minimal DataFrame with enough history for TabPFN."""
    start = pd.Timestamp("2026-01-05")  # Monday
    rows = []
    for i in range(n_days):
        d = start + pd.Timedelta(days=i)
        if d.dayofweek == 6:
            continue  # skip Sundays
        rows.append({
            "date": d, "item": item, "units_sold": 20.0 + i * 0.1,
            "day_of_week": d.dayofweek, "is_holiday": 0, "rainy": 0, "promo": 0,
            "unit_price_lkr": 280,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baseline fallback (thin item)
# ---------------------------------------------------------------------------
def test_baseline_fallback_thin_item():
    df = _make_df(n_days=30)  # < 60 days
    df_hash = "testhash"
    result = forecast(df, df_hash, "Croissant", horizon_days=7)
    assert result.method == "baseline_fallback"
    assert len(result.dates) == 7
    assert all(v >= 0 for v in result.median)


def test_baseline_result_has_intervals():
    df = _make_df(n_days=30)
    result = forecast(df, "hash", "Croissant", horizon_days=3)
    for lo, med, hi in zip(result.lower, result.median, result.upper):
        assert lo <= med <= hi


# ---------------------------------------------------------------------------
# TabPFN path (mocked)
# ---------------------------------------------------------------------------
class MockTabPFN:
    def fit(self, X, y):
        self._y_mean = float(np.mean(y))

    def predict(self, X, **kwargs):
        return np.full(len(X), self._y_mean)


def _fitted_mock_tabpfn(X, y):
    model = MockTabPFN()
    model.fit(X, y)
    return model


@patch("crumb.forecast._fit_tabpfn", side_effect=_fitted_mock_tabpfn)
def test_tabpfn_path(mock_fit):
    df = _make_df(n_days=80)
    clear_cache()
    result = forecast(df, "hash80", "Croissant", horizon_days=7)
    assert result.method == "tabpfn"
    assert len(result.dates) == 7
    assert all(v >= 0 for v in result.median)


@patch("crumb.forecast._fit_tabpfn", side_effect=RuntimeError("download failed"))
def test_tabpfn_download_failure_uses_baseline(mock_fit):
    df = _make_df(n_days=80)
    clear_cache()
    result = forecast(df, "hash-download-failure", "Croissant", horizon_days=5)
    assert result.method == "baseline_fallback"
    assert result.warning is not None
    assert len(result.median) == 5


@patch("crumb.forecast._fit_tabpfn", side_effect=_fitted_mock_tabpfn)
def test_predictions_clipped_at_zero(mock_fit):
    """Negative raw predictions should be clipped to 0."""
    df = _make_df(n_days=80)
    clear_cache()
    result = forecast(df, "hashclip", "Croissant", horizon_days=5)
    assert all(v >= 0 for v in result.lower)
    assert all(v >= 0 for v in result.median)
    assert all(v >= 0 for v in result.upper)


@patch("crumb.forecast._fit_tabpfn", side_effect=_fitted_mock_tabpfn)
def test_model_cache_reuse(mock_fit):
    """Second call with same hash should not refit the model."""
    df = _make_df(n_days=80)
    clear_cache()
    forecast(df, "hashcache", "Croissant", horizon_days=3)
    forecast(df, "hashcache", "Croissant", horizon_days=3)
    # fit should have been called exactly once
    assert mock_fit.call_count == 1


@patch("crumb.forecast._fit_tabpfn", side_effect=_fitted_mock_tabpfn)
def test_cache_cleared_on_clear_cache(mock_fit):
    df = _make_df(n_days=80)
    clear_cache()
    forecast(df, "hashclear", "Croissant", horizon_days=3)
    clear_cache()
    forecast(df, "hashclear", "Croissant", horizon_days=3)
    assert mock_fit.call_count == 2
