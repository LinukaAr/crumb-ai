"""
tests/test_backtest.py – tests for crumb.backtest module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crumb.backtest import _bias, _mae, _wape, run_backtest


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def test_mae_perfect():
    a = np.array([1.0, 2.0, 3.0])
    assert _mae(a, a) == 0.0


def test_wape_zero_actual():
    assert np.isnan(_wape(np.zeros(3), np.ones(3)))


def test_bias_positive():
    actual = np.array([10.0])
    pred   = np.array([12.0])
    assert _bias(actual, pred) > 0


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------
def _make_rich_df(n_days: int = 100, items=("Croissant", "Espresso")) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-05")
    rows = []
    for i in range(n_days):
        d = start + pd.Timedelta(days=i)
        if d.dayofweek == 6:
            continue
        for item in items:
            rows.append({
                "date": d, "item": item, "units_sold": 20.0 + i * 0.05,
                "day_of_week": d.dayofweek, "is_holiday": 0, "rainy": 0, "promo": 0,
                "unit_price_lkr": 280,
            })
    return pd.DataFrame(rows)


def test_run_backtest_returns_dataframe():
    df = _make_rich_df(n_days=60)
    result = run_backtest(df)
    assert isinstance(result, pd.DataFrame)
    assert "mae_naive" in result.columns
    assert "mae_ma" in result.columns


def test_run_backtest_has_overall_row():
    df = _make_rich_df(n_days=60)
    result = run_backtest(df)
    assert "__overall__" in result["item"].values


def test_baseline_metrics_finite_when_enough_data():
    df = _make_rich_df(n_days=60)
    result = run_backtest(df)
    item_row = result[result["item"] == "Croissant"].iloc[0]
    assert not np.isnan(item_row["mae_naive"])
    assert not np.isnan(item_row["mae_ma"])
