"""
tests/test_features.py – tests for crumb.features module.
"""
from __future__ import annotations

import pandas as pd

from crumb.features import FEATURE_COLS, build_features


def _make_df() -> pd.DataFrame:
    dates = pd.date_range("2026-01-05", periods=14, freq="D")
    rows = []
    for d in dates:
        rows.append({
            "date": d, "item": "Croissant", "units_sold": 10.0,
            "day_of_week": d.dayofweek, "is_holiday": 0, "rainy": 0, "promo": 0,
        })
    return pd.DataFrame(rows)


def test_historical_features_shape():
    df = _make_df()
    feat = build_features(df)
    assert len(feat) == len(df)
    for col in FEATURE_COLS:
        assert col in feat.columns, f"Missing column: {col}"


def test_future_features_shape():
    df = _make_df()
    future = pd.date_range("2026-01-20", periods=7, freq="D")
    feat = build_features(df, future_dates=future)
    assert len(feat) == 7
    for col in FEATURE_COLS:
        assert col in feat.columns


def test_trend_index_increases():
    df = _make_df()
    feat = build_features(df)
    trends = feat["trend_index"].tolist()
    assert trends == sorted(trends)
    assert trends[0] == 0


def test_weekend_flag():
    df = _make_df()
    feat = build_features(df)
    for _, row in feat.iterrows():
        dow = int(row["day_of_week"])
        expected = 1 if dow >= 5 else 0
        assert int(row["is_weekend"]) == expected


def test_holiday_override_applied():
    df = _make_df()
    future = pd.date_range("2026-01-20", periods=3, freq="D")
    holiday = ["2026-01-21"]
    feat = build_features(df, future_dates=future, holiday_overrides=holiday)
    row = feat[feat["date"] == pd.Timestamp("2026-01-21")]
    assert int(row["is_holiday"].iloc[0]) == 1
    row2 = feat[feat["date"] == pd.Timestamp("2026-01-20")]
    assert int(row2["is_holiday"].iloc[0]) == 0


def test_promo_override_applied():
    df = _make_df()
    future = pd.date_range("2026-01-20", periods=3, freq="D")
    feat = build_features(df, future_dates=future, promo_overrides={"2026-01-22": 1})
    row = feat[feat["date"] == pd.Timestamp("2026-01-22")]
    assert int(row["promo"].iloc[0]) == 1


def test_rainy_not_in_future_features():
    """rainy must not appear in future feature rows (unknown for future)."""
    df = _make_df()
    future = pd.date_range("2026-01-20", periods=3, freq="D")
    feat = build_features(df, future_dates=future)
    assert "rainy" not in feat.columns
