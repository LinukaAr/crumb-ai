"""
crumb/features.py – build calendar and trend features for model training and forecasting.

Only features knowable in advance are used for future-date predictions:
- day_of_week (int 0-6), is_weekend, day_of_month, month, week_of_year
- trend_index (days since earliest date in the full dataset)
- is_holiday (from data history; defaults to 0 for future unless user supplies list)
- promo (optional user override per date)

The ``rainy`` column is intentionally excluded from future-forecast features
because it is unknown in advance. It is kept in the anomaly context only.
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def build_features(
    df: pd.DataFrame,
    future_dates: Sequence[pd.Timestamp] | None = None,
    holiday_overrides: Sequence[str] | None = None,
    promo_overrides: dict[str, int] | None = None,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Return a feature DataFrame for *df* (historical) or *future_dates* (forecast).

    Parameters
    ----------
    df:
        Cleaned historical DataFrame as returned by :func:`crumb.data.load_csv`.
        Used to determine the reference start date for the trend index, and as
        the source of historical rows when ``future_dates`` is None.
    future_dates:
        If provided, build features for these future dates instead of the
        historical rows. ``df`` is still required for the trend reference.
    holiday_overrides:
        ISO date strings (``YYYY-MM-DD``) that should be marked ``is_holiday=1``
        in future features. Defaults to no holidays.
    promo_overrides:
        Mapping of ISO date string → 1/0. Dates absent from this dict default to 0.
    reference_date:
        The day-zero anchor for the trend index. Defaults to ``df["date"].min()``.

    Returns
    -------
    pd.DataFrame
        Columns: ``date, day_of_week, is_weekend, day_of_month, month,
        week_of_year, trend_index, is_holiday, promo``.
    """
    if reference_date is None:
        reference_date = df["date"].min()

    holiday_set: set[pd.Timestamp] = set()
    if holiday_overrides:
        for s in holiday_overrides:
            holiday_set.add(pd.Timestamp(s))

    promo_map: dict[pd.Timestamp, int] = {}
    if promo_overrides:
        for s, v in promo_overrides.items():
            promo_map[pd.Timestamp(s)] = int(bool(v))

    if future_dates is not None:
        # Build feature rows for the requested future dates
        dates = pd.DatetimeIndex(future_dates)
        feat = pd.DataFrame({"date": dates})
        feat["day_of_week"] = dates.dayofweek
        feat["is_weekend"] = (dates.dayofweek >= 5).astype(int)
        feat["day_of_month"] = dates.day
        feat["month"] = dates.month
        feat["week_of_year"] = dates.isocalendar().week.astype(int).values
        feat["trend_index"] = (dates - reference_date).days
        feat["is_holiday"] = feat["date"].apply(
            lambda d: 1 if pd.Timestamp(d) in holiday_set else 0
        )
        feat["promo"] = feat["date"].apply(
            lambda d: promo_map.get(pd.Timestamp(d), 0)
        )
        return feat.reset_index(drop=True)

    # Historical features (one row per row in df, preserving index)
    src = df.copy()
    dates = pd.DatetimeIndex(src["date"])
    feat = pd.DataFrame(index=src.index)
    feat["date"] = src["date"].values
    feat["day_of_week"] = dates.dayofweek
    feat["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    feat["day_of_month"] = dates.day
    feat["month"] = dates.month
    feat["week_of_year"] = dates.isocalendar().week.astype(int).values
    feat["trend_index"] = (dates - reference_date).days
    # For historical data: use the is_holiday flag from the CSV
    feat["is_holiday"] = src["is_holiday"].fillna(0).astype(int).values
    feat["promo"] = src["promo"].fillna(0).astype(int).values
    return feat.reset_index(drop=True)


#: Feature column names used for model training and prediction.
FEATURE_COLS: list[str] = [
    "day_of_week",
    "is_weekend",
    "day_of_month",
    "month",
    "week_of_year",
    "trend_index",
    "is_holiday",
    "promo",
]
