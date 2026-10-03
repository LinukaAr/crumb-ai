"""
crumb/baselines.py – simple, interpretable baseline forecasters.

Two methods:
1. Naive same-weekday: the most recent observation for the same weekday.
2. Moving average: mean of the last N same-weekday observations.

These are used when TabPFN cannot be fitted (insufficient history) and as
benchmarks in the backtest comparison table.
"""
from __future__ import annotations

import pandas as pd


def naive_same_weekday(series: pd.Series, target_date: pd.Timestamp) -> float:
    """
    Return the most recent observed value for the same weekday as *target_date*.

    Parameters
    ----------
    series:
        A ``pd.Series`` with a :class:`pd.DatetimeIndex` and float values
        (``units_sold``) for a single item.
    target_date:
        The date to forecast. Only dates strictly before this date are used.

    Returns
    -------
    float
        The last same-weekday value, or 0.0 if none exists.
    """
    dow = target_date.dayofweek
    past = series[series.index < target_date]
    same_dow = past[past.index.dayofweek == dow].dropna()
    if same_dow.empty:
        return 0.0
    return float(same_dow.iloc[-1])


def moving_average_weekday(
    series: pd.Series,
    target_date: pd.Timestamp,
    n: int = 4,
) -> float:
    """
    Return the mean of the last *n* same-weekday observations before *target_date*.

    Parameters
    ----------
    series:
        A ``pd.Series`` with a :class:`pd.DatetimeIndex` and float values.
    target_date:
        The date to forecast.
    n:
        Number of most-recent same-weekday values to average. Default: 4.

    Returns
    -------
    float
        The rolling mean, or 0.0 if no historical same-weekday values exist.
    """
    dow = target_date.dayofweek
    past = series[series.index < target_date]
    same_dow = past[past.index.dayofweek == dow].dropna()
    if same_dow.empty:
        return 0.0
    tail = same_dow.tail(n)
    return float(tail.mean())


def forecast_baseline(
    series: pd.Series,
    future_dates: list[pd.Timestamp],
    method: str = "moving_average",
) -> list[float]:
    """
    Generate baseline forecasts for a list of future dates.

    Parameters
    ----------
    series:
        Historical observations (DatetimeIndex, float).
    future_dates:
        Dates to forecast.
    method:
        ``"naive"`` for same-weekday naive, ``"moving_average"`` for 4-week MA.

    Returns
    -------
    list of float
        Forecasted values, clipped at 0.
    """
    preds: list[float] = []
    # For each future date, extend the series with previously predicted values
    # so the MA can chain predictions forward.
    rolling = series.copy()
    for dt in future_dates:
        if method == "naive":
            val = naive_same_weekday(rolling, dt)
        else:
            val = moving_average_weekday(rolling, dt)
        val = max(0.0, val)
        preds.append(val)
        # Append the prediction so subsequent dates can reference it
        rolling = pd.concat([rolling, pd.Series([val], index=[dt])])
    return preds
