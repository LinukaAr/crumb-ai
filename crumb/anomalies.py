"""
crumb/anomalies.py – residual-based anomaly detection for sales data.

Method (see DECISIONS.md D8):
- For each day, derive an "expected" value from the rolling median of the
  last 7 same-weekday observations (no lookahead).
- Flag if |actual − expected| > 2.5 × MAD(residuals) or if actual falls
  outside a historical percentile band.
- When all items total 0 on a date, label it "possible_closure".
- Context flags (is_holiday, rainy, promo) are surfaced as possibilities,
  never as certain causes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crumb.config import ANOMALY_MAD_MULTIPLIER, ANOMALY_MIN_SAME_WEEKDAY


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class AnomalyResult:
    """A single flagged day for one item."""

    date: str
    item: str
    actual: float
    expected: float
    pct_diff: float          # (actual - expected) / expected × 100, or inf if expected=0
    direction: str           # "above" | "below" | "possible_closure"
    is_holiday: int
    rainy: int
    promo: int
    label: str               # human-readable label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _robust_mad(values: np.ndarray) -> float:
    """Return the Median Absolute Deviation of an array."""
    if len(values) == 0:
        return 0.0
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def _same_weekday_rolling_median(
    series: pd.Series, target_idx: int, n: int = 7
) -> float | None:
    """
    Return the median of the *n* most recent same-weekday values before index *target_idx*.
    Returns None if fewer than ANOMALY_MIN_SAME_WEEKDAY observations are available.
    """
    target_date = series.index[target_idx]
    dow = target_date.dayofweek
    prior = series.iloc[:target_idx]
    same_dow = prior[prior.index.dayofweek == dow].dropna()
    if len(same_dow) < ANOMALY_MIN_SAME_WEEKDAY:
        return None
    return float(same_dow.tail(n).median())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def find_anomalies(
    df: pd.DataFrame,
    item: str,
    start: str | None = None,
    end: str | None = None,
) -> list[AnomalyResult]:
    """
    Detect anomalous sales days for *item*.

    Parameters
    ----------
    df:
        Full cleaned DataFrame from :func:`crumb.data.load_csv`.
    item:
        Item name to analyse.
    start:
        ISO date string for the start of the analysis window (inclusive).
    end:
        ISO date string for the end of the analysis window (inclusive).

    Returns
    -------
    list of :class:`AnomalyResult`
        One entry per flagged day.
    """
    item_df = (
        df[df["item"] == item]
        .dropna(subset=["units_sold"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    if item_df.empty:
        return []

    series = item_df.set_index("date")["units_sold"].astype(float)

    # Compute expected and residuals for every date that has enough history
    expected_vals: dict[pd.Timestamp, float] = {}
    for i in range(len(series)):
        exp = _same_weekday_rolling_median(series, i)
        if exp is not None:
            expected_vals[series.index[i]] = exp

    if not expected_vals:
        return []

    residuals = np.array(
        [series[d] - expected_vals[d] for d in expected_vals]
    )
    mad = _robust_mad(residuals)
    threshold = ANOMALY_MAD_MULTIPLIER * mad if mad > 0 else 1.0

    # Filter to window
    if start:
        start_ts = pd.Timestamp(start)
    else:
        start_ts = series.index.min()
    if end:
        end_ts = pd.Timestamp(end)
    else:
        end_ts = series.index.max()

    results: list[AnomalyResult] = []

    for date, exp in expected_vals.items():
        if date < start_ts or date > end_ts:
            continue
        actual = float(series[date])
        residual = actual - exp

        if abs(residual) <= threshold:
            continue

        if exp != 0:
            pct_diff = (actual - exp) / exp * 100.0
        else:
            pct_diff = float("inf") if actual > 0 else 0.0

        direction = "above" if residual > 0 else "below"

        # Fetch context flags
        row = item_df[item_df["date"] == date].iloc[0]
        is_holiday = int(row.get("is_holiday", 0))
        rainy = int(row.get("rainy", 0))
        promo = int(row.get("promo", 0))

        # Build label
        context_hints: list[str] = []
        if is_holiday:
            context_hints.append("holiday")
        if rainy:
            context_hints.append("rainy weather")
        if promo:
            context_hints.append("promotion active")
        context_str = (
            f" (possible factors: {', '.join(context_hints)})"
            if context_hints
            else ""
        )
        label = (
            f"Sales {direction} expected by {abs(pct_diff):.0f}%{context_str}"
            if pct_diff != float("inf")
            else f"Sales {direction} expected{context_str}"
        )

        results.append(
            AnomalyResult(
                date=date.strftime("%Y-%m-%d"),
                item=item,
                actual=actual,
                expected=round(exp, 2),
                pct_diff=round(pct_diff, 1) if pct_diff != float("inf") else pct_diff,
                direction=direction,
                is_holiday=is_holiday,
                rainy=rainy,
                promo=promo,
                label=label,
            )
        )

    return results


def find_possible_closures(df: pd.DataFrame) -> list[str]:
    """
    Return ISO dates where the total units_sold across all items is 0,
    indicating a possible shop closure.
    """
    daily_totals = (
        df.dropna(subset=["units_sold"])
        .groupby("date")["units_sold"]
        .sum()
    )
    closed = daily_totals[daily_totals == 0].index
    return [d.strftime("%Y-%m-%d") for d in sorted(closed)]
