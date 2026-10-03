"""
crumb/summary.py – pandas-based sales summaries and item rankings.

All numbers come from deterministic pandas operations. The LLM uses this
output to phrase answers; it never invents figures.
"""
from __future__ import annotations

import calendar
from typing import Literal

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Filter *df* to rows matching *period*.

    Accepts:
    - ``"last_7_days"``
    - ``"last_30_days"``
    - ``"month:YYYY-MM"``
    """
    today = pd.Timestamp.now().normalize()

    if period == "last_7_days":
        cutoff = today - pd.Timedelta(days=7)
        return df[df["date"] > cutoff]
    if period == "last_30_days":
        cutoff = today - pd.Timedelta(days=30)
        return df[df["date"] > cutoff]
    if period.startswith("month:"):
        ym = period[len("month:"):]
        year, month = map(int, ym.split("-"))
        start = pd.Timestamp(year=year, month=month, day=1)
        last_day = calendar.monthrange(year, month)[1]
        end = pd.Timestamp(year=year, month=month, day=last_day)
        return df[(df["date"] >= start) & (df["date"] <= end)]

    raise ValueError(
        f"Unknown period '{period}'. "
        "Use 'last_7_days', 'last_30_days', or 'month:YYYY-MM'."
    )


def _prior_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Return the period immediately before *period* (same length)."""
    today = pd.Timestamp.now().normalize()

    if period == "last_7_days":
        end = today - pd.Timedelta(days=7)
        start = end - pd.Timedelta(days=7)
        return df[(df["date"] > start) & (df["date"] <= end)]
    if period == "last_30_days":
        end = today - pd.Timedelta(days=30)
        start = end - pd.Timedelta(days=30)
        return df[(df["date"] > start) & (df["date"] <= end)]
    if period.startswith("month:"):
        ym = period[len("month:"):]
        year, month = map(int, ym.split("-"))
        # Previous calendar month
        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1
        start = pd.Timestamp(year=year, month=month, day=1)
        last_day = calendar.monthrange(year, month)[1]
        end = pd.Timestamp(year=year, month=month, day=last_day)
        return df[(df["date"] >= start) & (df["date"] <= end)]

    return pd.DataFrame()


_DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, item: str, period: str) -> dict:
    """
    Return a summary dictionary for *item* over *period*.

    Keys:
    - ``total``: total units_sold
    - ``avg_per_day``: average units per open day
    - ``best_weekday``: weekday name with highest average sales
    - ``worst_weekday``: weekday name with lowest average sales
    - ``vs_prior_pct``: % change vs previous period (None if unavailable)
    - ``period``: the period string
    - ``item``: the item name
    """
    period_df = _parse_period(df, period)
    if item != "__all__":
        period_df = period_df[period_df["item"] == item]
    period_df = period_df.dropna(subset=["units_sold"])

    if period_df.empty:
        return {
            "item": item,
            "period": period,
            "total": 0,
            "avg_per_day": 0.0,
            "best_weekday": None,
            "worst_weekday": None,
            "vs_prior_pct": None,
        }

    total = float(period_df["units_sold"].sum())
    n_days = period_df["date"].nunique()
    avg_per_day = total / n_days if n_days > 0 else 0.0

    by_dow = period_df.groupby("day_of_week")["units_sold"].mean()
    best_dow = int(by_dow.idxmax()) if not by_dow.empty else None
    worst_dow = int(by_dow.idxmin()) if not by_dow.empty else None

    best_weekday = _DOW_NAMES[best_dow] if best_dow is not None else None
    worst_weekday = _DOW_NAMES[worst_dow] if worst_dow is not None else None

    # vs prior period
    prior_df = _prior_period(df, period)
    if item != "__all__":
        prior_df = prior_df[prior_df["item"] == item]
    prior_df = prior_df.dropna(subset=["units_sold"])
    prior_total = float(prior_df["units_sold"].sum()) if not prior_df.empty else None

    if prior_total is not None and prior_total > 0:
        vs_prior_pct = round((total - prior_total) / prior_total * 100, 1)
    else:
        vs_prior_pct = None

    return {
        "item": item,
        "period": period,
        "total": round(total),
        "avg_per_day": round(avg_per_day, 1),
        "best_weekday": best_weekday,
        "worst_weekday": worst_weekday,
        "vs_prior_pct": vs_prior_pct,
    }


def top_items(
    df: pd.DataFrame,
    period: str,
    metric: Literal["units", "revenue"] = "units",
) -> list[dict]:
    """
    Return items ranked by *metric* over *period*.

    Each entry:
    - ``rank``, ``item``, ``total_units``, ``total_revenue_lkr``
    """
    period_df = _parse_period(df, period)
    period_df = period_df.dropna(subset=["units_sold"])

    if period_df.empty:
        return []

    grouped = period_df.groupby("item").agg(
        total_units=("units_sold", "sum"),
        total_revenue_lkr=(
            "units_sold",
            lambda x: (x * period_df.loc[x.index, "unit_price_lkr"]).sum(),
        ),
    )

    if metric == "revenue":
        grouped = grouped.sort_values("total_revenue_lkr", ascending=False)
    else:
        grouped = grouped.sort_values("total_units", ascending=False)

    rows = []
    for rank, (item, row) in enumerate(grouped.iterrows(), start=1):
        rows.append({
            "rank": rank,
            "item": item,
            "total_units": round(float(row["total_units"])),
            "total_revenue_lkr": round(float(row["total_revenue_lkr"]), 2),
        })
    return rows
