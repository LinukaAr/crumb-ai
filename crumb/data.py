"""
crumb/data.py – CSV loading, validation, cleaning, and aggregation.

Design decisions:
- Missing dates are filled with 0 only when the shop was plausibly open
  (not a Sunday and not a holiday present in the data). See DECISIONS.md D2.
- Column aliases are auto-detected before validation so users don't need
  exact column names.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import IO

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column alias maps
# ---------------------------------------------------------------------------
#: Acceptable column names that map to the canonical "item" column.
_ITEM_ALIASES: list[str] = ["item", "product", "sku", "name", "product_name"]

#: Acceptable column names that map to the canonical "units_sold" column.
_UNITS_ALIASES: list[str] = [
    "units_sold", "units", "qty", "quantity", "sold", "sales_qty", "count",
]

#: Acceptable column names that map to the canonical "date" column.
_DATE_ALIASES: list[str] = ["date", "day", "sale_date", "order_date", "timestamp"]


# ---------------------------------------------------------------------------
# DataReport
# ---------------------------------------------------------------------------
@dataclass
class DataReport:
    """Summary statistics returned alongside the cleaned DataFrame."""

    n_rows: int
    items: list[str]
    date_min: str
    date_max: str
    missing_days: int  # days in range excluded or filled with 0
    thin_items: list[str]  # items with fewer than MIN_HISTORY_ROWS rows


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _detect_column(df: pd.DataFrame, aliases: list[str], role: str) -> str:
    """
    Return the first column name in *df* that matches an alias (case-insensitive).

    Raises ValueError with helpful context if no match is found.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    available = ", ".join(df.columns.tolist())
    raise ValueError(
        f"Cannot find a '{role}' column. "
        f"Expected one of: {', '.join(aliases)}. "
        f"Columns found in your file: {available}."
    )


def _infer_holiday_dates(df: pd.DataFrame) -> set[pd.Timestamp]:
    """Return the set of dates marked as holidays in the dataset."""
    if "is_holiday" not in df.columns:
        return set()
    holiday_dates = df.loc[df["is_holiday"] == 1, "date"]
    return set(holiday_dates)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_csv(source: str | bytes | IO[bytes]) -> tuple[pd.DataFrame, DataReport]:
    """
    Load, validate, clean, and aggregate a sales CSV.

    Parameters
    ----------
    source:
        A file path string, raw bytes, or a file-like object.

    Returns
    -------
    df:
        Cleaned, validated DataFrame with columns:
        ``date, item, units_sold, unit_price_lkr, day_of_week,
          is_holiday, rainy, promo``.
    report:
        A :class:`DataReport` summarising the loaded data.

    Raises
    ------
    ValueError
        If required columns cannot be detected or the data is otherwise invalid.
    """
    from crumb.config import MIN_HISTORY_ROWS

    # ---- Read ---------------------------------------------------------------
    if isinstance(source, bytes):
        if not source.strip():
            raise ValueError("The uploaded CSV is empty.")
        source = io.BytesIO(source)
    try:
        raw = pd.read_csv(source, low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded CSV is empty.") from exc
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    if raw.empty:
        raise ValueError("The uploaded CSV is empty.")

    # ---- Detect required columns -------------------------------------------
    date_col = _detect_column(raw, _DATE_ALIASES, "date")
    item_col = _detect_column(raw, _ITEM_ALIASES, "item")
    units_col = _detect_column(raw, _UNITS_ALIASES, "units_sold")

    # Rename to canonical names
    rename: dict[str, str] = {
        date_col: "date",
        item_col: "item",
        units_col: "units_sold",
    }
    raw = raw.rename(columns=rename)

    # ---- Parse dates --------------------------------------------------------
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    bad_dates = raw["date"].isna().sum()
    if bad_dates > 0:
        raw = raw.dropna(subset=["date"])
        if raw.empty:
            raise ValueError("All date values were unparseable.")

    # ---- Coerce units_sold --------------------------------------------------
    raw["units_sold"] = pd.to_numeric(raw["units_sold"], errors="coerce").fillna(0.0)

    # ---- Optional columns (default to sensible values) ---------------------
    optional_float = {"unit_price_lkr": 0.0, "rainy": 0.0, "promo": 0.0, "is_holiday": 0.0}
    for col, default in optional_float.items():
        if col not in raw.columns:
            raw[col] = default
        else:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(default)

    # ---- day_of_week -------------------------------------------------------
    # Recalculate from date to ensure consistency regardless of CSV content.
    raw["day_of_week"] = raw["date"].dt.dayofweek  # 0=Mon … 6=Sun

    # ---- Aggregate duplicate (date, item) rows by summing units_sold -------
    agg_cols = {
        "units_sold": "sum",
        "unit_price_lkr": "mean",
        "is_holiday": "max",
        "rainy": "max",
        "promo": "max",
        "day_of_week": "first",
    }
    df = raw.groupby(["date", "item"], as_index=False).agg(agg_cols)
    df = df.sort_values(["date", "item"]).reset_index(drop=True)

    # ---- Fill missing dates ------------------------------------------------
    items = df["item"].unique().tolist()
    holiday_dates = _infer_holiday_dates(df)
    date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

    filled_rows: list[pd.DataFrame] = []
    missing_day_count = 0

    for item in items:
        item_df = df[df["item"] == item].set_index("date")
        item_df = item_df.reindex(date_range)

        for ts in date_range:
            if ts not in item_df.index or pd.isna(item_df.loc[ts, "units_sold"]):
                dow = ts.dayofweek  # 0=Mon … 6=Sun
                is_holiday = ts in holiday_dates
                if dow == 6 or is_holiday:
                    # Plausibly closed: mark as NaN so training ignores it
                    item_df.loc[ts, "units_sold"] = np.nan
                else:
                    # Plausibly open: treat as 0 sales day
                    item_df.loc[ts, "units_sold"] = 0.0
                missing_day_count += 1

        item_df["item"] = item
        item_df["day_of_week"] = item_df.index.dayofweek
        # Forward-fill optional flags for context (they rarely change)
        item_df[["unit_price_lkr", "is_holiday", "rainy", "promo"]] = (
            item_df[["unit_price_lkr", "is_holiday", "rainy", "promo"]]
            .ffill()
            .fillna(0.0)
        )
        filled_rows.append(item_df.reset_index().rename(columns={"index": "date"}))

    df = pd.concat(filled_rows, ignore_index=True)
    df = df.sort_values(["date", "item"]).reset_index(drop=True)

    # ---- DataReport --------------------------------------------------------
    item_counts = df.dropna(subset=["units_sold"]).groupby("item").size()
    thin_items = item_counts[item_counts < MIN_HISTORY_ROWS].index.tolist()

    report = DataReport(
        n_rows=len(df),
        items=sorted(items),
        date_min=df["date"].min().strftime("%Y-%m-%d"),
        date_max=df["date"].max().strftime("%Y-%m-%d"),
        missing_days=missing_day_count,
        thin_items=thin_items,
    )

    return df, report


def dataframe_hash(df: pd.DataFrame) -> str:
    """Return a stable SHA-256 hex digest of a DataFrame's CSV representation."""
    csv_bytes = df.to_csv(index=False).encode()
    return hashlib.sha256(csv_bytes).hexdigest()
