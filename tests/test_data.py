"""
tests/test_data.py – tests for crumb.data module.

Covers: column alias detection, date parsing, duplicate aggregation,
missing-date filling, bad-date handling, and DataReport fields.
"""
from __future__ import annotations

import textwrap

import pandas as pd
import pytest

from crumb.data import dataframe_hash, load_csv


# ---------------------------------------------------------------------------
# Helper to build CSV bytes from a string
# ---------------------------------------------------------------------------
def csv(text: str) -> bytes:
    return textwrap.dedent(text).strip().encode()


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------
MINIMAL_CSV = csv("""
date,item,units_sold,unit_price_lkr
2026-01-05,Croissant,10,280
2026-01-06,Croissant,12,280
2026-01-07,Croissant,15,280
""")


def test_load_minimal():
    df, report = load_csv(MINIMAL_CSV)
    assert "date" in df.columns
    assert "item" in df.columns
    assert "units_sold" in df.columns
    assert "Croissant" in report.items
    assert report.n_rows > 0


def test_column_alias_qty():
    data = csv("""
    sale_date,product,qty,unit_price_lkr
    2026-01-05,Croissant,10,280
    """)
    df, _report = load_csv(data)
    assert "units_sold" in df.columns
    assert "item" in df.columns
    assert "date" in df.columns


def test_column_alias_quantity():
    data = csv("""
    day,sku,quantity,unit_price_lkr
    2026-01-05,Bread,8,400
    """)
    _df, report = load_csv(data)
    assert "Bread" in report.items


def test_duplicate_rows_are_summed():
    data = csv("""
    date,item,units_sold,unit_price_lkr
    2026-01-05,Croissant,5,280
    2026-01-05,Croissant,7,280
    """)
    df, _ = load_csv(data)
    day = df[(df["date"] == pd.Timestamp("2026-01-05")) & (df["item"] == "Croissant")]
    assert float(day["units_sold"].iloc[0]) == 12.0


def test_bad_dates_are_dropped():
    """Rows with unparseable dates should be silently dropped."""
    data = csv("""
    date,item,units_sold
    2026-01-05,Bread,10
    not-a-date,Bread,99
    """)
    df, _ = load_csv(data)
    assert len(df[df["item"] == "Bread"]) >= 1
    # The bad row should not inflate totals above 10
    # (actual total depends on filled dates with 0, but 99 should not appear)
    assert 99 not in df["units_sold"].values


def test_empty_csv_raises():
    with pytest.raises(ValueError, match="empty"):
        load_csv(b"")


def test_missing_required_column():
    data = csv("""
    date,product,price
    2026-01-05,Bread,400
    """)
    with pytest.raises(ValueError, match="units_sold"):
        load_csv(data)


def test_missing_item_column():
    data = csv("""
    date,units_sold
    2026-01-05,10
    """)
    with pytest.raises(ValueError, match="item"):
        load_csv(data)


# ---------------------------------------------------------------------------
# Missing-date filling
# ---------------------------------------------------------------------------
def test_sunday_dates_are_nan():
    """2026-01-04 is a Sunday; it should be NaN, not 0."""
    data = csv("""
    date,item,units_sold,unit_price_lkr
    2026-01-05,Croissant,10,280
    2026-01-12,Croissant,12,280
    """)
    df, _ = load_csv(data)
    sunday = df[
        (df["item"] == "Croissant") & (df["date"].dt.dayofweek == 6)
    ]
    # All Sundays should be NaN
    assert sunday["units_sold"].isna().all()


def test_weekday_gap_filled_with_zero():
    """A missing weekday between two rows should be filled with 0."""
    data = csv("""
    date,item,units_sold,unit_price_lkr
    2026-01-05,Croissant,10,280
    2026-01-07,Croissant,12,280
    """)
    df, _ = load_csv(data)
    gap = df[(df["item"] == "Croissant") & (df["date"] == pd.Timestamp("2026-01-06"))]
    assert len(gap) == 1
    assert float(gap["units_sold"].iloc[0]) == 0.0


# ---------------------------------------------------------------------------
# DataReport
# ---------------------------------------------------------------------------
def test_data_report_fields():
    _df, report = load_csv(MINIMAL_CSV)
    assert isinstance(report.items, list)
    assert isinstance(report.date_min, str)
    assert isinstance(report.date_max, str)
    assert isinstance(report.missing_days, int)
    assert isinstance(report.thin_items, list)


def test_thin_items_flagged():
    """Items with < 60 rows should appear in thin_items."""
    data = csv("""
    date,item,units_sold
    2026-01-05,Bread,10
    """)
    _, report = load_csv(data)
    assert "Bread" in report.thin_items


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------
def test_dataframe_hash_stable():
    df, _ = load_csv(MINIMAL_CSV)
    h1 = dataframe_hash(df)
    h2 = dataframe_hash(df)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_dataframe_hash_differs_on_change():
    df, _ = load_csv(MINIMAL_CSV)
    h1 = dataframe_hash(df)
    df2 = df.copy()
    df2.loc[0, "units_sold"] = 9999.0
    h2 = dataframe_hash(df2)
    assert h1 != h2
