"""
scripts/run_backtest.py – print a Markdown table comparing TabPFN vs baselines.

Run from the repo root:
    python scripts/run_backtest.py data/sample_bakery_sales.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the crumb package importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from crumb.backtest import run_backtest
from crumb.data import load_csv


def _fmt(val: float | None, pct: bool = False) -> str:
    if val is None or pd.isna(val):
        return "-"
    if pct:
        return f"{val:.1%}"
    return f"{val:.1f}"


def main(csv_path: str = "data/sample_bakery_sales.csv") -> None:
    print(f"\nLoading {csv_path}...")
    df, report = load_csv(csv_path)
    print(f"  {report.n_rows} rows | {len(report.items)} items | {report.date_min} -> {report.date_max}")
    print(f"  Thin items (< 60 rows): {report.thin_items or 'none'}\n")

    print("Running rolling-origin backtest (hold-out: last 28 days, step: 7 days)...")
    results = run_backtest(df)

    # Print Markdown table
    header = (
        "| Item | N rows | MAE TabPFN | MAE Naive | MAE MA "
        "| WAPE TabPFN | WAPE Naive | WAPE MA |"
    )
    sep = (
        "|------|--------|------------|-----------|--------|"
        "-------------|------------|---------|"
    )
    print("\n### Backtest Results\n")
    print(header)
    print(sep)
    for _, row in results.iterrows():
        item = row["item"]
        n = int(row["n_rows"]) if not pd.isna(row["n_rows"]) else "-"
        print(
            f"| {item} | {n} "
            f"| {_fmt(row['mae_tabpfn'])} "
            f"| {_fmt(row['mae_naive'])} "
            f"| {_fmt(row['mae_ma'])} "
            f"| {_fmt(row['wape_tabpfn'], pct=True)} "
            f"| {_fmt(row['wape_naive'], pct=True)} "
            f"| {_fmt(row['wape_ma'], pct=True)} |"
        )
    print()


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_bakery_sales.csv"
    main(csv_path)
