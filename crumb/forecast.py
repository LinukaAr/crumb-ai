"""
crumb/forecast.py – TabPFN-based forecasting with prediction intervals.

Design:
- One TabPFNRegressor per item, fitted on historical rows that have
  non-NaN ``units_sold``.
- Quantile output: attempt ``predict(X, quantiles=[0.1, 0.5, 0.9])``.
  If the installed version doesn't support it, fall back to deriving
  intervals from backtest residuals (see DECISIONS.md D1).
- Models are cached by (item, data_hash) to keep chat fast.
- Items with < MIN_HISTORY_ROWS training rows use baseline fallback.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from crumb.config import (
    DEFAULT_HORIZON_DAYS,
    MIN_HISTORY_ROWS,
    QUANTILE_LOWER,
    QUANTILE_MED,
    QUANTILE_UPPER,
)
from crumb.features import FEATURE_COLS, build_features

# ---------------------------------------------------------------------------
# Model cache: (item, data_hash) -> fitted TabPFNRegressor
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def clear_cache() -> None:
    """Invalidate all cached models (called on new CSV upload)."""
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ForecastResult:
    """Output of :func:`forecast`."""

    item: str
    dates: list[str]          # ISO date strings
    median: list[float]       # clipped-at-0, rounded for display
    lower: list[float]        # p10 interval bound
    upper: list[float]        # p90 interval bound
    method: str               # "tabpfn" | "baseline_fallback"
    interval_method: str      # "tabpfn_quantile" | "residual_derived"
    backtest_metrics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _fit_tabpfn(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Fit and return a TabPFNRegressor."""
    try:
        from tabpfn import TabPFNRegressor  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "tabpfn is not installed. Run: pip install tabpfn"
        ) from exc

    model = TabPFNRegressor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    return model


def _predict_quantiles(
    model: Any,
    X_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (lower, median, upper) arrays from a fitted TabPFNRegressor.

    First attempts the native quantile API; falls back to residual-derived
    intervals if that API is unavailable.
    """
    quantiles = [QUANTILE_LOWER, QUANTILE_MED, QUANTILE_UPPER]
    # Try native quantile prediction
    try:
        result = model.predict(X_pred, output_type="quantiles", quantiles=quantiles)
        # result is expected to be a 2-D array of shape (n_samples, 3)
        if hasattr(result, "quantiles"):
            q = np.asarray(result.quantiles)
        else:
            q = np.asarray(result)
        if q.ndim == 2 and q.shape[1] == 3:
            return q[:, 0], q[:, 1], q[:, 2]
    except (TypeError, AttributeError, ValueError):
        pass

    # Fallback: point prediction + residual-derived intervals
    median = np.asarray(model.predict(X_pred), dtype=float)
    return None, median, None  # caller will fill bounds from residuals


def _residual_bounds(
    residuals: np.ndarray, median: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Derive lower/upper bounds from MAD of residuals."""
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    # 1.28 σ ≈ 80% interval using normal approximation
    sigma = 1.4826 * mad  # MAD to σ conversion for normal dist
    lower = median - 1.28 * sigma
    upper = median + 1.28 * sigma
    return lower, upper


def _get_item_series(df: pd.DataFrame, item: str) -> pd.Series:
    """Return a DatetimeIndex series of units_sold for *item*, dropping NaN."""
    sub = df[df["item"] == item][["date", "units_sold"]].copy()
    sub = sub.dropna(subset=["units_sold"])
    sub = sub.set_index("date").sort_index()
    return sub["units_sold"].astype(float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def forecast(
    df: pd.DataFrame,
    data_hash: str,
    item: str,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    overrides: dict | None = None,
    holiday_overrides: list[str] | None = None,
    promo_overrides: dict[str, int] | None = None,
) -> ForecastResult:
    """
    Forecast *item* for the next *horizon_days* days.

    Parameters
    ----------
    df:
        Full cleaned DataFrame from :func:`crumb.data.load_csv`.
    data_hash:
        SHA-256 hash of *df* for model caching.
    item:
        Item name to forecast.
    horizon_days:
        Number of calendar days to forecast ahead.
    overrides:
        Legacy dict kept for compatibility; prefer *holiday_overrides* and
        *promo_overrides*.
    holiday_overrides:
        List of ISO date strings for future holidays.
    promo_overrides:
        Mapping of ISO date string → 1/0 for planned promotions.

    Returns
    -------
    ForecastResult
    """
    from crumb.baselines import forecast_baseline

    if overrides:
        holiday_overrides = overrides.get("holidays", holiday_overrides)
        promo_overrides = overrides.get("promo", promo_overrides)

    # ---- Training data -----------------------------------------------------
    item_df = df[df["item"] == item].dropna(subset=["units_sold"])
    n_rows = len(item_df)

    series = _get_item_series(df, item)
    last_date: pd.Timestamp = series.index.max()
    future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(horizon_days)]

    # ---- Fallback: insufficient history ------------------------------------
    if n_rows < MIN_HISTORY_ROWS:
        preds = forecast_baseline(series, future_dates, method="moving_average")
        return ForecastResult(
            item=item,
            dates=[d.strftime("%Y-%m-%d") for d in future_dates],
            median=[round(max(0.0, v)) for v in preds],
            lower=[round(max(0.0, v * 0.75)) for v in preds],
            upper=[round(max(0.0, v * 1.25)) for v in preds],
            method="baseline_fallback",
            interval_method="heuristic_25pct",
        )

    # ---- Build training features ------------------------------------------
    ref_date = df["date"].min()
    feat_hist = build_features(df[df["item"] == item], reference_date=ref_date)
    # Align to non-NaN rows
    valid_mask = df[df["item"] == item]["units_sold"].notna().values
    X_train = feat_hist.loc[valid_mask, FEATURE_COLS].values.astype(float)
    y_train = item_df["units_sold"].values.astype(float)

    # ---- Cache: fit if needed ----------------------------------------------
    cache_key = (item, data_hash)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = _fit_tabpfn(X_train, y_train)
    model = _MODEL_CACHE[cache_key]

    # ---- Build future features --------------------------------------------
    feat_future = build_features(
        df,
        future_dates=future_dates,
        holiday_overrides=holiday_overrides,
        promo_overrides=promo_overrides,
        reference_date=ref_date,
    )
    X_pred = feat_future[FEATURE_COLS].values.astype(float)

    # ---- Predict quantiles ------------------------------------------------
    lower_arr, median_arr, upper_arr = _predict_quantiles(model, X_pred)

    if lower_arr is None:
        # Derive intervals from training residuals
        y_pred_train = np.asarray(model.predict(X_train), dtype=float)
        residuals = y_train - y_pred_train
        lower_arr, upper_arr = _residual_bounds(residuals, median_arr)
        interval_method = "residual_derived"
    else:
        interval_method = "tabpfn_quantile"

    # ---- Clip and round for display (store raw floats for internal use) ---
    def _clip_round(arr: np.ndarray) -> list[float]:
        return [round(max(0.0, float(v))) for v in arr]

    return ForecastResult(
        item=item,
        dates=[d.strftime("%Y-%m-%d") for d in future_dates],
        median=_clip_round(median_arr),
        lower=_clip_round(lower_arr),
        upper=_clip_round(upper_arr),
        method="tabpfn",
        interval_method=interval_method,
    )
