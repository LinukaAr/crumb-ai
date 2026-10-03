"""
crumb/backtest.py – rolling-origin backtest evaluation.

Methodology:
- Hold out the final 28 days of the series.
- Forecast 7 days at a time, stepping forward until all 28 held-out days
  are covered (4 folds).
- Metrics per item and overall: MAE, WAPE, Bias.
- Compares TabPFN against both baselines honestly (no cherry-picking).
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from crumb.baselines import forecast_baseline
from crumb.config import MIN_HISTORY_ROWS, TABPFN_N_ESTIMATORS
from crumb.features import FEATURE_COLS, build_features

_HOLDOUT_DAYS = 28
_STEP_DAYS = 7
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    total_actual = np.sum(np.abs(actual))
    if total_actual == 0:
        return float("nan")
    return float(np.sum(np.abs(actual - predicted)) / total_actual)


def _bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(predicted - actual))


# ---------------------------------------------------------------------------
# Single-item backtest
# ---------------------------------------------------------------------------
def _backtest_item(
    df: pd.DataFrame,
    item: str,
) -> dict[str, Any]:
    """
    Run rolling-origin backtest for one item.

    Returns a dict with keys: item, mae_tabpfn, wape_tabpfn, bias_tabpfn,
    mae_naive, wape_naive, bias_naive, mae_ma, wape_ma, bias_ma, n_rows, method.
    """
    item_df = df[df["item"] == item].dropna(subset=["units_sold"]).sort_values("date")
    n = len(item_df)
    ref_date = df["date"].min()

    if n <= _HOLDOUT_DAYS:
        return {
            "item": item,
            "mae_tabpfn": float("nan"),
            "wape_tabpfn": float("nan"),
            "bias_tabpfn": float("nan"),
            "mae_naive": float("nan"),
            "wape_naive": float("nan"),
            "bias_naive": float("nan"),
            "mae_ma": float("nan"),
            "wape_ma": float("nan"),
            "bias_ma": float("nan"),
            "n_rows": n,
            "method": "insufficient_data",
        }

    train_df = item_df.iloc[: n - _HOLDOUT_DAYS]
    test_df = item_df.iloc[n - _HOLDOUT_DAYS :]

    actuals_tabpfn: list[float] = []
    preds_tabpfn: list[float] = []
    actuals_naive: list[float] = []
    preds_naive: list[float] = []
    actuals_ma: list[float] = []
    preds_ma: list[float] = []

    use_tabpfn = len(train_df) >= MIN_HISTORY_ROWS

    for fold_start in range(0, _HOLDOUT_DAYS, _STEP_DAYS):
        fold_end = fold_start + _STEP_DAYS
        fold_test = test_df.iloc[fold_start:fold_end]
        fold_dates = fold_test["date"].tolist()
        fold_actual = fold_test["units_sold"].values.astype(float)

        # Series for baselines (include all training + earlier folds)
        available_df = pd.concat(
            [train_df, test_df.iloc[:fold_start]], ignore_index=True
        )
        series = available_df.set_index("date")["units_sold"].astype(float)

        # Baseline forecasts
        naive_preds = forecast_baseline(series, fold_dates, method="naive")
        ma_preds = forecast_baseline(series, fold_dates, method="moving_average")

        actuals_naive.extend(fold_actual)
        preds_naive.extend(naive_preds)
        actuals_ma.extend(fold_actual)
        preds_ma.extend(ma_preds)

        # TabPFN (if enough data)
        if use_tabpfn:
            fold_preds = _tabpfn_predict_fold(
                available_df, df, item, fold_dates, ref_date
            )
            if fold_preds is None:
                logger.warning(
                    "TabPFN is unavailable; reporting baseline-only backtest for %s.",
                    item,
                )
                use_tabpfn = False
                actuals_tabpfn.clear()
                preds_tabpfn.clear()
            else:
                actuals_tabpfn.extend(fold_actual)
                preds_tabpfn.extend(fold_preds)

    result: dict[str, Any] = {
        "item": item,
        "n_rows": n,
        "method": "tabpfn" if use_tabpfn else "baseline_only",
    }

    a_naive = np.array(actuals_naive)
    p_naive = np.array(preds_naive)
    a_ma = np.array(actuals_ma)
    p_ma = np.array(preds_ma)

    result["mae_naive"] = _mae(a_naive, p_naive)
    result["wape_naive"] = _wape(a_naive, p_naive)
    result["bias_naive"] = _bias(a_naive, p_naive)
    result["mae_ma"] = _mae(a_ma, p_ma)
    result["wape_ma"] = _wape(a_ma, p_ma)
    result["bias_ma"] = _bias(a_ma, p_ma)

    if use_tabpfn and actuals_tabpfn:
        a_t = np.array(actuals_tabpfn)
        p_t = np.array(preds_tabpfn)
        result["mae_tabpfn"] = _mae(a_t, p_t)
        result["wape_tabpfn"] = _wape(a_t, p_t)
        result["bias_tabpfn"] = _bias(a_t, p_t)
    else:
        result["mae_tabpfn"] = float("nan")
        result["wape_tabpfn"] = float("nan")
        result["bias_tabpfn"] = float("nan")

    return result


def _tabpfn_predict_fold(
    available_df: pd.DataFrame,
    full_df: pd.DataFrame,
    item: str,
    fold_dates: list[pd.Timestamp],
    ref_date: pd.Timestamp,
) -> list[float] | None:
    """Fit TabPFN on available_df and predict fold_dates."""
    try:
        from tabpfn import TabPFNRegressor  # type: ignore
        feat_hist = build_features(
            available_df.assign(item=item),
            reference_date=ref_date,
        )
        X_train = feat_hist[FEATURE_COLS].values.astype(float)
        y_train = available_df["units_sold"].values.astype(float)

        model = TabPFNRegressor(
            n_estimators=TABPFN_N_ESTIMATORS,
            show_progress_bar=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        feat_future = build_features(
            full_df, future_dates=fold_dates, reference_date=ref_date
        )
        X_pred = feat_future[FEATURE_COLS].values.astype(float)
        preds = np.asarray(model.predict(X_pred), dtype=float)
        return [max(0.0, float(v)) for v in preds]
    except Exception as exc:  # noqa: BLE001
        logger.warning("TabPFN backtest fold unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run rolling-origin backtest for all items in *df*.

    Returns a tidy DataFrame with one row per item and columns for MAE, WAPE,
    and Bias for TabPFN and both baselines.
    """
    items = df["item"].unique().tolist()
    rows = [_backtest_item(df, item) for item in items]
    result = pd.DataFrame(rows)

    # Overall aggregated row (mean of valid values)
    overall: dict[str, Any] = {"item": "__overall__", "n_rows": result["n_rows"].sum()}
    for metric in ["mae_tabpfn", "wape_tabpfn", "bias_tabpfn",
                   "mae_naive", "wape_naive", "bias_naive",
                   "mae_ma", "wape_ma", "bias_ma"]:
        overall[metric] = result[metric].mean(skipna=True)
    overall["method"] = "mixed"
    result = pd.concat([result, pd.DataFrame([overall])], ignore_index=True)

    return result
