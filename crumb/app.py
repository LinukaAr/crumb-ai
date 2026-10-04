"""
crumb/app.py – FastAPI application.

Endpoints:
  POST /api/upload      – upload CSV, return DataReport + item list
  GET  /api/forecast    – chart data (history, median, lower, upper, method, backtest)
  GET  /api/anomalies   – anomaly list for an item
  POST /api/chat        – {message} → {answer, tool, tool_result}
  GET  /api/health      – Ollama and model status

State is held in module-level variables (single-user, in-process).
Bound to 127.0.0.1 for privacy. 10 MB file-size limit enforced.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import Future
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from crumb.config import CRUMB_MODEL, MAX_UPLOAD_BYTES
from crumb.data import dataframe_hash, load_csv
from crumb.forecast import ForecastResult, clear_cache, forecast
from crumb.tools import ToolRegistry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
_STATE: dict[str, Any] = {
    "df": None,
    "data_hash": "",
    "report": None,
}
_registry = ToolRegistry()
_BACKTEST_CACHE: dict[tuple[str, str], dict[str, float | None]] = {}
_BACKTEST_IN_FLIGHT: set[tuple[str, str]] = set()
_FORECAST_CACHE: dict[tuple[str, str, int], ForecastResult] = {}
_FORECAST_IN_FLIGHT: dict[tuple[str, str, int], Future[ForecastResult]] = {}
_FORECAST_CACHE_LOCK = threading.Lock()
_ANOMALY_CACHE: dict[tuple[str, str, str | None, str | None], dict[str, Any]] = {}


def _backtest_metrics(result: dict[str, Any]) -> dict[str, float | None]:
    """Keep only the metrics exposed by the forecast endpoint."""
    return {
        "mae_tabpfn": result.get("mae_tabpfn"),
        "wape_tabpfn": result.get("wape_tabpfn"),
        "mae_naive": result.get("mae_naive"),
        "mae_ma": result.get("mae_ma"),
    }


def _compute_backtest(df: Any, data_hash: str, item: str) -> None:
    """Compute and cache one item's backtest outside the request path."""
    cache_key = (data_hash, item)
    try:
        from crumb.backtest import _backtest_item

        _BACKTEST_CACHE[cache_key] = _backtest_metrics(_backtest_item(df, item))
    except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Backtest failed for %s: %s", item, exc)
        _BACKTEST_CACHE[cache_key] = {}
    finally:
        _BACKTEST_IN_FLIGHT.discard(cache_key)


def _cached_forecast(
    df: Any,
    data_hash: str,
    item: str,
    days: int,
) -> ForecastResult:
    """Compute one forecast without duplicating concurrent model fits."""
    cache_key = (data_hash, item, days)
    with _FORECAST_CACHE_LOCK:
        cached = _FORECAST_CACHE.get(cache_key)
        if cached is not None:
            return cached
        pending = _FORECAST_IN_FLIGHT.get(cache_key)
        if pending is None:
            pending = Future()
            _FORECAST_IN_FLIGHT[cache_key] = pending
            owner = True
        else:
            owner = False

    if not owner:
        return pending.result()

    try:
        result = forecast(df, data_hash, item, horizon_days=days)
        with _FORECAST_CACHE_LOCK:
            _FORECAST_CACHE[cache_key] = result
            pending.set_result(result)
        return result
    except Exception as exc:
        with _FORECAST_CACHE_LOCK:
            pending.set_exception(exc)
        raise
    finally:
        with _FORECAST_CACHE_LOCK:
            _FORECAST_IN_FLIGHT.pop(cache_key, None)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Crumb starting — binding to 127.0.0.1 (local only)")
    yield
    logger.info("Crumb stopped")


app = FastAPI(
    title="Crumb",
    description="Local-first bakery sales forecasting assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):  # noqa: B008
    """
    Accept a CSV upload, load and validate it, cache state, and return
    the DataReport plus the item list.
    """
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
        )

    try:
        df, report = load_csv(content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Clear the model cache and update state
    clear_cache()
    _BACKTEST_CACHE.clear()
    _BACKTEST_IN_FLIGHT.clear()
    with _FORECAST_CACHE_LOCK:
        _FORECAST_CACHE.clear()
    _ANOMALY_CACHE.clear()
    h = dataframe_hash(df)
    _STATE["df"] = df
    _STATE["data_hash"] = h
    _STATE["report"] = report
    _registry.set_data(df, h)

    return {
        "report": {
            "n_rows": report.n_rows,
            "items": report.items,
            "date_min": report.date_min,
            "date_max": report.date_max,
            "missing_days": report.missing_days,
            "thin_items": report.thin_items,
        },
        "items": report.items,
    }


@app.get("/api/forecast")
async def get_forecast(
    background_tasks: BackgroundTasks,
    item: str = Query(..., description="Item name"),
    days: int = Query(14, ge=1, le=90, description="Forecast horizon in days"),
):
    """
    Return forecast data for Chart.js rendering:
    history (last 60 days), forecast (median, lower, upper), method, backtest metrics.
    """
    df = _STATE.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No data loaded. Upload a CSV first.")

    if item not in df["item"].unique():
        raise HTTPException(status_code=404, detail=f"Item '{item}' not found.")

    # History: last 60 non-NaN rows for the item
    item_df = (
        df[df["item"] == item]
        .dropna(subset=["units_sold"])
        .sort_values("date")
        .tail(60)
    )
    history = {
        "dates": [d.strftime("%Y-%m-%d") for d in item_df["date"]],
        "units": [round(float(u)) for u in item_df["units_sold"]],
    }

    # TabPFN fitting is CPU/GPU-heavy. Keep it out of the async event loop so
    # changing the item remains responsive while another item is fitting.
    result = await run_in_threadpool(
        _cached_forecast,
        df,
        _STATE["data_hash"],
        item,
        days,
    )

    # Backtests retrain models, so cache them and compute them after the fast
    # forecast response has been sent. The frontend refreshes while pending.
    cache_key = (_STATE["data_hash"], item)
    backtest_pending = cache_key not in _BACKTEST_CACHE
    if backtest_pending and cache_key not in _BACKTEST_IN_FLIGHT:
        _BACKTEST_IN_FLIGHT.add(cache_key)
        background_tasks.add_task(
            _compute_backtest,
            df.copy(deep=True),
            _STATE["data_hash"],
            item,
        )
    backtest_metrics = _BACKTEST_CACHE.get(cache_key, {})

    return {
        "item": item,
        "history": history,
        "forecast": {
            "dates": result.dates,
            "median": result.median,
            "lower": result.lower,
            "upper": result.upper,
        },
        "method": result.method,
        "interval_method": result.interval_method,
        "warning": result.warning,
        "backtest_metrics": backtest_metrics,
        "backtest_pending": backtest_pending,
        "model": CRUMB_MODEL,
    }


@app.get("/api/anomalies")
async def get_anomalies(
    item: str = Query(..., description="Item name"),
    start: str | None = Query(None, description="Start date YYYY-MM-DD"),
    end: str | None = Query(None, description="End date YYYY-MM-DD"),
):
    """Return anomaly list for an item."""
    df = _STATE.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No data loaded. Upload a CSV first.")

    if item not in df["item"].unique():
        raise HTTPException(status_code=404, detail=f"Item '{item}' not found.")

    cache_key = (_STATE["data_hash"], item, start, end)
    if cache_key in _ANOMALY_CACHE:
        return _ANOMALY_CACHE[cache_key]

    from dataclasses import asdict

    from crumb.anomalies import find_anomalies, find_possible_closures

    anomalies = find_anomalies(df, item, start=start, end=end)
    closures = find_possible_closures(df)

    response = {
        "item": item,
        "anomalies": [asdict(a) for a in anomalies],
        "possible_closure_dates": closures,
    }
    _ANOMALY_CACHE[cache_key] = response
    return response


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Process a plain-language question and return an AI-phrased answer.
    """
    if not _registry.has_data():
        return {
            "answer": "Please upload a CSV file first so I can answer questions about your sales.",
            "tool": None,
            "tool_result": {},
        }

    from crumb.llm import chat
    try:
        result = chat(request.message, _registry)
    except Exception as exc:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(exc))

    return result


@app.get("/api/health")
async def health():
    """Check Ollama and model status."""
    from crumb.llm import check_ollama_health
    ollama_status = check_ollama_health()
    data_loaded = _STATE.get("df") is not None
    report = _STATE.get("report")
    return {
        "data_loaded": data_loaded,
        "items": _STATE["report"].items if report else [],
        "ollama": ollama_status,
        "model": CRUMB_MODEL,
    }


# ---------------------------------------------------------------------------
# Static files (serve web/ directory)
# ---------------------------------------------------------------------------
import pathlib

_WEB_DIR = pathlib.Path(__file__).parent.parent / "web"

if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        index = _WEB_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        raise HTTPException(status_code=404, detail="index.html not found")
