"""
crumb/config.py – centralised configuration and constants.

All tuneable values live here. Override via environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Keep downloaded local TabPFN weights inside the project by default. This
# avoids platform-specific cache permission issues and keeps inference local.
TABPFN_MODEL_CACHE_DIR: str = os.environ.get(
    "TABPFN_MODEL_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent / ".tabpfn_models"),
)
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", TABPFN_MODEL_CACHE_DIR)

# ---------------------------------------------------------------------------
# Ollama / LLM
# ---------------------------------------------------------------------------
#: Default model name; override with CRUMB_MODEL env var.
CRUMB_MODEL: str = os.environ.get("CRUMB_MODEL", "gemma3:4b")

#: Base URL for the local Ollama HTTP API.
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

#: Timeout in seconds for individual Ollama requests.
OLLAMA_TIMEOUT: float = float(os.environ.get("OLLAMA_TIMEOUT", "60"))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
#: Maximum uploaded CSV file size in bytes (10 MB).
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024

#: Minimum number of daily rows required to use TabPFN (else baseline fallback).
MIN_HISTORY_ROWS: int = 60

# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
#: Default forecast horizon in days.
DEFAULT_HORIZON_DAYS: int = 14

#: Quantile levels for prediction intervals (lower, median, upper).
QUANTILE_LOWER: float = 0.10
QUANTILE_MED: float = 0.50
QUANTILE_UPPER: float = 0.90

# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
#: Number of median-absolute-deviations above which a residual is anomalous.
ANOMALY_MAD_MULTIPLIER: float = float(
    os.environ.get("CRUMB_ANOMALY_THRESHOLD", "2.5")
)

#: Minimum number of same-weekday observations needed before anomaly scoring.
ANOMALY_MIN_SAME_WEEKDAY: int = 4

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
#: Host the FastAPI app binds to (localhost-only by default for privacy).
APP_HOST: str = os.environ.get("CRUMB_HOST", "127.0.0.1")
APP_PORT: int = int(os.environ.get("CRUMB_PORT", "8000"))
