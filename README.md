# 🍞 Crumb – Local-First Bakery Sales Forecasting

> Built for a friend as part of the DEV Hacktoberfest Weekend Challenge: **Build for a Friend**.

Crumb is an AI-powered sales forecasting assistant for small bakeries and cafés. It lets you upload your sales history as a CSV, explore forecasts for any item, spot unusual sales days, and ask plain-language questions — **all without any data ever leaving your computer**.

![Screenshot placeholder](docs/screenshot.png)

---

## ✨ What it Does

| Feature | Description |
|---------|-------------|
| 📈 **Forecasting** | Predicts the next 1–30 days of sales per item, with a confidence range |
| 🔍 **Anomaly detection** | Flags unusual days (spikes, drops, closures) automatically |
| 💬 **Plain-language chat** | Ask "How many croissants on Saturday?" and get a friendly answer |
| 🔒 **100% local** | No cloud API calls, no telemetry, works offline |

---

## 🛠 Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

### 1. Pull the LLM

```bash
ollama pull gemma3:4b   # ~2 GB one-time download
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Crumb

```bash
uvicorn crumb.app:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.

---

## 📄 CSV Format

Upload any CSV with at least these columns (exact names or common aliases):

| Column | Required | Aliases |
|--------|----------|---------|
| `date` | ✅ | `day`, `sale_date`, `order_date` |
| `item` | ✅ | `product`, `sku`, `name` |
| `units_sold` | ✅ | `qty`, `quantity`, `sold`, `count` |
| `unit_price_lkr` | optional | — |
| `is_holiday` | optional | — |
| `rainy` | optional | — |
| `promo` | optional | — |

A sample file is at `data/sample_bakery_sales.csv` (synthetic data, 2026-01–09).

---

## 🏗 Architecture

```
Browser (vanilla JS + Chart.js)
         │  HTTP (127.0.0.1 only)
         ▼
  FastAPI app (crumb/app.py)
  ┌─────────┬──────────┬──────────┬──────────┐
  │ data.py │features.py│forecast.py│anomalies.py│
  │  (CSV)  │(calendar) │ (TabPFN) │  (MAD)    │
  └────┬────┴──────────┴──────────┴──────────┘
       │                   │
   pandas              TabPFNRegressor
                       (local weights)
         │  HTTP (127.0.0.1 only)
         ▼
   Ollama  (gemma3:4b, local GPU/CPU)
   Route → Execute → Phrase → Post-check
```

**LLM flow:**
1. **Route** – Ollama outputs `{"tool": "...", "args": {...}}` in JSON mode
2. **Execute** – a pure-Python tool runs (pandas / TabPFN), no LLM guessing
3. **Phrase** – Ollama rephrases the tool output in plain language
4. **Post-check** – all numbers in the reply are verified against the tool output

---

## 📊 Backtest Results

Run `python scripts/run_backtest.py` to reproduce:

<!-- BACKTEST TABLE PLACEHOLDER – run the script to fill this in -->

| Item | N rows | MAE TabPFN | MAE Naive | MAE MA | WAPE TabPFN | WAPE Naive | WAPE MA |
|------|--------|-----------|-----------|--------|------------|-----------|---------|
| *Run the script to see results* | | | | | | | |

Results are reported honestly — TabPFN doesn't always win against the baselines on short series.

---

## ⚠ Limitations

- **Synthetic sample data** – the included CSV is generated, not real sales data.
- **Small data** – TabPFN is designed for small tabular datasets. Items with fewer than 60 daily rows fall back to a moving-average baseline.
- **Forecast error** – all forecasts have uncertainty. Always check the range, not just the median.
- **Gemma 3:4b quality** – a small local model will sometimes phrase things oddly. Numbers always come from the deterministic tools.
- **Single user** – state is in-process memory; a server restart clears uploaded data.

---

## 📜 Licenses & Credits

| Component | License | Note |
|-----------|---------|------|
| **TabPFN** | [Apache 2.0](https://github.com/automl/tabpfn) | Local model weights; non-commercial for research use — check the TabPFN license for commercial use |
| **Gemma (via Ollama)** | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) | Local inference only |
| **Chart.js** | MIT | Vendored locally |
| **Crumb** | MIT | This project |

---

## 🤔 Why Open Source and Local?

1. **Privacy** – Your bakery's sales data never leaves your machine.
2. **Zero per-query cost** – No API bills, even if you ask 1 000 questions a day.
3. **Swap models freely** – Change `CRUMB_MODEL=llama3.2:3b` in your env and the whole system uses a different model.
4. **Transparency** – Every number Crumb shows comes from a deterministic pandas/TabPFN computation that you can audit.

---

## 🔧 Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `TABPFN_TOKEN` | *empty* | PriorLabs API Key for TabPFN weights |
| `TABPFN_MODEL_CACHE_DIR` | `.tabpfn_models` | Local directory for downloaded TabPFN weights |
| `HF_HUB_DISABLE_XET` | `1` | Use regular HTTP downloads for large model files |
| `HF_HUB_DOWNLOAD_TIMEOUT` | `60` | Hugging Face download timeout in seconds |
| `CRUMB_MODEL` | `gemma3:4b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API URL |
| `CRUMB_HOST` | `127.0.0.1` | Server bind address |
| `CRUMB_PORT` | `8000` | Server port |
| `CRUMB_ANOMALY_THRESHOLD` | `2.5` | MAD multiplier for anomaly flagging |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
ruff check crumb/ tests/ scripts/
```
