# Crumb: LLM Implementation Prompt

Paste everything below the line into your coding agent (Claude Code, Copilot CLI, R-CLI, etc.) from an empty repo.

---

## ROLE

You are a senior Python engineer. Build **Crumb**, a local-first sales forecasting assistant for a small shop (a bakery or café), end to end, working and tested. Make decisions yourself where this spec is silent, and list them in `DECISIONS.md`.

## CONTEXT

- Built for the DEV "Hacktoberfest Weekend Challenge: Build for a Friend". Open-source AI must be at the core.
- The user is a shop owner who is not technical. They upload a sales CSV and ask plain-language questions such as "How many croissants should I bake on Saturday?" or "Did anything odd happen to sales in June?"
- **Privacy is the core promise:** all data and all inference stay on the user's machine. No cloud LLM APIs. No telemetry.
- Sample data is at `data/sample_bakery_sales.csv` (synthetic). Columns: `date, item, units_sold, unit_price_lkr, day_of_week, is_holiday, rainy, promo`. The long format has one row per date and item.

## NON-NEGOTIABLE DESIGN RULES

1. **The LLM never computes or invents numbers.** All figures come from deterministic tools (pandas and TabPFN). The LLM only (a) chooses which tool to call and (b) phrases tool output in plain language.
2. **Every forecast shows its uncertainty and its backtest error.** Never present a single number as certain.
3. **Fail honestly.** If data is too thin (fewer than 60 daily rows for an item), say so and fall back to a simpler method, labelled as such.
4. **Everything runs offline** after the initial model downloads.
5. **No Git or GitHub actions.** Do not run `git commit`, `git push`, or open PRs. Edit files, run tests, and report results only.

## TECH STACK

- Python 3.11+, managed with `uv` or `pip` plus `requirements.txt`
- **TabPFN**: use the local `tabpfn` package (TabPFNRegressor), not the hosted client. Note its license terms in the README.
- **LLM**: Gemma through **Ollama** (default `gemma3:4b`, configurable with the `CRUMB_MODEL` env var). Call Ollama over its local HTTP API.
- Backend: **FastAPI**. Frontend: a single static page with vanilla JS and **Chart.js** (vendored locally, no CDN, so it works offline).
- Tests: `pytest`. Lint: `ruff`.

## REPO STRUCTURE

```
crumb/
  README.md
  DECISIONS.md
  LICENSE
  requirements.txt
  data/sample_bakery_sales.csv
  crumb/
    __init__.py
    config.py          # env vars, defaults, model name, thresholds
    data.py            # load, validate, clean, aggregate
    features.py        # calendar and trend features
    forecast.py        # TabPFN forecasting + intervals
    baselines.py       # naive same-weekday and moving average
    backtest.py        # rolling-origin evaluation, metrics
    anomalies.py       # residual-based anomaly detection
    summary.py         # pandas summaries
    tools.py           # tool registry + JSON schemas
    llm.py             # Ollama client, routing, answer phrasing
    app.py             # FastAPI app
  web/
    index.html
    app.js
    style.css
    vendor/chart.min.js
  tests/
    test_data.py test_features.py test_forecast.py
    test_backtest.py test_anomalies.py test_tools.py test_llm_routing.py
  scripts/
    run_backtest.py    # prints a comparison table for the README
```

## MODULE SPECIFICATIONS

### data.py
- Accept long format (`date,item,units_sold,...`). Auto-detect common column-name variants (`qty`, `quantity`, `sold`, `product`, `sku`) and map them. If mapping is ambiguous, return a clear error that lists the columns found.
- Parse dates robustly. Aggregate duplicate (date, item) rows by summing units.
- **Fill missing dates with 0 only if the shop was plausibly open.** Otherwise mark them as NaN and exclude them from training. Document this rule.
- Return a validated DataFrame plus a `DataReport` (rows, items, date range, missing days, items with insufficient history).

### features.py
Features known in advance for any future date:
- day of week (one-hot or integer), is_weekend, day of month, month, week of year
- trend index (days since start)
- `is_holiday` (taken from the CSV for history. For the future, accept a user-supplied holiday list in `config` or the UI, defaulting to none)
- `promo` as an optional user-set flag for future dates

Do **not** use `rainy` as a default feature for future forecasts because it is unknown. Allow an optional override per date. Use it in anomaly explanation only.

### forecast.py
- `forecast(item, horizon_days=14, overrides=None) -> ForecastResult`
- Train one TabPFNRegressor per item on `units_sold` using the features above. Predict the next `horizon_days`.
- Return the median plus a lower and upper bound (10th to 90th percentile). Use TabPFN's quantile output if the installed version supports it. Otherwise derive intervals from backtest residuals. Document which path is used.
- Clip predictions at 0 and round for display only.
- If history is under 60 days for the item, use the baseline and set `method="baseline_fallback"`.
- Cache fitted models per (item, data hash) to keep chat responses fast.

### baselines.py
- Naive: the same weekday last week.
- Moving average: the mean of the last 4 same weekdays.

### backtest.py
- Rolling-origin evaluation: hold out the final 28 days, forecasting 7 days at a time and moving forward.
- Metrics per item and overall: MAE, WAPE, and bias. Compare TabPFN against both baselines.
- Return a tidy DataFrame. `scripts/run_backtest.py` prints a Markdown table. **Report results honestly, even when TabPFN does not win.**

### anomalies.py
- `find_anomalies(item, start=None, end=None)`
- Use an expanding or rolling fit so that each day is predicted without seeing itself. Flag days where |actual − predicted| exceeds a threshold (default 2.5× the robust spread, such as MAD of residuals) or where the actual falls outside the interval.
- Each result contains: date, actual, expected, percent difference, direction, and optional context flags (`is_holiday`, `rainy`, `promo`) that the LLM can mention as possible explanations, never as certain causes.
- Include the closed-day case (all items at 0) as a distinct "possible closure" label.

### summary.py
- `summarize(item, period)` returns the total, average per day, best and worst weekday, and the change versus the previous period. `top_items(period)` ranks items by units and by revenue (`units × unit_price_lkr`).

### tools.py
Registry of exactly these tools, each with a JSON schema and a pure-Python implementation that returns JSON-serializable dicts:
1. `forecast(item, start_date, end_date)`
2. `find_anomalies(item, start_date, end_date)`
3. `summarize(item, period)` where period is one of `last_7_days | last_30_days | month:YYYY-MM`
4. `top_items(period, metric)` where metric is `units | revenue`
5. `list_items()`

Validate arguments strictly. Return `{"error": "..."}` for bad input instead of raising.

### llm.py
- Do **not** rely on native tool calling (Gemma via Ollama may not support it reliably). Use a two-step flow:
  1. **Route:** prompt the model to output a single JSON object `{"tool": "...", "args": {...}}` (use Ollama's `format: "json"`). Validate it against the tool schemas. Resolve relative dates ("Saturday", "next week", "last month") in **Python** using today's date, with the model only identifying the intent. On invalid JSON, retry once, then fall back to **keyword routing** (a simple rules-based router covering the five tools).
  2. **Phrase:** give the model the user question plus the tool result JSON, with a system prompt that says:
     - use only numbers present in the tool output
     - always state the range or uncertainty and the backtest error when forecasting
     - use plain, friendly, short language with no jargon
     - say "I can't tell from this data" when the data does not support an answer
     - never state a cause as fact, only as a possibility
- **Post-check:** extract all numbers from the model's answer and verify each appears in the tool output (allowing rounding and derived percentages). If a number cannot be verified, regenerate once, then fall back to a deterministic template answer.
- Handle Ollama being down or the model being missing with a clear message that explains how to start or pull it.

### app.py
Endpoints:
- `POST /api/upload`: upload a CSV, return the `DataReport` and the item list
- `GET /api/forecast?item=&days=`: JSON for the chart (history, median, lower, upper, method, backtest metrics)
- `GET /api/anomalies?item=`
- `POST /api/chat`: `{message}` returns `{answer, tool, tool_result}`
- `GET /api/health`: reports Ollama and model status

Keep state in memory per process (single user). Bind to `127.0.0.1` by default. Enforce a file size limit of 10 MB.

### web/
- A single page with: file upload, item selector, a Chart.js chart (history line, forecast line, shaded interval band, anomaly markers), a "Tomorrow" card (the median and range for the next day), and a chat panel.
- A visible badge, "Running locally, nothing leaves this computer", plus the current model name and forecast method.
- Readable, mobile-friendly, large tap targets, no external network requests (verify none are made).

## TESTING AND ACCEPTANCE CRITERIA

Write tests and make them pass. The project is done only when **all** of these hold:

1. `pytest` passes, including data-validation edge cases (bad dates, duplicate rows, renamed columns, missing days).
2. On `data/sample_bakery_sales.csv`, `find_anomalies` flags **2026-06-09** (closed), **2026-07-18/19** (promo spike), and **2026-08-10 to 08-12** (slow days) for most items.
3. `scripts/run_backtest.py` prints a Markdown table of MAE and WAPE for TabPFN vs. both baselines.
4. LLM routing tests, using a **mocked** Ollama client, cover each tool plus the invalid-JSON fallback.
5. The number-verification post-check rejects a fabricated number in a unit test.
6. `uvicorn crumb.app:app` starts, the page loads offline (Wi-Fi disabled), and a full flow of upload, chart, and chat answer works.
7. `ruff check` is clean.

## README REQUIREMENTS (for the project's own README, written last)

Include: what Crumb is and who it was built for (leave a `[FRIEND NAME]` placeholder), a screenshot placeholder, setup steps (Python, `ollama pull`, install, run), the CSV format, the architecture diagram (ASCII), the backtest table from `run_backtest.py`, an honest limitations section (small data, synthetic sample, forecast error), license notes for TabPFN and Gemma, and a "Why open source" section covering privacy, zero per-query cost, and the ability to swap models.

## WORKING STYLE

- Build in this order: data, features, baselines, forecast, backtest, anomalies, summary, tools, llm, app, web. Run tests after each module.
- Keep functions small, typed, and documented. No dead code.
- When a library API differs from this spec (for example TabPFN quantile output), adapt and record the change in `DECISIONS.md`.
- At the end, report: what works, what is untested, backtest results, and any deviations from this spec. Do not claim a check passed unless you ran it.
