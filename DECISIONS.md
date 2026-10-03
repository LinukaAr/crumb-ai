# Crumb – Design Decisions

This file records every design decision made where the spec was silent or where
a library API differed from expectations.

---

## D1 – TabPFN quantile output

**Decision:** Use `TabPFNRegressor.predict(X, quantiles=[0.1, 0.5, 0.9])` if the
installed version supports it (tabpfn ≥ 2.0 exposes this via the `output_type`
parameter or direct `quantiles` kwarg on `predict`). If the call raises
`TypeError`, fall back to generating intervals from backtest residuals:
`lower = median − 1.28 × MAD(residuals)` and `upper = median + 1.28 × MAD(residuals)`.
This is recorded per-forecast in `ForecastResult.interval_method`.

**Reason:** The spec says to document which path is used; we auto-detect at runtime.

---

## D2 – Missing-date filling strategy

**Decision:** A date is considered "plausibly open" if it is not a Sunday **and**
not a known holiday in the dataset. Missing dates meeting this criterion are
filled with `units_sold = 0`. Dates that are Sundays or known holidays are
marked `NaN` and excluded from training.

**Reason:** The spec says "Fill missing dates with 0 only if the shop was
plausibly open." We use Sunday + holiday as the closure heuristic because the
sample data has no sales on those days.

---

## D3 – Ollama JSON mode for routing

**Decision:** Use `/api/generate` with `format: "json"` for the routing step
(not `/api/chat`) to get the most reliable single-JSON-object output. Use
`/api/chat` for the phrase step since it benefits from conversation context.

**Reason:** The spec says "use Ollama's format: json". Testing showed
`/api/generate` with `format: "json"` produces cleaner single-object output
than the chat endpoint.

---

## D4 – Model cache key

**Decision:** Cache fitted `TabPFNRegressor` models by `(item, sha256_of_data)`
where the hash covers the full training DataFrame bytes. Cache is a module-level
dict; it is cleared on new CSV upload.

**Reason:** Spec says "cache fitted models per (item, data hash)". SHA-256 of
the DataFrame's CSV export provides a stable, cheap hash.

---

## D5 – Relative-date resolution

**Decision:** All relative temporal expressions ("Saturday", "next week",
"last month", "last 7 days") are resolved in Python using `datetime.date.today()`
**before** the phrase step. The routing step only identifies which tool and
which temporal intent (e.g., `"period": "last_7_days"`), not the actual dates.

**Reason:** The spec explicitly says "Resolve relative dates in Python using
today's date, with the model only identifying the intent."

---

## D6 – Chart.js vendoring

**Decision:** `chart.min.js` (v4.x) is included in the repo under
`web/vendor/chart.min.js`. It was downloaded once during development from
jsDelivr. End users never need an internet connection.

**Reason:** Spec says "vendored locally, no CDN".

---

## D7 – State management

**Decision:** Uploaded CSV data and fitted models are stored in module-level
variables in `app.py`. A new upload replaces all state and clears the model cache.

**Reason:** Spec says "Keep state in memory per process (single user)."

---

## D8 – Anomaly detection method

**Decision:** Use an expanding-window approach: for day N, fit a rolling median
(window = 7 same-weekday observations) on all data up to day N−1 to generate
`expected`. Flag if `|actual − expected| > 2.5 × MAD(residuals)`. When all
items are 0 on a date, label it "possible_closure" regardless of the threshold.

**Reason:** Expanding window ensures no lookahead; same-weekday rolling median
is simple and interpretable.

## D9 – Backtest without downloaded TabPFN weights

**Decision:** If local TabPFN weights cannot be loaded during a backtest, stop
the TabPFN folds for that item and report the two deterministic baseline
metrics with TabPFN metrics as unavailable. The script continues and labels
the item `baseline_only`.

**Reason:** A missing or unwritable model cache must not produce fabricated
TabPFN scores. The user can still inspect honest baseline performance while
fixing the local model-weight setup.
