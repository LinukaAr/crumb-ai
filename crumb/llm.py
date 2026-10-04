"""
crumb/llm.py – two-step LLM routing and answer-phrasing via Ollama.

Flow (see DECISIONS.md D3):
1. ROUTE: POST /api/generate with format:"json" asking the model to emit
   {tool, args}. Validate against tool schemas. Retry once on bad JSON,
   then fall back to keyword routing.
2. PHRASE: POST /api/chat with the tool result, asking the model to phrase
   the answer in plain friendly language.
3. POST-CHECK: extract all numbers from the model's reply and verify each
   appears in the tool output (allowing rounding ±5% and derived percentages).
   If any number fails, regenerate once, then fall back to a template answer.

Relative-date expressions (\"Saturday\", \"next week\") are resolved in Python
before any LLM call.
"""
from __future__ import annotations

import datetime
import json
import logging
import re
from typing import Any

import httpx
import pandas as pd

from crumb.config import (
    CRUMB_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_TOKENS,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_ROUTE_TOKENS,
    OLLAMA_TIMEOUT,
)
from crumb.tools import TOOL_SCHEMA_MAP, ToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ollama client helpers
# ---------------------------------------------------------------------------
_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"


def _ollama_generate(prompt: str, model: str = CRUMB_MODEL) -> str:
    """Call Ollama /api/generate with JSON mode. Returns the raw text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": 0, "num_predict": OLLAMA_ROUTE_TOKENS},
    }
    try:
        resp = httpx.post(_GENERATE_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except httpx.ConnectError:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Start it with: ollama serve"
        )
    except httpx.TimeoutException as exc:
        raise OllamaUnavailableError(
            f"Ollama took too long to respond while loading '{model}'. "
            "Try again once the model is warm, or use a smaller local model."
        ) from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise OllamaModelMissingError(
                f"Model '{model}' not found. Pull it with: ollama pull {model}"
            )
        raise


def _ollama_chat(messages: list[dict], model: str = CRUMB_MODEL) -> str:
    """Call Ollama /api/chat. Returns the assistant content string."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": 0.2, "num_predict": OLLAMA_CHAT_TOKENS},
    }
    try:
        resp = httpx.post(_CHAT_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except httpx.ConnectError:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Start it with: ollama serve"
        )
    except httpx.TimeoutException as exc:
        raise OllamaUnavailableError(
            f"Ollama took too long to respond while loading '{model}'. "
            "Try again once the model is warm, or use a smaller local model."
        ) from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise OllamaModelMissingError(
                f"Model '{model}' not found. Pull it with: ollama pull {model}"
            )
        raise


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class OllamaUnavailableError(RuntimeError):
    pass


class OllamaModelMissingError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Date-resolution helpers
# ---------------------------------------------------------------------------
def _today() -> datetime.date:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().date()


def _resolve_dates_for_forecast(args: dict) -> dict:
    """
    Ensure start_date and end_date are concrete ISO strings.
    If missing or relative, derive from today.
    """
    today = _today()
    args = dict(args)

    # Models sometimes emit a natural-language `date` field alongside a
    # generated range. Prefer the user's date phrase in that case.
    if args.get("date"):
        args["start_date"] = args["date"]
        args["end_date"] = args["date"]

    start_text = str(args.get("start_date") or today.isoformat())
    start_date = _resolve_weekday_or_relative(start_text, today)
    args["start_date"] = start_date.isoformat() if start_date else start_text

    end_text = args.get("end_date")
    if end_text:
        end_date = _resolve_weekday_or_relative(str(end_text), today)
        args["end_date"] = end_date.isoformat() if end_date else str(end_text)
        if start_date and end_date and end_date < start_date:
            args["end_date"] = start_date.isoformat()
    else:
        # A single relative day such as "Saturday" means that day, not a
        # seven-day window ending on an unrelated date.
        args["end_date"] = (
            start_date.isoformat()
            if start_date
            else (today + datetime.timedelta(days=6)).isoformat()
        )

    return args


_WEEKDAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _resolve_weekday_or_relative(text: str, today: datetime.date) -> datetime.date | None:
    """
    Try to parse a human date expression. Returns a date or None if *text*
    looks like an ISO date already.
    """
    text = text.strip().lower()
    if re.match(r"\d{4}-\d{2}-\d{2}", text):
        return None  # already ISO
    if text in ("today",):
        return today
    if text in ("tomorrow",):
        return today + datetime.timedelta(days=1)
    if text in ("yesterday",):
        return today - datetime.timedelta(days=1)
    if text.startswith("next "):
        day_name = text[5:]
        if day_name in _WEEKDAY_NAMES:
            target_dow = _WEEKDAY_NAMES[day_name]
            days_ahead = (target_dow - today.weekday() + 7) % 7
            days_ahead = days_ahead or 7
            return today + datetime.timedelta(days=days_ahead)
    if text.startswith("last "):
        day_name = text[5:]
        if day_name in _WEEKDAY_NAMES:
            target_dow = _WEEKDAY_NAMES[day_name]
            days_behind = (today.weekday() - target_dow + 7) % 7
            days_behind = days_behind or 7
            return today - datetime.timedelta(days=days_behind)
    if text in _WEEKDAY_NAMES:
        # "Saturday" → next Saturday
        target_dow = _WEEKDAY_NAMES[text]
        days_ahead = (target_dow - today.weekday() + 7) % 7
        days_ahead = days_ahead or 7
        return today + datetime.timedelta(days=days_ahead)
    return None


def _resolve_period(period: str | None) -> str:
    """Resolve relative period strings to canonical format."""
    if not period:
        return "last_7_days"
    period = period.strip().lower()
    if "7" in period or "week" in period and "last" in period:
        return "last_7_days"
    if "30" in period or "month" in period and "last" in period:
        return "last_30_days"
    if period in ("last_7_days", "last_30_days"):
        return period
    if period.startswith("month:"):
        return period
    return "last_7_days"


# ---------------------------------------------------------------------------
# Keyword-based fallback router
# ---------------------------------------------------------------------------
def _looks_like_forecast_intent(message: str) -> bool:
    """Recognise inventory questions that must use the forecast tool."""
    msg = message.casefold()
    weekdays = tuple(_WEEKDAY_NAMES)
    return any(
        word in msg
        for word in (
            "forecast", "bake", "make", "predict", "should i", "plan for",
            "expected", "next week", "next month", *weekdays,
        )
    )


def _looks_like_historical_intent(message: str) -> bool:
    """Recognise questions asking what was sold rather than what to make."""
    msg = message.casefold()
    return any(
        phrase in msg
        for phrase in (
            "sold", "sales", "sell count", "how many did", "what happened",
            "what happened on", "how did we do", "revenue", "earnings",
        )
    )


def _extract_date_text(message: str) -> str | None:
    """Extract common date phrases so date questions do not depend on the LLM."""
    patterns = (
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        (
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
            r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
            r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?"
            r"(?:,?\s+\d{4})?\b"
        ),
        r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def _normalise_date_text(text: str, registry: ToolRegistry) -> str | None:
    """Parse a user date and infer an omitted year from the uploaded data."""
    cleaned = re.sub(
        r"(\d)(st|nd|rd|th)\b",
        r"\1",
        text.strip(),
        flags=re.IGNORECASE,
    )
    try:
        parsed = pd.Timestamp(cleaned)
    except (TypeError, ValueError):
        # Dotted dates are commonly entered as day.month.
        match = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?", cleaned)
        if not match:
            return None
        day, month, year = (int(value) if value else None for value in match.groups())
        if year is None:
            year = int(pd.Timestamp(registry._df["date"].max()).year)
        try:
            parsed = pd.Timestamp(year=year, month=month, day=day)
        except (TypeError, ValueError):
            return None
    if not re.search(r"\d{4}", cleaned):
        # Prefer a matching date already present in the CSV for month/day
        matches = registry._df[
            (registry._df["date"].dt.month == parsed.month)
            & (registry._df["date"].dt.day == parsed.day)
        ]
        if not matches.empty:
            parsed = pd.Timestamp(matches["date"].max())
        else:
            parsed = parsed.replace(year=int(pd.Timestamp(registry._df["date"].max()).year))
    return parsed.strftime("%Y-%m-%d")


def _keyword_route(message: str, registry: ToolRegistry) -> tuple[str, dict]:
    """
    Simple rules-based router as fallback when the LLM produces invalid JSON.
    Returns (tool_name, args_dict).
    """
    msg = message.lower()
    items = (
        sorted(registry._df["item"].unique().tolist())
        if registry.has_data()
        else []
    )

    # Detect item mentions
    mentioned_item = None
    for it in items:
        if it.lower() in msg:
            mentioned_item = it
            break

    date_text = _extract_date_text(message)
    if _looks_like_historical_intent(msg) and date_text:
        item = mentioned_item or "__all__"
        date = _normalise_date_text(date_text, registry)
        if date:
            return "summarize", {"item": item, "date": date}

    if _looks_like_forecast_intent(msg) and not _looks_like_historical_intent(msg):
        item = mentioned_item or (items[0] if items else "unknown")
        today = _today()
        requested_weekday = next(
            (day for day in _WEEKDAY_NAMES if day in msg),
            None,
        )
        requested_date = (
            _resolve_weekday_or_relative(requested_weekday, today)
            if requested_weekday
            else None
        )
        start = (requested_date or today).isoformat()
        end = start if requested_date else (today + datetime.timedelta(days=6)).isoformat()
        return "forecast", {"item": item, "start_date": start, "end_date": end}

    if any(w in msg for w in ("anomal", "unusual", "odd", "weird", "strange", "spike", "drop")):
        item = mentioned_item or (items[0] if items else "unknown")
        return "find_anomalies", {"item": item}

    if any(w in msg for w in ("top", "best", "most popular", "rank", "best-sell")):
        period = "last_30_days"
        metric = "revenue" if "revenue" in msg or "money" in msg else "units"
        return "top_items", {"period": period, "metric": metric}

    if any(w in msg for w in ("list", "what item", "which item", "available")):
        return "list_items", {}

    # Default: summarize all items rather than silently choosing the first one.
    item = mentioned_item or "__all__"
    period = "last_7_days" if "week" in msg or "7" in msg else "last_30_days"
    return "summarize", {"item": item, "period": period}


def _has_confident_local_route(message: str) -> bool:
    """Return whether simple rules can route this question without Ollama."""
    msg = message.casefold()
    if _looks_like_forecast_intent(msg) and not _looks_like_historical_intent(msg):
        return True
    if _looks_like_historical_intent(msg) and _extract_date_text(msg):
        return True
    return any(
        phrase in msg
        for phrase in (
            "anomal", "unusual", "odd", "weird", "strange", "spike", "drop",
            "top", "best-selling", "most popular", "rank", "list", "available",
        )
    )


# ---------------------------------------------------------------------------
# Step 1: Route
# ---------------------------------------------------------------------------
_ROUTE_SYSTEM = """\
You are a routing assistant for a bakery sales tool. Given the user's question,
output a single JSON object with two keys:
- "tool": one of forecast, find_anomalies, summarize, top_items, list_items
- "args": an object with the required arguments

For dates, use ISO format YYYY-MM-DD if you know the exact date. For relative
expressions like "Saturday" or "next week", output them as-is (e.g. "saturday")
and they will be resolved by the system. For periods, use one of:
last_7_days, last_30_days, or month:YYYY-MM.
For a question about one date, use summarize with {"item": "...", "date": "YYYY-MM-DD"}.
Use item "__all__" when the question is about the whole uploaded dataset.
Historical questions containing "sold", "sales", or "what happened" must use summarize,
never forecast. Forecast is only for future planning or explicit forecasts.

Output ONLY valid JSON. No explanation.
"""


def _route(message: str, registry: ToolRegistry) -> tuple[str, dict]:
    """
    Determine which tool to call for *message*.

    Returns (tool_name, resolved_args).
    """
    if _has_confident_local_route(message):
        tool_name, args = _keyword_route(message, registry)
        if tool_name == "forecast":
            args = _resolve_dates_for_forecast(args)
        logger.debug("Fast-routed to %s %s", tool_name, args)
        return tool_name, args

    items_hint = ""
    if registry.has_data():
        items = sorted(registry._df["item"].unique().tolist())
        items_hint = f"\nAvailable items: {', '.join(items[:20])}"

    prompt = (
        f"{_ROUTE_SYSTEM}{items_hint}\n\n"
        f"Today is {_today().isoformat()}.\n\n"
        f"User question: {message}\n\n"
        "JSON:"
    )

    raw = ""
    for attempt in range(2):
        try:
            raw = _ollama_generate(prompt)
            data = json.loads(raw)
            tool_name = data.get("tool", "")
            args = data.get("args", {})
            if not isinstance(args, dict):
                raise TypeError("Tool args must be an object")
            if tool_name not in TOOL_SCHEMA_MAP:
                raise ValueError(f"Unknown tool: {tool_name}")
            if _looks_like_forecast_intent(message) and tool_name != "forecast":
                logger.info("Overriding route %s with forecast intent", tool_name)
                tool_name, args = _keyword_route(message, registry)
            elif _looks_like_historical_intent(message) and _extract_date_text(message):
                logger.info("Overriding route %s with historical date lookup", tool_name)
                tool_name, args = _keyword_route(message, registry)
            # Resolve dates
            if tool_name == "forecast":
                args = _resolve_dates_for_forecast(args)
                # Keep the forecast tool contract strict when the model adds
                # unsupported metadata such as `date`.
                args = {
                    key: args[key]
                    for key in ("item", "start_date", "end_date")
                    if key in args
                }
            elif tool_name == "summarize":
                date_text = args.get("date") or _extract_date_text(message)
                if _looks_like_historical_intent(message) and date_text:
                    args["date"] = _normalise_date_text(date_text, registry) or date_text
                if args.get("date") is None:
                    args.pop("date", None)
                args.setdefault("item", "__all__")
                args.setdefault("period", "last_30_days")
            if "period" in args:
                args["period"] = _resolve_period(args.get("period"))
            logger.debug("Routed to %s %s (attempt %d)", tool_name, args, attempt + 1)
            return tool_name, args
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("Route attempt %d failed: %s. Raw: %s", attempt + 1, exc, raw[:200])

    # Fall back to keyword routing
    logger.info("Falling back to keyword router for: %s", message)
    tool_name, args = _keyword_route(message, registry)
    if tool_name == "forecast":
        args = _resolve_dates_for_forecast(args)
    return tool_name, args


# ---------------------------------------------------------------------------
# Step 2: Phrase
# ---------------------------------------------------------------------------
_PHRASE_SYSTEM = """\
You are Crumb, a friendly assistant for a small bakery or café. Your job is to
explain sales data in plain, warm, concise language that a non-technical shop
owner can understand.

STRICT RULES:
1. Only use numbers that appear in the tool_result JSON. Do not invent numbers.
2. When giving a forecast, always mention the range (lower to upper) and the
   fact that it is an estimate.
3. Never state a cause as a fact – say "possibly" or "it might be because".
4. If the data does not support an answer, say "I can't tell from this data."
5. Keep answers short (2-4 sentences). No bullet lists unless listing items.
6. Use a warm, encouraging tone.
"""


def _phrase(message: str, tool_name: str, tool_result: dict) -> str:
    """Ask the LLM to phrase the tool result in plain language."""
    messages = [
        {"role": "system", "content": _PHRASE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"User question: {message}\n\n"
                f"Tool used: {tool_name}\n"
                f"Tool result (JSON): {json.dumps(tool_result, default=str)}\n\n"
                "Please write a friendly, plain-language answer based only on the data above."
            ),
        },
    ]
    return _ollama_chat(messages)


# ---------------------------------------------------------------------------
# Step 3: Post-check
# ---------------------------------------------------------------------------
def _extract_numbers(text: str) -> list[float]:
    """Extract all numbers (integer and decimal) from text."""
    return [float(m) for m in re.findall(r"\b\d+(?:\.\d+)?\b", text)]


def _flatten_numbers(obj: Any) -> set[float]:
    """Recursively extract all numeric values from a JSON-serialisable object."""
    nums: set[float] = set()
    if isinstance(obj, (int, float)):
        nums.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            nums |= _flatten_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            nums |= _flatten_numbers(v)
    return nums


def _verify_numbers(answer: str, tool_result: dict) -> bool:
    """
    Return True if every number in *answer* can be accounted for in *tool_result*.

    Allowances:
    - ±5% rounding/rounding difference
    - Derived percentages (0–100, present if a ratio of two tool numbers ≈ derived value)
    - Common round numbers from simple arithmetic (sum/count)
    """
    answer_nums = _extract_numbers(answer)
    if not answer_nums:
        return True

    tool_nums = _flatten_numbers(tool_result)
    if not tool_nums:
        return len(answer_nums) == 0

    for num in answer_nums:
        # Allow if within 5% of any tool number, or within ±2 absolute
        matched = any(
            abs(num - t) <= max(0.05 * max(abs(t), 1), 2)
            for t in tool_nums
        )
        if matched:
            continue
        # Allow derived percentages: check if num/100 ≈ ratio of two tool numbers
        # (e.g. "25% higher" where 25 = (A-B)/B*100)
        if 0 <= num <= 100:
            for a in tool_nums:
                for b in tool_nums:
                    if b != 0:
                        derived_pct = abs(a - b) / abs(b) * 100
                        if abs(num - derived_pct) <= 5:
                            matched = True
                            break
                if matched:
                    break
        if not matched:
            logger.warning("Number %.2f in answer not found in tool_result", num)
            return False
    return True


# ---------------------------------------------------------------------------
# Template fallback
# ---------------------------------------------------------------------------
def _template_answer(tool_name: str, tool_result: dict) -> str:
    """Generate a safe deterministic answer from tool_result."""
    if "error" in tool_result:
        return f"Sorry, I couldn't retrieve that information: {tool_result['error']}"

    if tool_name == "forecast":
        item = tool_result.get("item", "that item")
        if tool_result.get("dates"):
            date = tool_result["dates"][0]
            med = tool_result["median"][0]
            lo = tool_result["lower_p10"][0]
            hi = tool_result["upper_p90"][0]
            method = tool_result.get("method", "")
            note = " (baseline estimate, limited data)" if method == "baseline_fallback" else ""
            return (
                f"For {item} on {date}: I estimate around {med} units "
                f"(range: {lo}–{hi}{note})."
            )
    if tool_name == "find_anomalies":
        anomalies = tool_result.get("anomalies", [])
        if not anomalies:
            return "No unusual sales days were found in that period."
        dates = ", ".join(a["date"] for a in anomalies[:3])
        return f"I found unusual sales on: {dates}."
    if tool_name == "summarize":
        total = tool_result.get("total", "N/A")
        avg = tool_result.get("avg_per_day", "N/A")
        return f"Total sales: {total} units, averaging {avg} per day."
    if tool_name == "top_items":
        rankings = tool_result.get("rankings", [])
        if rankings:
            top = rankings[0]["item"]
            return f"The top item is {top}."
    if tool_name == "list_items":
        items = tool_result.get("items", [])
        return f"Available items: {', '.join(items)}."
    return "Here is the data: " + json.dumps(tool_result, default=str)[:300]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def chat(
    message: str,
    registry: ToolRegistry,
    model: str = CRUMB_MODEL,
) -> dict[str, Any]:
    """
    Process a user message end-to-end.

    Returns::

        {
            "answer": str,
            "tool": str,
            "tool_result": dict,
        }
    """
    # ---- Route -------------------------------------------------------------
    try:
        tool_name, args = _route(message, registry)
    except (OllamaUnavailableError, OllamaModelMissingError) as exc:
        return {"answer": str(exc), "tool": None, "tool_result": {}}

    # ---- Execute tool ------------------------------------------------------
    tool_result = registry.call(tool_name, args)

    # ---- Phrase ------------------------------------------------------------
    answer = ""
    try:
        answer = _phrase(message, tool_name, tool_result)
    except (OllamaUnavailableError, OllamaModelMissingError) as exc:
        return {"answer": str(exc), "tool": tool_name, "tool_result": tool_result}

    # ---- Post-check --------------------------------------------------------
    if not _verify_numbers(answer, tool_result):
        logger.info("Post-check failed, regenerating answer.")
        try:
            answer = _phrase(message, tool_name, tool_result)
        except (
            OllamaUnavailableError,
            OllamaModelMissingError,
            httpx.HTTPError,
            TypeError,
            ValueError,
        ) as exc:
            logger.warning("Answer regeneration failed: %s", exc)
        # Check again; if still failing, use template
        if not _verify_numbers(answer, tool_result):
            logger.warning("Post-check failed twice; using template answer.")
            answer = _template_answer(tool_name, tool_result)

    return {
        "answer": answer,
        "tool": tool_name,
        "tool_result": tool_result,
    }


def check_ollama_health(model: str = CRUMB_MODEL) -> dict[str, Any]:
    """Check whether Ollama is running and the model is available."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
        tags = resp.json()
        available_models = [m["name"] for m in tags.get("models", [])]
        model_ready = any(model in m for m in available_models)
        return {
            "ollama_running": True,
            "model": model,
            "model_ready": model_ready,
            "available_models": available_models,
        }
    except httpx.ConnectError:
        return {
            "ollama_running": False,
            "model": model,
            "model_ready": False,
            "available_models": [],
            "hint": f"Start Ollama with: ollama serve, then pull the model: ollama pull {model}",
        }
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        return {
            "ollama_running": False,
            "model": model,
            "model_ready": False,
            "error": str(exc),
        }
