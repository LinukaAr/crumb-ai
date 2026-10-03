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

from crumb.config import CRUMB_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT
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

    if "start_date" not in args or not args.get("start_date"):
        args["start_date"] = today.isoformat()
    if "end_date" not in args or not args.get("end_date"):
        # Default: 7-day horizon
        args["end_date"] = (today + datetime.timedelta(days=6)).isoformat()

    # Resolve weekday names to actual dates
    for key in ("start_date", "end_date"):
        val = str(args[key])
        resolved = _resolve_weekday_or_relative(val, today)
        if resolved:
            args[key] = resolved.isoformat()

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

    if any(w in msg for w in ("forecast", "bake", "make", "predict", "how many", "saturday", "sunday")):
        item = mentioned_item or (items[0] if items else "unknown")
        today = _today()
        start = today.isoformat()
        end = (today + datetime.timedelta(days=6)).isoformat()
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

    # Default: summarize
    item = mentioned_item or (items[0] if items else "unknown")
    period = "last_7_days" if "week" in msg or "7" in msg else "last_30_days"
    return "summarize", {"item": item, "period": period}


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

Output ONLY valid JSON. No explanation.
"""


def _route(message: str, registry: ToolRegistry) -> tuple[str, dict]:
    """
    Determine which tool to call for *message*.

    Returns (tool_name, resolved_args).
    """
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
            if tool_name not in TOOL_SCHEMA_MAP:
                raise ValueError(f"Unknown tool: {tool_name}")
            # Resolve dates
            if tool_name == "forecast":
                args = _resolve_dates_for_forecast(args)
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
