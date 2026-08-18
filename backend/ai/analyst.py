"""AI market analyst built on the local Ollama server.

PAPER TRADING ONLY — every analysis produced here is research output for a
simulated-trading platform. It is not financial advice and never triggers any
real order.

``analyze_market`` compresses a feature-enriched OHLCV frame into a compact
prompt (last-bar indicator snapshot + recent closes), asks the model for a
strictly structured JSON verdict, validates it with pydantic (with one
self-repair round-trip on malformed output) and persists every result to the
``ai_analyses`` table.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from backend.ai.ollama_client import OllamaClient, OllamaError
from backend.database.db import get_conn, utc_now_ms
from backend.indicators.features import feature_summary
from config.settings import settings

logger = logging.getLogger(__name__)

_LAST_CLOSES = 20

_JSON_SCHEMA = """{
  "opposing_case": "<1-2 sentences: the strongest argument AGAINST your final verdict>",
  "sentiment": "bullish" | "bearish" | "neutral",
  "confidence": <integer 0-100>,
  "risk_commentary": "<1-3 sentences on the main risks to this view>",
  "key_indicators": [
    {"name": "<indicator name>", "value": "<current reading>", "influence": "<how it sways the outlook>"}
  ],
  "reasoning": "<concise reasoning tying the indicators together>"
}"""

_SYSTEM_PROMPT = (
    "You are a senior market analyst at a quantitative RESEARCH firm. "
    "This platform is a research and PAPER-TRADING simulation only: your output "
    "is NOT financial advice and will never place a real order.\n\n"
    "The request comes from a research pipeline; do NOT assume anyone wants or "
    "intends to trade — judge the market strictly on its own evidence. Before "
    "you decide, first construct the strongest OPPOSING case (the best argument "
    'AGAINST the view you are leaning toward) and put it in "opposing_case"; '
    "only then commit to your verdict.\n\n"
    "Respond ONLY with a single valid JSON object — no prose, no markdown code "
    "fences, no extra keys — matching exactly this schema:\n"
    f"{_JSON_SCHEMA}"
)


class KeyIndicator(BaseModel):
    """One technical indicator reading and its influence on the outlook."""

    name: str
    value: str
    influence: str


class MarketAnalysis(BaseModel):
    """Structured AI market assessment (research only, not financial advice).

    ``opposing_case`` is the model's strongest argument AGAINST its own final
    verdict — a de-biasing device (the model must argue the other side before
    committing). It is backward-tolerant: replies without the field validate
    fine and default to ``""``.
    """

    model_config = ConfigDict(protected_namespaces=())

    sentiment: Literal["bullish", "bearish", "neutral"]
    confidence: int  # 0-100
    risk_commentary: str
    key_indicators: list[KeyIndicator]
    reasoning: str
    opposing_case: str = ""
    model_used: str = ""
    symbol: str = ""
    timeframe: str = ""


#: Window (bars) for the swing high/low and volume-trend structure summary.
_SWING_WINDOW = 20
#: Bars over which the SMA-20 slope decides the trend direction.
_TREND_SLOPE_BARS = 10
#: Slope dead-zone (fraction): |slope| below this reads as "sideways".
_TREND_DEAD_ZONE = 0.005


def _safe_float(value: Any) -> float | None:
    """Coerce to a JSON-safe float rounded to 6 decimals; NaN/inf/None → None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, 6)


def chart_context(df: pd.DataFrame) -> dict[str, Any]:
    """Summarise recent chart structure for the analyst prompt.

    Gives the model a compressed view of price structure: the 20-bar swing
    high/low with the percentage distance from the last close to each
    (support/resistance proximity), the SMA-20 trend direction (sign of the
    slope over the last 10 bars with a ±0.5% dead-zone), whether volume is
    rising or falling, and the last five closes.

    Purely descriptive and best-effort: missing columns or warm-up NaNs
    degrade individual fields to ``None`` instead of raising, and every value
    is JSON-safe (plain float/str/None).

    Args:
        df: Canonical OHLCV frame; a feature-enriched frame is welcome (an
            existing ``sma_20`` column is reused instead of recomputed).

    Returns:
        dict with keys ``swing_high_20``, ``swing_low_20``,
        ``dist_to_swing_high_pct``, ``dist_to_swing_low_pct``, ``trend_20``
        (``"up" | "down" | "sideways"`` or None), ``volume_trend``
        (``"rising" | "falling"`` or None) and ``last_5_closes``.
    """
    context: dict[str, Any] = {
        "swing_high_20": None,
        "swing_low_20": None,
        "dist_to_swing_high_pct": None,
        "dist_to_swing_low_pct": None,
        "trend_20": None,
        "volume_trend": None,
        "last_5_closes": [],
    }
    if df is None or df.empty or "close" not in df.columns:
        return context

    closes = df["close"]
    price = _safe_float(closes.iloc[-1])
    context["last_5_closes"] = [
        value for value in (_safe_float(c) for c in closes.tail(5)) if value is not None
    ]

    highs = df["high"] if "high" in df.columns else closes
    lows = df["low"] if "low" in df.columns else closes
    swing_high = _safe_float(highs.tail(_SWING_WINDOW).max())
    swing_low = _safe_float(lows.tail(_SWING_WINDOW).min())
    context["swing_high_20"] = swing_high
    context["swing_low_20"] = swing_low
    if price is not None and price > 0:
        if swing_high is not None:
            context["dist_to_swing_high_pct"] = round((swing_high - price) / price * 100.0, 4)
        if swing_low is not None:
            context["dist_to_swing_low_pct"] = round((price - swing_low) / price * 100.0, 4)

    sma20 = df["sma_20"] if "sma_20" in df.columns else closes.rolling(20).mean()
    if len(sma20) > _TREND_SLOPE_BARS:
        current = _safe_float(sma20.iloc[-1])
        past = _safe_float(sma20.iloc[-1 - _TREND_SLOPE_BARS])
        if current is not None and past is not None and past != 0:
            slope = (current - past) / abs(past)
            if slope > _TREND_DEAD_ZONE:
                context["trend_20"] = "up"
            elif slope < -_TREND_DEAD_ZONE:
                context["trend_20"] = "down"
            else:
                context["trend_20"] = "sideways"

    if "volume" in df.columns:
        volumes = [
            value
            for value in (_safe_float(v) for v in df["volume"].tail(_SWING_WINDOW))
            if value is not None
        ]
        if len(volumes) >= 2:
            half = len(volumes) // 2
            earlier_avg = sum(volumes[:half]) / half
            recent_avg = sum(volumes[half:]) / (len(volumes) - half)
            context["volume_trend"] = "rising" if recent_avg > earlier_avg else "falling"

    return context


def _build_prompt(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    news_context: dict[str, Any] | None = None,
    htf_context: str = "",
) -> str:
    """Build a compact analysis prompt from a feature-enriched frame.

    Args:
        symbol: Instrument symbol in the source's native format.
        timeframe: Candle timeframe (e.g. ``"1h"``).
        df: Feature-enriched canonical OHLCV frame.
        news_context: Optional fresh news/community sentiment snapshot (score,
            summary, top headlines, fear & greed) supplied by the intel layer
            via the auto-trader. ``None`` (the default) omits the block and
            leaves the prompt exactly as it was before the intel integration.
        htf_context: Optional newline-joined FACTUAL higher-timeframe lines
            (multi-day price changes, distance to the 4h EMA200) supplied by
            the auto-trader. Non-empty text is inserted as a
            "HIGHER-TIMEFRAME CONTEXT" block between the chart-structure and
            news blocks; the default ``""`` leaves the prompt byte-identical
            to the pre-v3 version.

    Returns:
        The user prompt string containing the last-bar indicator snapshot,
        a chart-structure summary (support/resistance distance, trend,
        volume behaviour), optional "HIGHER-TIMEFRAME CONTEXT" and
        "NEWS & SENTIMENT" blocks and the last 20 closing prices.
    """
    summary = feature_summary(df)
    structure = chart_context(df)
    closes = [round(float(c), 6) for c in df["close"].tail(_LAST_CLOSES).tolist()]
    htf_block = ""
    if htf_context:
        htf_block = (
            "HIGHER-TIMEFRAME CONTEXT (factual, longer-horizon reference "
            "points):\n"
            f"{htf_context}\n\n"
        )
    news_block = ""
    if news_context:
        news_block = (
            "NEWS & SENTIMENT (crowd/news mood aggregated from public "
            "sources; score runs -100 = very bearish to +100 = very "
            "bullish — weigh it against the technicals, do not follow it "
            "blindly):\n"
            f"{json.dumps(news_context, default=str)}\n\n"
        )
    return (
        f"Analyse {symbol} on the {timeframe} timeframe.\n\n"
        f"Technical snapshot of the most recent bar:\n"
        f"{json.dumps(summary, default=str)}\n\n"
        f"CHART STRUCTURE (price structure over the recent window — distance "
        f"to the 20-bar swing high/low i.e. resistance/support, SMA-20 trend "
        f"direction, volume behaviour):\n"
        f"{json.dumps(structure, default=str)}\n\n"
        f"{htf_block}"
        f"{news_block}"
        f"Last {len(closes)} closing prices (oldest to newest):\n"
        f"{json.dumps(closes)}\n\n"
        "Return your assessment as the single JSON object described in the "
        "system prompt."
    )


def _parse_analysis(raw: str) -> MarketAnalysis:
    """Parse and validate a raw model reply into a ``MarketAnalysis``.

    Normalises common model sloppiness before validation (case of the
    sentiment label, numeric confidence coerced to int and clamped into
    [0, 100], indicator fields stringified).

    Args:
        raw: Cleaned model output expected to be a JSON object.

    Returns:
        A validated ``MarketAnalysis`` (without symbol/timeframe/model set).

    Raises:
        ValueError: On any JSON/shape/validation problem
            (``json.JSONDecodeError`` and pydantic's ``ValidationError`` are
            both subclasses of ``ValueError``).
    """
    payload: Any = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"model response is not a JSON object: {type(payload).__name__}")

    if isinstance(payload.get("sentiment"), str):
        payload["sentiment"] = payload["sentiment"].strip().lower()

    raw_conf = payload.get("confidence", 0)
    try:
        confidence = int(round(float(raw_conf)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"confidence is not numeric: {raw_conf!r}") from exc
    payload["confidence"] = max(0, min(100, confidence))

    indicators = payload.get("key_indicators")
    if isinstance(indicators, list):
        payload["key_indicators"] = [
            {
                "name": str(item.get("name", "")),
                "value": str(item.get("value", "")),
                "influence": str(item.get("influence", "")),
            }
            for item in indicators
            if isinstance(item, dict)
        ]
    payload["risk_commentary"] = str(payload.get("risk_commentary") or "")
    payload["reasoning"] = str(payload.get("reasoning") or "")
    # De-bias field — backward-tolerant: a reply without it is still valid.
    payload["opposing_case"] = str(payload.get("opposing_case") or "")

    return MarketAnalysis.model_validate(payload)


#: Telemetry columns appended to ``ai_analyses``. Fresh installs get them from
#: ``schema.sql``; databases created before this feature are migrated at
#: runtime by ``_ensure_columns`` (ALTER TABLE ADD COLUMN is cheap and
#: idempotent thanks to the PRAGMA check).
_TELEMETRY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("prompt_hash", "TEXT NOT NULL DEFAULT ''"),
    ("latency_ms", "INTEGER"),
    ("attempts", "INTEGER"),
    ("fallback_used", "INTEGER NOT NULL DEFAULT 0"),
    ("repair_used", "INTEGER NOT NULL DEFAULT 0"),
)


def _prompt_hash(system_prompt: str, user_prompt: str) -> str:
    """Fingerprint the exact prompt bytes that produced a verdict.

    Args:
        system_prompt: The system prompt sent to the model.
        user_prompt: The user prompt sent to the model.

    Returns:
        The first 12 hex chars of sha256 over
        ``system_prompt + "\\n" + user_prompt`` — enough to tie every
        persisted analysis back to the prompt version that generated it.
    """
    joined = f"{system_prompt}\n{user_prompt}".encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:12]


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add any missing telemetry columns to ``ai_analyses`` (idempotent).

    Runtime-safe migration for databases created before the telemetry
    columns existed: ``PRAGMA table_info`` lists what is there and each
    missing column is added with ``ALTER TABLE ADD COLUMN``. Never raises
    into the caller — on failure the INSERT in ``_persist`` fails and is
    logged there.

    Args:
        conn: An open connection to the platform database.
    """
    try:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(ai_analyses)")}
        for name, decl in _TELEMETRY_COLUMNS:
            if name not in existing:
                conn.execute(f"ALTER TABLE ai_analyses ADD COLUMN {name} {decl}")
    except sqlite3.Error:
        logger.warning("Could not ensure ai_analyses telemetry columns", exc_info=True)


def _persist(
    analysis: MarketAnalysis,
    raw_response: str,
    telemetry: dict[str, Any] | None = None,
) -> None:
    """Write an analysis to the ``ai_analyses`` table.

    Persistence failures are logged but never mask a successfully produced
    analysis.

    Args:
        analysis: The validated analysis (symbol/timeframe/model already set).
        raw_response: The raw (cleaned) model reply that produced it.
        telemetry: Optional call telemetry: ``prompt_hash``, ``latency_ms``,
            ``attempts``, ``fallback_used`` (0/1), ``repair_used`` (0/1).
            Missing keys degrade to NULL / 0, never to an error.
    """
    telemetry = telemetry or {}
    latency_ms = telemetry.get("latency_ms")
    attempts = telemetry.get("attempts")
    try:
        with get_conn() as conn:
            _ensure_columns(conn)
            conn.execute(
                "INSERT INTO ai_analyses "
                "(created_at, model, symbol, timeframe, sentiment, confidence, "
                " risk_commentary, key_indicators, reasoning, raw_response, "
                " prompt_hash, latency_ms, attempts, fallback_used, repair_used) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    utc_now_ms(),
                    analysis.model_used,
                    analysis.symbol,
                    analysis.timeframe,
                    analysis.sentiment,
                    int(analysis.confidence),
                    analysis.risk_commentary,
                    json.dumps([ki.model_dump() for ki in analysis.key_indicators]),
                    analysis.reasoning,
                    raw_response,
                    str(telemetry.get("prompt_hash") or ""),
                    int(latency_ms) if latency_ms is not None else None,
                    int(attempts) if attempts is not None else None,
                    1 if telemetry.get("fallback_used") else 0,
                    1 if telemetry.get("repair_used") else 0,
                ),
            )
        logger.info(
            "Persisted AI analysis: %s %s → %s (confidence %d)",
            analysis.symbol, analysis.timeframe, analysis.sentiment, analysis.confidence,
        )
    except sqlite3.Error:
        logger.exception("Failed to persist AI analysis for %s", analysis.symbol)


def analyze_market(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    model: str | None = None,
    news_context: dict[str, Any] | None = None,
    allow_fallback: bool = True,
    htf_context: str = "",
) -> MarketAnalysis:
    """Run an AI market analysis over a feature-enriched OHLCV frame.

    Research only — the result is a structured opinion for paper-trading
    experimentation, never financial advice.

    Args:
        symbol: Instrument symbol in the source's native format.
        timeframe: Candle timeframe (``"1m" | "5m" | "15m" | "1h" | "4h" | "1d"``).
        df: Feature-enriched canonical OHLCV frame (``add_features`` output).
        model: Optional model override; defaults to ``settings.ollama_model``.
        news_context: Optional fresh news/community sentiment snapshot from
            the intel layer; when provided it is appended to the prompt as a
            "NEWS & SENTIMENT" block. The default ``None`` keeps the prompt
            (and therefore all pre-intel behaviour) unchanged.
        allow_fallback: Passed through to ``OllamaClient.chat``. When False
            the requested model is NEVER silently substituted with
            ``settings.ollama_fallback_model`` — a missing/failing model
            raises ``OllamaError`` instead (required by the auto-trader's
            second-judge gate, whose contract mandates a conservative skip).
        htf_context: Optional factual higher-timeframe lines injected into
            the prompt as a "HIGHER-TIMEFRAME CONTEXT" block (see
            ``_build_prompt``). The default ``""`` changes nothing.

    Returns:
        A validated ``MarketAnalysis`` with ``model_used``, ``symbol`` and
        ``timeframe`` populated and confidence clamped into [0, 100];
        ``model_used`` records the model that ACTUALLY answered (the
        resolved installed tag, or the fallback when it substituted). Every
        result is persisted to the ``ai_analyses`` table together with call
        telemetry (``prompt_hash``, ``latency_ms``, ``attempts``,
        ``fallback_used``, ``repair_used``) so each verdict is reproducible
        and the model's health is measurable.

    Raises:
        ValueError: When ``df`` is empty or missing the ``close`` column.
        OllamaError: When the model is unreachable or still produces
            unparseable output after one self-repair round-trip.
    """
    if df is None or df.empty:
        raise ValueError(f"No data available to analyse for {symbol} ({timeframe}).")
    if "close" not in df.columns:
        raise ValueError("DataFrame is missing the required 'close' column.")

    model_name = model or settings.ollama_model
    client = OllamaClient()
    prompt = _build_prompt(
        symbol, timeframe, df, news_context=news_context, htf_context=htf_context
    )

    raw = client.chat(
        prompt, system=_SYSTEM_PROMPT, model=model_name, json_mode=True,
        temperature=0.2, allow_fallback=allow_fallback,
    )
    # Telemetry for the persisted row: the prompt fingerprint always covers
    # the ORIGINAL analysis prompt (that is what makes the verdict
    # reproducible); latency/attempts accumulate over the repair round-trip
    # when it fires. A stubbed/legacy chat without stats degrades to NULLs.
    stats = dict(getattr(client, "last_call_stats", None) or {})
    telemetry: dict[str, Any] = {
        "prompt_hash": _prompt_hash(_SYSTEM_PROMPT, prompt),
        "latency_ms": stats.get("latency_ms"),
        "attempts": stats.get("attempts"),
        "fallback_used": 1 if stats.get("fallback_used") else 0,
        "repair_used": 0,
    }
    try:
        analysis = _parse_analysis(raw)
    except ValueError as exc:  # covers JSONDecodeError and ValidationError
        logger.warning(
            "AI analysis for %s failed to parse (%s) — attempting one repair round-trip",
            symbol, exc,
        )
        repair_prompt = (
            "Your previous reply was not a valid JSON object matching the "
            f"required schema.\nError: {exc}\n\n"
            f"Previous reply:\n{raw}\n\n"
            "Rewrite it as a single valid JSON object matching exactly this "
            f"schema (no prose, no markdown):\n{_JSON_SCHEMA}"
        )
        repaired = client.chat(
            repair_prompt, system=_SYSTEM_PROMPT, model=model_name,
            json_mode=True, temperature=0.0, allow_fallback=allow_fallback,
        )
        repair_stats = dict(getattr(client, "last_call_stats", None) or {})
        telemetry["repair_used"] = 1
        for key in ("latency_ms", "attempts"):
            base, extra = telemetry.get(key), repair_stats.get(key)
            if base is not None or extra is not None:
                telemetry[key] = int(base or 0) + int(extra or 0)
        if repair_stats.get("fallback_used"):
            telemetry["fallback_used"] = 1
        try:
            analysis = _parse_analysis(repaired)
            raw = repaired
        except (ValueError, ValidationError) as exc2:
            raise OllamaError(
                f"Model '{model_name}' produced unparseable analysis for {symbol} "
                f"even after a repair attempt: {exc2}"
            ) from exc2

    # Record the model that ACTUALLY answered — chat() may have resolved the
    # requested name to a differently-tagged install or (when allowed)
    # substituted the fallback model; persisting the requested name would
    # misattribute the verdict in ai_analyses and the UI.
    analysis.model_used = getattr(client, "last_model_used", None) or model_name
    analysis.symbol = symbol
    analysis.timeframe = timeframe

    _persist(analysis, raw, telemetry)
    return analysis


def get_analysis_history(symbol: str | None = None, limit: int = 50) -> list[dict]:
    """Read past analyses from the ``ai_analyses`` table, newest first.

    Args:
        symbol: Optional symbol filter (exact match).
        limit: Maximum number of rows to return.

    Returns:
        A list of dicts with keys ``id, created_at, model, symbol, timeframe,
        sentiment, confidence, risk_commentary, key_indicators, reasoning``;
        ``created_at`` is an ISO-8601 UTC string and ``key_indicators`` is a
        JSON-decoded list.
    """
    query = (
        "SELECT id, created_at, model, symbol, timeframe, sentiment, confidence, "
        "risk_commentary, key_indicators, reasoning FROM ai_analyses"
    )
    params: list[object] = []
    if symbol:
        query += " WHERE symbol = ?"
        params.append(symbol)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(int(limit))

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()

    items: list[dict] = []
    for row in rows:
        try:
            indicators = json.loads(row["key_indicators"] or "[]")
        except (TypeError, ValueError):
            logger.warning("Malformed key_indicators JSON in ai_analyses row %s", row["id"])
            indicators = []
        items.append(
            {
                "id": row["id"],
                "created_at": datetime.fromtimestamp(
                    row["created_at"] / 1000, tz=timezone.utc
                ).isoformat(),
                "model": row["model"],
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "sentiment": row["sentiment"],
                "confidence": row["confidence"],
                "risk_commentary": row["risk_commentary"],
                "key_indicators": indicators,
                "reasoning": row["reasoning"],
            }
        )
    return items
