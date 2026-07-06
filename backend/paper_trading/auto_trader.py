"""AI auto-trader — an autonomous PAPER-ONLY trading bot.

The :class:`AutoTrader` scans a crypto watchlist every cycle, ranks
opportunities by strategy-ensemble votes (:class:`VotingStrategy`) boosted by
fresh news sentiment (ranking only — news never qualifies a candidate),
optionally confirms the top candidates with the local LLM analyst
(``backend.ai.analyst.analyze_market``) — first the primary model for every
candidate, then the Fin-R1 second judge over the survivors (two passes, so
the GPU swaps models at most once per cycle) — and trades a **simulated**
account through the existing :class:`PaperTradingEngine` + :class:`RiskManager`.

It NEVER touches a real exchange: fills happen only against locally cached
public candles, exactly like every other part of this platform. No private
API keys exist anywhere in the project.

TRANSPARENCY: every decision — cycle start/end, scans, entries, exits, skips,
rejections, halts, errors — is appended to the ``bot_activity`` table with a
JSON detail payload, including a one-sentence plain-English ``explanation`` on
``enter``/``exit``/``skip``/``reject`` rows, so the dashboard can show the
user exactly why the bot did what it did.

Configuration (:class:`BotConfig`) persists in the ``account_state`` table
under the key ``'bot_config'`` and therefore survives API restarts. The
APScheduler job that drives periodic cycles is owned by ``backend.api.main``;
this module only exposes the thread-safe :meth:`AutoTrader.run_cycle`.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

from backend.ai import analyst as _analyst_module
from backend.ai.analyst import MarketAnalysis
from backend.ai.ollama_client import OllamaClient, OllamaError
from backend.data import service as _data_service
from backend.database.db import get_conn, init_db, load_ohlcv, utc_now_ms
from backend.indicators.features import feature_summary
from backend.paper_trading import regime
from backend.paper_trading.engine import PaperTradingEngine
from backend.paper_trading.risk import RiskManager
from backend.strategies.voting import VotingStrategy
from config.settings import settings

logger = logging.getLogger(__name__)

#: account_state keys used by the bot.
_CONFIG_KEY = "bot_config"
_LAST_CYCLE_TS_KEY = "bot_last_cycle_ts"
_LAST_CYCLE_SUMMARY_KEY = "bot_last_cycle_summary"
#: account_state key written by the API's Ollama watchdog (cross-module
#: contract). A MISSING or malformed key means ONLINE — a watchdog that never
#: ran must not block trading.
_OLLAMA_STATUS_KEY = "ollama_status"

#: The market-regime reference symbol: BTC leads the whole crypto market, so
#: the regime hard gate reads BTCUSDT alongside the candidate's own regime.
_REGIME_REFERENCE_SYMBOL = "BTCUSDT"

#: Minimum candles required before a symbol can be scanned/managed.
_MIN_SCAN_ROWS = 100
#: Incremental history depth fetched for each watchlist symbol per cycle.
_SCAN_LOOKBACK_DAYS = 30
#: Orders below this notional (USD) are skipped as dust.
_MIN_NOTIONAL_USD = 1.0
#: Quote-currency suffixes stripped for human-readable explanations.
_QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "USD")

#: Candle duration per supported timeframe (closed-candle/freshness checks).
_TIMEFRAME_DELTAS: dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}
#: Grace added to the staleness threshold (exchange/scheduler/clock lag).
_STALE_GRACE = pd.Timedelta(minutes=5)

#: News-sentiment gate (intel integration): a FRESH coin score at/below
#: -_SENTIMENT_BLOCK_SCORE blocks long entries (skip reason ``news_negative``)
#: and at/above +_SENTIMENT_BLOCK_SCORE blocks short entries
#: (``news_positive``). Contract-mandated thresholds.
_SENTIMENT_BLOCK_SCORE = 50
#: Sentiment older than this is ignored entirely (no gate) — news moods go
#: stale fast; a 3h-old headline must never veto a live technical signal.
_SENTIMENT_MAX_AGE_MS = 3 * 60 * 60 * 1000

_DEFAULT_WATCHLIST: tuple[str, ...] = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
)


# ---------------------------------------------------------------------------
# Late-bound proxies — resolved at call time so tests can monkeypatch either
# these names or the originals in their home modules.
# ---------------------------------------------------------------------------


def analyze_market(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    model: str | None = None,
    news_context: dict[str, Any] | None = None,
    allow_fallback: bool = True,
    htf_context: str = "",
) -> MarketAnalysis:
    """Proxy to :func:`backend.ai.analyst.analyze_market` (test patch point)."""
    return _analyst_module.analyze_market(
        symbol,
        timeframe,
        df,
        model=model,
        news_context=news_context,
        allow_fallback=allow_fallback,
        htf_context=htf_context,
    )


def update_symbol(
    source: str, symbol: str, timeframe: str, lookback_days: int = 365
) -> int:
    """Proxy to :func:`backend.data.service.update_symbol` (test patch point)."""
    return _data_service.update_symbol(
        source, symbol, timeframe, lookback_days=lookback_days
    )


def get_ohlcv(
    source: str,
    symbol: str,
    timeframe: str,
    limit: int = 500,
    with_features: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    """Proxy to :func:`backend.data.service.get_ohlcv` (test patch point)."""
    return _data_service.get_ohlcv(
        source,
        symbol,
        timeframe,
        limit=limit,
        with_features=with_features,
        refresh=refresh,
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _ms_to_iso(ms: int | None) -> str | None:
    """Convert epoch milliseconds to an ISO-8601 UTC string (None-safe)."""
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def _json_num(value: Any) -> float | None:
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


def _base_asset(symbol: str) -> str:
    """Human-readable base asset for a symbol (``BTCUSDT`` → ``BTC``)."""
    for suffix in _QUOTE_SUFFIXES:
        if symbol.endswith(suffix) and len(symbol) > len(suffix):
            return symbol[: -len(suffix)].rstrip("-/")
    return symbol


def _timeframe_delta(timeframe: str) -> pd.Timedelta:
    """Candle duration for a timeframe (defensive 1h default)."""
    return _TIMEFRAME_DELTAS.get(timeframe, pd.Timedelta(hours=1))


def _drop_unclosed_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Return ``df`` without a still-forming (unclosed) final candle.

    The data service deliberately re-fetches the in-progress candle, so the
    cached last row usually repaints until its period ends. Signals must be
    computed on CLOSED candles only — the same semantics the backtester
    validated ("signal at bar t → act at bar t+1, never on the forming bar").

    Args:
        df: Canonical (possibly feature-enriched) OHLCV frame.
        timeframe: Candle timeframe of the frame.

    Returns:
        The frame restricted to rows whose full period has already elapsed.
    """
    if df.empty:
        return df
    cutoff = pd.Timestamp.now(tz="UTC") - _timeframe_delta(timeframe)
    return df[df.index <= cutoff]


def _model_installed(model: str, installed: list[str]) -> bool:
    """True when ``model`` is available in the ``installed`` Ollama list.

    Contract rule for the second judge: match the exact name AND by the base
    name before the ``:`` tag (``mychen76/Fin-R1:Q5`` matches an installed
    ``mychen76/fin-r1:latest``). Comparison is case-insensitive because
    Ollama normalises model names to lowercase.

    Args:
        model: The configured model name (possibly with a ``:tag``).
        installed: Names returned by ``OllamaClient.list_models()``.

    Returns:
        True when the model (or a same-base-name variant) is installed.
    """
    wanted = str(model or "").strip().lower()
    if not wanted:
        return False
    wanted_base = wanted.split(":", 1)[0]
    for name in installed:
        have = str(name or "").strip().lower()
        if have == wanted or have.split(":", 1)[0] == wanted_base:
            return True
    return False


def _is_stale(df: pd.DataFrame, timeframe: str) -> bool:
    """True when the last (closed) candle is too old to trade on.

    Threshold: 2x the timeframe plus a small grace period — a healthy feed
    always has a closed candle younger than that. Trading on anything older
    (e.g. after an exchange outage/geo-block) would vote, size and fill on a
    frozen price while the real market keeps moving.

    Args:
        df: Canonical OHLCV frame (last row = most recent closed candle).
        timeframe: Candle timeframe of the frame.

    Returns:
        True when the frame is empty or its last candle exceeds the age limit.
    """
    if df.empty:
        return True
    age = pd.Timestamp.now(tz="UTC") - df.index[-1]
    return age > (2 * _timeframe_delta(timeframe) + _STALE_GRACE)


def _finite(value: Any) -> float | None:
    """Coerce to a finite float, or None (no rounding — for internal math)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _r_multiples(position: dict[str, Any]) -> tuple[float | None, float | None]:
    """R-multiple diagnostics for a closed position (contract v3).

    ``designed_r`` is the reward-to-risk ratio the entry was PLACED with
    (``|take_profit - entry| / |entry - stop_loss|``); ``realized_r`` is the
    actual outcome in units of the risked stop distance
    (``pnl / (|entry - stop_loss| * qty)``). Together they answer "was the
    trade designed well, and how much of the designed risk did it actually
    win or lose?".

    Args:
        position: Closed-position dict (engine shape) with ``entry_price``,
            ``stop_loss``, ``take_profit``, ``qty`` and ``pnl`` fields.

    Returns:
        ``(designed_r, realized_r)`` — each ``None`` when its inputs are
        missing/invalid or the stop distance is not positive.
    """
    entry = _finite(position.get("entry_price"))
    stop = _finite(position.get("stop_loss"))
    take = _finite(position.get("take_profit"))
    qty = _finite(position.get("qty"))
    pnl = _finite(position.get("pnl"))

    designed: float | None = None
    realized: float | None = None
    if entry is None or stop is None:
        return designed, realized
    stop_distance = abs(entry - stop)
    if stop_distance <= 0:
        return designed, realized
    if take is not None:
        designed = abs(take - entry) / stop_distance
    if qty is not None and qty > 0 and pnl is not None:
        denominator = stop_distance * qty
        if denominator > 0:
            realized = pnl / denominator
    return designed, realized


def _htf_context(info: regime.RegimeInfo) -> str:
    """Factual higher-timeframe context lines for the analyst prompt.

    Produces newline-joined FACT lines (multi-day price changes and the
    distance to the 4h EMA200) from a :class:`~backend.paper_trading.regime.
    RegimeInfo`. Deliberately contains NO trend verdicts ("uptrend"/
    "downtrend") — the analyst must judge the market on raw evidence, not on
    this module's regime classification (contract de-bias rule). Each part is
    omitted when unavailable; an empty string means "no context" and leaves
    the analyst prompt unchanged.

    Args:
        info: Regime snapshot for the candidate's symbol.

    Returns:
        Newline-joined factual lines, or ``""`` when nothing is available.
    """
    lines: list[str] = []
    pct_7d = _finite(getattr(info, "pct_change_7d", None))
    if pct_7d is not None:
        lines.append(f"7-day price change: {pct_7d:+.1f}%.")
    pct_30d = _finite(getattr(info, "pct_change_30d", None))
    if pct_30d is not None:
        lines.append(f"30-day price change: {pct_30d:+.1f}%.")
    close = _finite(getattr(info, "close", None))
    ema200 = _finite(getattr(info, "ema200", None))
    if close is not None and ema200 is not None and ema200 > 0:
        diff = (close / ema200 - 1.0) * 100.0
        side = "above" if diff >= 0 else "below"
        lines.append(f"Price is {abs(diff):.1f}% {side} the 4h EMA200.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------


class BotConfig(BaseModel):
    """Persisted configuration of the AI auto-trader (paper only).

    Stored as JSON in the ``account_state`` table under the key
    ``'bot_config'`` so it survives process restarts.

    Attributes:
        watchlist: Symbols scanned every cycle (source-native format).
        source: Market-data source for the whole bot (``binance`` default).
        timeframe: Candle timeframe used for signals and fills.
        interval_minutes: Minutes between scheduled cycles.
        use_ai: When True, entries require LLM confirmation via
            ``analyze_market`` (sentiment must match the trade direction with
            at least ``min_ai_confidence`` confidence).
        min_ai_confidence: Minimum AI confidence (0-100) to allow an entry.
        max_ai_calls_per_cycle: Cap on LLM calls (and therefore AI-confirmed
            entries) per cycle.
        allow_short: Permit short entries when the ensemble votes negative.
        min_vote: Minimum absolute sum of the 4 strategy votes to shortlist.
        max_position_fraction: Cap on a single position's notional as a
            fraction of account equity.
        running: Whether the periodic scheduler job should be active.
        use_second_judge: When True (and ``use_ai``), entries additionally
            require confirmation from the finance-specialist second judge
            model (Fin-R1). A not-yet-installed judge model degrades
            gracefully: one warning per cycle, entries proceed unjudged.
        second_judge_model: Ollama model name of the second judge.
        judge_min_confidence: Minimum second-judge confidence (0-100) for an
            entry to survive the judge gate.
        sentiment_rank_weight: Weight of the news-sentiment bonus in the
            shortlist ranking (clamped to [0.0, 2.0] on every update; 0
            disables the bonus). News only reorders qualified candidates —
            it never qualifies one.
        regime_gate_enabled: When True, every candidate passes the market-
            regime hard gate BEFORE any AI call: shorts are allowed only
            when both the coin and BTCUSDT are in a 4h downtrend, and longs
            are blocked in exactly that double-downtrend state
            (``regime.regime_allows``).
        cost_gate_multiple: The expected per-bar move (ATR/price) must be at
            least this multiple of the round-trip fee+slippage cost or the
            candidate is skipped before any AI call. Clamped to [0.0, 10.0]
            on every update; 0 disables the gate.
    """

    watchlist: list[str] = Field(default_factory=lambda: list(_DEFAULT_WATCHLIST))
    source: str = "binance"
    timeframe: str = "1h"
    interval_minutes: int = Field(15, ge=1)
    use_ai: bool = True
    min_ai_confidence: int = Field(60, ge=0, le=100)
    max_ai_calls_per_cycle: int = Field(3, ge=0)
    allow_short: bool = False
    min_vote: int = Field(2, ge=1, le=4)
    max_position_fraction: float = Field(0.35, gt=0.0, le=1.0)
    running: bool = False
    use_second_judge: bool = True
    second_judge_model: str = "mychen76/Fin-R1:Q5"
    judge_min_confidence: int = Field(55, ge=0, le=100)
    sentiment_rank_weight: float = 0.5
    regime_gate_enabled: bool = True
    cost_gate_multiple: float = 3.0

    @field_validator("sentiment_rank_weight")
    @classmethod
    def _clamp_sentiment_rank_weight(cls, value: float) -> float:
        """Clamp the ranking weight to the contract range [0.0, 2.0].

        Clamping (rather than rejecting) is contract-mandated for
        ``set_config``; running it as a validator also sanitises persisted
        values on every load. Non-finite input degrades safely (NaN → 0.0,
        +inf → 2.0).
        """
        return min(2.0, max(0.0, float(value)))

    @field_validator("cost_gate_multiple")
    @classmethod
    def _clamp_cost_gate_multiple(cls, value: float) -> float:
        """Clamp the cost-gate multiple to the contract range [0.0, 10.0].

        Mirrors ``_clamp_sentiment_rank_weight``: clamp instead of reject,
        so ``set_config`` and persisted-value loads both sanitise. Non-finite
        input degrades safely (NaN → 0.0, +inf → 10.0); 0 disables the gate.
        """
        return min(10.0, max(0.0, float(value)))

    @field_validator("watchlist")
    @classmethod
    def _clean_watchlist(cls, value: list[str]) -> list[str]:
        """Strip/upper-case symbols, drop blanks and duplicates (order kept)."""
        cleaned: list[str] = []
        for item in value:
            symbol = str(item).strip().upper()
            if symbol and symbol not in cleaned:
                cleaned.append(symbol)
        if not cleaned:
            raise ValueError("watchlist must contain at least one symbol")
        return cleaned


# ---------------------------------------------------------------------------
# AutoTrader
# ---------------------------------------------------------------------------


class AutoTrader:
    """Autonomous simulated trader driving the :class:`PaperTradingEngine`.

    Designed as a singleton living next to the engine inside the API process.
    :meth:`run_cycle` is thread-safe (a non-blocking ``threading.Lock`` skips
    a cycle when one is already in flight) and NEVER raises — every
    per-symbol failure is caught, logged and written to ``bot_activity``.
    """

    def __init__(self, engine: PaperTradingEngine) -> None:
        """Bind the bot to an engine and make sure the schema exists.

        Args:
            engine: The shared paper-trading engine (simulation only).
        """
        init_db()  # idempotent — guarantees bot_activity exists
        self.engine = engine
        self.risk = engine.risk
        self._cycle_lock = threading.Lock()
        self._config_lock = threading.RLock()
        logger.info("AutoTrader ready (PAPER TRADING ONLY — no real orders, ever)")

    # ------------------------------------------------------------------ #
    # Configuration                                                       #
    # ------------------------------------------------------------------ #

    def get_config(self) -> BotConfig:
        """Load the persisted bot configuration (defaults when absent/corrupt).

        Returns:
            The current :class:`BotConfig`.
        """
        with self._connect() as conn:
            raw = self.engine._state_get(conn, _CONFIG_KEY, None)
        if raw is None:
            return BotConfig()
        try:
            return BotConfig.model_validate(raw)
        except ValidationError:
            logger.warning("Persisted bot_config is invalid — using defaults")
            return BotConfig()

    def set_config(self, updates: dict[str, Any]) -> BotConfig:
        """Apply a partial update to the persisted configuration.

        Unknown keys are ignored with a warning; invalid values raise
        pydantic's ``ValidationError`` (a ``ValueError`` subclass, so the API
        maps it to HTTP 400). ``sentiment_rank_weight`` is CLAMPED to
        [0.0, 2.0] instead of rejected (contract rule).

        Args:
            updates: Subset of :class:`BotConfig` fields to change.

        Returns:
            The updated, persisted :class:`BotConfig`.
        """
        with self._config_lock:
            merged = self.get_config().model_dump()
            known = set(BotConfig.model_fields)
            unknown = sorted(set(updates) - known)
            if unknown:
                logger.warning("Ignoring unknown bot config keys: %s", unknown)
            merged.update({k: v for k, v in updates.items() if k in known})
            config = BotConfig.model_validate(merged)
            with self._connect() as conn:
                self.engine._state_set(conn, _CONFIG_KEY, config.model_dump())
            logger.info(
                "Bot config updated (running=%s, %d symbols, every %d min, use_ai=%s)",
                config.running,
                len(config.watchlist),
                config.interval_minutes,
                config.use_ai,
            )
            return config

    def start(self) -> BotConfig:
        """Mark the bot as running (persisted). Scheduling is the API's job.

        Returns:
            The updated :class:`BotConfig` with ``running=True``.
        """
        config = self.set_config({"running": True})
        logger.info(
            "AutoTrader STARTED (paper only) — %d symbols every %d min",
            len(config.watchlist),
            config.interval_minutes,
        )
        return config

    def stop(self) -> BotConfig:
        """Mark the bot as stopped (persisted).

        Returns:
            The updated :class:`BotConfig` with ``running=False``.
        """
        config = self.set_config({"running": False})
        logger.info("AutoTrader stopped (paper only)")
        return config

    def status(self) -> dict[str, Any]:
        """Current bot status for the API/dashboard.

        Returns:
            dict with keys ``running``, ``last_cycle_ts`` (ISO or None),
            ``next_cycle_ts`` (ISO or None — estimated from the last cycle;
            the API overlays the scheduler's exact time when available),
            ``last_cycle_summary`` (dict or None), ``watchlist``, ``equity``
            and ``open_positions``.
        """
        config = self.get_config()
        portfolio = self.engine.get_portfolio()
        with self._connect() as conn:
            last_ms = self.engine._state_get(conn, _LAST_CYCLE_TS_KEY, None)
            last_summary = self.engine._state_get(conn, _LAST_CYCLE_SUMMARY_KEY, None)
        next_iso: str | None = None
        if config.running and last_ms is not None:
            next_iso = _ms_to_iso(int(last_ms) + config.interval_minutes * 60_000)
        return {
            "running": config.running,
            "last_cycle_ts": _ms_to_iso(int(last_ms)) if last_ms is not None else None,
            "next_cycle_ts": next_iso,
            "last_cycle_summary": last_summary,
            "watchlist": config.watchlist,
            "equity": float(portfolio["equity"]),
            "open_positions": int(portfolio["open_positions"]),
        }

    # ------------------------------------------------------------------ #
    # Activity feed                                                        #
    # ------------------------------------------------------------------ #

    def get_activity(self, limit: int = 100) -> list[dict[str, Any]]:
        """Most recent ``bot_activity`` rows, newest first.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            list of ``{"ts": iso, "symbol": str, "action": str, "detail": dict}``.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, symbol, action, detail FROM bot_activity "
                "ORDER BY ts DESC, id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            try:
                detail = json.loads(row["detail"])
            except (TypeError, json.JSONDecodeError):
                detail = {}
            items.append(
                {
                    "ts": _ms_to_iso(row["ts"]),
                    "symbol": row["symbol"],
                    "action": row["action"],
                    "detail": detail if isinstance(detail, dict) else {},
                }
            )
        return items

    # ------------------------------------------------------------------ #
    # The trading cycle                                                    #
    # ------------------------------------------------------------------ #

    def run_cycle(self) -> dict[str, Any]:
        """Run one full scan → manage → confirm → trade pass. NEVER raises.

        Thread-safe: when a cycle is already running the call returns
        immediately with ``{"status": "busy", ...}`` instead of blocking.

        Returns:
            Cycle summary dict (``status``, ``scanned``, ``entered``,
            ``exited``, ``skipped``, ``rejected``, ``errors``, ``equity``,
            timing fields; plus ``halt_reason``/``error`` when applicable).
        """
        if not self._cycle_lock.acquire(blocking=False):
            logger.info("Bot cycle skipped — a cycle is already running")
            return {"status": "busy", "reason": "a cycle is already running"}
        try:
            try:
                return self._run_cycle_locked()
            except Exception as exc:  # noqa: BLE001 — run_cycle must never raise
                logger.exception("Bot cycle crashed — recorded and swallowed")
                self._log("error", detail={"where": "run_cycle", "reason": str(exc)})
                summary = {
                    "status": "error",
                    "error": str(exc),
                    "scanned": 0,
                    "entered": 0,
                    "exited": 0,
                    "skipped": 0,
                    "rejected": 0,
                    "errors": 1,
                    "equity": None,
                    "finished_at": _ms_to_iso(utc_now_ms()),
                }
                self._record_cycle(summary)
                return summary
        finally:
            self._cycle_lock.release()

    # ------------------------------------------------------------------ #
    # Cycle internals                                                      #
    # ------------------------------------------------------------------ #

    def _run_cycle_locked(self) -> dict[str, Any]:
        """The actual cycle body (contract steps 1-7); lock already held."""
        started_ms = utc_now_ms()
        config = self.get_config()
        counters = {
            "scanned": 0,
            "entered": 0,
            "exited": 0,
            "skipped": 0,
            "rejected": 0,
            "errors": 0,
        }
        self._log(
            "cycle_start",
            detail={
                "watchlist": config.watchlist,
                "source": config.source,
                "timeframe": config.timeframe,
                "use_ai": config.use_ai,
                "allow_short": config.allow_short,
                "min_vote": config.min_vote,
            },
        )

        voter = VotingStrategy()

        # --- 2) manage open positions (SL/TP + ensemble-flip exits) ----------
        # Runs even when trading is halted: process_tick is the ONLY code
        # path that evaluates stop-loss/take-profit on open positions, and a
        # halt (daily loss / circuit breaker) is exactly when those stops
        # must keep working. RiskManager explicitly allows risk-reducing
        # exits during a halt — only NEW entries are gated below.
        self._manage_positions(voter, counters)

        # --- 3) halt gate: a halt skips new entries only ----------------------
        portfolio = self.engine.get_portfolio()
        if portfolio.get("trading_halted"):
            reason = str(portfolio.get("halt_reason") or "trading_halted")
            self._log(
                "halt",
                detail={
                    "reason": reason,
                    "explanation": (
                        "Managed open positions, then stopped before scanning: "
                        f"trading is halted — {reason}"
                    ),
                },
            )
            return self._finish_cycle(counters, started_ms, status="halted", halt_reason=reason)

        # --- 4) scan the watchlist -------------------------------------------
        # Per-coin coach playbook — read ONCE per cycle (contract: one query,
        # not one per symbol). Empty when the coach/intel layer is absent.
        playbook = self._load_playbook()
        candidates: list[dict[str, Any]] = []
        for symbol in config.watchlist:
            entry = playbook.get(symbol)
            if entry is not None and entry.get("bench"):
                counters["skipped"] += 1
                self._log(
                    "skip",
                    symbol=symbol,
                    detail={
                        "reason": "coach_bench",
                        "coach_reasoning": str(entry.get("reasoning") or ""),
                        "explanation": (
                            f"Skipped {symbol}: the coach benched this coin "
                            "after reviewing its recent results — no new "
                            "entries until it is un-benched."
                        ),
                    },
                )
                continue
            candidate = self._scan_symbol(config, voter, symbol, counters)
            if candidate is not None:
                candidates.append(candidate)

        # --- 5) shortlist & rank ----------------------------------------------
        shortlist = self._shortlist(config, candidates, playbook)

        # AI-offline cycle skip (watchdog integration): when the persisted
        # watchdog status says Ollama is down, drop the whole shortlist with
        # ONE skip row instead of timing out per candidate. A missing or
        # malformed status key means ONLINE (a watchdog that never ran must
        # not block trading).
        if config.use_ai and shortlist:
            offline, since_ms = self._ollama_offline()
            if offline:
                counters["skipped"] += len(shortlist)
                self._log(
                    "skip",
                    detail={
                        "reason": "ai_offline",
                        "since_ms": since_ms,
                        "skipped_candidates": len(shortlist),
                        "explanation": (
                            "Skipped all AI confirmations this cycle: the local "
                            "AI (Ollama) is offline — the watchdog is trying to "
                            "restart it. AI-gated entries stay off until it is "
                            "back."
                        ),
                    },
                )
                shortlist = []

        # Per-cycle regime cache: at most ONE regime.get_regime call per
        # symbol per cycle; BTCUSDT is computed at most once and reused for
        # every candidate (contract).
        regime_cache: dict[str, regime.RegimeInfo] = {}

        # --- 6) AI confirmation gate + entries ---------------------------------
        # TWO PASSES (contract, GPU model-swap thrash avoidance): first the
        # primary gate for ALL candidates (the primary model stays loaded),
        # then the Fin-R1 second-judge pass over the survivors — at most one
        # model swap per cycle. Survivors that clear both gates are entered.
        survivors: list[dict[str, Any]] = []
        halted = False
        for candidate in shortlist:
            if self._halted_mid_cycle(candidate["symbol"]):
                halted = True
                break
            try:
                if self._confirm_candidate(config, candidate, counters, regime_cache):
                    survivors.append(candidate)
            except Exception as exc:  # noqa: BLE001 — per-candidate isolation
                counters["errors"] += 1
                logger.exception(
                    "Bot primary confirmation failed for %s", candidate["symbol"]
                )
                self._log(
                    "error",
                    symbol=candidate["symbol"],
                    detail={"where": "confirm", "reason": str(exc)},
                )

        if halted:
            survivors = []
        judge_active = bool(survivors) and self._second_judge_ready(config)
        for candidate in survivors:
            if self._halted_mid_cycle(candidate["symbol"]):
                break
            try:
                if judge_active and not self._confirm_judge(config, candidate, counters):
                    continue
                self._enter_candidate(config, candidate, counters, playbook)
            except Exception as exc:  # noqa: BLE001 — per-candidate isolation
                counters["errors"] += 1
                logger.exception("Bot entry failed for %s", candidate["symbol"])
                self._log(
                    "error",
                    symbol=candidate["symbol"],
                    detail={"where": "enter", "reason": str(exc)},
                )

        # --- 7) equity snapshot + cycle_end -------------------------------------
        return self._finish_cycle(counters, started_ms, status="ok")

    def _halted_mid_cycle(self, symbol: str) -> bool:
        """Mid-cycle halt gate for the entry loops (contract step 6).

        Trading can become halted while entries are being confirmed (the
        scalper ticks concurrently; a first entry's costs can trip the daily
        loss limit on a small account), so the gate is re-checked before
        every AI confirmation and every order submission.

        Args:
            symbol: Candidate symbol, recorded on the ``halt`` activity row.

        Returns:
            True when trading is halted (a ``halt`` row was logged and the
            caller must stop entering).
        """
        portfolio = self.engine.get_portfolio()
        if not portfolio.get("trading_halted"):
            return False
        reason = str(portfolio.get("halt_reason") or "trading_halted")
        self._log(
            "halt",
            symbol=symbol,
            detail={
                "reason": reason,
                "explanation": (
                    "Stopped entering new positions mid-cycle: trading "
                    f"became halted — {reason}"
                ),
            },
        )
        return True

    def _manage_positions(
        self, voter: VotingStrategy, counters: dict[str, int]
    ) -> None:
        """Contract step 2: tick every open-position symbol, exit on flips.

        For each distinct open ``(symbol, source, timeframe)``:
        ``process_tick(refresh=True)`` (fills SL/TP, marks to market), then
        recompute the ensemble stance on featured data — CLOSED candles only,
        and only when the data is fresh — and market-close any position whose
        side the ensemble now opposes. Every close is logged as an ``exit``
        row; per-symbol failures are logged and isolated. Runs even while
        trading is halted (exits are risk-reducing and always allowed).

        Scalper-owned positions (ids in the ``account_state`` key
        ``'scalper_position_ids'``) are SKIPPED entirely: the fast scalper
        script manages its own exits (fixed-percent SL/TP + time-stop), so
        the bot neither ensemble-flip-closes nor double-logs them.
        """
        scalper_ids = self._scalper_owned_ids()
        seen: set[tuple[str, str, str]] = set()
        for pos in self.engine.get_positions("open"):
            if int(pos["id"]) in scalper_ids:
                continue  # scalper-owned — the scalper manages its own exits
            key = (str(pos["symbol"]), str(pos["source"]), str(pos["timeframe"]))
            if key in seen:
                continue
            seen.add(key)
            symbol, source, timeframe = key
            try:
                tick = self.engine.process_tick(
                    symbol, source=source, timeframe=timeframe, refresh=True
                )
                for closed in tick.get("closed", []):
                    if int(closed.get("id") or 0) in scalper_ids:
                        continue  # scalper-owned — the scalper logs its own exits
                    counters["exited"] += 1
                    self._log_exit(closed, str(closed.get("exit_reason") or "stop_loss"))

                df = get_ohlcv(source, symbol, timeframe, limit=500, with_features=True)
                df = _drop_unclosed_candle(df, timeframe)
                if len(df) < _MIN_SCAN_ROWS:
                    logger.info(
                        "Not enough data (%d rows) to manage %s — leaving position as is",
                        len(df),
                        symbol,
                    )
                    continue
                if _is_stale(df, timeframe):
                    logger.warning(
                        "Stale data for %s:%s:%s (last closed candle %s) — "
                        "skipping the ensemble-flip check (SL/TP already ticked)",
                        source,
                        symbol,
                        timeframe,
                        df.index[-1].isoformat(),
                    )
                    continue
                stance = int(voter.vote_breakdown(df)["ensemble"].iloc[-1])
                for open_pos in self.engine.get_positions("open"):
                    pos_key = (
                        str(open_pos["symbol"]),
                        str(open_pos["source"]),
                        str(open_pos["timeframe"]),
                    )
                    if pos_key != key:
                        continue
                    if int(open_pos["id"]) in scalper_ids:
                        continue  # scalper-owned — never ensemble-flip-close it
                    flipped = (open_pos["side"] == "long" and stance < 0) or (
                        open_pos["side"] == "short" and stance > 0
                    )
                    if not flipped:
                        continue
                    closed = self.engine.close_position(int(open_pos["id"]))
                    counters["exited"] += 1
                    self._log_exit(closed, "ensemble_flip")
            except Exception as exc:  # noqa: BLE001 — per-symbol isolation
                counters["errors"] += 1
                logger.exception("Bot position management failed for %s", symbol)
                self._log(
                    "error", symbol=symbol, detail={"where": "manage", "reason": str(exc)}
                )

    def _scalper_owned_ids(self) -> set[int]:
        """Position ids owned by the fast scalper script (never raises).

        The scalper (``backend.paper_trading.scalper``) manages its own
        exits, so the bot must skip those positions in
        :meth:`_manage_positions`. Imported lazily to avoid a circular
        dependency (the scalper module imports this module's helpers); any
        failure conservatively returns an empty set so the bot's original
        behaviour is preserved.

        Returns:
            Set of scalper-owned position ids (possibly empty).
        """
        try:
            from backend.paper_trading.scalper import load_scalper_position_ids

            with self._connect() as conn:
                return load_scalper_position_ids(conn)
        except Exception:  # noqa: BLE001 — ownership info is best-effort
            logger.exception("Could not load scalper-owned position ids")
            return set()

    # ------------------------------------------------------------------ #
    # Intel/coach integration (all late-bound and fail-open: the bot keeps  #
    # its original behaviour when the intel layer is absent or broken)      #
    # ------------------------------------------------------------------ #

    def _load_playbook(self) -> dict[str, dict[str, Any]]:
        """The coach's per-coin playbook, keyed by symbol. Never raises.

        Imported late (contract) so the platform still works when the
        coach/intel layer is absent or broken, and so tests can monkeypatch
        either this method or ``coach.load_playbook_map``. One DB query per
        cycle; any failure returns ``{}`` (no playbook = no behaviour change).

        Returns:
            ``{symbol: playbook entry dict}`` (possibly empty).
        """
        try:
            from backend.paper_trading.coach import load_playbook_map

            return load_playbook_map()
        except Exception:  # noqa: BLE001 — the playbook is an optional layer
            logger.exception("Coach playbook unavailable — trading without it")
            return {}

    def _ollama_offline(self) -> tuple[bool, int | None]:
        """Read the watchdog's persisted Ollama status. Never raises.

        The API's watchdog job writes ``account_state['ollama_status']``
        (``{"available": bool, "since_ms": int, "restarts": int}``). Contract
        read rule: a MISSING or malformed key means ONLINE — a watchdog that
        never ran must not block trading.

        Returns:
            ``(offline, since_ms)`` — ``offline`` is True only when the
            status exists and explicitly says ``available == False``;
            ``since_ms`` is when that state began (None when unknown).
        """
        try:
            with self._connect() as conn:
                status = self.engine._state_get(conn, _OLLAMA_STATUS_KEY, None)
        except Exception:  # noqa: BLE001 — a broken status read = online
            logger.exception("Could not read the Ollama watchdog status")
            return False, None
        if not isinstance(status, dict) or status.get("available") is not False:
            return False, None
        try:
            raw_since = status.get("since_ms")
            since_ms = int(raw_since) if raw_since is not None else None
        except (TypeError, ValueError):
            since_ms = None
        return True, since_ms

    def _regime_cached(
        self,
        config: BotConfig,
        symbol: str,
        cache: dict[str, regime.RegimeInfo],
    ) -> regime.RegimeInfo:
        """Per-cycle cached market-regime lookup (contract: one call/symbol).

        ``regime.get_regime`` reads cached candles only and never raises; the
        cache guarantees at most one computation per symbol per cycle (and
        therefore BTCUSDT at most once per cycle, reused for every
        candidate).

        Args:
            config: Active bot configuration (supplies the data source).
            symbol: Symbol whose 4h regime is wanted.
            cache: The cycle's regime cache, mutated in place.

        Returns:
            The (possibly cached) :class:`~backend.paper_trading.regime.RegimeInfo`.
        """
        info = cache.get(symbol)
        if info is None:
            info = regime.get_regime(config.source, symbol)
            cache[symbol] = info
        return info

    def _fresh_sentiment(self, symbol: str) -> dict[str, Any] | None:
        """Latest FRESH news-sentiment row for ``symbol``. Never raises.

        Reads the ``coin_sentiment`` table (written by the intel layer)
        directly — the table schema is contract-fixed, and a plain read keeps
        this working regardless of the intel module's health. Rows older than
        :data:`_SENTIMENT_MAX_AGE_MS` (3h) are treated as absent: no gate and
        no prompt block on stale news.

        Args:
            symbol: Watchlist symbol (source-native format).

        Returns:
            dict with ``ts`` (epoch ms), ``sentiment``, ``score`` (clamped to
            [-100, 100]), ``confidence``, ``summary`` and ``headlines``
            (list) — or ``None`` when unavailable/stale/malformed.
        """
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT ts, sentiment, score, confidence, summary, headlines "
                    "FROM coin_sentiment WHERE symbol = ? "
                    "ORDER BY ts DESC, id DESC LIMIT 1",
                    (symbol,),
                ).fetchone()
        except sqlite3.Error as exc:
            logger.debug("coin_sentiment unavailable (%s) — no sentiment gate", exc)
            return None
        if row is None:
            return None
        try:
            ts = int(row["ts"])
            score = max(-100, min(100, int(row["score"])))
            confidence = max(0, min(100, int(row["confidence"])))
        except (TypeError, ValueError):
            logger.warning("Malformed coin_sentiment row for %s — ignored", symbol)
            return None
        if utc_now_ms() - ts > _SENTIMENT_MAX_AGE_MS:
            return None  # stale news never gates a live technical signal
        try:
            headlines = json.loads(row["headlines"] or "[]")
        except (TypeError, json.JSONDecodeError):
            headlines = []
        return {
            "ts": ts,
            "sentiment": str(row["sentiment"] or "neutral"),
            "score": score,
            "confidence": confidence,
            "summary": str(row["summary"] or ""),
            "headlines": headlines if isinstance(headlines, list) else [],
        }

    def _sentiment_bonus(
        self, config: BotConfig, symbol: str, vote_sum: int
    ) -> float:
        """Ranking-only bonus from fresh news sentiment (contract upgrade 1).

        ``bonus = sentiment_rank_weight × direction × (score / 100)`` where
        direction is +1 for long candidates (``vote_sum > 0``) and -1 for
        short ones, and ``score`` comes from the freshest ``coin_sentiment``
        row via :meth:`_fresh_sentiment` (same 3h staleness rule). Missing or
        stale sentiment → 0.0. The bonus ONLY reorders already-qualified
        candidates — the ``min_vote`` gate never reads it. Never raises.

        Args:
            config: Active bot configuration (supplies the weight).
            symbol: Candidate symbol.
            vote_sum: Last-bar strategy vote sum (sign = trade direction).

        Returns:
            The (possibly negative) ranking bonus; 0.0 when unavailable.
        """
        if vote_sum == 0 or config.sentiment_rank_weight <= 0.0:
            return 0.0
        sentiment = self._fresh_sentiment(symbol)
        if sentiment is None:
            return 0.0
        direction = 1.0 if vote_sum > 0 else -1.0
        score = float(sentiment.get("score") or 0)
        return config.sentiment_rank_weight * direction * (score / 100.0)

    def _news_context(
        self, symbol: str, sentiment: dict[str, Any]
    ) -> dict[str, Any]:
        """Compact "NEWS & SENTIMENT" payload for the analyst prompt.

        Args:
            symbol: The coin being analysed.
            sentiment: A fresh sentiment dict from :meth:`_fresh_sentiment`.

        Returns:
            JSON-safe dict with the score, summary, top 3 headlines and the
            fear & greed index (``None`` when the intel module is absent or
            its keyless public fetch fails — never raises).
        """
        top_headlines = [
            {
                "title": str(item.get("title") or ""),
                "source": str(item.get("source") or ""),
            }
            for item in sentiment.get("headlines") or []
            if isinstance(item, dict)
        ][:3]
        fear_greed: dict[str, Any] | None = None
        try:
            # Late-bound intel import (contract): absent/broken module → None.
            from backend.intel.news import fetch_fear_greed

            fear_greed = fetch_fear_greed()
        except Exception:  # noqa: BLE001 — the intel layer is optional
            logger.debug("Fear & greed unavailable for the news context")
        return {
            "symbol": symbol,
            "sentiment": sentiment.get("sentiment"),
            "score": sentiment.get("score"),
            "confidence": sentiment.get("confidence"),
            "summary": sentiment.get("summary"),
            "top_headlines": top_headlines,
            "fear_greed": fear_greed,
            "as_of": _ms_to_iso(int(sentiment.get("ts") or 0)),
        }

    def _scan_symbol(
        self,
        config: BotConfig,
        voter: VotingStrategy,
        symbol: str,
        counters: dict[str, int],
    ) -> dict[str, Any] | None:
        """Contract step 4: refresh + feature one watchlist symbol and vote.

        Votes are computed on CLOSED candles only (the still-forming last
        candle is dropped) and the symbol is skipped as ``stale_data`` when
        the last closed candle is older than ~2x the timeframe — never vote,
        AI-confirm or size on a frozen/repainting price.

        Returns:
            Candidate dict (``symbol``, ``df``, ``votes``, ``vote_sum``,
            ``momentum``) or None when the symbol was skipped/errored.
        """
        try:
            try:
                update_symbol(
                    config.source,
                    symbol,
                    config.timeframe,
                    lookback_days=_SCAN_LOOKBACK_DAYS,
                )
            except Exception as exc:  # noqa: BLE001 — stale cache may still work
                counters["errors"] += 1
                logger.warning(
                    "Bot data update failed for %s:%s:%s — %s",
                    config.source,
                    symbol,
                    config.timeframe,
                    exc,
                )
                self._log(
                    "error",
                    symbol=symbol,
                    detail={"where": "scan_update", "reason": str(exc)},
                )
            df = get_ohlcv(
                config.source, symbol, config.timeframe, limit=500, with_features=True
            )
        except Exception as exc:  # noqa: BLE001 — per-symbol isolation
            counters["errors"] += 1
            logger.warning("Bot scan failed for %s — %s", symbol, exc)
            self._log("error", symbol=symbol, detail={"where": "scan", "reason": str(exc)})
            return None

        df = _drop_unclosed_candle(df, config.timeframe)
        if len(df) < _MIN_SCAN_ROWS:
            counters["skipped"] += 1
            self._log(
                "skip",
                symbol=symbol,
                detail={
                    "reason": f"insufficient_data: {len(df)} rows < {_MIN_SCAN_ROWS}",
                    "explanation": (
                        f"Skipped {symbol}: only {len(df)} closed candles cached — at "
                        f"least {_MIN_SCAN_ROWS} are needed to evaluate the strategies."
                    ),
                },
            )
            return None

        if _is_stale(df, config.timeframe):
            counters["skipped"] += 1
            last_iso = df.index[-1].isoformat()
            self._log(
                "skip",
                symbol=symbol,
                detail={
                    "reason": f"stale_data: last closed candle {last_iso}",
                    "explanation": (
                        f"Skipped {symbol}: the newest closed candle is from "
                        f"{last_iso}, too old for the {config.timeframe} timeframe — "
                        "trading on stale prices is not allowed."
                    ),
                },
            )
            return None

        try:
            breakdown = voter.vote_breakdown(df)
        except Exception as exc:  # noqa: BLE001 — per-symbol isolation
            counters["errors"] += 1
            logger.exception("Vote computation failed for %s", symbol)
            self._log("error", symbol=symbol, detail={"where": "vote", "reason": str(exc)})
            return None

        last_votes = breakdown.iloc[-1]
        votes = {
            name: int(last_votes[name]) for name in breakdown.columns if name != "ensemble"
        }
        vote_sum = int(sum(votes.values()))
        last_row = df.iloc[-1]
        momentum = _json_num(last_row.get("momentum_10"))
        # Sentiment-boosted ranking (contract): the bonus reorders qualified
        # candidates only — the min_vote gate below never reads it.
        sentiment_bonus = self._sentiment_bonus(config, symbol, vote_sum)
        rank_score = abs(vote_sum) + sentiment_bonus
        counters["scanned"] += 1
        self._log(
            "scan",
            symbol=symbol,
            detail={
                "vote_sum": vote_sum,
                "momentum_10": momentum,
                "rsi_14": _json_num(last_row.get("rsi_14")),
                "price": _json_num(last_row.get("close")),
                "sentiment_bonus": _json_num(sentiment_bonus),
                "rank_score": _json_num(rank_score),
            },
        )
        return {
            "symbol": symbol,
            "df": df,
            "votes": votes,
            "vote_sum": vote_sum,
            "momentum": momentum if momentum is not None else 0.0,
            "sentiment_bonus": sentiment_bonus,
            "rank_score": rank_score,
        }

    def _shortlist(
        self,
        config: BotConfig,
        candidates: list[dict[str, Any]],
        playbook: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Contract step 5: filter, rank and cap the entry candidates.

        Keeps symbols with no open position, ``|vote_sum| >= min_vote`` and
        (unless ``allow_short``) a long-side vote only; ranks by
        ``rank_score = |vote_sum| + sentiment_bonus`` (sentiment-boosted
        ranking — fresh news reorders qualified candidates but NEVER
        qualifies one, the ``min_vote`` gate is unchanged) then
        ``|momentum_10|`` descending; caps at the number of free position
        slots (further capped by ``max_ai_calls_per_cycle`` when the AI gate
        is on).

        Coach playbook integration: a coin's ``min_vote_override`` replaces
        ``config.min_vote`` for that coin, and its ``side_bias`` filters the
        allowed entry direction (``long_only``/``short_only``) — both are
        silent shortlist filters, like the ``min_vote`` gate itself.
        """
        playbook = playbook or {}

        def _required_vote(candidate: dict[str, Any]) -> int:
            entry = playbook.get(candidate["symbol"]) or {}
            override = entry.get("min_vote_override")
            return int(override) if override else config.min_vote

        def _side_allowed(candidate: dict[str, Any]) -> bool:
            entry = playbook.get(candidate["symbol"]) or {}
            bias = str(entry.get("side_bias") or "both")
            if candidate["vote_sum"] > 0:
                return bias != "short_only"
            return config.allow_short and bias != "long_only"

        def _rank_score(candidate: dict[str, Any]) -> float:
            """Sentiment-boosted rank score, backfilled for hand-built candidates."""
            bonus = candidate.get("sentiment_bonus")
            if bonus is None:
                bonus = self._sentiment_bonus(
                    config, candidate["symbol"], int(candidate["vote_sum"])
                )
                candidate["sentiment_bonus"] = bonus
            score = abs(candidate["vote_sum"]) + float(bonus)
            candidate["rank_score"] = score
            return score

        open_symbols = {str(p["symbol"]) for p in self.engine.get_positions("open")}
        eligible = [
            c
            for c in candidates
            if c["symbol"] not in open_symbols
            and abs(c["vote_sum"]) >= _required_vote(c)
            and _side_allowed(c)
        ]
        eligible.sort(key=lambda c: (_rank_score(c), abs(c["momentum"])), reverse=True)

        portfolio = self.engine.get_portfolio()
        free_slots = max(
            0, settings.max_open_positions - int(portfolio.get("open_positions") or 0)
        )
        cap = min(free_slots, config.max_ai_calls_per_cycle) if config.use_ai else free_slots
        return eligible[:cap]

    def _confirm_candidate(
        self,
        config: BotConfig,
        candidate: dict[str, Any],
        counters: dict[str, int],
        regime_cache: dict[str, regime.RegimeInfo] | None = None,
    ) -> bool:
        """Entry pass 1 (contract step 6): hard gates + primary AI gate.

        Gate order (contract v3 — the cheap hard gates run BEFORE any LLM
        call, saving GPU time):

        1. REGIME gate (when ``regime_gate_enabled``): the shared
           ``regime.regime_allows`` predicate — shorts only in a coin+BTC 4h
           double-downtrend, longs blocked in exactly that state.
        2. COST gate (when ``cost_gate_multiple > 0``): the expected per-bar
           move (ATR/price) must cover ``cost_gate_multiple`` × the
           round-trip fee+slippage cost.
        3. News-sentiment veto, then the primary AI gate (both unchanged).

        Intel integration: a FRESH (< 3h) news-sentiment score at/below -50
        blocks a long (skip ``news_negative``) and at/above +50 blocks a
        short (``news_positive``); the fresh sentiment is also forwarded to
        ``analyze_market`` as its ``news_context``. The primary AI must agree
        with the trade direction at ``min_ai_confidence``+ (conservative: an
        unavailable AI blocks the entry).

        Side effects on ``candidate`` (consumed by the judge/entry passes):
        ``ai`` (primary transparency payload, None when ``use_ai`` is off),
        ``ai_second`` (initialised to None), ``news_context`` and
        ``htf_context`` (factual higher-timeframe lines for both analyst
        calls, built from the per-cycle regime cache even when the regime
        gate is disabled).

        Args:
            config: Active bot configuration.
            candidate: Shortlisted candidate dict from :meth:`_scan_symbol`.
            counters: Cycle counters, mutated in place.
            regime_cache: The cycle's shared regime cache (None → private
                one-off cache, keeping direct calls working).

        Returns:
            True when the candidate survives to the second-judge/entry pass.
        """
        symbol: str = candidate["symbol"]
        df: pd.DataFrame = candidate["df"]
        vote_sum: int = candidate["vote_sum"]
        direction = "long" if vote_sum > 0 else "short"
        candidate["ai"] = None
        candidate["ai_second"] = None
        candidate["news_context"] = None
        if regime_cache is None:
            regime_cache = {}

        # Higher-timeframe context for BOTH analyst calls — factual lines
        # only, computed once per candidate from the per-cycle regime cache
        # (works even when the regime gate is disabled).
        symbol_info = self._regime_cached(config, symbol, regime_cache)
        candidate["htf_context"] = _htf_context(symbol_info)

        # 1) REGIME hard gate — above the ensemble result, before any AI call.
        if config.regime_gate_enabled:
            btc_info = self._regime_cached(
                config, _REGIME_REFERENCE_SYMBOL, regime_cache
            )
            if not regime.regime_allows(
                direction, symbol_info.regime, btc_info.regime
            ):
                counters["skipped"] += 1
                if direction == "short":
                    explanation = (
                        f"Skipped a short in {symbol}: shorts are only allowed "
                        "when both the coin and Bitcoin are in a 4h downtrend "
                        f"— {symbol} is {symbol_info.regime} and BTC is "
                        f"{btc_info.regime}."
                    )
                else:
                    explanation = (
                        f"Skipped a long in {symbol}: longs are blocked while "
                        "both the coin and Bitcoin are in a 4h downtrend — "
                        f"{symbol} is {symbol_info.regime} and BTC is "
                        f"{btc_info.regime}."
                    )
                self._log(
                    "skip",
                    symbol=symbol,
                    detail={
                        "reason": "regime_block",
                        "side": direction,
                        "symbol_regime": symbol_info.regime,
                        "btc_regime": btc_info.regime,
                        "close": _json_num(symbol_info.close),
                        "ema50": _json_num(symbol_info.ema50),
                        "ema200": _json_num(symbol_info.ema200),
                        "vote_sum": vote_sum,
                        "votes": candidate["votes"],
                        "explanation": explanation,
                    },
                )
                return False

        # 2) COST gate — the expected move must be worth the round trip.
        if config.cost_gate_multiple > 0:
            last_row = df.iloc[-1]
            gate = regime.cost_gate(
                _finite(last_row.get("atr_14")),
                _finite(last_row.get("close")),
                settings.commission_rate,
                settings.slippage_rate,
                config.cost_gate_multiple,
            )
            if not gate.get("passes"):
                counters["skipped"] += 1
                expected = _json_num(gate.get("expected_move_pct"))
                needed = _json_num(gate.get("needed_pct")) or 0.0
                move_txt = (
                    f"the typical {config.timeframe} move is "
                    f"{expected * 100:.2f}% of price"
                    if expected is not None
                    else f"the typical {config.timeframe} move is unknown"
                )
                self._log(
                    "skip",
                    symbol=symbol,
                    detail={
                        "reason": "cost_gate",
                        "expected_move_pct": expected,
                        "needed_pct": needed,
                        "explanation": (
                            f"Skipped {symbol}: {move_txt}, but fees + slippage "
                            f"need at least {needed * 100:.2f}% of expected "
                            "movement to be worth trading."
                        ),
                    },
                )
                return False

        # 3) News-sentiment gate: strongly negative fresh news blocks longs,
        # strongly positive blocks shorts. Stale (>3h) or absent sentiment
        # applies no gate at all.
        sentiment = self._fresh_sentiment(symbol)
        if sentiment is not None:
            score = int(sentiment.get("score") or 0)
            gate_reason: str | None = None
            if direction == "long" and score <= -_SENTIMENT_BLOCK_SCORE:
                gate_reason = "news_negative"
            elif direction == "short" and score >= _SENTIMENT_BLOCK_SCORE:
                gate_reason = "news_positive"
            if gate_reason is not None:
                counters["skipped"] += 1
                self._log(
                    "skip",
                    symbol=symbol,
                    detail={
                        "reason": gate_reason,
                        "score": score,
                        "sentiment": str(sentiment.get("sentiment") or ""),
                        "summary": str(sentiment.get("summary") or ""),
                        "explanation": (
                            f"Skipped {symbol}: fresh news sentiment scores "
                            f"{score:+d} (-100..100), too "
                            f"{'negative for a long' if gate_reason == 'news_negative' else 'positive for a short'}"
                            " entry right now."
                        ),
                    },
                )
                return False
        candidate["news_context"] = (
            self._news_context(symbol, sentiment) if sentiment is not None else None
        )

        # AI confirmation gate (conservative: unavailable AI blocks the entry).
        if config.use_ai:
            try:
                analysis = analyze_market(
                    symbol,
                    config.timeframe,
                    df,
                    news_context=candidate["news_context"],
                    htf_context=str(candidate.get("htf_context") or ""),
                )
            except OllamaError as exc:
                counters["errors"] += 1
                logger.warning(
                    "AI confirmation unavailable for %s — skipping candidate: %s",
                    symbol,
                    exc,
                )
                self._log(
                    "error",
                    symbol=symbol,
                    detail={
                        "where": "ai_confirm",
                        "reason": str(exc),
                        "explanation": (
                            f"Skipped {symbol}: the AI analyst was unavailable, and "
                            "entering without confirmation is not allowed."
                        ),
                    },
                )
                return False
            ai_detail = {
                "sentiment": analysis.sentiment,
                "confidence": int(analysis.confidence),
                "reasoning": analysis.reasoning,
                "risk_commentary": analysis.risk_commentary,
                "opposing_case": str(getattr(analysis, "opposing_case", "") or ""),
            }
            candidate["ai"] = ai_detail
            wanted = "bullish" if direction == "long" else "bearish"
            if analysis.sentiment != wanted or int(analysis.confidence) < config.min_ai_confidence:
                counters["skipped"] += 1
                self._log(
                    "skip",
                    symbol=symbol,
                    detail={
                        "reason": (
                            f"ai_gate: sentiment={analysis.sentiment}, "
                            f"confidence={int(analysis.confidence)}"
                        ),
                        "ai": ai_detail,
                        "explanation": (
                            f"Skipped {symbol}: AI is {analysis.sentiment} at "
                            f"{int(analysis.confidence)}% confidence, but a {direction} "
                            f"entry requires {wanted} sentiment with at least "
                            f"{config.min_ai_confidence}% confidence."
                        ),
                    },
                )
                return False
        return True

    def _second_judge_ready(self, config: BotConfig) -> bool:
        """Whether the second judge gates this cycle's entries. Never raises.

        Contract model-availability rule: the judge only runs when the AI
        gate is on, ``use_second_judge`` is enabled AND
        ``second_judge_model`` is installed in Ollama (matched exactly and by
        the base name before ``:``). A configured-but-missing model — e.g.
        the Fin-R1 pull has not finished yet — logs ONE warning-level
        ``error`` activity row for the cycle and entries proceed WITHOUT the
        judge.

        That fail-open path requires an AUTHORITATIVE model list.
        ``list_models`` also returns ``[]`` on any server/transport failure
        (its never-raises contract), which is indistinguishable from "no
        models installed" — so an empty list is retried once, and if it is
        still empty the judge stays ACTIVE: availability is unknown, and the
        strict no-fallback judge call in :meth:`_confirm_judge` will surface
        a genuinely unavailable server/model as ``OllamaError`` and skip
        each candidate conservatively. A transient ``/api/tags`` blip must
        never silently drop the risk gate for the whole cycle.

        Args:
            config: Active bot configuration.

        Returns:
            True when the judge pass must run for the surviving candidates.
        """
        if not (config.use_ai and config.use_second_judge):
            return False
        model = config.second_judge_model
        client = OllamaClient()
        installed = client.list_models()
        if not installed:
            # [] is also list_models' failure value — retry once before
            # treating the catalogue as authoritative.
            installed = client.list_models()
        if _model_installed(model, installed):
            return True
        if not installed:
            logger.warning(
                "Could not obtain an authoritative Ollama model list — keeping "
                "the second judge (%s) ACTIVE; its strict no-fallback call "
                "skips candidates conservatively if the server/model is "
                "really unavailable",
                model,
            )
            return True
        logger.warning(
            "Second-judge model %r is not installed (%d models present) — "
            "entering WITHOUT the second judge this cycle",
            model,
            len(installed),
        )
        self._log(
            "error",
            detail={
                "where": "second_judge",
                "reason": f"judge_model_missing: {model}",
                "explanation": (
                    f"The second-judge model {model} is not installed in Ollama "
                    "yet — entries proceed with the primary AI confirmation only "
                    "until the model pull finishes."
                ),
            },
        )
        return False

    def _confirm_judge(
        self,
        config: BotConfig,
        candidate: dict[str, Any],
        counters: dict[str, int],
    ) -> bool:
        """Entry pass 2 (contract step 6): the Fin-R1 second-judge gate.

        Re-runs ``analyze_market`` with ``model=config.second_judge_model``
        on the SAME frame, news context and higher-timeframe context the
        primary gate saw — in STRICT
        no-fallback mode (``allow_fallback=False``): a second opinion from
        the generic fallback model would be worthless (it may even be the
        same model the primary gate fell back to, confirming itself), so a
        missing or failing judge must surface as ``OllamaError`` here. The
        entry survives only when the judge's sentiment matches the trade
        direction with at least ``judge_min_confidence`` confidence —
        otherwise the candidate is skipped with a ``judge_disagreement``
        reason. A failing judge (``OllamaError``) skips conservatively with
        an ``error`` row.

        Side effect: stores ``candidate["ai_second"]``
        (``{sentiment, confidence, model}``) for the transparency payloads.

        Args:
            config: Active bot configuration.
            candidate: Candidate that already passed the primary gate.
            counters: Cycle counters, mutated in place.

        Returns:
            True when the judge agrees and the entry may proceed.
        """
        symbol: str = candidate["symbol"]
        direction = "long" if candidate["vote_sum"] > 0 else "short"
        model = config.second_judge_model
        try:
            analysis = analyze_market(
                symbol,
                config.timeframe,
                candidate["df"],
                model=model,
                news_context=candidate.get("news_context"),
                allow_fallback=False,
                htf_context=str(candidate.get("htf_context") or ""),
            )
        except OllamaError as exc:
            counters["errors"] += 1
            logger.warning(
                "Second judge (%s) failed for %s — skipping candidate: %s",
                model,
                symbol,
                exc,
            )
            self._log(
                "error",
                symbol=symbol,
                detail={
                    "where": "second_judge",
                    "reason": str(exc),
                    "explanation": (
                        f"Skipped {symbol}: the second judge ({model}) is "
                        "installed but failed to answer, and entering without "
                        "its confirmation is not allowed."
                    ),
                },
            )
            return False
        confidence = int(analysis.confidence)
        ai_second = {
            "sentiment": analysis.sentiment,
            "confidence": confidence,
            "model": model,
        }
        candidate["ai_second"] = ai_second
        wanted = "bullish" if direction == "long" else "bearish"
        if analysis.sentiment != wanted or confidence < config.judge_min_confidence:
            counters["skipped"] += 1
            self._log(
                "skip",
                symbol=symbol,
                detail={
                    "reason": (
                        f"judge_disagreement: {model} said "
                        f"{analysis.sentiment} at {confidence}%"
                    ),
                    "ai": candidate.get("ai"),
                    "ai_second": ai_second,
                    "explanation": (
                        f"Skipped {symbol}: the second judge ({model}) said "
                        f"{analysis.sentiment} at {confidence}% confidence, but a "
                        f"{direction} entry requires {wanted} sentiment with at "
                        f"least {config.judge_min_confidence}% confidence."
                    ),
                },
            )
            return False
        return True

    def _enter_candidate(
        self,
        config: BotConfig,
        candidate: dict[str, Any],
        counters: dict[str, int],
        playbook: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Contract step 6 (final): size and submit one confirmed entry order.

        Runs AFTER :meth:`_confirm_candidate` (hard gates + sentiment veto +
        primary AI gate) and, when active, :meth:`_confirm_judge` — the
        candidate dict carries their ``ai``/``ai_second`` transparency
        payloads. The coach playbook's ``size_multiplier`` scales the
        quantity AFTER the existing caps, with the ``max_position_fraction``,
        cash and dust guards re-applied (a multiplier above 1.0 can never
        breach the per-coin fraction cap). Contract v3 additions:
        ``RiskManager.derisk_multiplier`` shrinks the final quantity while
        the account is in drawdown or near its soft daily stop, and a full-
        context ``check_order`` (price + stop distance → portfolio heat cap,
        direction cap) runs immediately before ``submit_order``.
        """
        playbook = playbook or {}
        symbol: str = candidate["symbol"]
        df: pd.DataFrame = candidate["df"]
        vote_sum: int = candidate["vote_sum"]
        votes: dict[str, int] = candidate["votes"]
        side = "buy" if vote_sum > 0 else "sell"
        direction = "long" if side == "buy" else "short"
        ai_detail: dict[str, Any] | None = candidate.get("ai")

        # Sizing: min(ATR risk size, max fraction of equity, 95% of free cash).
        # The engine fills market orders at the LATEST cached close at submit
        # time, which can drift from the scan-time close (data scheduler
        # upserts, slow AI confirmations) — so re-read it now and size/place
        # stops against the same price the fill will use, otherwise a price
        # jump between scan and fill could breach the caps (even drive cash
        # negative on a small account).
        last_row = df.iloc[-1]
        price = float(last_row["close"])
        try:
            latest = load_ohlcv(config.source, symbol, config.timeframe, limit=1)
            if not latest.empty:
                price = float(latest["close"].iloc[-1])
        except Exception as exc:  # noqa: BLE001 — fall back to the scan close
            logger.warning(
                "Could not re-read the latest cached close for %s — sizing off "
                "the scan-time close: %s",
                symbol,
                exc,
            )
        atr_raw = last_row.get("atr_14")
        atr_value = float(atr_raw) if pd.notna(atr_raw) else 0.0
        portfolio = self.engine.get_portfolio()
        equity = float(portfolio["equity"])
        cash = float(portfolio["cash"])
        if price <= 0 or not math.isfinite(price):
            counters["skipped"] += 1
            self._log(
                "skip",
                symbol=symbol,
                detail={
                    "reason": f"invalid_price: {price!r}",
                    "explanation": f"Skipped {symbol}: latest cached price is invalid.",
                },
            )
            return
        qty = min(
            self.risk.position_size(equity, atr_value, price),
            (config.max_position_fraction * equity) / price,
            (0.95 * cash) / price,
        )
        # Coach playbook size multiplier — applied AFTER the existing caps
        # (contract), then the per-coin fraction cap and the cash guard are
        # re-applied so a multiplier above 1.0 can never exceed
        # max_position_fraction of equity ("35% max per coin" is a hard,
        # documented risk limit) or spend beyond free cash; the dust guard
        # below runs on the final quantity.
        playbook_entry = playbook.get(symbol) or {}
        try:
            size_multiplier = float(playbook_entry.get("size_multiplier", 1.0))
        except (TypeError, ValueError):
            size_multiplier = 1.0
        if math.isfinite(size_multiplier) and size_multiplier != 1.0:
            qty = min(
                qty * size_multiplier,
                (config.max_position_fraction * equity) / price,
                (0.95 * cash) / price,
            )
        # De-risk multiplier (contract v3) — the FINAL sizing step: shrink new
        # entries by 0.8 per 10% of drawdown from peak equity, halved again
        # while the soft daily stop is active. The bot still enters during a
        # soft daily stop — just at the reduced size.
        derisk_multiplier = RiskManager.derisk_multiplier(
            equity,
            portfolio["peak_equity"],
            portfolio["realized_pnl_today"],
            settings.daily_loss_limit,
        )
        qty *= derisk_multiplier
        notional = qty * price
        if qty <= 0 or notional < _MIN_NOTIONAL_USD:
            counters["skipped"] += 1
            self._log(
                "skip",
                symbol=symbol,
                detail={
                    "reason": (
                        f"dust: notional ${max(notional, 0.0):.2f} < "
                        f"${_MIN_NOTIONAL_USD:.2f}"
                    ),
                    "explanation": (
                        f"Skipped {symbol}: the risk-based position size works out to "
                        f"${max(notional, 0.0):.2f}, below the "
                        f"${_MIN_NOTIONAL_USD:.2f} minimum."
                    ),
                },
            )
            return

        # ATR-based protective levels (price -/+ ATR x settings multipliers).
        sl_offset = settings.atr_stop_multiplier * atr_value
        tp_offset = settings.atr_take_profit_multiplier * atr_value
        if direction == "long":
            stop_loss: float | None = price - sl_offset
            take_profit: float | None = price + tp_offset
        else:
            stop_loss = price + sl_offset
            take_profit = price - tp_offset
        if stop_loss is not None and stop_loss <= 0:
            stop_loss = None
        if take_profit is not None and take_profit <= 0:
            take_profit = None

        # Re-check for an open position right before submitting: the
        # shortlist's open-symbols snapshot is minutes old by now (AI
        # confirmations can take up to ollama_timeout each) and the scalper
        # ticks every couple of minutes — it may have entered this symbol
        # meanwhile. A second position would concentrate far more than
        # max_position_fraction in one coin, or open an opposite-side hedge
        # on another timeframe that nets ~zero price PnL while paying
        # commissions and slippage on both legs.
        if any(
            str(p["symbol"]) == symbol for p in self.engine.get_positions("open")
        ):
            counters["skipped"] += 1
            self._log(
                "skip",
                symbol=symbol,
                detail={
                    "reason": "position_exists",
                    "explanation": (
                        f"Skipped {symbol}: a position was opened in it "
                        "(e.g. by the fast scalper) while this cycle was "
                        "confirming the entry."
                    ),
                },
            )
            return

        # Pre-submit risk check with FULL order context (contract v3): the
        # engine's own check_order call has no price/stop, so the portfolio
        # heat cap and the direction cap are enforced here, where the stop
        # distance is known.
        check = self.risk.check_order(
            portfolio,
            self.engine.get_positions("open"),
            side=side,
            qty=qty,
            symbol=symbol,
            source=config.source,
            timeframe=config.timeframe,
            price=price,
            stop_loss=stop_loss,
        )
        if not check.allowed:
            counters["rejected"] += 1
            self._log(
                "reject",
                symbol=symbol,
                detail={
                    "reason": check.reason,
                    "explanation": (
                        f"Entry order for {symbol} was blocked by the risk "
                        f"manager before submission: {check.reason}"
                    ),
                },
            )
            return

        try:
            order = self.engine.submit_order(
                symbol,
                side,
                "market",
                qty=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                source=config.source,
                timeframe=config.timeframe,
            )
        except ValueError as exc:
            counters["rejected"] += 1
            self._log(
                "reject",
                symbol=symbol,
                detail={
                    "reason": str(exc),
                    "explanation": (
                        f"Entry order for {symbol} was rejected by the risk "
                        f"manager: {exc}"
                    ),
                },
            )
            return

        fill_price = float(order.get("fill_price") or price)
        notional = qty * fill_price
        features = feature_summary(df)
        indicators = {
            key: features.get(key)
            for key in (
                "rsi_14",
                "macd_state",
                "momentum_10",
                "atr_14",
                "price_vs_sma20",
                "bb_position",
            )
        }
        counters["entered"] += 1
        self._log(
            "enter",
            symbol=symbol,
            detail={
                "side": direction,
                "qty": _json_num(qty),
                "price": _json_num(fill_price),
                "notional": _json_num(notional),
                "stop_loss": _json_num(stop_loss),
                "take_profit": _json_num(take_profit),
                "vote_sum": vote_sum,
                "votes": votes,
                "rank_score": _json_num(candidate.get("rank_score", abs(vote_sum))),
                "sentiment_bonus": _json_num(candidate.get("sentiment_bonus", 0.0)),
                "shadow_flags": regime.shadow_flags(
                    df, datetime.now(timezone.utc).hour
                ),
                "derisk_multiplier": _json_num(derisk_multiplier),
                "indicators": indicators,
                "ai": ai_detail,
                "ai_second": candidate.get("ai_second"),
                "explanation": self._entry_explanation(
                    symbol,
                    direction,
                    qty,
                    notional,
                    votes,
                    indicators,
                    ai_detail,
                    stop_loss,
                    take_profit,
                ),
            },
        )

    def _finish_cycle(
        self,
        counters: dict[str, int],
        started_ms: int,
        status: str,
        halt_reason: str = "",
    ) -> dict[str, Any]:
        """Contract step 7: snapshot equity, log ``cycle_end``, persist summary."""
        equity: float | None = None
        try:
            portfolio = self.engine.get_portfolio()
            equity = float(portfolio["equity"])
            self._snapshot_equity(portfolio)
        except Exception:  # noqa: BLE001 — the summary must still be produced
            logger.exception("Could not snapshot equity at cycle end")

        self._log(
            "cycle_end",
            detail={
                "scanned": counters["scanned"],
                "entered": counters["entered"],
                "exited": counters["exited"],
                "skipped": counters["skipped"],
                "equity": equity,
            },
        )
        finished_ms = utc_now_ms()
        summary: dict[str, Any] = {
            "status": status,
            "scanned": counters["scanned"],
            "entered": counters["entered"],
            "exited": counters["exited"],
            "skipped": counters["skipped"],
            "rejected": counters["rejected"],
            "errors": counters["errors"],
            "equity": equity,
            "started_at": _ms_to_iso(started_ms),
            "finished_at": _ms_to_iso(finished_ms),
            "duration_s": round((finished_ms - started_ms) / 1000.0, 3),
        }
        if halt_reason:
            summary["halt_reason"] = halt_reason
        self._record_cycle(summary)
        logger.info(
            "Bot cycle finished (%s): scanned=%d entered=%d exited=%d skipped=%d "
            "rejected=%d errors=%d equity=%s",
            status,
            counters["scanned"],
            counters["entered"],
            counters["exited"],
            counters["skipped"],
            counters["rejected"],
            counters["errors"],
            equity,
        )
        return summary

    # ------------------------------------------------------------------ #
    # Explanations & logging                                               #
    # ------------------------------------------------------------------ #

    def _entry_explanation(
        self,
        symbol: str,
        direction: str,
        qty: float,
        notional: float,
        votes: dict[str, int],
        indicators: dict[str, Any],
        ai_detail: dict[str, Any] | None,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> str:
        """One plain-English sentence describing an entry (contract mandate)."""
        base = _base_asset(symbol)
        target_vote = 1 if direction == "long" else -1
        n_agree = sum(1 for v in votes.values() if v == target_vote)
        rsi_val = indicators.get("rsi_14")
        rsi_txt = (
            f"RSI {rsi_val:.0f}" if isinstance(rsi_val, (int, float)) else "RSI n/a"
        )
        ai_txt = (
            f"AI {ai_detail['sentiment']} at {ai_detail['confidence']}%"
            if ai_detail is not None
            else "AI confirmation off"
        )
        sl_txt = f"stop ${stop_loss:,.2f}" if stop_loss is not None else "no stop"
        tp_txt = f"target ${take_profit:,.2f}" if take_profit is not None else "no target"
        verb = "Bought" if direction == "long" else "Shorted"
        return (
            f"{verb} {qty:.6g} {base} (${notional:,.2f}): {n_agree}/4 strategies "
            f"vote {direction}, {rsi_txt}, {ai_txt} — {sl_txt}, {tp_txt}."
        )

    def _log_exit(self, position: dict[str, Any], reason: str) -> None:
        """Write an ``exit`` activity row for a closed position.

        Contract v3: every exit row carries the R-multiple diagnostics
        ``designed_r`` (reward:risk the entry was placed with) and
        ``realized_r`` (actual PnL in units of the risked stop distance) —
        both ``None`` when the position had no valid stop geometry.
        """
        symbol = str(position.get("symbol") or "")
        side = str(position.get("side") or "long")
        qty = float(position.get("qty") or 0.0)
        entry_price = float(position.get("entry_price") or 0.0)
        exit_raw = position.get("exit_price")
        exit_price = float(exit_raw) if exit_raw is not None else entry_price
        pnl = float(position.get("pnl") or 0.0)
        entry_notional = qty * entry_price
        pnl_pct = pnl / entry_notional if entry_notional > 0 else 0.0
        designed_r, realized_r = _r_multiples(position)

        if reason == "ensemble_flip":
            why = f"the strategy ensemble flipped against the {side} stance"
        elif reason == "stop_loss":
            why = "the protective stop-loss was hit"
        elif reason == "take_profit":
            why = "the take-profit target was reached"
        else:
            why = f"exit reason: {reason}"
        explanation = (
            f"Closed {side} {qty:.6g} {_base_asset(symbol)} at ${exit_price:,.2f} "
            f"because {why} — PnL ${pnl:+,.2f} ({pnl_pct:+.2%})."
        )
        self._log(
            "exit",
            symbol=symbol,
            detail={
                "reason": reason,
                "pnl": _json_num(pnl),
                "pnl_pct": _json_num(pnl_pct),
                "designed_r": _json_num(designed_r),
                "realized_r": _json_num(realized_r),
                "explanation": explanation,
            },
        )

    def _log(
        self, action: str, symbol: str = "", detail: dict[str, Any] | None = None
    ) -> None:
        """Append a row to ``bot_activity`` (best-effort — never raises)."""
        try:
            payload = json.dumps(detail or {}, default=str)
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO bot_activity (ts, symbol, action, detail) "
                    "VALUES (?, ?, ?, ?)",
                    (utc_now_ms(), symbol, action, payload),
                )
        except Exception:  # noqa: BLE001 — logging must never break a cycle
            logger.exception(
                "Failed to write bot_activity row (action=%s, symbol=%s)", action, symbol
            )

    # ------------------------------------------------------------------ #
    # Persistence helpers                                                  #
    # ------------------------------------------------------------------ #

    def _record_cycle(self, summary: dict[str, Any]) -> None:
        """Persist the last-cycle timestamp and summary (best-effort)."""
        try:
            with self._connect() as conn:
                self.engine._state_set(conn, _LAST_CYCLE_TS_KEY, utc_now_ms())
                self.engine._state_set(conn, _LAST_CYCLE_SUMMARY_KEY, summary)
        except Exception:  # noqa: BLE001 — best-effort bookkeeping
            logger.exception("Could not persist bot cycle summary")

    def _snapshot_equity(self, portfolio: dict[str, Any]) -> None:
        """Append an ``equity_snapshots`` row (same shape the engine writes)."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO equity_snapshots "
                "(ts, equity, cash, unrealized_pnl, realized_pnl_today) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    utc_now_ms(),
                    float(portfolio["equity"]),
                    float(portfolio["cash"]),
                    float(portfolio["unrealized_pnl"]),
                    float(portfolio["realized_pnl_today"]),
                ),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection that commits on success and always closes."""
        conn = get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
