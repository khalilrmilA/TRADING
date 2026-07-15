"""Fast mechanical scalper + hourly AI parameter supervisor — PAPER ONLY.

Two-tier design (see CONTRACTS.md, section ``backend/paper_trading/scalper.py``):

* :class:`Scalper` — a **fast mechanical script** with no LLM in the trading
  loop. Every ``interval_minutes`` it reads CLOSED 15m candles for the bot's
  watchlist, manages its own open scalp positions (engine-level SL/TP plus a
  time-stop) and enters new scalps on a fixed three-condition momentum rule
  (EMA trend + RSI band + VWAP side).
* :meth:`Scalper.tune_with_ai` — an **AI supervisor** that reviews the
  script's recent results hourly and re-tunes its parameters, with every
  proposed value CLAMPED into :data:`HARD_BOUNDS` no matter what the model
  says. Stops can never be disabled.

All orders go through the existing :class:`PaperTradingEngine` +
:class:`RiskManager` (global position cap, daily-loss halt and circuit
breaker gate every order). Fills happen only against locally cached public
candles — nothing here can ever place a real order or touch an API key.

Ownership separation: the ids of positions the scalper opened are tracked in
the ``account_state`` key ``'scalper_position_ids'`` (JSON list, pruned on
close). The :class:`AutoTrader` skips those ids in its position management,
and the scalper manages ONLY its own — the two traders share one simulated
account without stepping on each other's exits.

TRANSPARENCY: every decision is appended to ``bot_activity`` with actions
``scalp_enter`` / ``scalp_exit`` / ``scalp_skip`` / ``scalp_tune`` (plus
plain ``error`` rows), each carrying a one-sentence plain-English
``explanation``. Routine non-events (no entry signal on a symbol, symbol
already holds a position) are logged at DEBUG only — at a 1-2 minute cadence
they would otherwise drown the feed.

Stopping the script (API ``POST /api/scalper/stop`` → :meth:`Scalper.stop`)
market-closes ALL scalper-owned positions (``scalp_exit`` reason
``script_stopped``): a stopped script no longer runs its time-stop, so open
scalps are deliberately not left behind. Disabling via ``set_params`` alone
does NOT close positions — use ``stop()``.

Improvement Pack v3 (see CONTRACTS.md, section ``## Improvement Pack v3``):

* **Regime hard gate** — scalp shorts fire only when the symbol AND BTCUSDT
  are both in a 4h downtrend (longs are blocked only in that same state);
  the shared predicate is :func:`backend.paper_trading.regime.regime_allows`.
* **Cost gate** — an entry needs the 15m ATR to be at least
  ``cost_gate_multiple`` × the round-trip fee+slippage cost (0 disables).
* **ATR geometry** — TP/SL scale with the last closed candle's ATR, always
  clamped inside :data:`HARD_BOUNDS` (the fixed ``tp_pct``/``sl_pct`` remain
  the fallback and the tuner's object).
* **Soft daily stop** — at 80% of the daily loss limit the scalper stops
  opening NEW scalps for the tick (managing exits always continues).
* **De-risk ratchet** — every entry size is multiplied by
  :meth:`RiskManager.derisk_multiplier` (drawdown ratchet + soft-stop halving).
* **Shrink-only tuner** — until 100 closed scalp trades exist, the AI
  supervisor may never widen the stop or raise ``max_positions`` /
  ``position_fraction`` / ``max_trades_per_day``; raising
  ``position_fraction`` after a losing tune window (martingale) is ALWAYS
  banned. Offending fields are kept at their current values and audited in
  the ``scalp_tune`` detail (``shrink_only_clamped`` / ``martingale_blocked``).
* **R logging** — every ``scalp_exit`` carries ``designed_r``/``realized_r``;
  every ``scalp_enter`` carries observation-only ``shadow_flags``.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

from backend.ai.ollama_client import OllamaClient, OllamaError
from backend.data import service as _data_service
from backend.database.db import get_conn, init_db, load_ohlcv, utc_now_ms

# Contract-directed reuse of the AutoTrader's tested helpers (closed-candle,
# staleness, timeframe and formatting utilities) — imported, not copy-pasted,
# so both traders keep exactly one behaviour for these guards.
from backend.paper_trading.auto_trader import (  # noqa: PLC2701 — sanctioned reuse
    _CONFIG_KEY as _BOT_CONFIG_KEY,
    _MIN_NOTIONAL_USD,
    _MIN_SCAN_ROWS,
    _TIMEFRAME_DELTAS,
    BotConfig,
    _base_asset,
    _drop_unclosed_candle,
    _is_stale,
    _json_num,
    _ms_to_iso,
    _timeframe_delta,
)
from backend.paper_trading import regime
from backend.paper_trading.engine import (
    LAST_RESET_MS_KEY as _LAST_RESET_KEY,
    PaperTradingEngine,
)
from backend.paper_trading.risk import RiskManager
from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistence keys (account_state)
# ---------------------------------------------------------------------------

#: Persisted :class:`ScalperParams` (contract-mandated key).
_PARAMS_KEY = "scalper_params"
#: Open scalper-owned position ids, JSON list (contract-mandated key).
SCALPER_POSITION_IDS_KEY = "scalper_position_ids"
#: Per-symbol anti-churn cooldowns: ``{symbol: until_epoch_ms}``.
_COOLDOWNS_KEY = "scalper_cooldowns"
#: UTC-day-scoped entry counter: ``{"date": iso_date, "count": int}``.
_TRADES_TODAY_KEY = "scalper_trades_today"
#: Last AI tune record (before/after/applied/reasoning/model/ts).
_LAST_TUNE_KEY = "scalper_last_tune"

#: Incremental history depth fetched per symbol per tick (15m candles).
_SCAN_LOOKBACK_DAYS = 7

#: Watchdog-owned ``account_state`` key (cross-module contract, section 8 of
#: Improvement Pack v3). READ-ONLY here: a missing or malformed value means
#: ONLINE — a watchdog that never ran must not block tuning.
_OLLAMA_STATUS_KEY = "ollama_status"

#: The market-regime reference symbol (both traders share this anchor).
_BTC_SYMBOL = "BTCUSDT"

# ---------------------------------------------------------------------------
# Improvement Pack v3 — binding module constants (CONTRACTS.md section 6)
# ---------------------------------------------------------------------------

#: Stop distance as a multiple of the last closed candle's ATR (ATR geometry).
ATR_GEOMETRY_SL_MULT = 1.5
#: Target distance as a multiple of the ATR-scaled stop distance.
ATR_GEOMETRY_TP_MULT = 2.0
#: Closed scalp trades required before the AI tuner may size anything UP.
SHRINK_ONLY_MIN_TRADES = 100

#: Params the shrink-only tuner may never RAISE before
#: :data:`SHRINK_ONLY_MIN_TRADES` closed trades exist (widen stop / more
#: concurrent scalps / bigger size / more trades per day = sizing UP).
_SHRINK_ONLY_FIELDS = (
    "sl_pct",
    "max_positions",
    "position_fraction",
    "max_trades_per_day",
)

# ---------------------------------------------------------------------------
# Hard safety bounds — enforced with clamping WHATEVER the source of a change
# (API params update, AI tuner proposal, or a drifted persisted value).
# Stops can NEVER be disabled: tp_pct/sl_pct always exist and always > 0.
# ---------------------------------------------------------------------------

HARD_BOUNDS: dict[str, tuple[float, float]] = {
    "tp_pct": (0.006, 0.03),
    "sl_pct": (0.004, 0.02),
    "time_stop_bars": (4, 64),
    "position_fraction": (0.02, 0.10),
    "max_positions": (1, 10),
    "interval_minutes": (1, 10),
    "cooldown_bars": (1, 20),
    "max_trades_per_day": (5, 100),
    "cost_gate_multiple": (0.0, 10.0),
}

#: Bounded params that must be integers after clamping.
_INT_PARAMS = frozenset(
    {
        "time_stop_bars",
        "max_positions",
        "interval_minutes",
        "cooldown_bars",
        "max_trades_per_day",
    }
)

#: Minimum gap kept between sl_pct and tp_pct (sl_pct must stay BELOW tp_pct).
_SL_TP_MIN_GAP = 0.001

#: Params the AI supervisor may change. ``enabled`` and ``timeframe`` are
#: deliberately excluded — only the user (API start/stop, params endpoint)
#: controls whether the script runs and which candles it reads. The v3 gate
#: knobs ``use_atr_geometry`` and ``cost_gate_multiple`` are likewise
#: user-only (contract): the AI supervisor may not loosen its own guardrails.
_TUNABLE_PARAMS = frozenset(
    {
        "tp_pct",
        "sl_pct",
        "time_stop_bars",
        "position_fraction",
        "max_positions",
        "interval_minutes",
        "cooldown_bars",
        "max_trades_per_day",
        "rsi_long_min",
        "rsi_long_max",
        "allowed_sides",
    }
)


# ---------------------------------------------------------------------------
# Late-bound data-service proxies (test patch points, same pattern the
# AutoTrader uses — patch either these names or the service originals).
# ---------------------------------------------------------------------------


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


def _utc_today() -> str:
    """Current UTC calendar date as an ISO string (daily counter key)."""
    return datetime.now(timezone.utc).date().isoformat()


def load_scalper_position_ids(conn: sqlite3.Connection) -> set[int]:
    """Read the scalper-owned open position ids from ``account_state``.

    Shared with :meth:`AutoTrader._manage_positions`, which must skip these
    ids (the scalper manages its own exits). Malformed entries are ignored.

    Args:
        conn: Open database connection.

    Returns:
        Set of position ids currently owned by the scalper (may be empty).
    """
    raw = PaperTradingEngine._state_get(conn, SCALPER_POSITION_IDS_KEY, [])
    ids: set[int] = set()
    if isinstance(raw, list):
        for item in raw:
            try:
                ids.add(int(item))
            except (TypeError, ValueError):
                logger.warning("Ignoring malformed scalper position id %r", item)
    return ids


def _clamp_params(data: dict[str, Any]) -> dict[str, Any]:
    """Clamp a FULL params dict into :data:`HARD_BOUNDS` (+ sanity rules).

    Applied on every read/write path so the bounds hold whatever the source
    of a change (user API call, AI tuner, drifted persisted state). Rules on
    top of the plain per-field bounds:

    * ``sl_pct`` must stay at least :data:`_SL_TP_MIN_GAP` BELOW ``tp_pct``
      (a stop can never sit at/above the target);
    * ``rsi_long_min``/``rsi_long_max`` are clamped into ``[0, 100]`` and
      swapped when inverted;
    * unparseable/non-finite numerics fall back to the field default.

    Args:
        data: Complete params dict (every :class:`ScalperParams` field).

    Returns:
        A new dict with every bounded value inside its hard bounds.
    """
    defaults = ScalperParams()
    out = dict(data)

    def _coerce(name: str) -> float:
        raw = out.get(name, getattr(defaults, name))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float(getattr(defaults, name))
        if not math.isfinite(value):
            value = float(getattr(defaults, name))
        return value

    for name, (low, high) in HARD_BOUNDS.items():
        value = min(max(_coerce(name), low), high)
        if name in _INT_PARAMS:
            out[name] = int(min(max(round(value), int(low)), int(high)))
        else:
            out[name] = value

    if out["sl_pct"] > out["tp_pct"] - _SL_TP_MIN_GAP:
        clamped_sl = max(HARD_BOUNDS["sl_pct"][0], out["tp_pct"] - _SL_TP_MIN_GAP)
        logger.info(
            "Clamping sl_pct %.4f below tp_pct %.4f -> %.4f (stops never disabled)",
            out["sl_pct"],
            out["tp_pct"],
            clamped_sl,
        )
        out["sl_pct"] = clamped_sl

    for name in ("rsi_long_min", "rsi_long_max"):
        out[name] = min(max(_coerce(name), 0.0), 100.0)
    if out["rsi_long_min"] > out["rsi_long_max"]:
        out["rsi_long_min"], out["rsi_long_max"] = (
            out["rsi_long_max"],
            out["rsi_long_min"],
        )
    return out


def _bounds_text() -> str:
    """Human/LLM-readable rendering of :data:`HARD_BOUNDS` for the prompt."""
    parts = [f"{name} [{low}, {high}]" for name, (low, high) in HARD_BOUNDS.items()]
    return (
        "; ".join(parts)
        + "; sl_pct must stay below tp_pct; rsi bands within [0, 100]; "
        "stops can NEVER be disabled."
    )


# ---------------------------------------------------------------------------
# Parameters model
# ---------------------------------------------------------------------------


class ScalperParams(BaseModel):
    """Persisted fast-scalper configuration (``account_state['scalper_params']``).

    Every numeric knob is clamped into :data:`HARD_BOUNDS` on read and write.
    Defaults are fee-aware: with a round trip costing about 0.3% (2 x 0.1%
    commission + 2 x 0.05% slippage), the +1.2% target / -0.8% stop pair keeps
    a meaningful edge after costs.

    Attributes:
        enabled: Whether the periodic tick job should run (persisted).
        timeframe: Candle timeframe the rules read (closed candles only).
        interval_minutes: How often the script ticks.
        tp_pct: Take-profit as a fraction of the entry price (e.g. 0.012).
        sl_pct: Stop-loss as a fraction of the entry price (below ``tp_pct``).
        time_stop_bars: Market-exit after this many bars if neither SL nor TP
            has been hit.
        position_fraction: Fraction of account equity per scalp.
        max_positions: The scalper's own sub-cap on concurrent scalps (the
            global ``settings.max_open_positions`` still applies on top).
        rsi_long_min: Lower RSI bound of the long entry band.
        rsi_long_max: Upper RSI bound of the long entry band (short entries
            use the mirrored band ``[100-max, 100-min]``).
        allowed_sides: Which entry directions are permitted.
        cooldown_bars: Per-symbol wait (in bars) after an exit (anti-churn).
        max_trades_per_day: Hard anti-runaway cap on entries per UTC day.
        disabled_symbols: Symbols benched by the AI tuner (or the user).
        use_atr_geometry: Scale TP/SL off the last closed candle's ATR
            (clamped inside :data:`HARD_BOUNDS`); when off — or when the ATR
            is unavailable — the fixed ``tp_pct``/``sl_pct`` apply. User-only
            (not AI-tunable).
        cost_gate_multiple: Entry cost gate — the candle's ATR must be at
            least this multiple of the round-trip fee+slippage cost, else the
            scalp is skipped (``0`` disables the gate). User-only (not
            AI-tunable).
        research_mode: Data-collection mode — trade EVERY signal on EVERY
            watchlist coin: skips the soft daily stop, the disabled/benched
            symbol filters, the coach side-bias filter, the regime gate and
            the cost gate, and pauses the AI tuner (config frozen). Exits
            (SL/TP/time-stop) and HARD_BOUNDS still fully apply; the
            engine's hard daily halt remains the last-resort backstop.
            User-only (not AI-tunable). PAPER TRADING ONLY.
    """

    enabled: bool = False
    timeframe: str = "15m"
    interval_minutes: int = 2
    tp_pct: float = 0.012
    sl_pct: float = 0.008
    time_stop_bars: int = 16
    position_fraction: float = 0.06
    max_positions: int = 8
    rsi_long_min: float = 50.0
    rsi_long_max: float = 72.0
    allowed_sides: Literal["long", "short", "both"] = "both"
    cooldown_bars: int = 3
    max_trades_per_day: int = 40
    disabled_symbols: list[str] = Field(default_factory=list)
    use_atr_geometry: bool = True
    cost_gate_multiple: float = 3.0
    research_mode: bool = False

    @field_validator("timeframe")
    @classmethod
    def _valid_timeframe(cls, value: str) -> str:
        """Restrict the timeframe to the platform's canonical set."""
        tf = str(value).strip()
        if tf not in _TIMEFRAME_DELTAS:
            raise ValueError(
                f"invalid timeframe {tf!r} — expected one of "
                f"{sorted(_TIMEFRAME_DELTAS)}"
            )
        return tf

    @field_validator("disabled_symbols")
    @classmethod
    def _clean_disabled(cls, value: list[str]) -> list[str]:
        """Strip/upper-case symbols, drop blanks and duplicates (order kept)."""
        cleaned: list[str] = []
        for item in value:
            symbol = str(item).strip().upper()
            if symbol and symbol not in cleaned:
                cleaned.append(symbol)
        return cleaned


# ---------------------------------------------------------------------------
# Scalper
# ---------------------------------------------------------------------------


class Scalper:
    """Fast mechanical paper-scalper driving the shared :class:`PaperTradingEngine`.

    Designed as a singleton living next to the engine (and the AutoTrader)
    inside the API process. :meth:`run_tick` is thread-safe — a non-blocking
    ``threading.Lock`` skips a tick when one is already in flight — and NEVER
    raises: every per-symbol failure is caught, logged and written to
    ``bot_activity``. PAPER TRADING ONLY.
    """

    def __init__(self, engine: PaperTradingEngine) -> None:
        """Bind the scalper to the shared engine; make sure the schema exists.

        Args:
            engine: The shared paper-trading engine (simulation only).
        """
        init_db()  # idempotent — guarantees bot_activity/account_state exist
        self.engine = engine
        self.risk = engine.risk
        self._tick_lock = threading.Lock()
        self._params_lock = threading.RLock()
        self._state_lock = threading.RLock()
        logger.info("Scalper ready (PAPER TRADING ONLY — no real orders, ever)")

    # ------------------------------------------------------------------ #
    # Parameters                                                          #
    # ------------------------------------------------------------------ #

    def get_params(self) -> ScalperParams:
        """Load the persisted params, clamped to :data:`HARD_BOUNDS`.

        Returns:
            The current :class:`ScalperParams` (defaults when absent/corrupt).
        """
        with self._connect() as conn:
            raw = self.engine._state_get(conn, _PARAMS_KEY, None)
        if raw is None:
            return ScalperParams()
        if not isinstance(raw, dict):
            logger.warning("Persisted scalper_params is not a dict — using defaults")
            return ScalperParams()
        known = set(ScalperParams.model_fields)
        merged = ScalperParams().model_dump()
        merged.update({k: v for k, v in raw.items() if k in known})
        try:
            return ScalperParams.model_validate(_clamp_params(merged))
        except ValidationError:
            logger.warning("Persisted scalper_params is invalid — using defaults")
            return ScalperParams()

    def set_params(self, updates: dict[str, Any]) -> ScalperParams:
        """Apply a partial params update, clamp to :data:`HARD_BOUNDS`, persist.

        Unknown keys are ignored with a warning; ``None`` values are treated
        as "not provided". Non-numeric bounded values fall back to their
        defaults during clamping; invalid enum/timeframe values raise
        pydantic's ``ValidationError`` (a ``ValueError`` subclass → HTTP 400).

        Args:
            updates: Subset of :class:`ScalperParams` fields to change.

        Returns:
            The updated, clamped, persisted :class:`ScalperParams`.
        """
        with self._params_lock:
            merged = self.get_params().model_dump()
            known = set(ScalperParams.model_fields)
            unknown = sorted(set(updates) - known)
            if unknown:
                logger.warning("Ignoring unknown scalper param keys: %s", unknown)
            merged.update(
                {k: v for k, v in updates.items() if k in known and v is not None}
            )
            params = ScalperParams.model_validate(_clamp_params(merged))
            with self._connect() as conn:
                self.engine._state_set(conn, _PARAMS_KEY, params.model_dump())
            logger.info(
                "Scalper params updated (enabled=%s, tf=%s, every %d min, "
                "tp=%.2f%%, sl=%.2f%%, sides=%s)",
                params.enabled,
                params.timeframe,
                params.interval_minutes,
                params.tp_pct * 100,
                params.sl_pct * 100,
                params.allowed_sides,
            )
            return params

    def start(self) -> ScalperParams:
        """Enable the scalper (persisted). Scheduling is the API's job.

        Returns:
            The updated :class:`ScalperParams` with ``enabled=True``.
        """
        params = self.set_params({"enabled": True})
        logger.info(
            "Scalper STARTED (paper only) — %s candles every %d min",
            params.timeframe,
            params.interval_minutes,
        )
        return params

    def stop(self) -> dict[str, Any]:
        """Disable the scalper and market-close ALL scalper-owned positions.

        Contract-documented behaviour: a stopped script no longer runs its
        time-stop, so instead of leaving half-managed scalps behind, every
        owned open position is closed at market (``scalp_exit`` with reason
        ``script_stopped``) and its ownership record pruned.

        The close pass serializes behind any in-flight tick (blocking acquire
        of the tick lock): a tick that started before the stop request could
        otherwise submit one last entry AFTER the positions were closed,
        leaving an orphan scalp that nothing would ever manage. The disable
        flag is persisted FIRST so the in-flight tick's own pre-submit
        re-check (see :meth:`_enter_one`) also aborts late entries.

        Returns:
            dict: ``{"params": ScalperParams dict, "closed_positions": n}``.
        """
        params = self.set_params({"enabled": False})
        closed_count = 0
        with self._tick_lock:
            for pid in sorted(self._owned_ids()):
                try:
                    position = self.engine.close_position(pid)
                    self._log_scalp_exit(position, "script_stopped", params)
                    closed_count += 1
                except ValueError as exc:
                    # Already closed / no price — prune either way below.
                    logger.warning(
                        "Could not market-close scalper position #%d on stop: %s",
                        pid,
                        exc,
                    )
                    self._log(
                        "error",
                        detail={
                            "where": "scalp_stop",
                            "reason": f"position #{pid}: {exc}",
                            "explanation": (
                                f"Could not close scalp position #{pid} while "
                                f"stopping the script: {exc}"
                            ),
                        },
                    )
                self._prune_owned(pid)
        logger.info(
            "Scalper stopped (paper only) — %d open scalp position(s) closed",
            closed_count,
        )
        return {"params": params.model_dump(), "closed_positions": closed_count}

    def get_last_tune(self) -> dict[str, Any] | None:
        """Return the persisted record of the last AI tune, or None.

        Returns:
            dict with ``ts``, ``before``, ``after``, ``applied``,
            ``reasoning`` and ``model`` — or ``None`` when never tuned.
        """
        with self._connect() as conn:
            record = self.engine._state_get(conn, _LAST_TUNE_KEY, None)
        return record if isinstance(record, dict) else None

    # ------------------------------------------------------------------ #
    # The fast tick                                                        #
    # ------------------------------------------------------------------ #

    def run_tick(self) -> dict[str, Any]:
        """Run one fast manage-then-enter pass. NEVER raises.

        Thread-safe: when a tick is already running the call returns
        immediately with ``{"status": "busy", ...}`` instead of blocking.

        Returns:
            Tick summary dict (``status``, ``entered``, ``exited``,
            ``skipped``, ``errors``, ``equity``, ``open``, ``trades_today``).
        """
        if not self._tick_lock.acquire(blocking=False):
            logger.info("Scalper tick skipped — a tick is already running")
            return {"status": "busy", "reason": "a scalper tick is already running"}
        try:
            try:
                return self._run_tick_locked()
            except Exception as exc:  # noqa: BLE001 — run_tick must never raise
                logger.exception("Scalper tick crashed — recorded and swallowed")
                self._log(
                    "error",
                    detail={
                        "where": "scalp_tick",
                        "reason": str(exc),
                        "explanation": f"The scalper tick failed unexpectedly: {exc}",
                    },
                )
                return {"status": "error", "error": str(exc)}
        finally:
            self._tick_lock.release()

    def _run_tick_locked(self) -> dict[str, Any]:
        """The actual tick body (contract steps 1-3); lock already held."""
        params = self.get_params()
        if not params.enabled:
            return {"status": "disabled"}

        counters = {"entered": 0, "exited": 0, "skipped": 0, "errors": 0}
        bot_config = self._bot_config()
        disabled = set(params.disabled_symbols)
        # Coach playbook — read ONCE per tick (contract: one query, not one
        # per symbol; {} when the coach/intel layer is absent). Coach-benched
        # coins are filtered like the AI tuner's disabled_symbols: silently at
        # DEBUG — at a 1-2 minute cadence a persistent bench would otherwise
        # drown the feed (the AutoTrader logs the explicit ``coach_bench``
        # skip rows at its 15-minute cadence).
        playbook = self._load_playbook()
        benched = {s for s, entry in playbook.items() if entry.get("bench")}
        if benched:
            logger.debug("Scalper: coach-benched symbols filtered: %s", sorted(benched))
        if params.research_mode:
            # Research mode: every watchlist coin stays in play — the AI
            # tuner's bench list and the coach's bench are both ignored.
            watchlist = list(bot_config.watchlist)
        else:
            watchlist = [
                s for s in bot_config.watchlist if s not in disabled and s not in benched
            ]
        source = bot_config.source

        # --- 2) manage own positions (engine SL/TP tick + time-stop) --------
        # Runs even while trading is halted: exits are risk-reducing and the
        # RiskManager always allows them.
        self._manage_own_positions(params, counters)

        # --- 3) entries (halt gate: a halt blocks NEW scalps only) ----------
        portfolio = self.engine.get_portfolio()
        status = "ok"
        halt_reason = ""
        if portfolio.get("trading_halted"):
            status = "halted"
            halt_reason = str(portfolio.get("halt_reason") or "trading_halted")
            counters["skipped"] += 1
            self._log(
                "scalp_skip",
                detail={
                    "reason": halt_reason,
                    "explanation": (
                        "Skipped all new scalp entries: trading is halted — "
                        f"{halt_reason}. Open scalp positions were still managed."
                    ),
                },
            )
        elif not params.research_mode and RiskManager.soft_daily_stop_active(
            float(portfolio.get("equity") or 0.0),
            float(portfolio.get("realized_pnl_today") or 0.0),
            settings.daily_loss_limit,
        ):
            # Soft daily stop (v3): at 80% of the daily loss limit the scalper
            # stands down for the tick — no NEW scalps, one skip row, exits
            # above were still managed. The slower bot keeps trading at half
            # size (its own de-risk multiplier handles that).
            status = "soft_stop"
            counters["skipped"] += 1
            self._log(
                "scalp_skip",
                detail={
                    "reason": "soft_daily_stop",
                    "realized_pnl_today": _json_num(
                        portfolio.get("realized_pnl_today")
                    ),
                    "daily_limit_pct": _json_num(settings.daily_loss_limit),
                    "explanation": (
                        "Skipped all new scalps: today's realized loss is at "
                        "80% of the daily loss limit — the scalper stands "
                        "down while the slower bot may still trade at half "
                        "size."
                    ),
                },
            )
        else:
            # Per-tick regime cache (v3): BTCUSDT and each symbol computed at
            # most once per tick, so regime skips can never spam the feed.
            regime_cache: dict[str, regime.RegimeInfo] = {}
            self._enter_positions(
                params, watchlist, source, counters, playbook, regime_cache
            )

        portfolio = self.engine.get_portfolio()
        summary: dict[str, Any] = {
            "status": status,
            **counters,
            "equity": float(portfolio["equity"]),
            "open": len(self._owned_ids()),
            "trades_today": self._trades_today(),
        }
        if halt_reason:
            summary["halt_reason"] = halt_reason
        logger.info(
            "Scalper tick finished (%s): entered=%d exited=%d skipped=%d "
            "errors=%d open=%d trades_today=%d",
            status,
            counters["entered"],
            counters["exited"],
            counters["skipped"],
            counters["errors"],
            summary["open"],
            summary["trades_today"],
        )
        return summary

    # ------------------------------------------------------------------ #
    # Manage own positions                                                 #
    # ------------------------------------------------------------------ #

    def _manage_own_positions(
        self, params: ScalperParams, counters: dict[str, int]
    ) -> None:
        """Contract step 2: reconcile, tick SL/TP and enforce the time-stop.

        Three passes over the scalper-owned position ids:

        1. RECONCILE — ids whose positions were closed outside the scalper's
           own tick (manual close, an engine SL/TP fired during another
           trader's tick) get a ``scalp_exit`` row (reason inferred from the
           exit price vs the stored SL/TP levels), a cooldown and are pruned.
        2. TICK — ``engine.process_tick(refresh=True)`` once per distinct
           owned (symbol, source, timeframe): the engine evaluates the
           stop-loss/take-profit attached at entry and marks to market.
        3. TIME-STOP — any owned position open for ``time_stop_bars`` bars or
           more is market-closed (reason ``time_stop``).

        Per-position failures are isolated, counted and logged.
        """
        owned = self._owned_ids()
        if not owned:
            return

        # --- 1) reconcile externally closed positions ------------------------
        all_positions = {int(p["id"]): p for p in self.engine.get_positions("all")}
        for pid in sorted(owned):
            position = all_positions.get(pid)
            if position is None:
                logger.warning(
                    "Scalper-owned position #%d no longer exists — pruning", pid
                )
                self._prune_owned(pid)
                continue
            if position.get("status") != "open":
                reason = self._infer_exit_reason(position)
                self._log_scalp_exit(position, reason, params)
                counters["exited"] += 1
                self._prune_owned(pid)
                self._set_cooldown(str(position["symbol"]), params)

        # --- 2) tick every owned symbol once (engine-level SL/TP) ------------
        owned = self._owned_ids()
        seen: set[tuple[str, str, str]] = set()
        for pid in sorted(owned):
            position = all_positions.get(pid)
            if position is None:
                continue
            key = (
                str(position["symbol"]),
                str(position["source"]),
                str(position["timeframe"]),
            )
            if key in seen:
                continue
            seen.add(key)
            try:
                tick = self.engine.process_tick(
                    key[0], source=key[1], timeframe=key[2], refresh=True
                )
            except Exception as exc:  # noqa: BLE001 — per-symbol isolation
                counters["errors"] += 1
                logger.exception("Scalper manage tick failed for %s", key[0])
                self._log(
                    "error",
                    symbol=key[0],
                    detail={
                        "where": "scalp_manage",
                        "reason": str(exc),
                        "explanation": (
                            f"Could not refresh/tick {key[0]} while managing "
                            f"scalp positions: {exc}"
                        ),
                    },
                )
                continue
            still_owned = self._owned_ids()
            for closed_pos in tick.get("closed", []):
                cid = int(closed_pos.get("id") or 0)
                if cid not in still_owned:
                    continue
                reason = str(closed_pos.get("exit_reason") or "stop_loss")
                self._log_scalp_exit(closed_pos, reason, params)
                counters["exited"] += 1
                self._prune_owned(cid)
                self._set_cooldown(str(closed_pos.get("symbol") or key[0]), params)

        # --- 3) time-stop on whatever is still open --------------------------
        owned = self._owned_ids()
        if not owned:
            return
        open_now = {int(p["id"]): p for p in self.engine.get_positions("open")}
        now = datetime.now(timezone.utc)
        for pid in sorted(owned):
            position = open_now.get(pid)
            if position is None:
                continue  # closed during the tick pass; next tick reconciles
            try:
                opened_at = datetime.fromisoformat(str(position.get("opened_at")))
            except (TypeError, ValueError):
                logger.warning(
                    "Position #%d has an unreadable opened_at — time-stop skipped",
                    pid,
                )
                continue
            bar = _timeframe_delta(
                str(position.get("timeframe") or params.timeframe)
            )
            if now - opened_at < params.time_stop_bars * bar:
                continue
            try:
                closed_pos = self.engine.close_position(pid)
            except ValueError as exc:
                counters["errors"] += 1
                self._log(
                    "error",
                    symbol=str(position.get("symbol") or ""),
                    detail={
                        "where": "scalp_time_stop",
                        "reason": str(exc),
                        "explanation": (
                            f"Time-stop close of position #{pid} failed: {exc}"
                        ),
                    },
                )
                continue
            self._log_scalp_exit(closed_pos, "time_stop", params)
            counters["exited"] += 1
            self._prune_owned(pid)
            self._set_cooldown(str(closed_pos.get("symbol") or ""), params)

    @staticmethod
    def _infer_exit_reason(position: dict[str, Any]) -> str:
        """Best-effort exit reason for a position closed outside our tick.

        Positions closed by the engine's SL/TP during ANOTHER trader's tick
        carry no reason in the DB row — infer it by comparing the exit price
        to the stored protective levels (0.3% tolerance covers slippage).

        Args:
            position: Serialized (closed) position dict.

        Returns:
            ``"take_profit"``, ``"stop_loss"`` or ``"closed_externally"``.
        """
        try:
            exit_price = float(position.get("exit_price") or 0.0)
        except (TypeError, ValueError):
            return "closed_externally"
        if exit_price <= 0:
            return "closed_externally"
        side = str(position.get("side") or "long")
        tol = 0.003
        tp = position.get("take_profit")
        if tp is not None:
            tp = float(tp)
            if (side == "long" and exit_price >= tp * (1 - tol)) or (
                side == "short" and exit_price <= tp * (1 + tol)
            ):
                return "take_profit"
        sl = position.get("stop_loss")
        if sl is not None:
            sl = float(sl)
            if (side == "long" and exit_price <= sl * (1 + tol)) or (
                side == "short" and exit_price >= sl * (1 - tol)
            ):
                return "stop_loss"
        return "closed_externally"

    # ------------------------------------------------------------------ #
    # Entries                                                              #
    # ------------------------------------------------------------------ #

    def _enter_positions(
        self,
        params: ScalperParams,
        watchlist: list[str],
        source: str,
        counters: dict[str, int],
        playbook: dict[str, dict[str, Any]] | None = None,
        regime_cache: dict[str, regime.RegimeInfo] | None = None,
    ) -> None:
        """Contract step 3: scan the watchlist and enter eligible scalps.

        Per-symbol gates, cheapest first: daily trade cap / scalper sub-cap /
        global position cap (each logged ONCE and the loop stops), an
        existing open position in the symbol (silent — resting state), the
        anti-churn cooldown, then data guards and the entry rule (followed by
        the v3 regime and cost gates inside :meth:`_scan_and_enter_symbol`).

        The coach ``playbook`` (already loaded once per tick) filters the
        entry side (``side_bias``) and scales the size (``size_multiplier``)
        further down the pipeline; benched coins were removed from
        ``watchlist`` before this method ran. ``regime_cache`` is the
        per-tick memo for :func:`regime.get_regime` (one call per symbol per
        tick, BTCUSDT included).
        """
        playbook = playbook or {}
        open_positions = self.engine.get_positions("open")
        open_symbols = {str(p["symbol"]) for p in open_positions}
        n_open_global = len(open_positions)
        n_owned = len(self._owned_ids())
        trades_today = self._trades_today()

        for symbol in watchlist:
            if trades_today >= params.max_trades_per_day:
                counters["skipped"] += 1
                self._log(
                    "scalp_skip",
                    detail={
                        "reason": (
                            f"max_trades_per_day: {trades_today} >= "
                            f"{params.max_trades_per_day}"
                        ),
                        "explanation": (
                            f"Stopped scanning: the scalper already made "
                            f"{trades_today} trades today, at its daily cap of "
                            f"{params.max_trades_per_day} (anti-runaway guard)."
                        ),
                    },
                )
                break
            if n_owned >= params.max_positions:
                counters["skipped"] += 1
                self._log(
                    "scalp_skip",
                    detail={
                        "reason": (
                            f"max_positions: {n_owned} >= {params.max_positions}"
                        ),
                        "explanation": (
                            f"Stopped scanning: the scalper already holds "
                            f"{n_owned} positions, at its own cap of "
                            f"{params.max_positions}."
                        ),
                    },
                )
                break
            if n_open_global >= settings.max_open_positions:
                counters["skipped"] += 1
                self._log(
                    "scalp_skip",
                    detail={
                        "reason": (
                            f"global max_open_positions: {n_open_global} >= "
                            f"{settings.max_open_positions}"
                        ),
                        "explanation": (
                            "Stopped scanning: the whole account already holds "
                            f"{n_open_global} open positions, at the global cap "
                            f"of {settings.max_open_positions}."
                        ),
                    },
                )
                break
            if symbol in open_symbols:
                logger.debug("Scalper: %s already has an open position", symbol)
                continue
            if self._in_cooldown(symbol):
                counters["skipped"] += 1
                self._log(
                    "scalp_skip",
                    symbol=symbol,
                    detail={
                        "reason": "cooldown",
                        "explanation": (
                            f"Skipped {symbol}: waiting out the "
                            f"{params.cooldown_bars}-bar cooldown after its last "
                            "scalp exit (anti-churn)."
                        ),
                    },
                )
                continue

            try:
                entered = self._scan_and_enter_symbol(
                    params, symbol, source, counters, playbook, regime_cache
                )
            except Exception as exc:  # noqa: BLE001 — per-symbol isolation
                counters["errors"] += 1
                logger.exception("Scalper entry failed for %s", symbol)
                self._log(
                    "error",
                    symbol=symbol,
                    detail={
                        "where": "scalp_enter",
                        "reason": str(exc),
                        "explanation": (
                            f"Could not evaluate/enter {symbol}: {exc}"
                        ),
                    },
                )
                continue
            if entered:
                trades_today = self._trades_today()
                n_owned += 1
                n_open_global += 1
                open_symbols.add(symbol)

    def _scan_and_enter_symbol(
        self,
        params: ScalperParams,
        symbol: str,
        source: str,
        counters: dict[str, int],
        playbook: dict[str, dict[str, Any]] | None = None,
        regime_cache: dict[str, regime.RegimeInfo] | None = None,
    ) -> bool:
        """Refresh one symbol, evaluate the entry rule, place the order.

        Signals act on CLOSED candles only (the still-forming last candle is
        dropped) and stale data is never traded on — both guards reused from
        the AutoTrader. The coach playbook's ``side_bias`` filters the fired
        direction (a silent non-event at DEBUG, like ``allowed_sides``).

        After the signal fires and the side-bias filter passes, the v3 hard
        gates run (both logged as ``scalp_skip`` rows when they block):

        1. **Regime gate** — shorts need the symbol AND BTCUSDT in a 4h
           downtrend; longs are blocked only in that same double-downtrend
           (shared :func:`regime.regime_allows` predicate).
        2. **Cost gate** — the last closed candle's ATR must be at least
           ``cost_gate_multiple`` × the round-trip fee+slippage cost.

        Returns:
            True when a scalp was entered.
        """
        playbook = playbook or {}
        try:
            update_symbol(
                source, symbol, params.timeframe, lookback_days=_SCAN_LOOKBACK_DAYS
            )
        except Exception as exc:  # noqa: BLE001 — stale cache may still work
            counters["errors"] += 1
            logger.warning(
                "Scalper data update failed for %s:%s:%s — %s",
                source,
                symbol,
                params.timeframe,
                exc,
            )
            self._log(
                "error",
                symbol=symbol,
                detail={
                    "where": "scalp_scan_update",
                    "reason": str(exc),
                    "explanation": (
                        f"Data refresh for {symbol} failed ({exc}) — falling "
                        "back to the cached candles (the staleness guard "
                        "protects against trading a frozen price)."
                    ),
                },
            )
        df = get_ohlcv(
            source, symbol, params.timeframe, limit=500, with_features=True
        )
        df = _drop_unclosed_candle(df, params.timeframe)
        if len(df) < _MIN_SCAN_ROWS:
            counters["skipped"] += 1
            self._log(
                "scalp_skip",
                symbol=symbol,
                detail={
                    "reason": f"insufficient_data: {len(df)} rows < {_MIN_SCAN_ROWS}",
                    "explanation": (
                        f"Skipped {symbol}: only {len(df)} closed "
                        f"{params.timeframe} candles cached — at least "
                        f"{_MIN_SCAN_ROWS} are needed for the entry rule."
                    ),
                },
            )
            return False
        if _is_stale(df, params.timeframe):
            counters["skipped"] += 1
            last_iso = df.index[-1].isoformat()
            self._log(
                "scalp_skip",
                symbol=symbol,
                detail={
                    "reason": f"stale_data: last closed candle {last_iso}",
                    "explanation": (
                        f"Skipped {symbol}: the newest closed candle is from "
                        f"{last_iso}, too old for the {params.timeframe} "
                        "timeframe — trading on stale prices is not allowed."
                    ),
                },
            )
            return False

        last = df.iloc[-1]
        raw = {
            name: last.get(name)
            for name in ("ema_12", "ema_26", "rsi_14", "vwap", "close")
        }
        if any(value is None or pd.isna(value) for value in raw.values()):
            logger.debug("Scalper: %s indicators still warming up", symbol)
            return False
        values = {name: float(value) for name, value in raw.items()}

        direction = self._entry_signal(
            params,
            values["ema_12"],
            values["ema_26"],
            values["rsi_14"],
            values["vwap"],
            values["close"],
        )
        if direction is None:
            logger.debug("Scalper: no entry signal on %s", symbol)
            return False
        bias = str((playbook.get(symbol) or {}).get("side_bias") or "both")
        if not params.research_mode and (
            (direction == "long" and bias == "short_only")
            or (direction == "short" and bias == "long_only")
        ):
            logger.debug(
                "Scalper: %s %s entry filtered by coach side_bias=%s",
                symbol,
                direction,
                bias,
            )
            return False

        if params.research_mode:
            # Research mode: regime gate, cost gate and side-bias are all
            # bypassed — every fired signal becomes a trade so the results
            # can be studied later. Exits still manage every position.
            return self._enter_one(
                params,
                symbol,
                source,
                direction,
                values,
                counters,
                playbook,
                atr_14=self._last_atr(last),
                frame=df,
            )

        # --- v3 gate 1: market regime (shared predicate with the bot) --------
        symbol_info = self._regime_for(regime_cache, source, symbol)
        btc_info = self._regime_for(regime_cache, source, _BTC_SYMBOL)
        if not regime.regime_allows(direction, symbol_info.regime, btc_info.regime):
            counters["skipped"] += 1
            if direction == "short":
                explanation = (
                    f"Skipped a short scalp in {symbol}: shorts are only "
                    "allowed when both the coin and Bitcoin are in a 4h "
                    f"downtrend — {symbol} is {symbol_info.regime} and BTC "
                    f"is {btc_info.regime}."
                )
            else:
                explanation = (
                    f"Skipped a long scalp in {symbol}: longs are paused "
                    "while both the coin and Bitcoin are in a 4h downtrend "
                    f"— {symbol} is {symbol_info.regime} and BTC is "
                    f"{btc_info.regime}."
                )
            self._log(
                "scalp_skip",
                symbol=symbol,
                detail={
                    "reason": "regime_block",
                    "side": direction,
                    "symbol_regime": symbol_info.regime,
                    "btc_regime": btc_info.regime,
                    "close": _json_num(symbol_info.close),
                    "ema50": _json_num(symbol_info.ema50),
                    "ema200": _json_num(symbol_info.ema200),
                    "explanation": explanation,
                },
            )
            return False

        # --- v3 gate 2: trade cost (ATR must be worth the round trip) --------
        atr_14 = self._last_atr(last)
        gate = regime.cost_gate(
            atr_14,
            values["close"],
            settings.commission_rate,
            settings.slippage_rate,
            params.cost_gate_multiple,
        )
        if not gate.get("passes"):
            counters["skipped"] += 1
            expected = gate.get("expected_move_pct")
            needed = float(gate.get("needed_pct") or 0.0)
            move_txt = (
                f"{float(expected):.2%} of price"
                if isinstance(expected, (int, float))
                else "unknown (no ATR yet)"
            )
            self._log(
                "scalp_skip",
                symbol=symbol,
                detail={
                    "reason": "cost_gate",
                    "expected_move_pct": _json_num(expected),
                    "needed_pct": _json_num(needed),
                    "explanation": (
                        f"Skipped {symbol}: the {params.timeframe} range is "
                        f"{move_txt} but fees + slippage need at least "
                        f"{needed:.2%} of expected movement to be worth "
                        "trading."
                    ),
                },
            )
            return False

        return self._enter_one(
            params,
            symbol,
            source,
            direction,
            values,
            counters,
            playbook,
            atr_14=atr_14,
            frame=df,
        )

    @staticmethod
    def _entry_signal(
        params: ScalperParams,
        ema_12: float,
        ema_26: float,
        rsi_14: float,
        vwap_value: float,
        close: float,
    ) -> str | None:
        """Evaluate the mechanical entry rule on one closed candle.

        LONG when ``ema_12 > ema_26`` AND ``rsi_14`` inside
        ``[rsi_long_min, rsi_long_max]`` AND ``close > vwap``. SHORT is the
        exact mirror (EMA trend down, RSI inside ``[100-max, 100-min]``,
        close below VWAP). ``allowed_sides`` gates each direction.

        Args:
            params: Current scalper parameters.
            ema_12 / ema_26: Latest EMA values.
            rsi_14: Latest RSI value.
            vwap_value: Latest per-UTC-day VWAP.
            close: Latest closed-candle close.

        Returns:
            ``"long"``, ``"short"`` or ``None`` when no rule fires.
        """
        long_ok = (
            ema_12 > ema_26
            and params.rsi_long_min <= rsi_14 <= params.rsi_long_max
            and close > vwap_value
        )
        short_ok = (
            ema_12 < ema_26
            and (100.0 - params.rsi_long_max)
            <= rsi_14
            <= (100.0 - params.rsi_long_min)
            and close < vwap_value
        )
        if long_ok and params.allowed_sides in ("long", "both"):
            return "long"
        if short_ok and params.allowed_sides in ("short", "both"):
            return "short"
        return None

    @staticmethod
    def _regime_for(
        cache: dict[str, regime.RegimeInfo] | None, source: str, symbol: str
    ) -> regime.RegimeInfo:
        """Per-tick memoized :func:`regime.get_regime` lookup.

        The cache (one dict per tick, created in :meth:`_run_tick_locked`)
        guarantees at most ONE regime computation per symbol per tick —
        BTCUSDT included — so regime skips log at most once per symbol per
        tick and the 4h resample never runs twice for the same coin.

        Args:
            cache: The tick's ``{symbol: RegimeInfo}`` memo (None → uncached).
            source: Exchange source (the bot config's source).
            symbol: Symbol to classify.

        Returns:
            The (possibly cached) :class:`regime.RegimeInfo`.
        """
        if cache is None:
            return regime.get_regime(source, symbol)
        if symbol not in cache:
            cache[symbol] = regime.get_regime(source, symbol)
        return cache[symbol]

    @staticmethod
    def _last_atr(last: pd.Series) -> float | None:
        """The last closed candle's ``atr_14``, or None when missing/invalid."""
        try:
            value = float(last.get("atr_14"))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _enter_one(
        self,
        params: ScalperParams,
        symbol: str,
        source: str,
        direction: str,
        values: dict[str, float],
        counters: dict[str, int],
        playbook: dict[str, dict[str, Any]] | None = None,
        atr_14: float | None = None,
        frame: pd.DataFrame | None = None,
    ) -> bool:
        """Size and submit one scalp entry with SL/TP attached.

        Sizing: ``position_fraction`` of equity, capped at 95% of free cash,
        then scaled by the coach playbook's ``size_multiplier`` (contract:
        AFTER the existing caps, with the ``position_fraction``, cash and
        dust guards re-applied — a multiplier above 1.0 can never breach the
        per-scalp fraction cap), then by the v3 de-risk multiplier as the
        FINAL sizing step before the dust guard. The latest cached close is
        re-read immediately before sizing (the engine fills market orders at
        that price) so the stops sit at the intended percentages of the
        actual fill basis.

        Stop geometry (v3): when ``use_atr_geometry`` is on and ``atr_14``
        is valid, the stop is ``ATR_GEOMETRY_SL_MULT`` ATRs from the price
        and the target ``ATR_GEOMETRY_TP_MULT`` × that stop distance — both
        clamped inside :data:`HARD_BOUNDS`, so nothing can escape the bounds.
        Otherwise the fixed ``params.sl_pct``/``params.tp_pct`` apply.

        Args:
            params: Current scalper parameters.
            symbol: Symbol to enter.
            source: Exchange source.
            direction: ``"long"`` or ``"short"``.
            values: Last-closed-bar indicator readout from the scan.
            counters: The tick's mutable counters.
            playbook: Coach playbook map (already loaded once per tick).
            atr_14: Last closed candle's ATR (None → fixed-percent stops).
            frame: The scanned 15m frame (shadow-flag diagnostics only).

        Returns:
            True when the order filled and the position is now owned.
        """
        playbook = playbook or {}
        portfolio = self.engine.get_portfolio()
        equity = float(portfolio["equity"])
        cash = float(portfolio["cash"])

        price = values["close"]
        try:
            latest = load_ohlcv(source, symbol, params.timeframe, limit=1)
            if not latest.empty:
                price = float(latest["close"].iloc[-1])
        except Exception as exc:  # noqa: BLE001 — fall back to the scan close
            logger.warning(
                "Scalper could not re-read the latest close for %s — sizing off "
                "the scan-time close: %s",
                symbol,
                exc,
            )
        if price <= 0 or not math.isfinite(price):
            counters["skipped"] += 1
            self._log(
                "scalp_skip",
                symbol=symbol,
                detail={
                    "reason": f"invalid_price: {price!r}",
                    "explanation": f"Skipped {symbol}: latest cached price is invalid.",
                },
            )
            return False

        # ATR geometry (v3): stops scale with realized volatility, ALWAYS
        # clamped inside HARD_BOUNDS (nothing may escape the bounds). The
        # clamp order guarantees sl_pct_eff < tp_pct_eff. Disabled or invalid
        # ATR falls back to the fixed (tuner-owned) percentages.
        tp_pct_eff = params.tp_pct
        sl_pct_eff = params.sl_pct
        atr_geometry = False
        if (
            params.use_atr_geometry
            and atr_14 is not None
            and math.isfinite(atr_14)
            and atr_14 > 0
        ):
            sl_low, sl_high = HARD_BOUNDS["sl_pct"]
            tp_low, tp_high = HARD_BOUNDS["tp_pct"]
            sl_pct_eff = min(
                max(ATR_GEOMETRY_SL_MULT * atr_14 / price, sl_low), sl_high
            )
            tp_pct_eff = min(
                max(ATR_GEOMETRY_TP_MULT * sl_pct_eff, tp_low), tp_high
            )
            atr_geometry = True

        qty = min(
            (params.position_fraction * equity) / price,
            (0.95 * cash) / price,
        )
        # Coach playbook size multiplier — applied AFTER the existing caps
        # (contract), then the per-scalp fraction cap and the cash guard are
        # re-applied so a multiplier above 1.0 can never exceed
        # position_fraction of equity (a hard-bounded safety knob) or spend
        # beyond free cash; the dust guard below runs on the final quantity.
        try:
            size_multiplier = float(
                (playbook.get(symbol) or {}).get("size_multiplier", 1.0)
            )
        except (TypeError, ValueError):
            size_multiplier = 1.0
        if math.isfinite(size_multiplier) and size_multiplier != 1.0:
            qty = min(
                qty * size_multiplier,
                (params.position_fraction * equity) / price,
                (0.95 * cash) / price,
            )
        # De-risk ratchet (v3): the FINAL sizing step before the dust guard —
        # the deeper the account's drawdown (and during a soft daily stop),
        # the smaller every new scalp gets. Always in (0, 1].
        derisk_multiplier = RiskManager.derisk_multiplier(
            equity,
            float(portfolio.get("peak_equity") or 0.0),
            float(portfolio.get("realized_pnl_today") or 0.0),
            settings.daily_loss_limit,
        )
        qty *= derisk_multiplier
        notional = qty * price
        if qty <= 0 or notional < _MIN_NOTIONAL_USD:
            counters["skipped"] += 1
            self._log(
                "scalp_skip",
                symbol=symbol,
                detail={
                    "reason": (
                        f"dust: notional ${max(notional, 0.0):.2f} < "
                        f"${_MIN_NOTIONAL_USD:.2f}"
                    ),
                    "explanation": (
                        f"Skipped {symbol}: {params.position_fraction:.0%} of "
                        f"equity works out to ${max(notional, 0.0):.2f}, below "
                        f"the ${_MIN_NOTIONAL_USD:.2f} minimum."
                    ),
                },
            )
            return False

        if direction == "long":
            side = "buy"
            stop_loss = price * (1.0 - sl_pct_eff)
            take_profit = price * (1.0 + tp_pct_eff)
        else:
            side = "sell"
            stop_loss = price * (1.0 + sl_pct_eff)
            take_profit = price * (1.0 - tp_pct_eff)

        # Re-read the enabled flag right before submitting: stop() may have
        # been requested while this tick was scanning the watchlist, and an
        # entry that lands after stop() closed the owned positions would sit
        # unmanaged forever (no time-stop, no SL/TP ticks).
        if not self.get_params().enabled:
            counters["skipped"] += 1
            self._log(
                "scalp_skip",
                symbol=symbol,
                detail={
                    "reason": "disabled",
                    "explanation": (
                        f"Skipped {symbol}: the scalper was disabled while "
                        "this tick was running — no new scalps after stop."
                    ),
                },
            )
            return False

        # Re-check for an open position right before submitting: the scan
        # loop's open-symbols snapshot goes stale over a multi-second pass and
        # the AutoTrader may have entered this symbol meanwhile — a second
        # position would over-concentrate the account (or open an
        # opposite-side hedge on another timeframe that only burns fees).
        open_now = self.engine.get_positions("open")
        if any(str(p["symbol"]) == symbol for p in open_now):
            counters["skipped"] += 1
            self._log(
                "scalp_skip",
                symbol=symbol,
                detail={
                    "reason": "position_exists",
                    "explanation": (
                        f"Skipped {symbol}: another trader opened a position "
                        "in it while this tick was scanning."
                    ),
                },
            )
            return False

        # Full-context pre-submit risk check (v3): with the exact price and
        # stop this entry would use, the RiskManager can enforce the
        # portfolio heat cap and the same-direction cap precisely (the
        # engine's own in-submit check runs degraded, without price/stop).
        risk_check = self.risk.check_order(
            portfolio,
            open_now,
            side=side,
            qty=qty,
            symbol=symbol,
            source=source,
            timeframe=params.timeframe,
            price=price,
            stop_loss=stop_loss,
        )
        if not risk_check.allowed:
            counters["skipped"] += 1
            self._log(
                "scalp_skip",
                symbol=symbol,
                detail={
                    "reason": risk_check.reason,
                    "explanation": (
                        f"Skipped {symbol}: the portfolio risk manager "
                        f"blocked this scalp — {risk_check.reason}."
                    ),
                },
            )
            return False

        before_ids = {int(p["id"]) for p in open_now}
        try:
            order = self.engine.submit_order(
                symbol,
                side,
                "market",
                qty=qty,
                stop_loss=stop_loss,
                take_profit=take_profit,
                source=source,
                timeframe=params.timeframe,
            )
        except ValueError as exc:
            counters["skipped"] += 1
            self._log(
                "scalp_skip",
                symbol=symbol,
                detail={
                    "reason": f"risk_rejected: {exc}",
                    "explanation": (
                        f"Scalp entry for {symbol} was rejected by the risk "
                        f"manager: {exc}"
                    ),
                },
            )
            return False

        fill_price = float(order.get("fill_price") or price)
        notional = qty * fill_price
        self._increment_trades_today()

        # Ownership attribution: prefer the id the engine reports for the
        # position THIS fill opened (race-free). Fall back to a before/after
        # diff scoped to this trader's (source, timeframe) so a concurrent
        # AutoTrader fill in the same symbol+side can never be claimed.
        raw_opened = order.get("opened_position_id")
        position_id: int | None = int(raw_opened) if raw_opened is not None else None
        if position_id is None:
            new_positions = [
                p
                for p in self.engine.get_positions("open")
                if int(p["id"]) not in before_ids
                and str(p["symbol"]) == symbol
                and str(p["side"]) == direction
                and str(p["source"]) == source
                and str(p["timeframe"]) == params.timeframe
            ]
            if new_positions:
                position_id = max(int(p["id"]) for p in new_positions)
        if position_id is not None:
            self._add_owned(position_id)
        else:
            logger.error(
                "Scalp order for %s filled but no new %s position was found — "
                "ownership not recorded",
                symbol,
                direction,
            )
            self._log(
                "error",
                symbol=symbol,
                detail={
                    "where": "scalp_ownership",
                    "reason": "filled order produced no new position",
                    "explanation": (
                        f"A scalp {direction} order for {symbol} filled but no "
                        "new position could be attributed to the scalper."
                    ),
                },
            )

        rsi_14 = values["rsi_14"]
        vwap_side = "above" if values["close"] > values["vwap"] else "below"
        ema_state = "bullish" if values["ema_12"] > values["ema_26"] else "bearish"
        # Shadow diagnostics (v3): PURE OBSERVATION for later research — the
        # flags never gate anything and regime.shadow_flags never raises.
        shadow = regime.shadow_flags(
            frame if frame is not None else pd.DataFrame(),
            datetime.now(timezone.utc).hour,
        )
        counters["entered"] += 1
        self._log(
            "scalp_enter",
            symbol=symbol,
            detail={
                "side": direction,
                "qty": _json_num(qty),
                "price": _json_num(fill_price),
                "notional": _json_num(notional),
                "stop_loss": _json_num(stop_loss),
                "take_profit": _json_num(take_profit),
                "rsi_14": _json_num(rsi_14),
                "vwap_side": vwap_side,
                "ema_state": ema_state,
                # Contract: tp_pct/sl_pct carry the EFFECTIVE values used for
                # THIS entry (ATR-scaled when atr_geometry is true).
                "tp_pct": _json_num(tp_pct_eff),
                "sl_pct": _json_num(sl_pct_eff),
                "atr_geometry": atr_geometry,
                "derisk_multiplier": _json_num(derisk_multiplier),
                "shadow_flags": shadow,
                "position_id": position_id,
                "explanation": self._entry_explanation(
                    params,
                    symbol,
                    direction,
                    qty,
                    notional,
                    fill_price,
                    stop_loss,
                    take_profit,
                    rsi_14,
                    vwap_side,
                    ema_state,
                    tp_pct=tp_pct_eff,
                    sl_pct=sl_pct_eff,
                ),
            },
        )
        return True

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    def stats(self, since_ms: int | None = None) -> dict[str, Any]:
        """Aggregate the scalper's closed-trade results from its exit rows.

        Every scalp close writes exactly one ``scalp_exit`` activity row
        (whether it was closed by TP/SL, time-stop, script stop or an
        external actor), so the feed doubles as the trade ledger.

        Args:
            since_ms: Only count exits at/after this epoch-ms timestamp
                (``None`` → since the last account reset, or all time when the
                account was never reset — ``bot_activity`` survives
                ``engine.reset()``, and pre-reset trades must not be reported
                against the fresh account).

        Returns:
            dict: ``{trades, wins, losses, win_rate, pnl, profit_factor,
            pnl_by_symbol, open, trades_today}``.
        """
        with self._connect() as conn:
            reset_ms = self.engine._state_get(conn, _LAST_RESET_KEY, None)
        if reset_ms is not None:
            try:
                floor_ms = int(reset_ms)
                since_ms = (
                    floor_ms if since_ms is None else max(int(since_ms), floor_ms)
                )
            except (TypeError, ValueError):
                logger.warning("Ignoring malformed %s value", _LAST_RESET_KEY)

        query = (
            "SELECT ts, symbol, detail FROM bot_activity WHERE action='scalp_exit'"
        )
        args: tuple[Any, ...] = ()
        if since_ms is not None:
            query += " AND ts >= ?"
            args = (int(since_ms),)
        query += " ORDER BY id ASC"
        with self._connect() as conn:
            rows = conn.execute(query, args).fetchall()

        trades = wins = losses = 0
        total_pnl = gross_wins = gross_losses = 0.0
        pnl_by_symbol: dict[str, float] = {}
        for row in rows:
            try:
                detail = json.loads(row["detail"])
            except (TypeError, json.JSONDecodeError):
                detail = {}
            if not isinstance(detail, dict):
                detail = {}
            pnl = detail.get("pnl")
            pnl = float(pnl) if isinstance(pnl, (int, float)) else 0.0
            trades += 1
            total_pnl += pnl
            if pnl > 0:
                wins += 1
                gross_wins += pnl
            elif pnl < 0:
                losses += 1
                gross_losses += -pnl
            symbol = str(row["symbol"] or "")
            if symbol:
                pnl_by_symbol[symbol] = pnl_by_symbol.get(symbol, 0.0) + pnl

        if gross_losses > 0:
            profit_factor = min(gross_wins / gross_losses, 100.0)
        else:
            profit_factor = 100.0 if gross_wins > 0 else 0.0

        owned = self._owned_ids()
        open_ids = {int(p["id"]) for p in self.engine.get_positions("open")}
        return {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / trades, 6) if trades else 0.0,
            "pnl": round(total_pnl, 6),
            "profit_factor": round(profit_factor, 6),
            "pnl_by_symbol": {s: round(v, 6) for s, v in pnl_by_symbol.items()},
            "open": len(owned & open_ids),
            "trades_today": self._trades_today(),
        }

    # ------------------------------------------------------------------ #
    # AI supervisor                                                        #
    # ------------------------------------------------------------------ #

    def tune_with_ai(self) -> dict[str, Any]:
        """One AI supervisor pass: review results, re-tune within HARD_BOUNDS.

        Builds a compact performance report (current params, stats since the
        last tune, per-symbol PnL, 5 best/worst recent trades), asks the
        local LLM for conservative changes as JSON, then validates and CLAMPS
        every proposed value into :data:`HARD_BOUNDS` before applying it via
        :meth:`set_params`. ``OllamaError`` or unparseable JSON changes
        NOTHING (logged as an ``error`` activity row).

        Returns:
            dict: ``{"applied": {param: new_value}, "reasoning": str}`` —
            plus an ``"error"`` key (and empty ``applied``) on failure, or a
            ``"status": "disabled"`` key when the scalper is not enabled.
        """
        params = self.get_params()
        if not params.enabled:
            # No blind tuning of a script that is not running (e.g. the
            # hourly job outliving a stop/reset): burning an LLM pass on
            # stale results and rewriting params the user is not using is
            # never useful.
            logger.info("AI tune skipped — the scalper is disabled")
            self._log(
                "scalp_skip",
                detail={
                    "reason": "disabled",
                    "explanation": (
                        "AI tune skipped: the scalper is disabled — "
                        "parameters were left unchanged."
                    ),
                },
            )
            return {"applied": {}, "reasoning": "", "status": "disabled"}

        if params.research_mode:
            # Research mode freezes the config: the tuner must not shrink
            # params or bench coins while the user is collecting unfiltered
            # trade data.
            logger.info("AI tune skipped — research mode freezes the config")
            self._log(
                "scalp_skip",
                detail={
                    "reason": "research_mode",
                    "explanation": (
                        "AI tune skipped: research mode is on — the "
                        "configuration is frozen while unfiltered trade "
                        "data is being collected."
                    ),
                },
            )
            return {"applied": {}, "reasoning": "", "status": "research_mode"}

        # AI-offline tune skip (v3): the Ollama watchdog owns the persisted
        # 'ollama_status'; a MISSING or malformed value means ONLINE (a
        # watchdog that never ran must not block tuning). Offline → change
        # nothing and let the hourly job try again later.
        with self._connect() as conn:
            ollama_status = self.engine._state_get(conn, _OLLAMA_STATUS_KEY, None)
        if (
            isinstance(ollama_status, dict)
            and ollama_status.get("available") is False
        ):
            since_raw = ollama_status.get("since_ms")
            since_ms = (
                int(since_raw) if isinstance(since_raw, (int, float)) else None
            )
            logger.info("AI tune skipped — Ollama is offline (watchdog status)")
            self._log(
                "scalp_skip",
                detail={
                    "reason": "ai_offline",
                    "since_ms": since_ms,
                    "explanation": (
                        "AI tune skipped: the local AI (Ollama) is offline — "
                        "the watchdog is trying to restart it; scalper "
                        "parameters were left unchanged."
                    ),
                },
            )
            return {"applied": {}, "reasoning": "", "status": "ai_offline"}

        before = params.model_dump()
        last_tune = self.get_last_tune()
        since_ms = last_tune.get("ts_ms") if isinstance(last_tune, dict) else None
        report = {
            "current_params": before,
            "stats_since_last_tune": self.stats(
                since_ms=int(since_ms) if since_ms else None
            ),
            "recent_trades": self._recent_trades_for_report(),
        }
        model = settings.ollama_model
        prompt = (
            "PERFORMANCE REPORT of the mechanical paper-trading scalper you "
            "supervise:\n"
            f"{json.dumps(report, indent=2, default=str)}\n\n"
            f"HARD BOUNDS (any value outside is clamped): {_bounds_text()}\n\n"
            "Rules:\n"
            "- Be conservative: small incremental changes only.\n"
            "- Fee-aware: a round trip costs about 0.3% (commission + "
            "slippage), so tp_pct must stay comfortably above that.\n"
            "- Prefer benching losing symbols (disabled_symbols) over "
            "widening stops.\n"
            f"- Shrink-only phase: until {SHRINK_ONLY_MIN_TRADES} closed "
            "trades exist, proposals that RAISE sl_pct, max_positions, "
            "position_fraction or max_trades_per_day are ignored; raising "
            "position_fraction after a losing stretch is always ignored "
            "(no martingale).\n"
            "- Only propose changes you can justify from the report; an empty "
            '"changes" object is a perfectly good answer.\n\n'
            "Respond ONLY with JSON exactly in this shape:\n"
            '{"changes": {"param_name": value, ...}, '
            '"disabled_symbols": ["SYMBOL", ...], '
            '"reasoning": "one short paragraph"}\n'
            '"disabled_symbols" is the FULL new bench list (it replaces the '
            "old one)."
        )
        system = (
            "You supervise a mechanical PAPER-TRADING scalper script — "
            "simulated money only, research, never real orders. It trades "
            f"{params.timeframe} candles with fixed rules: long when "
            "EMA12>EMA26, RSI inside a band and price above VWAP (short "
            "mirrored); fixed-percent take-profit/stop-loss; a time-stop "
            "after N bars. You review its recent results and propose "
            "conservative parameter changes as JSON. Not financial advice."
        )

        try:
            response = OllamaClient().chat(
                prompt, system=system, json_mode=True, temperature=0.2
            )
        except OllamaError as exc:
            logger.warning("AI tune skipped — Ollama unavailable: %s", exc)
            self._log(
                "error",
                detail={
                    "where": "scalp_tune",
                    "reason": str(exc),
                    "explanation": (
                        "The AI supervisor could not run (Ollama error) — "
                        "scalper parameters were left unchanged."
                    ),
                },
            )
            return {"applied": {}, "reasoning": "", "error": str(exc)}

        try:
            proposal = json.loads(response)
            if not isinstance(proposal, dict):
                raise ValueError(
                    f"expected a JSON object, got {type(proposal).__name__}"
                )
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("AI tune returned unusable JSON: %s", exc)
            self._log(
                "error",
                detail={
                    "where": "scalp_tune",
                    "reason": f"bad JSON from model: {exc}",
                    "explanation": (
                        "The AI supervisor returned unusable JSON — scalper "
                        "parameters were left unchanged."
                    ),
                },
            )
            return {"applied": {}, "reasoning": "", "error": f"bad JSON: {exc}"}

        reasoning = str(proposal.get("reasoning") or "")
        updates = self._sanitize_ai_proposal(
            proposal.get("changes"), proposal.get("disabled_symbols")
        )
        # Shrink-only tuner + martingale ban (v3): until enough closed trades
        # exist the AI may only shrink risk, and it may NEVER size up after a
        # losing window — offending fields are DROPPED from the update so the
        # live persisted value always wins (never re-written from the tune's
        # start-of-run snapshot, which a concurrent user change may have
        # superseded).
        updates, shrink_only_clamped, martingale_blocked = self._shrink_only_filter(
            updates,
            before,
            trades=int(self.stats().get("trades") or 0),
            pnl_since_last_tune=report["stats_since_last_tune"].get("pnl"),
        )
        try:
            new_params = self.set_params(updates) if updates else params
        except (ValidationError, ValueError) as exc:
            logger.warning("AI tune proposal failed validation: %s", exc)
            self._log(
                "error",
                detail={
                    "where": "scalp_tune",
                    "reason": f"proposal failed validation: {exc}",
                    "explanation": (
                        "The AI supervisor's proposal failed validation — "
                        "scalper parameters were left unchanged."
                    ),
                },
            )
            return {"applied": {}, "reasoning": reasoning, "error": str(exc)}

        after = new_params.model_dump()
        applied = {key: after[key] for key in after if before.get(key) != after[key]}
        now_ms = utc_now_ms()
        record = {
            "ts": _ms_to_iso(now_ms),
            "ts_ms": now_ms,
            "before": before,
            "after": after,
            "applied": applied,
            "reasoning": reasoning,
            "model": model,
            "shrink_only_clamped": shrink_only_clamped,
            "martingale_blocked": martingale_blocked,
        }
        with self._connect() as conn:
            self.engine._state_set(conn, _LAST_TUNE_KEY, record)
        if applied:
            changes_txt = ", ".join(
                f"{key}: {before.get(key)} -> {value}" for key, value in applied.items()
            )
            explanation = f"AI supervisor re-tuned the scalper ({changes_txt})."
        else:
            explanation = "AI supervisor reviewed the scalper and changed nothing."
        if shrink_only_clamped:
            explanation += (
                f" Shrink-only guard kept {', '.join(shrink_only_clamped)} "
                f"unchanged (fewer than {SHRINK_ONLY_MIN_TRADES} closed "
                "trades — the AI may only reduce risk until then)."
            )
        if martingale_blocked:
            explanation += (
                " Martingale ban kept position_fraction unchanged — sizing "
                "up right after losses is never allowed."
            )
        self._log(
            "scalp_tune",
            detail={
                "before": before,
                "after": after,
                "applied": applied,
                "reasoning": reasoning,
                "model": model,
                "shrink_only_clamped": shrink_only_clamped,
                "martingale_blocked": martingale_blocked,
                "explanation": explanation,
            },
        )
        logger.info("AI tune finished: %s", explanation)
        return {
            "applied": applied,
            "reasoning": reasoning,
            "shrink_only_clamped": shrink_only_clamped,
            "martingale_blocked": martingale_blocked,
        }

    @staticmethod
    def _shrink_only_filter(
        updates: dict[str, Any],
        current: dict[str, Any],
        trades: int,
        pnl_since_last_tune: Any,
    ) -> tuple[dict[str, Any], list[str], bool]:
        """Apply the v3 shrink-only tuner clamps and the martingale ban.

        Shrink-only rule: while fewer than :data:`SHRINK_ONLY_MIN_TRADES`
        closed scalp trades exist (since the last account reset), a proposal
        may NEVER raise ``sl_pct`` (a wider stop), ``max_positions``,
        ``position_fraction`` or ``max_trades_per_day`` — with too little
        evidence the AI may only reduce risk. Each offending field is
        REMOVED from the returned update dict (so ``set_params`` never
        touches it — the live persisted value stays authoritative, even when
        the user changed it while the LLM call was in flight) and reported
        in the returned clamp-audit list.

        Martingale ban (ALWAYS, whatever the trade count): raising
        ``position_fraction`` right after a losing tune window — "double
        down to win it back" — is dropped outright (same key removal).

        Args:
            updates: Sanitized proposal dict (not mutated; a copy is
                filtered and returned).
            current: Current persisted params dump (the tune's ``before``).
            trades: Closed scalp trades since the last account reset.
            pnl_since_last_tune: ``stats_since_last_tune["pnl"]`` from the
                tune report (non-numeric values are treated as 0.0).

        Returns:
            ``(filtered updates, shrink_only_clamped, martingale_blocked)``.
        """
        out = dict(updates)
        shrink_only_clamped: list[str] = []

        def _raises(field: str) -> bool:
            """True when the proposal raises ``field`` above its current value."""
            if field not in out:
                return False
            try:
                return float(out[field]) > float(current.get(field, 0.0))
            except (TypeError, ValueError):
                return False

        # Evaluate the martingale condition against the ORIGINAL proposal,
        # before the shrink-only pass rewrites position_fraction.
        position_fraction_raised = _raises("position_fraction")

        if trades < SHRINK_ONLY_MIN_TRADES:
            for field in _SHRINK_ONLY_FIELDS:
                if _raises(field):
                    logger.info(
                        "Shrink-only tuner: keeping %s at %r (proposed %r; "
                        "%d/%d closed trades)",
                        field,
                        current.get(field),
                        out[field],
                        trades,
                        SHRINK_ONLY_MIN_TRADES,
                    )
                    # Drop the key instead of re-writing current.get(field):
                    # `current` is the tune-start snapshot and set_params
                    # would persist it, silently overwriting any parameter
                    # change the user made during the (slow) LLM call — the
                    # guard could then RAISE a risk knob above its live value.
                    out.pop(field, None)
                    shrink_only_clamped.append(field)

        try:
            window_pnl = float(pnl_since_last_tune)
        except (TypeError, ValueError):
            window_pnl = 0.0
        if not math.isfinite(window_pnl):
            window_pnl = 0.0

        martingale_blocked = False
        if position_fraction_raised and window_pnl < 0:
            logger.info(
                "Martingale ban: keeping position_fraction at %r after a "
                "losing tune window (pnl %.2f)",
                current.get("position_fraction"),
                window_pnl,
            )
            # Key removal (not a stale re-write) — see the shrink-only note.
            out.pop("position_fraction", None)
            martingale_blocked = True
        return out, shrink_only_clamped, martingale_blocked

    @staticmethod
    def _sanitize_ai_proposal(changes: Any, disabled_symbols: Any) -> dict[str, Any]:
        """Reduce a raw AI proposal to a safe :meth:`set_params` update dict.

        Unknown/untunable params, non-numeric numerics and invalid enum
        values are dropped with a warning — never applied. The surviving
        values are still clamped by ``set_params``.

        Args:
            changes: The model's ``changes`` object (any JSON value).
            disabled_symbols: The model's full replacement bench list.

        Returns:
            Sanitized update dict (possibly empty).
        """
        updates: dict[str, Any] = {}
        if isinstance(changes, dict):
            for key, value in changes.items():
                if key not in _TUNABLE_PARAMS:
                    logger.warning(
                        "AI tuner proposed unknown/untunable param %r — ignored", key
                    )
                    continue
                if key == "allowed_sides":
                    side = str(value).strip().lower()
                    if side in ("long", "short", "both"):
                        updates[key] = side
                    else:
                        logger.warning(
                            "AI tuner proposed invalid allowed_sides %r — ignored",
                            value,
                        )
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    logger.warning(
                        "AI tuner proposed non-numeric %s=%r — ignored", key, value
                    )
                    continue
                if not math.isfinite(number):
                    logger.warning(
                        "AI tuner proposed non-finite %s=%r — ignored", key, value
                    )
                    continue
                updates[key] = number
        elif changes is not None:
            logger.warning("AI tuner 'changes' is not an object — ignored")
        if isinstance(disabled_symbols, list):
            updates["disabled_symbols"] = [
                str(s).strip().upper() for s in disabled_symbols if str(s).strip()
            ]
        elif disabled_symbols is not None:
            logger.warning("AI tuner 'disabled_symbols' is not a list — ignored")
        return updates

    def _recent_trades_for_report(self, limit: int = 50) -> dict[str, Any]:
        """The 5 best and 5 worst recent scalp trades for the tune report."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, symbol, detail FROM bot_activity "
                "WHERE action='scalp_exit' ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        trades: list[dict[str, Any]] = []
        for row in rows:
            try:
                detail = json.loads(row["detail"])
            except (TypeError, json.JSONDecodeError):
                detail = {}
            if not isinstance(detail, dict):
                detail = {}
            pnl = detail.get("pnl")
            trades.append(
                {
                    "ts": _ms_to_iso(row["ts"]),
                    "symbol": str(row["symbol"] or ""),
                    "reason": str(detail.get("reason") or ""),
                    "pnl": float(pnl) if isinstance(pnl, (int, float)) else 0.0,
                    "pnl_pct": detail.get("pnl_pct"),
                }
            )
        by_pnl = sorted(trades, key=lambda t: t["pnl"])
        return {"worst": by_pnl[:5], "best": list(reversed(by_pnl[-5:]))}

    # ------------------------------------------------------------------ #
    # Explanations & logging                                               #
    # ------------------------------------------------------------------ #

    def _entry_explanation(
        self,
        params: ScalperParams,
        symbol: str,
        direction: str,
        qty: float,
        notional: float,
        fill_price: float,
        stop_loss: float,
        take_profit: float,
        rsi_14: float,
        vwap_side: str,
        ema_state: str,
        tp_pct: float | None = None,
        sl_pct: float | None = None,
    ) -> str:
        """One plain-English sentence describing a scalp entry.

        ``tp_pct``/``sl_pct`` are the EFFECTIVE percentages used for this
        entry (ATR-scaled when geometry priced the stops); they default to
        the fixed params for backward compatibility.
        """
        tp_pct = params.tp_pct if tp_pct is None else tp_pct
        sl_pct = params.sl_pct if sl_pct is None else sl_pct
        base = _base_asset(symbol)
        verb = "Scalped long" if direction == "long" else "Scalped short"
        ema_txt = "EMA12 above EMA26" if ema_state == "bullish" else "EMA12 below EMA26"
        if direction == "long":
            band_lo, band_hi = params.rsi_long_min, params.rsi_long_max
        else:
            band_lo = 100.0 - params.rsi_long_max
            band_hi = 100.0 - params.rsi_long_min
        return (
            f"{verb} {qty:.6g} {base} (${notional:,.2f}) at ${fill_price:,.2f}: "
            f"{ema_txt}, RSI {rsi_14:.0f} in the {band_lo:.0f}-{band_hi:.0f} "
            f"entry band, price {vwap_side} VWAP — target ${take_profit:,.2f} "
            f"(+{tp_pct:.2%}), stop ${stop_loss:,.2f} "
            f"(-{sl_pct:.2%}), time-stop after {params.time_stop_bars} bars."
        )

    @staticmethod
    def _r_multiples(
        position: dict[str, Any],
    ) -> tuple[float | None, float | None]:
        """R-multiples of a closed position (contract null-safety rules).

        ``designed_r`` — how many times the risked stop distance the target
        promised: ``|take_profit - entry_price| / |entry_price - stop_loss|``
        (None when the stop or target is missing/invalid or the stop distance
        is not positive). ``realized_r`` — how many times the risked dollars
        the trade actually made: ``pnl / (|entry_price - stop_loss| * qty)``
        (None when the stop is missing/invalid, ``qty <= 0`` or the
        denominator is not positive).

        Args:
            position: Serialized (closed) position dict.

        Returns:
            ``(designed_r, realized_r)`` — either side may be None.
        """

        def _num(value: Any) -> float | None:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) else None

        entry = _num(position.get("entry_price"))
        stop = _num(position.get("stop_loss"))
        target = _num(position.get("take_profit"))
        qty = _num(position.get("qty"))
        pnl = _num(position.get("pnl"))

        stop_distance = 0.0
        if entry is not None and stop is not None:
            stop_distance = abs(entry - stop)

        designed_r: float | None = None
        if stop_distance > 0 and target is not None and entry is not None:
            designed_r = abs(target - entry) / stop_distance

        realized_r: float | None = None
        if stop_distance > 0 and qty is not None and qty > 0:
            denominator = stop_distance * qty
            if denominator > 0:
                realized_r = (pnl if pnl is not None else 0.0) / denominator
        return designed_r, realized_r

    def _log_scalp_exit(
        self, position: dict[str, Any], reason: str, params: ScalperParams
    ) -> None:
        """Write a ``scalp_exit`` activity row for a closed scalp position."""
        symbol = str(position.get("symbol") or "")
        side = str(position.get("side") or "long")
        qty = float(position.get("qty") or 0.0)
        entry_price = float(position.get("entry_price") or 0.0)
        exit_raw = position.get("exit_price")
        exit_price = float(exit_raw) if exit_raw is not None else entry_price
        pnl = float(position.get("pnl") or 0.0)
        entry_notional = qty * entry_price
        pnl_pct = pnl / entry_notional if entry_notional > 0 else 0.0
        designed_r, realized_r = self._r_multiples(position)

        # Actual protective-level distances (ATR geometry can price stops
        # away from the fixed params) — fall back to the params percentages.
        tp_pct_txt = params.tp_pct
        sl_pct_txt = params.sl_pct
        if entry_price > 0:
            tp_raw = position.get("take_profit")
            sl_raw = position.get("stop_loss")
            try:
                if tp_raw is not None and math.isfinite(float(tp_raw)):
                    tp_pct_txt = abs(float(tp_raw) - entry_price) / entry_price
                if sl_raw is not None and math.isfinite(float(sl_raw)):
                    sl_pct_txt = abs(float(sl_raw) - entry_price) / entry_price
            except (TypeError, ValueError):
                pass

        if reason == "take_profit":
            why = f"the +{tp_pct_txt:.2%} take-profit target was hit"
        elif reason == "stop_loss":
            why = f"the -{sl_pct_txt:.2%} protective stop-loss was hit"
        elif reason == "time_stop":
            why = (
                f"it sat {params.time_stop_bars}+ bars without hitting "
                "target or stop"
            )
        elif reason == "script_stopped":
            why = "the scalper script was stopped (open scalps close on stop)"
        elif reason == "closed_externally":
            why = "it was closed outside the scalper"
        else:
            why = f"exit reason: {reason}"
        explanation = (
            f"Closed scalp {side} {qty:.6g} {_base_asset(symbol)} at "
            f"${exit_price:,.2f} because {why} — PnL ${pnl:+,.2f} ({pnl_pct:+.2%})."
        )
        self._log(
            "scalp_exit",
            symbol=symbol,
            detail={
                "reason": reason,
                "pnl": _json_num(pnl),
                "pnl_pct": _json_num(pnl_pct),
                "side": side,
                "qty": _json_num(qty),
                # R-multiples (v3): reward promised / realized in units of the
                # dollars risked at the stop (null when no stop was recorded).
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
        except Exception:  # noqa: BLE001 — logging must never break a tick
            logger.exception(
                "Failed to write bot_activity row (action=%s, symbol=%s)",
                action,
                symbol,
            )

    # ------------------------------------------------------------------ #
    # Ownership, cooldowns, daily counter                                  #
    # ------------------------------------------------------------------ #

    def _owned_ids(self) -> set[int]:
        """Current scalper-owned position ids (fresh read from account_state)."""
        with self._connect() as conn:
            return load_scalper_position_ids(conn)

    def _add_owned(self, position_id: int) -> None:
        """Record ownership of a newly opened scalp position."""
        with self._state_lock, self._connect() as conn:
            ids = load_scalper_position_ids(conn)
            ids.add(int(position_id))
            self.engine._state_set(conn, SCALPER_POSITION_IDS_KEY, sorted(ids))

    def _prune_owned(self, position_id: int) -> None:
        """Drop a closed/vanished position id from the ownership record."""
        with self._state_lock, self._connect() as conn:
            ids = load_scalper_position_ids(conn)
            ids.discard(int(position_id))
            self.engine._state_set(conn, SCALPER_POSITION_IDS_KEY, sorted(ids))

    def _in_cooldown(self, symbol: str) -> bool:
        """True while ``symbol`` is inside its post-exit anti-churn window."""
        with self._connect() as conn:
            cooldowns = self.engine._state_get(conn, _COOLDOWNS_KEY, {})
        if not isinstance(cooldowns, dict):
            return False
        try:
            until_ms = int(cooldowns.get(symbol, 0))
        except (TypeError, ValueError):
            return False
        return utc_now_ms() < until_ms

    def _set_cooldown(self, symbol: str, params: ScalperParams) -> None:
        """Start the per-symbol cooldown after an exit; prune expired entries."""
        if not symbol:
            return
        now_ms = utc_now_ms()
        until_ms = now_ms + int(
            params.cooldown_bars
            * _timeframe_delta(params.timeframe).total_seconds()
            * 1000
        )
        with self._state_lock, self._connect() as conn:
            cooldowns = self.engine._state_get(conn, _COOLDOWNS_KEY, {})
            if not isinstance(cooldowns, dict):
                cooldowns = {}
            cooldowns = {
                s: int(t)
                for s, t in cooldowns.items()
                if isinstance(t, (int, float)) and int(t) > now_ms
            }
            cooldowns[symbol] = until_ms
            self.engine._state_set(conn, _COOLDOWNS_KEY, cooldowns)
        logger.info(
            "Cooldown set for %s until %s (%d bars)",
            symbol,
            _ms_to_iso(until_ms),
            params.cooldown_bars,
        )

    def _trades_today(self) -> int:
        """Entries made so far this UTC day (persisted anti-runaway counter)."""
        with self._connect() as conn:
            record = self.engine._state_get(conn, _TRADES_TODAY_KEY, None)
        if isinstance(record, dict) and record.get("date") == _utc_today():
            try:
                return max(0, int(record.get("count", 0)))
            except (TypeError, ValueError):
                return 0
        return 0

    def _increment_trades_today(self) -> int:
        """Increment (and UTC-day-roll) the persisted daily entry counter."""
        with self._state_lock:
            count = self._trades_today() + 1
            with self._connect() as conn:
                self.engine._state_set(
                    conn, _TRADES_TODAY_KEY, {"date": _utc_today(), "count": count}
                )
        return count

    # ------------------------------------------------------------------ #
    # Misc helpers                                                         #
    # ------------------------------------------------------------------ #

    def _load_playbook(self) -> dict[str, dict[str, Any]]:
        """The coach's per-coin playbook, keyed by symbol. Never raises.

        Imported late (contract) so the platform still works when the
        coach/intel layer is absent or broken, and so tests can monkeypatch
        either this method or ``coach.load_playbook_map``. One DB query per
        tick; any failure returns ``{}`` (no playbook = original behaviour).

        Returns:
            ``{symbol: playbook entry dict}`` (possibly empty).
        """
        try:
            from backend.paper_trading.coach import load_playbook_map

            return load_playbook_map()
        except Exception:  # noqa: BLE001 — the playbook is an optional layer
            logger.exception("Coach playbook unavailable — scalping without it")
            return {}

    def _bot_config(self) -> BotConfig:
        """The AutoTrader's persisted config (source + watchlist provider)."""
        with self._connect() as conn:
            raw = self.engine._state_get(conn, _BOT_CONFIG_KEY, None)
        if raw is None:
            return BotConfig()
        try:
            return BotConfig.model_validate(raw)
        except ValidationError:
            logger.warning("Persisted bot_config is invalid — using defaults")
            return BotConfig()

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
