"""Pydantic request/response models for the Trading AI Platform API.

PAPER TRADING ONLY — these schemas describe a research/backtest/paper-trading
API. Nothing here places real orders or references private API keys.

Request models mirror the JSON bodies documented in ``CONTRACTS.md`` (section
``backend/api/``) exactly, with the contract's defaults. Response models are
provided for the endpoints whose shapes the API layer fully controls; dynamic
payloads (backtest results, paper-engine records, AI analyses) are passed
through as plain JSON dicts by ``backend.api.main``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Shared enumerations (see "Canonical OHLCV DataFrame" in CONTRACTS.md).
Source = Literal["binance", "bybit", "yahoo"]
Timeframe = Literal["1m", "5m", "15m", "1h", "4h", "1d"]
OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop"]


# ---------------------------------------------------------------------------
# Request models (POST bodies)
# ---------------------------------------------------------------------------


class FetchDataRequest(BaseModel):
    """Body for ``POST /api/data/fetch``."""

    source: Source = "binance"
    symbol: str = Field(
        ..., min_length=1, description="Symbol in the source's native format, e.g. BTCUSDT or AAPL."
    )
    timeframe: Timeframe = "1h"
    lookback_days: int = Field(365, ge=1, le=3650, description="History window when nothing is cached yet.")


class AnalysisRequest(BaseModel):
    """Body for ``POST /api/analysis``."""

    source: Source = "binance"
    symbol: str = Field(..., min_length=1)
    timeframe: Timeframe = "1h"
    model: str | None = Field(
        None, description="Optional Ollama model override; defaults to settings.ollama_model."
    )


class BacktestRequest(BaseModel):
    """Body for ``POST /api/backtest``."""

    source: Source = "binance"
    symbol: str = Field(..., min_length=1)
    timeframe: Timeframe = "1h"
    strategy: str = Field(..., min_length=1, description="Registry key, e.g. 'trend_following'.")
    limit: int = Field(2000, ge=1, description="Most recent cached candles to backtest on.")
    params: dict[str, Any] = Field(default_factory=dict, description="Strategy constructor kwargs.")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional BacktestConfig field overrides (e.g. initial_capital, allow_short).",
    )


class WalkForwardRequest(BaseModel):
    """Body for ``POST /api/backtest/walkforward``."""

    source: Source = "binance"
    symbol: str = Field(..., min_length=1)
    timeframe: Timeframe = "1h"
    strategy: str = Field(..., min_length=1)
    n_splits: int = Field(4, ge=2, le=20)
    train_ratio: float = Field(0.7, gt=0.0, lt=1.0)
    limit: int = Field(3000, ge=1)


class PaperOrderRequest(BaseModel):
    """Body for ``POST /api/paper/orders`` (simulated orders only)."""

    symbol: str = Field(..., min_length=1)
    side: OrderSide
    order_type: OrderType = "market"
    qty: float | None = Field(None, gt=0, description="None → risk-based auto-sizing from ATR.")
    limit_price: float | None = Field(None, gt=0)
    stop_price: float | None = Field(None, gt=0)
    stop_loss: float | None = Field(None, gt=0)
    take_profit: float | None = Field(None, gt=0)
    source: Source = "binance"
    timeframe: Timeframe = "1h"


class TickRequest(BaseModel):
    """Body for ``POST /api/paper/tick``."""

    symbol: str = Field(..., min_length=1)
    source: Source = "binance"
    timeframe: Timeframe = "1h"
    refresh: bool = Field(False, description="True → refresh from the public data API before processing.")


class BotConfigUpdate(BaseModel):
    """Body for ``POST /api/bot/config`` — partial AI auto-trader config update.

    Every field is optional: only the fields actually provided are applied
    (the route passes ``model_dump(exclude_unset=True)`` to
    ``AutoTrader.set_config``). Mirrors ``BotConfig`` in
    ``backend.paper_trading.auto_trader`` — paper trading only.
    """

    watchlist: list[str] | None = Field(
        None, min_length=1, description="Symbols the bot scans each cycle."
    )
    source: Source | None = None
    timeframe: Timeframe | None = None
    interval_minutes: int | None = Field(None, ge=1, le=1440)
    use_ai: bool | None = Field(None, description="LLM confirmation gate for entries.")
    min_ai_confidence: int | None = Field(None, ge=0, le=100)
    max_ai_calls_per_cycle: int | None = Field(None, ge=0, le=20)
    allow_short: bool | None = None
    min_vote: int | None = Field(None, ge=1, le=4)
    max_position_fraction: float | None = Field(None, gt=0.0, le=1.0)
    running: bool | None = None
    regime_gate_enabled: bool | None = Field(
        None,
        description=(
            "Improvement Pack v3 market-regime hard gate (shorts only in a "
            "coin+BTC 4h double-downtrend, longs blocked in that same state)."
        ),
    )
    cost_gate_multiple: float | None = Field(
        None,
        description=(
            "Improvement Pack v3 entry cost gate: expected per-bar move "
            "(ATR/price) must cover this multiple of the round-trip "
            "fee+slippage cost. 0 disables. Deliberately NOT range-checked "
            "here — BotConfig clamps it to [0, 10] instead of rejecting."
        ),
    )
    trailing_enabled: bool | None = Field(
        None,
        description=(
            "'Let winners run' trailing stops: once a bot position gains "
            "trail_activate_pct, its stop follows the best close at "
            "trail_distance_pct behind and the take-profit is removed."
        ),
    )
    trail_activate_pct: float | None = Field(
        None,
        description="Gain that activates trailing. Not range-checked here — BotConfig clamps to [0.005, 0.2].",
    )
    trail_distance_pct: float | None = Field(
        None,
        description="Trail distance behind the best close. Not range-checked here — BotConfig clamps to [0.005, 0.1].",
    )
    use_second_judge: bool | None = Field(
        None, description="Require the finance-specialist second judge to agree before entering."
    )
    second_judge_model: str | None = Field(None, min_length=1)
    judge_min_confidence: int | None = Field(None, ge=0, le=100)
    sentiment_rank_weight: float | None = Field(
        None,
        description=(
            "News-sentiment weight in candidate RANKING (never qualifies a "
            "trade). Not range-checked here — BotConfig clamps to [0, 2]."
        ),
    )


class ScalperParamsUpdate(BaseModel):
    """Body for ``POST /api/scalper/params`` — partial fast-scalper params update.

    Every field is optional: only the fields actually provided are applied
    (the route passes ``model_dump(exclude_unset=True)`` to
    ``Scalper.set_params``). Numeric values are deliberately NOT range-checked
    here — ``Scalper.set_params`` CLAMPS everything into the module's
    ``HARD_BOUNDS`` (per CONTRACTS.md) instead of rejecting with a 422.
    ``enabled`` is intentionally absent: use ``POST /api/scalper/start`` /
    ``/stop`` to toggle the script (stop also market-closes open scalps).
    Mirrors ``ScalperParams`` in ``backend.paper_trading.scalper`` — paper
    trading only.
    """

    timeframe: Timeframe | None = None
    interval_minutes: int | None = None
    tp_pct: float | None = None
    sl_pct: float | None = None
    time_stop_bars: int | None = None
    position_fraction: float | None = None
    max_positions: int | None = None
    rsi_long_min: float | None = None
    rsi_long_max: float | None = None
    allowed_sides: Literal["long", "short", "both"] | None = None
    cooldown_bars: int | None = None
    max_trades_per_day: int | None = None
    disabled_symbols: list[str] | None = Field(
        None, description="Symbols the scalper must not trade (full replacement list)."
    )
    use_atr_geometry: bool | None = Field(
        None,
        description=(
            "Improvement Pack v3 ATR stop geometry (TP/SL scaled off the last "
            "closed candle's ATR). USER-ONLY knob: this endpoint is the only "
            "sanctioned mutation path — the AI tuner may never touch it."
        ),
    )
    cost_gate_multiple: float | None = Field(
        None,
        description=(
            "Improvement Pack v3 entry cost gate: the 15m ATR must cover this "
            "multiple of the round-trip fee+slippage cost. 0 disables. "
            "USER-ONLY knob (not AI-tunable); clamped to HARD_BOUNDS [0, 10] "
            "by set_params."
        ),
    )
    research_mode: bool | None = Field(
        None,
        description=(
            "Data-collection mode: trade EVERY signal on EVERY watchlist "
            "coin — soft daily stop, bench lists, side-bias, regime gate and "
            "cost gate are all bypassed and the AI tuner is paused (config "
            "frozen). Exits and HARD_BOUNDS still apply. USER-ONLY knob "
            "(not AI-tunable). PAPER TRADING ONLY."
        ),
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: str = "ok"
    ollama_available: bool = False
    db: str = "ok"
    paper_only: bool = True
    ollama_since_ms: int | None = Field(
        None,
        description=(
            "Epoch ms (UTC) when Ollama's current available/unavailable state "
            "began, from the watchdog's persisted status; null when the "
            "watchdog has never written it."
        ),
    )


class ConfiguredModels(BaseModel):
    """The Ollama models configured in settings."""

    default: str
    fallback: str
    code: str


class ModelsResponse(BaseModel):
    """Response for ``GET /api/models``."""

    configured: ConfiguredModels
    installed: list[str] = Field(default_factory=list)
    ollama_available: bool = False


class CoverageInfo(BaseModel):
    """First/last cached candle timestamps (ISO-8601 UTC), or nulls when empty."""

    first: str | None = None
    last: str | None = None


class FetchDataResponse(BaseModel):
    """Response for ``POST /api/data/fetch``."""

    rows_added: int
    coverage: CoverageInfo


class OHLCVResponse(BaseModel):
    """Response for ``GET /api/data/ohlcv``."""

    symbol: str
    timeframe: str
    rows: int
    data: list[dict[str, Any]] = Field(default_factory=list)


class ItemsResponse(BaseModel):
    """Generic ``{"items": [...]}`` list wrapper used by several GET endpoints."""

    items: list[dict[str, Any]] = Field(default_factory=list)


class StatusResponse(BaseModel):
    """Simple status acknowledgement (e.g. ``POST /api/paper/reset``)."""

    status: str
