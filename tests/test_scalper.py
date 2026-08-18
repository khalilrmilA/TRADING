"""Tests for backend/paper_trading/scalper.py — fast scalper + AI tuner (paper only).

Everything runs offline and deterministically:
    * NO network — the autouse conftest fixture blocks sockets, and an autouse
      fixture here additionally no-ops ``update_symbol`` everywhere the
      scalper/bot could look it up, so every pass runs purely off the seeded
      per-test tmp SQLite cache.
    * NO Ollama — ``OllamaClient.chat`` is patched at the CLASS level (so any
      import/instantiation style is covered). Outside the tune tests it FAILS
      the test if called: the fast script must never put an LLM in its loop.
    * Deterministic candles only — the 15m frames below are exact arithmetic
      patterns (no randomness) whose last-closed-bar entry conditions were
      verified robust across 100 UTC-day-boundary anchors, so the per-day
      VWAP reset can never flip an assertion regardless of when tests run:

        =============  ==========  ============  ============
        frame          ema12>ema26  rsi_14        close>vwap
        =============  ==========  ============  ============
        _long_ok_df    True (+.36)  58.9 in band  True (+.13)
        _rsi_hot_df    True         100.0 (hot)   True
        _ema_down_df   False (-.08) 54.0 in band  True
        _below_vwap_df True (+.52)  58.2 in band  False (-.10)
        _short_ok_df   False (-.36) 41.1 short-ok False (-.13)
        =============  ==========  ============  ============

      (band = default long band [50, 72]; short band = mirrored [28, 50].)

The VWAP determinism trick: the decision bar carries ~1e9 volume vs ~200 for
every other bar, so the day-cumulative VWAP is pinned to that bar's typical
price no matter where the UTC day boundary falls.

The autouse conftest fixtures give every test a fresh tmp database and pin
``settings.initial_capital`` to the contract default of 100,000.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

import backend.data.service as data_service_mod
import backend.paper_trading.auto_trader as auto_trader_mod
import backend.paper_trading.regime as regime_mod
import backend.paper_trading.scalper as scalper_mod
from backend.ai.ollama_client import OllamaClient, OllamaError
from backend.database.db import get_conn, upsert_ohlcv
from backend.indicators.features import add_features
from backend.paper_trading.auto_trader import AutoTrader
from backend.paper_trading.engine import PaperTradingEngine
from backend.paper_trading.scalper import HARD_BOUNDS, Scalper, ScalperParams
from backend.strategies.voting import VotingStrategy
from config.settings import settings

SOURCE = "binance"
TIMEFRAME = "15m"
BAR = pd.Timedelta(minutes=15)
BAR_MS = 15 * 60 * 1000

#: Decision-bar volume dominates the UTC-day VWAP (see module docstring).
HUGE_VOLUME = 1e9
BASE_VOLUME = 200.0

LONG_SYMBOL = "SCLLONGUSDT"
RSI_HOT_SYMBOL = "SCLHOTUSDT"
EMA_DOWN_SYMBOL = "SCLEMAUSDT"
BELOW_VWAP_SYMBOL = "SCLVWAPUSDT"
SHORT_SYMBOL = "SCLSHORTUSDT"
GOOD_SYMBOL = "SCLGOODUSDT"
BAD_SYMBOL = "SCLBADUSDT"
OWNED_SYMBOL = "SCLOWNUSDT"
CONTROL_SYMBOL = "SCLCTRLUSDT"

#: account_state key the contract mandates for scalper position ownership.
OWNERSHIP_KEY = "scalper_position_ids"

#: Mandatory scalp_enter transparency payload (CONTRACTS.md).
ENTER_DETAIL_KEYS = {
    "side",
    "qty",
    "price",
    "notional",
    "stop_loss",
    "take_profit",
    "rsi_14",
    "vwap_side",
    "ema_state",
    "explanation",
}

#: Mandatory stats() keys (CONTRACTS.md).
STATS_KEYS = {
    "trades",
    "wins",
    "losses",
    "win_rate",
    "pnl",
    "profit_factor",
    "pnl_by_symbol",
    "open",
    "trades_today",
}


# --------------------------------------------------------------------------- #
# Synthetic 15m market data (pure arithmetic — fully deterministic)            #
# --------------------------------------------------------------------------- #


def _make_ohlcv(close: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
    """Wrap a close/volume path into a canonical 15m UTC OHLCV frame.

    The frame is anchored so its last candle is the most recent fully CLOSED
    15m period — the scalper's unclosed-candle guard keeps it and the
    staleness guard treats it as live data.

    Args:
        close: Closing prices (all positive).
        volume: Per-bar volumes (all positive).

    Returns:
        Canonical OHLCV frame (ms-resolution index so DB roundtrips are exact).
    """
    n = len(close)
    last_closed = pd.Timestamp.now(tz="UTC").floor("15min") - BAR
    start = last_closed - (n - 1) * BAR
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    index = pd.date_range(
        start, periods=n, freq="15min", tz="UTC", name="timestamp"
    ).as_unit("ms")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    ).astype("float64")
    assert (df["low"] > 0).all()
    return df


def _decision_bar_volume(n: int) -> np.ndarray:
    """Base volume with a huge decision (last) bar that pins the day VWAP."""
    volume = np.full(n, BASE_VOLUME)
    volume[-1] = HUGE_VOLUME
    return volume


def _long_ok_df(n: int = 250) -> pd.DataFrame:
    """All three LONG conditions true: ema up, RSI ~58.9, close > vwap.

    A -0.3/+0.4 sawtooth (net +0.05/bar uptrend) ending on an UP bar; RSI
    settles around 59, ema_12 leads ema_26 by ~0.36 and the huge-volume up
    decision bar pins VWAP ~0.13 below its close.
    """
    deltas = np.tile([-0.3, 0.4], n // 2)  # ends on +0.4 (up bar)
    close = 100.0 + np.cumsum(deltas)
    return _make_ohlcv(close, _decision_bar_volume(n))


def _rsi_hot_df(n: int = 250) -> pd.DataFrame:
    """ONLY the RSI condition violated: every bar up → RSI 100 > 72."""
    close = 100.0 + 0.5 * np.arange(1, n + 1, dtype=np.float64)
    return _make_ohlcv(close, _decision_bar_volume(n))


def _ema_down_df(n_down: int = 220, recovery: int = 3) -> pd.DataFrame:
    """ONLY the EMA condition violated: ema_12 < ema_26 (gap ~-0.08).

    A shallow -0.25/+0.2 ripple downtrend followed by a 3-bar recovery: the
    recovery lifts RSI to ~54 (in band) and the huge-volume up decision bar
    puts close above VWAP, but the EMAs still reflect the downtrend.
    """
    deltas = np.concatenate(
        [np.tile([-0.25, 0.2], n_down // 2), np.full(recovery, 0.15)]
    )
    close = 120.0 + np.cumsum(deltas)
    return _make_ohlcv(close, _decision_bar_volume(n_down + recovery))


def _below_vwap_df(n: int = 250) -> pd.DataFrame:
    """ONLY the VWAP condition violated: close < vwap (gap ~-0.10).

    A +0.45/-0.3 sawtooth uptrend ending on a DOWN bar: ema up (+0.52), RSI
    ~58.2 (in band), but the huge-volume down decision bar pins VWAP above
    its close.
    """
    deltas = np.tile([0.45, -0.3], n // 2)  # ends on -0.3 (down bar)
    close = 100.0 + np.cumsum(deltas)
    return _make_ohlcv(close, _decision_bar_volume(n))


def _short_ok_df(n: int = 250) -> pd.DataFrame:
    """All three SHORT conditions true (exact mirror of ``_long_ok_df``).

    ema down (-0.36), RSI ~41.1 within the mirrored band [28, 50], and the
    huge-volume down decision bar pins close below VWAP (-0.13).
    """
    deltas = np.tile([0.3, -0.4], n // 2)  # ends on -0.4 (down bar)
    close = 150.0 + np.cumsum(deltas)
    return _make_ohlcv(close, _decision_bar_volume(n))


def _bearish_votes_df(n: int = 240) -> pd.DataFrame:
    """Down-trending frame whose LAST bar draws a -2 strategy vote sum.

    Mirror of test_auto_trader's trending fixture: flat base, then a steady
    decline ending in a decisive -2% breakdown close on a 4x volume spike.
    Last-bar votes (verified in the test that uses it): trend_following -1,
    breakout -1, rsi_macd -1, mean_reversion +1 → ensemble stance -1.
    """
    rng = np.random.default_rng(7)
    i = np.arange(n, dtype=np.float64)
    base = 200.0 + 0.6 * np.sin(i[: n // 2] / 5.0)
    trend = base[-1] - 0.45 * (i[: n - n // 2] + 1) - 0.3 * np.sin(i[: n - n // 2] / 4.0)
    close = np.concatenate([base, trend]) + rng.normal(0.0, 0.05, n)
    close[-1] = close[-2] * 0.98  # decisive breakdown close on the last bar
    volume = np.full(n, 200.0) + rng.uniform(0.0, 20.0, n)
    volume[-1] = 800.0  # volume spike confirms the breakdown
    return _make_ohlcv(close, volume)


def _seed(symbol: str, df: pd.DataFrame) -> None:
    """Pre-seed the tmp-DB 15m candle cache for one symbol."""
    upsert_ohlcv(SOURCE, symbol, TIMEFRAME, df)


def _entry_conditions(df: pd.DataFrame) -> dict[str, Any]:
    """Long-entry condition readout of a frame's last (decision) bar."""
    last = add_features(df).iloc[-1]
    return {
        "ema_up": bool(last["ema_12"] > last["ema_26"]),
        "rsi": float(last["rsi_14"]),
        "above_vwap": bool(last["close"] > last["vwap"]),
    }


def _seed_exit_trigger(
    symbol: str, df: pd.DataFrame, level: float, direction: str
) -> None:
    """Make the cached price path touch ``level`` after an entry.

    Overwrites the last closed candle AND appends the next (currently
    forming) candle so the touch is visible both to the engine's
    ``process_tick`` SL/TP evaluation (new candle → new watermark) and to any
    closed-candle price check the scalper might run itself.

    Args:
        symbol: Symbol whose cache to extend.
        df: The frame originally seeded for the symbol.
        level: Price level to touch (the position's TP or SL).
        direction: ``"up"`` to spike through ``level`` from below (TP on a
            long), ``"down"`` to dip through it (SL on a long).
    """
    last = df.iloc[-1]
    prev_close = float(df["close"].iloc[-2])
    t_last = df.index[-1]
    t_next = t_last + BAR
    if direction == "up":
        mod_close = level * 1.0005
        modified = {
            "open": float(last["open"]),
            "high": max(float(last["high"]), level * 1.003),
            "low": float(last["low"]),
            "close": mod_close,
            "volume": float(last["volume"]),
        }
        nxt = {
            "open": mod_close,
            "high": level * 1.004,
            # Stays above a -0.8% stop: the dip is < 0.5% below the entry bar.
            "low": prev_close * 0.999,
            "close": level * 1.001,
            "volume": BASE_VOLUME,
        }
    else:
        mod_close = level * 0.999
        modified = {
            "open": float(last["open"]),
            "high": float(last["high"]),
            "low": min(float(last["low"]), level * 0.997),
            "close": mod_close,
            "volume": float(last["volume"]),
        }
        nxt = {
            "open": mod_close,
            # Stays below a +1.2% target: the bounce is < 0.2% above entry.
            "high": float(last["high"]),
            "low": level * 0.996,
            "close": level * 0.9975,
            "volume": BASE_VOLUME,
        }
    frame = pd.DataFrame(
        [modified, nxt],
        index=pd.DatetimeIndex([t_last, t_next], name="timestamp"),
    ).astype("float64")
    upsert_ohlcv(SOURCE, symbol, TIMEFRAME, frame)


# --------------------------------------------------------------------------- #
# Scalper / engine helpers                                                     #
# --------------------------------------------------------------------------- #


def _make_scalper() -> tuple[PaperTradingEngine, Scalper]:
    """Build a fresh engine + scalper pair against the per-test tmp database."""
    engine = PaperTradingEngine()
    return engine, Scalper(engine)


def _set_watchlist(engine: PaperTradingEngine, symbols: list[str]) -> None:
    """Persist the bot watchlist the scalper scans (bot config minus benched)."""
    AutoTrader(engine).set_config(
        {
            "watchlist": symbols,
            "source": SOURCE,
            "timeframe": TIMEFRAME,
            "use_ai": False,
        }
    )


def _configure(scalper: Scalper, **overrides: Any) -> ScalperParams:
    """Enable the scalper and apply per-test parameter overrides.

    Baseline: the Improvement Pack v3 cost gate and ATR stop geometry are
    switched OFF via their contract-sanctioned user knobs (``0 disables`` /
    ``use_atr_geometry=False``) so this legacy suite keeps exercising the
    fixed-percent entry/exit mechanics its assertions pin. The v3 gate,
    geometry and tuner behaviors have their own binding acceptance suite in
    tests/test_improvements_traders.py.
    """
    updates: dict[str, Any] = {
        "enabled": True,
        "cost_gate_multiple": 0.0,  # 0 disables the v3 entry cost gate
        "use_atr_geometry": False,  # fixed tp_pct/sl_pct levels, as pinned below
    }
    updates.update(overrides)
    return scalper.set_params(updates)


def _open_symbols(engine: PaperTradingEngine) -> set[str]:
    """Symbols with an open paper position."""
    return {str(p["symbol"]) for p in engine.get_positions("open")}


def _scalp_activity(action: str | None = None) -> list[dict[str, Any]]:
    """Read the ``bot_activity`` audit log (oldest first), decoding detail JSON."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT ts, symbol, action, detail FROM bot_activity ORDER BY id ASC"
        ).fetchall()
    finally:
        conn.close()
    items: list[dict[str, Any]] = []
    for row in rows:
        try:
            detail = json.loads(row["detail"] or "{}")
        except (TypeError, ValueError):
            detail = {}
        items.append(
            {
                "ts": int(row["ts"]),
                "symbol": row["symbol"],
                "action": row["action"],
                "detail": detail if isinstance(detail, dict) else {},
            }
        )
    if action is not None:
        items = [item for item in items if item["action"] == action]
    return items


def _read_state(key: str) -> Any:
    """Read + JSON-decode one ``account_state`` value (None when absent)."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT value FROM account_state WHERE key=?", (key,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        return json.loads(row["value"])
    except (TypeError, ValueError):
        return None


def _write_state(key: str, value: Any) -> None:
    """JSON-encode + persist one ``account_state`` value."""
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO account_state (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        conn.commit()
    finally:
        conn.close()


def _owned_ids() -> list[int]:
    """Scalper-owned position ids from the contract-mandated state key."""
    raw = _read_state(OWNERSHIP_KEY)
    if not isinstance(raw, list):
        return []
    return [int(x) for x in raw]


def _backdate_position(position_id: int, bars: int) -> None:
    """Shift a position's ``opened_at`` back by ``bars`` 15m bars (age it)."""
    conn = get_conn()
    try:
        conn.execute(
            "UPDATE paper_positions SET opened_at = opened_at - ? WHERE id=?",
            (bars * BAR_MS, int(position_id)),
        )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Ollama chat stub (class-level patch covers every lookup/instantiation style) #
# --------------------------------------------------------------------------- #


class _ChatStub:
    """Recording ``OllamaClient.chat`` replacement with a fixed reply/error.

    Installed as a class attribute; it is not a function descriptor, so the
    client instance is NOT prepended to the call — the signature below
    mirrors ``chat(prompt, system=..., model=..., json_mode=..., ...)``.
    """

    def __init__(self, response: str = "", error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[str] = []

    def __call__(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        json_mode: bool = True,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        self.calls.append(str(prompt))
        if self.error is not None:
            raise self.error
        return self.response


def _patch_chat(
    monkeypatch: pytest.MonkeyPatch,
    response: str = "",
    error: Exception | None = None,
) -> _ChatStub:
    """Replace ``OllamaClient.chat`` for the tune tests (records prompts)."""
    stub = _ChatStub(response=response, error=error)
    monkeypatch.setattr(OllamaClient, "chat", stub)
    return stub


# --------------------------------------------------------------------------- #
# Autouse guards                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _pin_contract_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every settings knob the sizing/risk assertions depend on.

    ``max_open_positions`` is raised to 10 so the GLOBAL cap never masks the
    scalper's own guards (its sub-cap, cooldown and daily-trade limits are
    what these tests exercise).
    """
    monkeypatch.setattr(settings, "commission_rate", 0.001)
    monkeypatch.setattr(settings, "slippage_rate", 0.0005)
    monkeypatch.setattr(settings, "risk_per_trade", 0.01)
    monkeypatch.setattr(settings, "max_open_positions", 10)
    monkeypatch.setattr(settings, "daily_loss_limit", 0.03)
    monkeypatch.setattr(settings, "circuit_breaker_drawdown", 0.10)
    monkeypatch.setattr(settings, "atr_stop_multiplier", 2.0)
    monkeypatch.setattr(settings, "atr_take_profit_multiplier", 3.0)


@pytest.fixture(autouse=True)
def _offline_market_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """No-op every data refresh so passes run purely off the seeded cache.

    Patches ``update_symbol`` at its source module and on any module that
    re-exported it as a proxy (covers both import styles, plus the engine's
    call-time import inside ``process_tick``).
    """

    def _noop_update(*args: Any, **kwargs: Any) -> int:
        return 0

    monkeypatch.setattr(data_service_mod, "update_symbol", _noop_update)
    for module in (scalper_mod, auto_trader_mod):
        if hasattr(module, "update_symbol"):
            monkeypatch.setattr(module, "update_symbol", _noop_update)


@pytest.fixture(autouse=True)
def _no_llm_in_the_fast_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything calls the LLM outside an explicit tune stub.

    The scalper is contractually a fast MECHANICAL script — ``run_tick`` must
    never talk to Ollama. Tune tests overwrite this patch with ``_patch_chat``
    (same function-scoped monkeypatch, so the later setattr wins).
    """

    def _forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError(
            "OllamaClient.chat must not be called here — the fast scalper loop "
            "is LLM-free and tune tests must stub the chat call explicitly."
        )

    monkeypatch.setattr(OllamaClient, "chat", _forbidden)
    monkeypatch.setattr(OllamaClient, "is_available", lambda self: True)
    monkeypatch.setattr(
        OllamaClient, "list_models", lambda self: [settings.ollama_model]
    )


# --------------------------------------------------------------------------- #
# Fixture self-check — the crafted frames must read the way the tests assume   #
# --------------------------------------------------------------------------- #


def test_synthetic_frames_have_expected_conditions() -> None:
    """Pin each frame's last-bar entry-condition truth values (see docstring)."""
    long_ok = _entry_conditions(_long_ok_df())
    assert long_ok["ema_up"] is True
    assert 50.0 < long_ok["rsi"] < 72.0
    assert long_ok["above_vwap"] is True

    rsi_hot = _entry_conditions(_rsi_hot_df())
    assert rsi_hot["ema_up"] is True
    assert rsi_hot["rsi"] > 72.0  # the ONLY violated long condition
    assert rsi_hot["above_vwap"] is True

    ema_down = _entry_conditions(_ema_down_df())
    assert ema_down["ema_up"] is False  # the ONLY violated long condition
    assert 50.0 < ema_down["rsi"] < 72.0
    assert ema_down["above_vwap"] is True

    below_vwap = _entry_conditions(_below_vwap_df())
    assert below_vwap["ema_up"] is True
    assert 50.0 < below_vwap["rsi"] < 72.0
    assert below_vwap["above_vwap"] is False  # the ONLY violated long condition

    short_ok = _entry_conditions(_short_ok_df())
    assert short_ok["ema_up"] is False
    assert 28.0 < short_ok["rsi"] < 50.0  # mirrored band for rsi bounds (50, 72)
    assert short_ok["above_vwap"] is False


# --------------------------------------------------------------------------- #
# Params: defaults, clamping, persistence                                      #
# --------------------------------------------------------------------------- #


def test_scalper_params_contract_defaults() -> None:
    """ScalperParams defaults match CONTRACTS.md exactly; HARD_BOUNDS exists."""
    params = ScalperParams()
    assert params.enabled is False
    assert params.timeframe == "15m"
    assert params.interval_minutes == 2
    assert params.tp_pct == pytest.approx(0.012)
    assert params.sl_pct == pytest.approx(0.008)
    assert params.time_stop_bars == 16
    assert params.position_fraction == pytest.approx(0.06)
    assert params.max_positions == 8
    assert params.rsi_long_min == pytest.approx(50.0)
    assert params.rsi_long_max == pytest.approx(72.0)
    assert params.allowed_sides == "both"
    assert params.cooldown_bars == 3
    assert params.max_trades_per_day == 40
    assert params.disabled_symbols == []
    assert HARD_BOUNDS  # module constant, non-empty


def test_set_params_clamps_out_of_bounds_values() -> None:
    """Every out-of-bounds value is clamped to its HARD_BOUNDS edge."""
    _, scalper = _make_scalper()

    high = scalper.set_params(
        {
            "tp_pct": 1.0,
            "sl_pct": 0.9,
            "time_stop_bars": 1000,
            "position_fraction": 0.5,
            "max_positions": 50,
            "interval_minutes": 99,
            "cooldown_bars": 99,
            "max_trades_per_day": 9999,
        }
    )
    assert high.tp_pct == pytest.approx(0.03)
    assert high.sl_pct == pytest.approx(0.02)
    assert high.time_stop_bars == 64
    assert high.position_fraction == pytest.approx(0.10)
    assert high.max_positions == 10
    assert high.interval_minutes == 10
    assert high.cooldown_bars == 20
    assert high.max_trades_per_day == 100

    low = scalper.set_params(
        {
            "sl_pct": 0.000001,
            "time_stop_bars": 1,
            "position_fraction": 0.0001,
            "max_positions": 0,
            "interval_minutes": 0,
            "cooldown_bars": 0,
            "max_trades_per_day": 1,
        }
    )
    assert low.sl_pct == pytest.approx(0.004)
    assert low.time_stop_bars == 4
    assert low.position_fraction == pytest.approx(0.02)
    assert low.max_positions == 1
    assert low.interval_minutes == 1
    assert low.cooldown_bars == 1
    assert low.max_trades_per_day == 5


def test_set_params_keeps_stop_below_target() -> None:
    """The sl_pct < tp_pct invariant survives any update (stops never off)."""
    _, scalper = _make_scalper()
    params = scalper.set_params({"sl_pct": 0.02, "tp_pct": 0.006})
    assert params.sl_pct < params.tp_pct
    assert 0.004 <= params.sl_pct <= 0.02
    assert 0.006 <= params.tp_pct <= 0.03


def test_params_persist_across_instances() -> None:
    """set_params persists under 'scalper_params' and survives a fresh Scalper."""
    engine, scalper = _make_scalper()
    scalper.set_params(
        {"tp_pct": 0.02, "cooldown_bars": 5, "disabled_symbols": [BAD_SYMBOL]}
    )

    persisted = _read_state("scalper_params")
    assert isinstance(persisted, dict)
    assert persisted["tp_pct"] == pytest.approx(0.02)

    fresh = Scalper(engine).get_params()
    assert fresh.tp_pct == pytest.approx(0.02)
    assert fresh.cooldown_bars == 5
    assert BAD_SYMBOL in fresh.disabled_symbols


# --------------------------------------------------------------------------- #
# run_tick: enable gate + entry-rule truth table                               #
# --------------------------------------------------------------------------- #


def test_run_tick_disabled_is_noop() -> None:
    """With enabled=False (the default) a tick trades nothing at all."""
    _seed(LONG_SYMBOL, _long_ok_df())
    engine, scalper = _make_scalper()
    _set_watchlist(engine, [LONG_SYMBOL])
    assert scalper.get_params().enabled is False

    result = scalper.run_tick()

    assert isinstance(result, dict)
    assert engine.get_positions("open") == []
    assert engine.get_orders() == []
    assert _scalp_activity("scalp_enter") == []


def test_long_entry_when_all_conditions_align() -> None:
    """ema up + RSI in band + close > vwap → one long scalp with full detail."""
    _seed(LONG_SYMBOL, _long_ok_df())
    engine, scalper = _make_scalper()
    _set_watchlist(engine, [LONG_SYMBOL])
    _configure(scalper)

    result = scalper.run_tick()
    assert isinstance(result, dict)

    positions = engine.get_positions("open")
    assert len(positions) == 1
    position = positions[0]
    assert position["symbol"] == LONG_SYMBOL
    assert position["side"] == "long"
    assert position["timeframe"] == TIMEFRAME

    entry = float(position["entry_price"])
    qty = float(position["qty"])
    notional = qty * entry
    # position_fraction sizing: 6% of the 100k account (± slippage rounding).
    assert 5_600.0 <= notional <= 6_300.0

    # Protective levels at entry -/+ sl_pct/tp_pct (0.8% / 1.2% defaults);
    # tolerance covers computing the offsets off the pre-slippage close.
    stop_loss = float(position["stop_loss"])
    take_profit = float(position["take_profit"])
    assert stop_loss < entry < take_profit
    assert 0.9900 <= stop_loss / entry <= 0.9930
    assert 1.0105 <= take_profit / entry <= 1.0135

    # Ownership tag: the id is tracked in account_state 'scalper_position_ids'.
    assert int(position["id"]) in _owned_ids()

    enters = _scalp_activity("scalp_enter")
    assert len(enters) == 1
    assert enters[0]["symbol"] == LONG_SYMBOL
    detail = enters[0]["detail"]
    assert ENTER_DETAIL_KEYS <= set(detail.keys())
    assert detail["side"] == "long"
    assert isinstance(detail["explanation"], str) and detail["explanation"]
    assert float(detail["qty"]) > 0
    assert float(detail["price"]) > 0


@pytest.mark.parametrize(
    ("symbol", "frame_factory", "broken_condition"),
    [
        (RSI_HOT_SYMBOL, _rsi_hot_df, "rsi_out_of_band"),
        (EMA_DOWN_SYMBOL, _ema_down_df, "ema_down"),
        (BELOW_VWAP_SYMBOL, _below_vwap_df, "close_below_vwap"),
    ],
)
def test_no_long_entry_when_one_condition_fails(
    symbol: str, frame_factory: Any, broken_condition: str
) -> None:
    """Each frame violates exactly ONE long condition → no entry at all."""
    _seed(symbol, frame_factory())
    engine, scalper = _make_scalper()
    _set_watchlist(engine, [symbol])
    _configure(scalper)

    result = scalper.run_tick()

    assert isinstance(result, dict), broken_condition
    assert engine.get_positions("open") == []
    assert engine.get_orders() == []  # not even a rejected order was submitted
    assert _scalp_activity("scalp_enter") == []


def test_short_entry_mirrored(monkeypatch: pytest.MonkeyPatch) -> None:
    """ema down + mirrored RSI band + close < vwap → one short scalp.

    The v3 regime hard gate only allows shorts when the coin AND BTCUSDT are
    both in a 4h downtrend, so ``regime.get_regime`` is pinned to a
    double-downtrend here (the gate itself is covered by the binding suite in
    tests/test_improvements_traders.py).
    """
    _seed(SHORT_SYMBOL, _short_ok_df())
    engine, scalper = _make_scalper()
    _set_watchlist(engine, [SHORT_SYMBOL])
    _configure(scalper)  # allowed_sides defaults to "both"

    def _downtrend_regime(
        source: str, symbol: str, ref_timeframe: str = "1h"
    ) -> regime_mod.RegimeInfo:
        return regime_mod.RegimeInfo(
            regime="downtrend",
            close=95.0,
            ema50=97.0,
            ema200=105.0,
            bars_used=300,
            pct_change_7d=-5.0,
            pct_change_30d=-12.0,
        )

    monkeypatch.setattr(regime_mod, "get_regime", _downtrend_regime)

    scalper.run_tick()

    positions = engine.get_positions("open")
    assert len(positions) == 1
    position = positions[0]
    assert position["symbol"] == SHORT_SYMBOL
    assert position["side"] == "short"

    # Mirrored protective levels: stop ABOVE entry, target BELOW.
    entry = float(position["entry_price"])
    stop_loss = float(position["stop_loss"])
    take_profit = float(position["take_profit"])
    assert take_profit < entry < stop_loss
    assert 1.0070 <= stop_loss / entry <= 1.0100
    assert 0.9870 <= take_profit / entry <= 0.9900

    enters = _scalp_activity("scalp_enter")
    assert len(enters) == 1
    assert enters[0]["detail"]["side"] == "short"


def test_allowed_sides_long_only_blocks_short() -> None:
    """A perfect short setup is ignored when allowed_sides='long'."""
    _seed(SHORT_SYMBOL, _short_ok_df())
    engine, scalper = _make_scalper()
    _set_watchlist(engine, [SHORT_SYMBOL])
    _configure(scalper, allowed_sides="long")

    scalper.run_tick()

    assert engine.get_positions("open") == []
    assert _scalp_activity("scalp_enter") == []


def test_allowed_sides_short_only_blocks_long() -> None:
    """A perfect long setup is ignored when allowed_sides='short'."""
    _seed(LONG_SYMBOL, _long_ok_df())
    engine, scalper = _make_scalper()
    _set_watchlist(engine, [LONG_SYMBOL])
    _configure(scalper, allowed_sides="short")

    scalper.run_tick()

    assert engine.get_positions("open") == []
    assert _scalp_activity("scalp_enter") == []


# --------------------------------------------------------------------------- #
# Exits: take-profit, stop-loss, time-stop                                     #
# --------------------------------------------------------------------------- #


def test_take_profit_exit_closes_position() -> None:
    """A candle touching the TP closes the scalp at a profit and prunes the tag."""
    df = _long_ok_df()
    _seed(LONG_SYMBOL, df)
    engine, scalper = _make_scalper()
    _set_watchlist(engine, [LONG_SYMBOL])
    _configure(scalper)

    scalper.run_tick()
    position = engine.get_positions("open")[0]
    position_id = int(position["id"])
    assert position_id in _owned_ids()

    _seed_exit_trigger(LONG_SYMBOL, df, float(position["take_profit"]), "up")
    scalper.run_tick()

    assert engine.get_positions("open") == []
    closed = [
        p for p in engine.get_positions("closed") if int(p["id"]) == position_id
    ]
    assert len(closed) == 1
    assert float(closed[0]["pnl"]) > 0  # +1.2% target comfortably beats fees

    exits = _scalp_activity("scalp_exit")
    assert exits, "a scalp exit must be logged to bot_activity"
    assert exits[-1]["symbol"] == LONG_SYMBOL
    detail = exits[-1]["detail"]
    assert {"reason", "pnl", "explanation"} <= set(detail.keys())
    assert isinstance(detail["explanation"], str) and detail["explanation"]

    # Ownership pruned on close (contract: JSON list, pruned on close).
    assert position_id not in _owned_ids()

    stats = scalper.stats()
    assert STATS_KEYS <= set(stats.keys())
    assert int(stats["open"]) == 0


def test_stop_loss_exit_closes_position() -> None:
    """A candle dipping through the SL closes the scalp at a loss."""
    df = _long_ok_df()
    _seed(LONG_SYMBOL, df)
    engine, scalper = _make_scalper()
    _set_watchlist(engine, [LONG_SYMBOL])
    _configure(scalper)

    scalper.run_tick()
    position = engine.get_positions("open")[0]
    position_id = int(position["id"])

    _seed_exit_trigger(LONG_SYMBOL, df, float(position["stop_loss"]), "down")
    scalper.run_tick()

    assert engine.get_positions("open") == []
    closed = [
        p for p in engine.get_positions("closed") if int(p["id"]) == position_id
    ]
    assert len(closed) == 1
    assert float(closed[0]["pnl"]) < 0
    assert _scalp_activity("scalp_exit"), "the stop-out must be logged"
    assert position_id not in _owned_ids()


def test_time_stop_exits_aged_position() -> None:
    """A position older than time_stop_bars is market-closed by the tick."""
    _seed(LONG_SYMBOL, _long_ok_df())
    engine, scalper = _make_scalper()
    _set_watchlist(engine, [LONG_SYMBOL])
    _configure(scalper, time_stop_bars=4)

    scalper.run_tick()
    position = engine.get_positions("open")[0]
    position_id = int(position["id"])

    # Age the position past the 4-bar limit (no new candles: SL/TP untouched).
    _backdate_position(position_id, bars=6)
    scalper.run_tick()

    assert engine.get_positions("open") == []
    closed = [
        p for p in engine.get_positions("closed") if int(p["id"]) == position_id
    ]
    assert len(closed) == 1

    exits = [e for e in _scalp_activity("scalp_exit") if e["symbol"] == LONG_SYMBOL]
    assert exits, "the time-stop exit must be logged"
    assert "time" in str(exits[-1]["detail"].get("reason", "")).lower()
    assert position_id not in _owned_ids()


# --------------------------------------------------------------------------- #
# Anti-churn guards: cooldown + max_trades_per_day                             #
# --------------------------------------------------------------------------- #


def test_cooldown_blocks_immediate_reentry() -> None:
    """After an exit the symbol must sit out cooldown_bars; others may trade.

    GOOD_SYMBOL (added to the watchlist only after the exit) is the control:
    it enters on the very tick where LONG_SYMBOL — whose entry conditions
    still hold, self-checked below — is blocked purely by its cooldown.
    """
    _seed(LONG_SYMBOL, _long_ok_df())
    engine, scalper = _make_scalper()
    _set_watchlist(engine, [LONG_SYMBOL, GOOD_SYMBOL])
    _configure(scalper, time_stop_bars=4, cooldown_bars=3)

    scalper.run_tick()
    assert _open_symbols(engine) == {LONG_SYMBOL}
    position_id = int(engine.get_positions("open")[0]["id"])

    # Exit via the time-stop (leaves the symbol's candles untouched, so its
    # entry conditions remain TRUE for the re-entry attempt below).
    _backdate_position(position_id, bars=6)
    scalper.run_tick()
    assert engine.get_positions("open") == []

    conditions = _entry_conditions(_long_ok_df())
    assert conditions["ema_up"] and conditions["above_vwap"]
    assert 50.0 < conditions["rsi"] < 72.0

    _seed(GOOD_SYMBOL, _long_ok_df())
    scalper.run_tick()

    # Control entered; the cooled-down symbol did not re-enter.
    assert _open_symbols(engine) == {GOOD_SYMBOL}
    long_enters = [
        e for e in _scalp_activity("scalp_enter") if e["symbol"] == LONG_SYMBOL
    ]
    assert len(long_enters) == 1, "cooldown must block the immediate re-entry"


def test_max_trades_per_day_guard() -> None:
    """Once trades_today reaches the cap, no further scalp entries fire."""
    symbols = [f"SCLM{i}USDT" for i in range(5)]
    for symbol in symbols:
        _seed(symbol, _long_ok_df())
    engine, scalper = _make_scalper()
    _set_watchlist(engine, [*symbols, GOOD_SYMBOL])
    _configure(scalper, max_trades_per_day=5)  # the HARD_BOUNDS minimum

    scalper.run_tick()
    assert _open_symbols(engine) == set(symbols)  # 5 entries booked today

    # A brand-new perfect setup appears — the daily guard must block it.
    _seed(GOOD_SYMBOL, _long_ok_df())
    scalper.run_tick()

    assert _open_symbols(engine) == set(symbols)
    assert GOOD_SYMBOL not in {
        e["symbol"] for e in _scalp_activity("scalp_enter")
    }
    stats = scalper.stats()
    assert int(stats["trades_today"]) >= 5


# --------------------------------------------------------------------------- #
# tune_with_ai (OllamaClient.chat mocked at the class — no server, no network) #
# --------------------------------------------------------------------------- #


def test_tune_clamps_out_of_range_ai_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wild AI proposals are clamped to HARD_BOUNDS; unknown params ignored.

    ``position_fraction`` and ``max_trades_per_day`` start AT their hard-bound
    ceilings: below SHRINK_ONLY_MIN_TRADES closed trades the v3 shrink-only
    tuner phase would revert any RAISE of them (that behavior has its own
    binding tests in tests/test_improvements_traders.py) — starting at the
    ceiling keeps this test pinned on pure HARD_BOUNDS clamping.
    """
    engine, scalper = _make_scalper()
    _set_watchlist(engine, [LONG_SYMBOL])
    _configure(scalper, position_fraction=0.10, max_trades_per_day=100)
    stub = _patch_chat(
        monkeypatch,
        response=json.dumps(
            {
                "changes": {
                    "tp_pct": 0.5,
                    "sl_pct": 0.000001,
                    "position_fraction": 0.9,
                    "time_stop_bars": 1000,
                    "max_trades_per_day": 100000,
                    "leverage": 100,  # unknown param — must be ignored
                },
                "disabled_symbols": [],
                "reasoning": "synthetic aggressive tune (must be clamped)",
            }
        ),
    )

    result = scalper.tune_with_ai()

    assert stub.calls, "the tuner must consult the (mocked) LLM"
    assert isinstance(result, dict)
    params = scalper.get_params()
    assert params.tp_pct == pytest.approx(0.03)
    assert params.sl_pct == pytest.approx(0.004)
    assert params.position_fraction == pytest.approx(0.10)
    assert params.time_stop_bars == 64
    assert params.max_trades_per_day == 100
    assert params.sl_pct < params.tp_pct
    assert "leverage" not in params.model_dump()
    assert _scalp_activity("scalp_tune"), "an applied tune must be logged"


def test_tune_with_malformed_json_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unparseable AI output leaves every parameter exactly as it was."""
    engine, scalper = _make_scalper()
    _set_watchlist(engine, [LONG_SYMBOL])
    _configure(scalper, tp_pct=0.015, cooldown_bars=4)
    before = scalper.get_params().model_dump()
    _patch_chat(monkeypatch, response="sorry, no JSON from me today ][")

    result = scalper.tune_with_ai()  # must not raise

    assert isinstance(result, dict)
    assert scalper.get_params().model_dump() == before


def test_tune_ollama_error_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreachable LLM aborts the tune quietly: log an error, change nothing."""
    engine, scalper = _make_scalper()
    _set_watchlist(engine, [LONG_SYMBOL])
    _configure(scalper)
    before = scalper.get_params().model_dump()
    _patch_chat(monkeypatch, error=OllamaError("model unreachable (synthetic)"))

    result = scalper.tune_with_ai()  # must not raise

    assert isinstance(result, dict)
    assert scalper.get_params().model_dump() == before
    actions = {item["action"] for item in _scalp_activity()}
    assert actions & {"error", "scalp_error"}, "the failure must be logged"


def test_tune_can_bench_symbols_via_disabled_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AI-benched symbols persist in disabled_symbols and stop being traded."""
    _seed(GOOD_SYMBOL, _long_ok_df())
    _seed(BAD_SYMBOL, _long_ok_df())
    engine, scalper = _make_scalper()
    _set_watchlist(engine, [GOOD_SYMBOL, BAD_SYMBOL])
    _configure(scalper)
    _patch_chat(
        monkeypatch,
        response=json.dumps(
            {
                "changes": {},
                "disabled_symbols": [BAD_SYMBOL],
                "reasoning": "synthetic: this symbol keeps losing — bench it",
            }
        ),
    )

    scalper.tune_with_ai()
    assert BAD_SYMBOL in scalper.get_params().disabled_symbols

    scalper.run_tick()

    # Identical perfect setups, but the benched symbol is never traded.
    assert _open_symbols(engine) == {GOOD_SYMBOL}
    assert BAD_SYMBOL not in {e["symbol"] for e in _scalp_activity("scalp_enter")}


# --------------------------------------------------------------------------- #
# Ownership separation: AutoTrader must leave scalper positions alone          #
# --------------------------------------------------------------------------- #


def test_auto_trader_manage_skips_scalper_owned_positions() -> None:
    """The bot's ensemble-flip exit never touches scalper-owned positions.

    Two identical LONG positions sit on strongly BEARISH data (last-bar
    strategy vote sum -2 → ensemble stance -1, self-checked below). The bot's
    manage pass must flip-close the unowned control position and leave the
    scalper-tagged one open — the scalper alone manages its own scalps.
    """
    bearish = _bearish_votes_df()
    _seed(OWNED_SYMBOL, bearish)
    _seed(CONTROL_SYMBOL, bearish)

    # Self-check: the ensemble opposes a long on this data (flip guaranteed).
    breakdown = VotingStrategy().vote_breakdown(add_features(bearish))
    assert int(breakdown.iloc[-1][["trend_following", "mean_reversion",
                                   "breakout", "rsi_macd"]].sum()) <= -2
    assert int(breakdown["ensemble"].iloc[-1]) == -1

    engine = PaperTradingEngine()
    engine.submit_order(
        OWNED_SYMBOL, "buy", "market", qty=1.0, source=SOURCE, timeframe=TIMEFRAME
    )
    engine.submit_order(
        CONTROL_SYMBOL, "buy", "market", qty=1.0, source=SOURCE, timeframe=TIMEFRAME
    )
    assert _open_symbols(engine) == {OWNED_SYMBOL, CONTROL_SYMBOL}

    owned_id = next(
        int(p["id"])
        for p in engine.get_positions("open")
        if p["symbol"] == OWNED_SYMBOL
    )
    _write_state(OWNERSHIP_KEY, [owned_id])  # tag it as scalper-owned

    trader = AutoTrader(engine)
    trader.set_config(
        {
            "watchlist": [OWNED_SYMBOL, CONTROL_SYMBOL],
            "source": SOURCE,
            "timeframe": TIMEFRAME,
            "use_ai": False,
            "allow_short": False,
        }
    )

    result = trader.run_cycle()
    assert isinstance(result, dict)

    open_symbols = _open_symbols(engine)
    assert CONTROL_SYMBOL not in open_symbols, (
        "control: the ensemble flip must close the unowned long"
    )
    assert OWNED_SYMBOL in open_symbols, (
        "the bot must NOT ensemble-flip-close a scalper-owned position"
    )
    owned_position = next(
        p for p in engine.get_positions("open") if p["symbol"] == OWNED_SYMBOL
    )
    assert int(owned_position["id"]) == owned_id
