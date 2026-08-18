"""Protective-level validation + raw_response exposure (audit follow-up).

An audit found two gaps (paper only, like everything else):

* The engine's fill path accepted stop_loss/take_profit levels on the WRONG
  side of the fill price (e.g. a short with its stop BELOW entry) — the level
  then insta-triggered on the next evaluated candle with adverse slippage.
  The fix validates AT FILL TIME (the only moment the reference price is
  known): the invalid level is dropped (set to NULL), a human-readable
  explanation is appended to the order's ``note``, and the order itself
  still fills. ``update_protective_levels`` refuses a wrong-side FIRST stop
  install the same way (``stop_moved: False`` — the existing not-moved
  signalling), while tighten moves keep their established semantics.
* ``ai_analyses.raw_response`` was write-only. ``GET /api/analysis/history``
  now takes ``include_raw`` (default False) which adds each row's
  ``raw_response``; the default payload is unchanged.

Everything runs offline and deterministically: NO network (conftest blocks
sockets), NO Ollama (history rows are seeded straight into the tmp DB),
synthetic seeded candles only. Endpoint functions are called directly with
explicit keyword arguments — the suite has no HTTP-level tests (same
convention as tests/test_watchdog.py).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

import backend.api.main as main_mod
from backend.database.db import get_conn, upsert_ohlcv, utc_now_ms
from backend.paper_trading.engine import PaperTradingEngine

SOURCE = "binance"
SYMBOL = "VALUSDT"
TIMEFRAME = "1h"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _seed_flat_candles(
    n: int = 30, close: float = 100.0, start: str = "2024-01-01"
) -> pd.DataFrame:
    """Seed n flat candles (close constant, high/low ±1) into the DB cache."""
    index = pd.date_range(start, periods=n, freq="1h", tz="UTC", name="timestamp")
    closes = np.full(n, close, dtype=np.float64)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.full(n, 250.0),
        },
        index=index,
    )
    upsert_ohlcv(SOURCE, SYMBOL, TIMEFRAME, df)
    return df


def _append_candle(
    ts: str, open_: float, high: float, low: float, close: float
) -> None:
    """Append one crafted candle (e.g. one that fills a pending limit)."""
    index = pd.DatetimeIndex([pd.Timestamp(ts, tz="UTC")], name="timestamp")
    df = pd.DataFrame(
        {
            "open": [open_],
            "high": [high],
            "low": [low],
            "close": [close],
            "volume": [250.0],
        },
        index=index,
    )
    upsert_ohlcv(SOURCE, SYMBOL, TIMEFRAME, df)


def _seed_analysis(symbol: str, raw: str, created_ms: int) -> int:
    """Insert one ``ai_analyses`` row directly (no Ollama) and return its id."""
    conn = get_conn()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO ai_analyses (created_at, model, symbol, timeframe, "
                "sentiment, confidence, risk_commentary, key_indicators, "
                "reasoning, raw_response) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    created_ms,
                    "test-model",
                    symbol,
                    TIMEFRAME,
                    "neutral",
                    55,
                    "test risk commentary",
                    json.dumps(["rsi_14"]),
                    "test reasoning",
                    raw,
                ),
            )
            return int(cur.lastrowid)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Fill-time validation — market orders (long side)                             #
# --------------------------------------------------------------------------- #


def test_long_wrong_side_stop_dropped_valid_tp_kept(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A long's stop AT/ABOVE the fill is dropped (note + warning); TP stays."""
    _seed_flat_candles(close=100.0)
    engine = PaperTradingEngine()

    with caplog.at_level(logging.WARNING, logger="backend.paper_trading.engine"):
        order = engine.submit_order(
            SYMBOL, "buy", "market", qty=1.0, stop_loss=105.0, take_profit=110.0
        )

    # The order itself still fills — only the invalid level is dropped.
    assert order["status"] == "filled"
    assert "stop_loss" in order["note"]
    assert "dropped" in order["note"]
    assert any("stop_loss" in r.getMessage() for r in caplog.records)

    position = engine.get_positions("open")[0]
    assert position["stop_loss"] is None
    assert float(position["take_profit"]) == pytest.approx(110.0)

    # Regression: the old behaviour insta-triggered the 105 "stop" on the
    # very next candle (trigger_low 99 <= 105) with adverse slippage.
    _append_candle("2024-01-02 06:00", 100.0, 101.0, 99.0, 100.0)
    result = engine.process_tick(SYMBOL, source=SOURCE, timeframe=TIMEFRAME)
    assert result["closed"] == []
    assert len(engine.get_positions("open")) == 1


def test_long_wrong_side_take_profit_dropped_valid_stop_kept() -> None:
    """A long's TP AT/BELOW the fill is dropped; the valid stop stays."""
    _seed_flat_candles(close=100.0)
    engine = PaperTradingEngine()

    order = engine.submit_order(
        SYMBOL, "buy", "market", qty=1.0, stop_loss=95.0, take_profit=90.0
    )
    assert order["status"] == "filled"
    assert "take_profit" in order["note"]
    assert "stop_loss" not in order["note"]

    position = engine.get_positions("open")[0]
    assert float(position["stop_loss"]) == pytest.approx(95.0)
    assert position["take_profit"] is None


# --------------------------------------------------------------------------- #
# Fill-time validation — market orders (short side, mirrored)                  #
# --------------------------------------------------------------------------- #


def test_short_wrong_side_stop_dropped_valid_tp_kept() -> None:
    """A short's stop AT/BELOW the fill is dropped; the valid TP stays."""
    _seed_flat_candles(close=100.0)
    engine = PaperTradingEngine()

    order = engine.submit_order(
        SYMBOL, "sell", "market", qty=1.0, stop_loss=95.0, take_profit=90.0
    )
    assert order["status"] == "filled"
    assert "stop_loss" in order["note"]
    assert "dropped" in order["note"]

    position = engine.get_positions("open")[0]
    assert position["side"] == "short"
    assert position["stop_loss"] is None
    assert float(position["take_profit"]) == pytest.approx(90.0)

    # The old behaviour: trigger_high 101 >= 95 → insta "stop" with slippage.
    _append_candle("2024-01-02 06:00", 100.0, 101.0, 99.0, 100.0)
    result = engine.process_tick(SYMBOL, source=SOURCE, timeframe=TIMEFRAME)
    assert result["closed"] == []
    assert len(engine.get_positions("open")) == 1


def test_short_wrong_side_take_profit_dropped_valid_stop_kept() -> None:
    """A short's TP AT/ABOVE the fill is dropped; the valid stop stays."""
    _seed_flat_candles(close=100.0)
    engine = PaperTradingEngine()

    order = engine.submit_order(
        SYMBOL, "sell", "market", qty=1.0, stop_loss=105.0, take_profit=104.0
    )
    assert order["status"] == "filled"
    assert "take_profit" in order["note"]
    assert "stop_loss" not in order["note"]

    position = engine.get_positions("open")[0]
    assert float(position["stop_loss"]) == pytest.approx(105.0)
    assert position["take_profit"] is None


def test_valid_levels_kept_and_note_stays_empty() -> None:
    """Correct-side levels are installed untouched and the note stays empty."""
    _seed_flat_candles(close=100.0)
    engine = PaperTradingEngine()

    order = engine.submit_order(
        SYMBOL, "buy", "market", qty=1.0, stop_loss=95.0, take_profit=110.0
    )
    assert order["status"] == "filled"
    assert order["note"] == ""

    position = engine.get_positions("open")[0]
    assert float(position["stop_loss"]) == pytest.approx(95.0)
    assert float(position["take_profit"]) == pytest.approx(110.0)


# --------------------------------------------------------------------------- #
# Fill-time validation — pending orders validate against the FILL price        #
# --------------------------------------------------------------------------- #


def test_pending_limit_fill_validates_at_fill_price() -> None:
    """A pending limit's levels are judged at its fill, not the submit price.

    The buy limit at 95 carries stop_loss=99 / take_profit=98 — both below
    the market at submit time (100), but only the stop is on the wrong side
    of the eventual 95 fill.
    """
    _seed_flat_candles(n=30, close=100.0)  # last candle: 2024-01-02 05:00 UTC
    engine = PaperTradingEngine()

    order = engine.submit_order(
        SYMBOL,
        "buy",
        "limit",
        qty=1.0,
        limit_price=95.0,
        stop_loss=99.0,
        take_profit=98.0,
    )
    assert order["status"] == "pending"
    assert order["note"] == ""  # nothing validated (or dropped) yet

    # Low 94 pierces the 95 limit; high 96.5 stays under the 98 TP.
    _append_candle("2024-01-02 06:00", 96.0, 96.5, 94.0, 96.0)
    result = engine.process_tick(SYMBOL, source=SOURCE, timeframe=TIMEFRAME)
    assert len(result["filled"]) == 1
    filled = result["filled"][0]
    assert filled["fill_price"] == pytest.approx(95.0)
    assert "stop_loss" in filled["note"]
    assert "take_profit" not in filled["note"]

    # Without the drop, trigger_low 94 <= 99 would have insta-stopped it out.
    assert result["closed"] == []
    position = engine.get_positions("open")[0]
    assert position["stop_loss"] is None
    assert float(position["take_profit"]) == pytest.approx(98.0)


# --------------------------------------------------------------------------- #
# update_protective_levels — wrong-side FIRST install is refused               #
# --------------------------------------------------------------------------- #


def test_update_protective_levels_refuses_wrong_side_first_install_long() -> None:
    """A long's first stop at/above the mark is refused (stop_moved False)."""
    _seed_flat_candles(close=100.0)
    engine = PaperTradingEngine()

    engine.submit_order(SYMBOL, "buy", "market", qty=1.0)  # no stop attached
    position = engine.get_positions("open")[0]
    position_id = int(position["id"])
    entry = float(position["entry_price"])

    # Above the mark → would trigger immediately → refused, stop stays NULL.
    result = engine.update_protective_levels(position_id, stop_loss=106.0)
    assert result["stop_moved"] is False
    assert result["stop_loss"] is None

    # Exactly at the mark is refused too (>= is wrong-side for a long).
    result = engine.update_protective_levels(position_id, stop_loss=entry)
    assert result["stop_moved"] is False
    assert result["stop_loss"] is None

    # A correct-side first install goes through.
    result = engine.update_protective_levels(position_id, stop_loss=95.0)
    assert result["stop_moved"] is True
    assert float(result["stop_loss"]) == pytest.approx(95.0)


def test_update_protective_levels_refuses_wrong_side_first_install_short() -> None:
    """A short's first stop at/below the mark is refused (stop_moved False)."""
    _seed_flat_candles(close=100.0)
    engine = PaperTradingEngine()

    engine.submit_order(SYMBOL, "sell", "market", qty=1.0)  # no stop attached
    position = engine.get_positions("open")[0]
    assert position["side"] == "short"
    position_id = int(position["id"])
    entry = float(position["entry_price"])

    # Below the mark → would trigger immediately → refused, stop stays NULL.
    result = engine.update_protective_levels(position_id, stop_loss=95.0)
    assert result["stop_moved"] is False
    assert result["stop_loss"] is None

    # Exactly at the mark is refused too (<= is wrong-side for a short).
    result = engine.update_protective_levels(position_id, stop_loss=entry)
    assert result["stop_moved"] is False
    assert result["stop_loss"] is None

    # A correct-side first install goes through.
    result = engine.update_protective_levels(position_id, stop_loss=110.0)
    assert result["stop_moved"] is True
    assert float(result["stop_loss"]) == pytest.approx(110.0)


# --------------------------------------------------------------------------- #
# GET /api/analysis/history — raw_response only when asked                     #
# --------------------------------------------------------------------------- #


def test_history_default_payload_has_no_raw_response() -> None:
    """Without include_raw the payload is unchanged — no raw_response key."""
    assert any(
        getattr(route, "path", "") == "/api/analysis/history"
        for route in main_mod.app.routes
    )
    _seed_analysis("BTCUSDT", "raw reply A", utc_now_ms() - 1_000)
    _seed_analysis("BTCUSDT", "raw reply B", utc_now_ms())

    payload = main_mod.analysis_history(symbol=None, limit=50, include_raw=False)
    assert len(payload["items"]) == 2
    for item in payload["items"]:
        assert "raw_response" not in item


def test_history_include_raw_adds_each_rows_raw_response() -> None:
    """include_raw=True adds raw_response per row; other keys are untouched."""
    id_a = _seed_analysis("BTCUSDT", "raw reply A", utc_now_ms() - 1_000)
    id_b = _seed_analysis("ETHUSDT", "raw reply B", utc_now_ms())

    payload = main_mod.analysis_history(symbol=None, limit=50, include_raw=True)
    by_id = {int(item["id"]): item for item in payload["items"]}
    assert by_id[id_a]["raw_response"] == "raw reply A"
    assert by_id[id_b]["raw_response"] == "raw reply B"

    # The symbol filter still applies with include_raw.
    filtered = main_mod.analysis_history(
        symbol="ETHUSDT", limit=50, include_raw=True
    )
    assert [item["raw_response"] for item in filtered["items"]] == ["raw reply B"]

    # raw_response is the ONLY key added relative to the default payload.
    base = main_mod.analysis_history(symbol="ETHUSDT", limit=50, include_raw=False)
    assert set(filtered["items"][0]) == set(base["items"][0]) | {"raw_response"}
