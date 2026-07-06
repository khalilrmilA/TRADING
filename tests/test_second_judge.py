"""Tests for the SECOND JUDGE (Fin-R1) + SENTIMENT-BOOSTED RANKING upgrade.

Binding spec: CONTRACTS.md, section "SECOND JUDGE (Fin-R1) + SENTIMENT-BOOSTED
RANKING" — two additions to the AutoTrader entry pipeline:

    1. Sentiment-boosted ranking in ``_shortlist``:
       ``rank_score = |vote_sum| + sentiment_rank_weight × direction ×
       (score / 100)`` using the freshest ``coin_sentiment`` row (3h rule);
       missing/stale sentiment → bonus 0. News only REORDERS qualified
       candidates — the ``min_vote`` gate is unchanged.
    2. A second LLM judge in ``_enter_candidate`` (after the primary AI gate
       passes): ``analyze_market(model=config.second_judge_model)`` must agree
       on direction with ``confidence >= judge_min_confidence``; a model that
       is not installed (per ``OllamaClient.list_models``, prefix-before-':'
       matching included) logs ONE warning 'error' activity row per cycle and
       the entry proceeds WITHOUT the judge; an ``OllamaError`` from an
       installed judge conservatively skips the candidate.

Everything runs offline and deterministically:
    * NO network — the autouse conftest fixture blocks sockets outright and
      an autouse fixture here additionally no-ops ``update_symbol`` so scans
      run purely off the pre-seeded tmp-DB candle cache.
    * NO Ollama — ``analyze_market`` is monkeypatched at its source module
      AND on the ``auto_trader`` module (wherever the bot looked it up) with
      a stub that DISPATCHES ON ITS ``model`` ARGUMENT: calls with
      ``model == config.second_judge_model`` receive the second judge's
      canned verdict, every other call receives the primary analyst's.
      ``OllamaClient.list_models`` is patched at the CLASS level so any
      import/instantiation style auto_trader uses sees the canned catalogue;
      ``OllamaClient.chat`` fails the test if anything reaches it directly.
    * Synthetic candles only — the same seeded trending frame as
      tests/test_auto_trader.py (last-bar strategy vote sum +2, self-checked
      below in ``test_trending_frame_votes_self_check``).

The autouse conftest fixtures give every test a fresh tmp database and pin
``settings.initial_capital`` to the contract default of 100,000.

NOTE: this file is the acceptance suite for the JUDGE agent's backend edits;
until those land in ``backend/paper_trading/auto_trader.py`` these tests fail
by design (the main suite run ignores this file for that reason).
"""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

import backend.ai.analyst as analyst_mod
import backend.data.service as data_service_mod
import backend.paper_trading.auto_trader as auto_trader_mod
from backend.ai.analyst import MarketAnalysis
from backend.ai.ollama_client import OllamaClient, OllamaError
from backend.database.db import get_conn, upsert_ohlcv, utc_now_ms
from backend.indicators.features import add_features
from backend.paper_trading.auto_trader import AutoTrader, BotConfig
from backend.paper_trading.engine import PaperTradingEngine
from backend.strategies.voting import VotingStrategy
from config.settings import settings

SOURCE = "binance"
TIMEFRAME = "1h"

#: Contract default of ``BotConfig.second_judge_model``.
DEFAULT_JUDGE_MODEL = "mychen76/Fin-R1:Q5"

#: Symbols used by the judge tests (identical seeded trending candles).
TREND_SYMBOL = "TRENDUSDT"
TREND_TWIN_SYMBOL = "TRENDBUSDT"
#: Symbols used by the sentiment-ranking tests (identical trending candles,
#: so |vote_sum| and |momentum_10| tie exactly and only the bonus can reorder).
RANK_A_SYMBOL = "RANKAUSDT"
RANK_B_SYMBOL = "RANKBUSDT"

#: The four member strategies whose last-bar votes the bot sums for the scan.
MEMBER_STRATEGIES = ("trend_following", "mean_reversion", "breakout", "rsi_macd")

#: Empirically pinned last-bar vote sum of the trending frame (self-checked
#: below; byte-identical construction to tests/test_auto_trader.py).
TREND_VOTE_SUM = 2

#: 3h freshness horizon of the sentiment rules (contract: stale > 3h = 0).
_HOUR_MS = 3_600_000


# --------------------------------------------------------------------------- #
# Synthetic market data (seeded — fully deterministic)                         #
# --------------------------------------------------------------------------- #


def _make_ohlcv(close: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
    """Wrap a close/volume path into a canonical hourly UTC OHLCV frame.

    The frame ends at the most recent CLOSED hourly candle so the bot's
    closed-candle and data-freshness gates treat it as live data.

    Args:
        close: Closing prices (all positive).
        volume: Per-bar volumes (all positive).

    Returns:
        Canonical OHLCV frame (ms-resolution index so DB roundtrips are exact).
    """
    n = len(close)
    last_closed = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
    start = last_closed - pd.Timedelta(hours=n - 1)
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    index = pd.date_range(
        start, periods=n, freq="1h", tz="UTC", name="timestamp"
    ).as_unit("ms")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    ).astype("float64")


def _trending_df(n: int = 240) -> pd.DataFrame:
    """Strongly up-trending frame whose LAST bar draws a +2 strategy vote sum.

    Byte-identical construction to tests/test_auto_trader.py: flat base for
    indicator warm-up, steady climb, decisive +2% breakout close on a 4x
    volume spike. Self-checked in ``test_trending_frame_votes_self_check``.

    Args:
        n: Number of hourly candles.

    Returns:
        Canonical OHLCV frame ending at the most recent closed hour.
    """
    rng = np.random.default_rng(7)
    i = np.arange(n, dtype=np.float64)
    base = 100.0 + 0.6 * np.sin(i[: n // 2] / 5.0)
    trend = base[-1] + 0.45 * (i[: n - n // 2] + 1) + 0.3 * np.sin(i[: n - n // 2] / 4.0)
    close = np.concatenate([base, trend]) + rng.normal(0.0, 0.05, n)
    close[-1] = close[-2] * 1.02  # decisive breakout close on the last bar
    volume = np.full(n, 200.0) + rng.uniform(0.0, 20.0, n)
    volume[-1] = 800.0  # volume spike confirms the breakout
    return _make_ohlcv(close, volume)


def _seed_symbol(symbol: str, df: pd.DataFrame) -> None:
    """Pre-seed the tmp-DB candle cache for one symbol."""
    upsert_ohlcv(SOURCE, symbol, TIMEFRAME, df)


# --------------------------------------------------------------------------- #
# Bot helpers                                                                  #
# --------------------------------------------------------------------------- #


def _make_trader() -> tuple[PaperTradingEngine, AutoTrader]:
    """Build a fresh engine + trader pair against the per-test tmp database."""
    engine = PaperTradingEngine()
    return engine, AutoTrader(engine)


def _configure(trader: AutoTrader, **overrides: Any) -> BotConfig:
    """Apply the offline judge-test baseline config plus per-test overrides.

    Baseline: AI gate ON with the second judge ON at its contract defaults —
    individual tests flip ``use_ai``/``use_second_judge`` off as needed.

    Args:
        trader: The AutoTrader under test.
        **overrides: BotConfig fields to change on top of the baseline.

    Returns:
        The updated, persisted BotConfig.
    """
    updates: dict[str, Any] = {
        "watchlist": [TREND_SYMBOL],
        "source": SOURCE,
        "timeframe": TIMEFRAME,
        "use_ai": True,
        "min_ai_confidence": 60,
        "max_ai_calls_per_cycle": 3,
        "min_vote": 2,
        "allow_short": False,
        "use_second_judge": True,
        "second_judge_model": DEFAULT_JUDGE_MODEL,
        "judge_min_confidence": 55,
        # This module tests the judge pipeline; the Improvement Pack v3 cost
        # gate is disabled via its contract knob (0 disables) — the low-ATR
        # fixtures here would otherwise be cost-gated before any AI call.
        # The v3 gates have their own binding suite in
        # tests/test_improvements_traders.py.
        "cost_gate_multiple": 0.0,
    }
    updates.update(overrides)
    return trader.set_config(updates)


def _patch_lookup(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    replacement: Callable[..., Any],
    source_module: Any,
) -> None:
    """Patch a callable at its source module AND where auto_trader looked it up.

    Covers both import styles: ``from x import name`` (rebinds the name into
    the ``auto_trader`` module namespace) and ``import x; x.name(...)``
    (resolved on the source module at call time).

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        name: Attribute name to replace.
        replacement: The stub callable.
        source_module: The module that originally defines ``name``.
    """
    monkeypatch.setattr(source_module, name, replacement, raising=True)
    if hasattr(auto_trader_mod, name):
        monkeypatch.setattr(auto_trader_mod, name, replacement)


class _JudgeAwareAnalystStub:
    """Recording ``analyze_market`` replacement that dispatches on ``model``.

    Calls whose ``model`` argument equals the configured second-judge model
    return (or raise) the judge verdict; every other call — ``model=None`` or
    the primary/default model — returns (or raises) the primary verdict.
    """

    def __init__(
        self,
        judge_model: str,
        primary: tuple[str, int] | Exception = ("bullish", 90),
        judge: tuple[str, int] | Exception = ("bullish", 80),
    ) -> None:
        self.judge_model = judge_model
        self.primary = primary
        self.judge = judge
        #: Every call as (symbol, model) in call order.
        self.calls: list[tuple[str, str | None]] = []

    @property
    def primary_calls(self) -> list[tuple[str, str | None]]:
        """Calls served with the primary verdict (model != judge model)."""
        return [c for c in self.calls if c[1] != self.judge_model]

    @property
    def judge_calls(self) -> list[tuple[str, str | None]]:
        """Calls served with the judge verdict (model == judge model)."""
        return [c for c in self.calls if c[1] == self.judge_model]

    def __call__(
        self,
        symbol: str,
        timeframe: str = TIMEFRAME,
        df: pd.DataFrame | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> MarketAnalysis:
        self.calls.append((str(symbol), model))
        verdict = self.judge if model == self.judge_model else self.primary
        if isinstance(verdict, Exception):
            raise verdict
        sentiment, confidence = verdict
        return MarketAnalysis(
            sentiment=sentiment,  # type: ignore[arg-type]
            confidence=int(confidence),
            risk_commentary="synthetic stub commentary",
            key_indicators=[],
            reasoning="synthetic stub reasoning",
            model_used=str(model) if model else "stub-primary",
            symbol=str(symbol),
            timeframe=str(timeframe),
        )


def _patch_dual_analyst(
    monkeypatch: pytest.MonkeyPatch,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    primary: tuple[str, int] | Exception = ("bullish", 90),
    judge: tuple[str, int] | Exception = ("bullish", 80),
) -> _JudgeAwareAnalystStub:
    """Install the model-dispatching ``analyze_market`` stub everywhere.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        judge_model: Model string that selects the judge verdict.
        primary: Verdict for primary-analyst calls (tuple or exception).
        judge: Verdict for second-judge calls (tuple or exception).

    Returns:
        The recording stub.
    """
    stub = _JudgeAwareAnalystStub(judge_model, primary=primary, judge=judge)
    _patch_lookup(monkeypatch, "analyze_market", stub, analyst_mod)
    return stub


def _forbid_analyst(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``analyze_market`` with a stub that fails the test when called."""

    def _boom(*args: Any, **kwargs: Any) -> MarketAnalysis:
        raise AssertionError("analyze_market must not be called when use_ai=False")

    _patch_lookup(monkeypatch, "analyze_market", _boom, analyst_mod)


def _patch_list_models(
    monkeypatch: pytest.MonkeyPatch, models: list[str]
) -> list[str]:
    """Patch ``OllamaClient.list_models`` (class level) to a canned catalogue.

    Class-level patching covers every lookup/instantiation style auto_trader
    may use (``OllamaClient().list_models()`` however the class was imported).

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        models: Installed model names the stub reports.

    Returns:
        The canned list (defensive copy handed to each call).
    """
    installed = list(models)
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: list(installed))
    return installed


# --------------------------------------------------------------------------- #
# Database helpers                                                             #
# --------------------------------------------------------------------------- #


def _bot_activity(action: str | None = None) -> list[dict[str, Any]]:
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


def _insert_sentiment(
    symbol: str,
    score: int,
    age_ms: int = 0,
    sentiment: str | None = None,
    confidence: int = 80,
) -> None:
    """Seed one ``coin_sentiment`` row aged ``age_ms`` into the past.

    Args:
        symbol: Coin the sentiment row belongs to.
        score: Sentiment score in [-100, 100].
        age_ms: How far in the past the row's timestamp lies.
        sentiment: Label; derived from the score's sign when None.
        confidence: Sentiment confidence in [0, 100].
    """
    if sentiment is None:
        sentiment = "bearish" if score < 0 else "bullish" if score > 0 else "neutral"
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO coin_sentiment "
            "(ts, symbol, sentiment, score, confidence, summary, headlines, model) "
            "VALUES (?, ?, ?, ?, ?, ?, '[]', 'test')",
            (
                utc_now_ms() - int(age_ms),
                symbol,
                sentiment,
                int(score),
                int(confidence),
                f"seeded test sentiment for {symbol}",
            ),
        )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Autouse guards                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _pin_contract_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every settings knob the sizing/risk assertions depend on.

    A local ``.env`` may override any of these; the tests do exact math with
    the contract defaults, so pin them per test (monkeypatch restores).
    """
    monkeypatch.setattr(settings, "commission_rate", 0.001)
    monkeypatch.setattr(settings, "slippage_rate", 0.0005)
    monkeypatch.setattr(settings, "risk_per_trade", 0.01)
    monkeypatch.setattr(settings, "max_open_positions", 5)
    monkeypatch.setattr(settings, "daily_loss_limit", 0.03)
    monkeypatch.setattr(settings, "circuit_breaker_drawdown", 0.10)
    monkeypatch.setattr(settings, "atr_stop_multiplier", 2.0)
    monkeypatch.setattr(settings, "atr_take_profit_multiplier", 3.0)


@pytest.fixture(autouse=True)
def _offline_market_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """No-op every candle refresh so cycles run purely off the seeded cache.

    Patches ``update_symbol`` at its source module and on the auto_trader
    proxy (covers both import styles the bot supports).
    """

    def _noop_update(*args: Any, **kwargs: Any) -> int:
        return 0

    _patch_lookup(monkeypatch, "update_symbol", _noop_update, data_service_mod)


@pytest.fixture(autouse=True)
def _no_llm_unless_stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every Ollama code path canned; fail loudly on a raw chat call.

    ``analyze_market`` is always stubbed per test, so nothing should ever
    reach ``OllamaClient.chat`` — if it does, the test fails immediately.
    ``list_models`` defaults to a catalogue that INCLUDES the judge model;
    availability-rule tests override it with ``_patch_list_models`` (the same
    function-scoped monkeypatch, so the later setattr wins).
    """

    def _forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError(
            "OllamaClient.chat must not be called here — second-judge tests "
            "are offline and stub analyze_market/list_models explicitly."
        )

    monkeypatch.setattr(OllamaClient, "chat", _forbidden)
    monkeypatch.setattr(OllamaClient, "is_available", lambda self: True)
    _patch_list_models(monkeypatch, [settings.ollama_model, DEFAULT_JUDGE_MODEL])


# --------------------------------------------------------------------------- #
# Fixture self-check — the synthetic frame must vote the way the tests assume  #
# --------------------------------------------------------------------------- #


def test_trending_frame_votes_self_check() -> None:
    """Pin the last-bar vote sum every ranking/entry assertion relies on."""
    breakdown = VotingStrategy().vote_breakdown(add_features(_trending_df()))
    last = breakdown.iloc[-1]
    votes = {name: int(last[name]) for name in MEMBER_STRATEGIES}
    assert sum(votes.values()) == TREND_VOTE_SUM, (
        f"trending frame must vote sum {TREND_VOTE_SUM} on its last bar, "
        f"got {votes}"
    )
    assert votes["trend_following"] == 1  # unambiguously long-side


# --------------------------------------------------------------------------- #
# BotConfig additions: defaults, clamping, persistence                         #
# --------------------------------------------------------------------------- #


def test_config_defaults_clamps_and_persistence() -> None:
    """New judge fields default per contract; set_config clamps the weight.

    ``sentiment_rank_weight`` must be clamped to [0.0, 2.0] in ``set_config``
    (not rejected), and all four new fields must persist like existing ones.
    """
    engine, trader = _make_trader()

    # Contract defaults.
    cfg = trader.get_config()
    assert cfg.use_second_judge is True
    assert cfg.second_judge_model == DEFAULT_JUDGE_MODEL
    assert cfg.judge_min_confidence == 55
    assert cfg.sentiment_rank_weight == pytest.approx(0.5)

    # Clamp above / below / in-range.
    assert trader.set_config(
        {"sentiment_rank_weight": 99.0}
    ).sentiment_rank_weight == pytest.approx(2.0)
    assert trader.set_config(
        {"sentiment_rank_weight": -3.0}
    ).sentiment_rank_weight == pytest.approx(0.0)
    assert trader.set_config(
        {"sentiment_rank_weight": 1.25}
    ).sentiment_rank_weight == pytest.approx(1.25)

    updated = trader.set_config(
        {
            "use_second_judge": False,
            "second_judge_model": "custom/Fin-R1:Q8",
            "judge_min_confidence": 70,
        }
    )
    assert updated.use_second_judge is False
    assert updated.second_judge_model == "custom/Fin-R1:Q8"
    assert updated.judge_min_confidence == 70

    # Persistence: a brand-new trader over the same DB sees the same values.
    cfg2 = AutoTrader(engine).get_config()
    assert cfg2.use_second_judge is False
    assert cfg2.second_judge_model == "custom/Fin-R1:Q8"
    assert cfg2.judge_min_confidence == 70
    assert cfg2.sentiment_rank_weight == pytest.approx(1.25)


# --------------------------------------------------------------------------- #
# Second judge — agreement path                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "installed_name",
    [DEFAULT_JUDGE_MODEL, "mychen76/Fin-R1:latest"],
    ids=["exact_tag", "prefix_match"],
)
def test_agreeing_judge_enters_with_ai_second(
    monkeypatch: pytest.MonkeyPatch, installed_name: str
) -> None:
    """Primary and judge both bullish → entry with ``ai_second`` populated.

    The availability rule matches the installed name by prefix before ':'
    too, so a differently-tagged install of the same model still counts.
    """
    _seed_symbol(TREND_SYMBOL, _trending_df())
    stub = _patch_dual_analyst(
        monkeypatch, primary=("bullish", 90), judge=("bullish", 70)
    )
    _patch_list_models(monkeypatch, [settings.ollama_model, installed_name])

    engine, trader = _make_trader()
    _configure(trader)

    result = trader.run_cycle()
    assert isinstance(result, dict)

    # Both models were consulted exactly once, judge with the configured name.
    assert len(stub.primary_calls) == 1
    assert stub.judge_calls == [(TREND_SYMBOL, DEFAULT_JUDGE_MODEL)]

    positions = engine.get_positions("open")
    assert len(positions) == 1
    assert positions[0]["symbol"] == TREND_SYMBOL
    assert positions[0]["side"] == "long"

    enters = _bot_activity("enter")
    assert len(enters) == 1
    detail = enters[0]["detail"]
    # `ai` keeps the primary result's shape (UI back-compat).
    assert {"sentiment", "confidence", "reasoning", "risk_commentary"} <= set(
        detail["ai"].keys()
    )
    assert detail["ai"]["sentiment"] == "bullish"
    assert int(detail["ai"]["confidence"]) == 90
    # New transparency field: the second judge's verdict.
    ai_second = detail["ai_second"]
    assert ai_second is not None
    assert {"sentiment", "confidence", "model"} <= set(ai_second.keys())
    assert ai_second["sentiment"] == "bullish"
    assert int(ai_second["confidence"]) == 70
    assert ai_second["model"] == DEFAULT_JUDGE_MODEL


# --------------------------------------------------------------------------- #
# Second judge — veto paths                                                    #
# --------------------------------------------------------------------------- #


def test_judge_direction_disagreement_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A confident judge on the WRONG side vetoes the entry (judge_disagreement)."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    stub = _patch_dual_analyst(
        monkeypatch, primary=("bullish", 90), judge=("bearish", 80)
    )

    engine, trader = _make_trader()
    _configure(trader)

    trader.run_cycle()

    assert len(stub.judge_calls) == 1  # the primary gate passed first
    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []

    skips = [item for item in _bot_activity("skip") if item["symbol"] == TREND_SYMBOL]
    assert skips, "a judge veto must be logged as 'skip'"
    reason = str(skips[-1]["detail"].get("reason") or "")
    assert reason.startswith("judge_disagreement"), reason
    assert "bearish" in reason  # `<model> said <sentiment> at <conf>%`


def test_judge_low_confidence_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    """Judge agrees on direction but below judge_min_confidence → skip."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    stub = _patch_dual_analyst(
        monkeypatch, primary=("bullish", 90), judge=("bullish", 40)
    )

    engine, trader = _make_trader()
    _configure(trader, judge_min_confidence=55)

    trader.run_cycle()

    assert len(stub.judge_calls) == 1
    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []

    skips = [item for item in _bot_activity("skip") if item["symbol"] == TREND_SYMBOL]
    assert skips, "a low-confidence judge verdict must be logged as 'skip'"
    reason = str(skips[-1]["detail"].get("reason") or "")
    assert reason.startswith("judge_disagreement"), reason
    assert "40" in reason  # `... at <conf>%`


# --------------------------------------------------------------------------- #
# Second judge — availability rule                                             #
# --------------------------------------------------------------------------- #


def test_missing_judge_model_enters_with_one_warning_and_null_ai_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Judge model not installed → ONE 'error' row per cycle, entries proceed.

    The feature waits for the model pull to finish: the judge is never called,
    every survivor of the primary gate enters, ``ai_second`` stays null, and
    exactly one warning 'error' activity row is written for the whole cycle
    (two candidates here prove the per-cycle dedup).
    """
    frame = _trending_df()
    _seed_symbol(TREND_SYMBOL, frame)
    _seed_symbol(TREND_TWIN_SYMBOL, frame)
    stub = _patch_dual_analyst(
        monkeypatch, primary=("bullish", 90), judge=("bullish", 99)
    )
    _patch_list_models(monkeypatch, [settings.ollama_model])  # no Fin-R1 at all

    engine, trader = _make_trader()
    _configure(trader, watchlist=[TREND_SYMBOL, TREND_TWIN_SYMBOL])

    result = trader.run_cycle()
    assert isinstance(result, dict)

    assert stub.judge_calls == [], "an uninstalled judge model must not be called"
    assert len(stub.primary_calls) == 2

    positions = {str(p["symbol"]) for p in engine.get_positions("open")}
    assert positions == {TREND_SYMBOL, TREND_TWIN_SYMBOL}

    enters = _bot_activity("enter")
    assert {item["symbol"] for item in enters} == {TREND_SYMBOL, TREND_TWIN_SYMBOL}
    for item in enters:
        assert item["detail"].get("ai_second") is None
        assert item["detail"]["ai"]["sentiment"] == "bullish"  # primary intact

    errors = _bot_activity("error")
    assert len(errors) == 1, (
        f"exactly ONE warning 'error' row per cycle is mandated, got {errors}"
    )
    blob = json.dumps(errors[0], default=str).lower()
    assert "judge" in blob or "fin-r1" in blob


def test_judge_ollama_error_skips_candidate_with_error_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed judge that raises OllamaError conservatively skips the entry."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    stub = _patch_dual_analyst(
        monkeypatch,
        primary=("bullish", 90),
        judge=OllamaError("judge model unreachable (synthetic)"),
    )

    engine, trader = _make_trader()
    _configure(trader)

    result = trader.run_cycle()  # must not raise
    assert isinstance(result, dict)

    assert len(stub.judge_calls) == 1
    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []

    errors = [item for item in _bot_activity("error") if item["symbol"] == TREND_SYMBOL]
    assert errors, "the judge failure must be logged as 'error'"


# --------------------------------------------------------------------------- #
# Second judge — two-pass evaluation (GPU model-swap thrash avoidance)         #
# --------------------------------------------------------------------------- #


def test_two_pass_evaluation_runs_all_primary_calls_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All primary-gate calls precede every judge call (one model swap only)."""
    frame = _trending_df()
    _seed_symbol(TREND_SYMBOL, frame)
    _seed_symbol(TREND_TWIN_SYMBOL, frame)
    stub = _patch_dual_analyst(
        monkeypatch, primary=("bullish", 90), judge=("bullish", 80)
    )

    engine, trader = _make_trader()
    _configure(trader, watchlist=[TREND_SYMBOL, TREND_TWIN_SYMBOL])

    trader.run_cycle()

    assert len(stub.primary_calls) == 2
    assert len(stub.judge_calls) == 2
    first_judge_idx = min(
        i for i, (_, model) in enumerate(stub.calls) if model == DEFAULT_JUDGE_MODEL
    )
    last_primary_idx = max(
        i for i, (_, model) in enumerate(stub.calls) if model != DEFAULT_JUDGE_MODEL
    )
    assert last_primary_idx < first_judge_idx, (
        f"primary pass must finish before the judge pass starts: {stub.calls}"
    )

    positions = {str(p["symbol"]) for p in engine.get_positions("open")}
    assert positions == {TREND_SYMBOL, TREND_TWIN_SYMBOL}


# --------------------------------------------------------------------------- #
# Sentiment-boosted ranking                                                    #
# --------------------------------------------------------------------------- #


def test_fresh_sentiment_bonus_reorders_equal_vote_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh positive sentiment lifts an equal-vote candidate to the top.

    Both symbols share identical candles (equal |vote_sum| and |momentum_10|),
    so without a bonus the stable ranking keeps the watchlist order and the
    FIRST symbol would take the single free slot. A fresh +60 score on the
    second symbol adds ``1.0 × (+1) × 0.6 = 0.6`` rank points and must win.
    """
    frame = _trending_df()
    _seed_symbol(RANK_A_SYMBOL, frame)
    _seed_symbol(RANK_B_SYMBOL, frame)
    _insert_sentiment(RANK_B_SYMBOL, score=60, age_ms=60_000)  # 1 minute old
    _forbid_analyst(monkeypatch)
    monkeypatch.setattr(settings, "max_open_positions", 1)  # single free slot

    engine, trader = _make_trader()
    _configure(
        trader,
        watchlist=[RANK_A_SYMBOL, RANK_B_SYMBOL],
        use_ai=False,
        use_second_judge=False,
        sentiment_rank_weight=1.0,
    )

    trader.run_cycle()

    positions = engine.get_positions("open")
    assert len(positions) == 1
    assert positions[0]["symbol"] == RANK_B_SYMBOL, (
        "the sentiment-boosted candidate must outrank its equal-vote twin"
    )

    enters = _bot_activity("enter")
    assert len(enters) == 1 and enters[0]["symbol"] == RANK_B_SYMBOL
    detail = enters[0]["detail"]
    # Contract transparency: rank_score/sentiment_bonus in the enter detail.
    assert float(detail["sentiment_bonus"]) == pytest.approx(0.6)
    assert float(detail["rank_score"]) == pytest.approx(TREND_VOTE_SUM + 0.6)


def test_stale_sentiment_gives_zero_bonus_and_no_reorder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sentiment older than 3h contributes bonus 0 — no reordering happens."""
    frame = _trending_df()
    _seed_symbol(RANK_A_SYMBOL, frame)
    _seed_symbol(RANK_B_SYMBOL, frame)
    _insert_sentiment(RANK_B_SYMBOL, score=60, age_ms=4 * _HOUR_MS)  # stale
    _forbid_analyst(monkeypatch)
    monkeypatch.setattr(settings, "max_open_positions", 1)  # single free slot

    engine, trader = _make_trader()
    _configure(
        trader,
        watchlist=[RANK_A_SYMBOL, RANK_B_SYMBOL],
        use_ai=False,
        use_second_judge=False,
        sentiment_rank_weight=1.0,
    )

    trader.run_cycle()

    positions = engine.get_positions("open")
    assert len(positions) == 1
    assert positions[0]["symbol"] == RANK_A_SYMBOL, (
        "stale sentiment must not lift the second candidate over the first"
    )

    enters = _bot_activity("enter")
    assert len(enters) == 1 and enters[0]["symbol"] == RANK_A_SYMBOL
    detail = enters[0]["detail"]
    assert float(detail["sentiment_bonus"]) == pytest.approx(0.0)
    assert float(detail["rank_score"]) == pytest.approx(float(TREND_VOTE_SUM))
