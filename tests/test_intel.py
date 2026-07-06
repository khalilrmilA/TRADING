"""Tests for the intel layer — backend/intel/* + backend/paper_trading/coach.py.

Everything runs offline and deterministically:
    * NO network — the autouse conftest fixture blocks sockets outright, and
      every fetcher test additionally monkeypatches ``requests.get`` with a
      URL-routed stub serving canned payloads (a small Google News RSS XML
      string plus a malformed one, a reddit hot ``.json`` dict and a Fear &
      Greed JSON response).
    * NO Ollama — ``OllamaClient.chat`` is patched at the CLASS level (so any
      import/instantiation style is covered). Outside an explicit stub it
      FAILS the test if called.
    * The news module's 10-minute TTL cache is cleared around every test via
      its public ``clear_cache()`` so canned routes never leak across tests.

Coverage (CONTRACTS.md, "backend/intel/ + backend/paper_trading/coach.py"):
    * Fetchers: normalized dicts, ``[]``/``None`` on any failure (malformed
      XML, HTTP errors, network exceptions), TTL cache serves the second call
      without a second ``requests.get``.
    * SentimentAgent: batch scoring persists clamped rows plus one GLOBAL
      row; a batch whose (mocked) chat returns invalid JSON — or raises an
      OllamaError — is skipped without raising.
    * Coach: <3 closed trades → no playbook changes; out-of-range LLM
      proposals are clamped; applied changes write ``coach_tune`` activity;
      LLM failure changes nothing.
    * Trader integration: a benched coin is skipped by the AutoTrader
      (reason ``coach_bench``); ``size_multiplier`` scales the submitted qty;
      fresh sentiment score -60 blocks a long entry (reason
      ``news_negative``); stale (>3h) sentiment does not block.

The autouse conftest fixtures give every test a fresh tmp database and pin
``settings.initial_capital`` to the contract default of 100,000.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pytest
import requests

import backend.data.service as data_service_mod
import backend.intel.news as news_mod
import backend.paper_trading.auto_trader as auto_trader_mod
from backend.ai.ollama_client import OllamaClient, OllamaError
from backend.database.db import get_conn, upsert_ohlcv, utc_now_ms
from backend.intel.news import (
    COIN_NAMES,
    fetch_fear_greed,
    fetch_google_news,
    fetch_reddit_hot,
)
from backend.intel.sentiment import GLOBAL_SYMBOL, SentimentAgent
from backend.paper_trading.auto_trader import AutoTrader, BotConfig
from backend.paper_trading.coach import Coach, load_playbook_map
from backend.paper_trading.engine import PaperTradingEngine
from backend.strategies.voting import VotingStrategy
from backend.indicators.features import add_features
from config.settings import settings

SOURCE = "binance"
TIMEFRAME = "1h"

#: Symbols used by the trader-integration tests (synthetic candles).
TREND_SYMBOL = "TRENDUSDT"
SIZE_FULL_SYMBOL = "SIZEAUSDT"
SIZE_HALF_SYMBOL = "SIZEBUSDT"
NEWS_SYMBOL = "NEWSUSDT"

#: Symbols used by the Coach tests (closed trades seeded directly).
COACH_HI = "COACHAUSDT"
COACH_LO = "COACHBUSDT"
COACH_FEW = "COACHFUSDT"

#: Empirically pinned last-bar vote sum of the trending frame (self-checked
#: below in ``test_trending_frame_votes_self_check``; same frame as
#: tests/test_auto_trader.py).
TREND_VOTE_SUM = 2

#: 3h freshness horizon of the sentiment gate (contract: stale > 3h = no gate).
_HOUR_MS = 3_600_000

_RSS_TITLES = (
    "Bitcoin climbs above $120,000 as ETF inflows accelerate",
    "Exchange outage rattles crypto traders for a second day",
    "Analysts split on the next leg for major altcoins",
)

_REDDIT_TITLES = (
    "Daily discussion: crypto markets are wild again",
    "Bitcoin ETF inflows hit a new weekly record",
    "Which altcoins are you watching this cycle?",
)

_MALFORMED_RSS = '<?xml version="1.0"?><rss><channel><item><title>Broken feed'


# --------------------------------------------------------------------------- #
# Canned network payloads                                                      #
# --------------------------------------------------------------------------- #


def _rss_xml(titles: tuple[str, ...] = _RSS_TITLES) -> str:
    """A small, valid Google News RSS search feed with one item per title."""
    items = "".join(
        f"""
    <item>
      <title>{title}</title>
      <link>https://news.example.com/{i}</link>
      <guid isPermaLink="false">unit-test-guid-{i}</guid>
      <pubDate>Fri, 03 Jul 2026 1{i}:00:00 GMT</pubDate>
      <description>{title} — canned summary.</description>
      <source url="https://news.example.com">Example News</source>
    </item>"""
        for i, title in enumerate(titles)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0"><channel>'
        "<title>unit test feed</title>"
        "<link>https://news.google.com/rss</link>"
        f"{items}"
        "</channel></rss>"
    )


def _reddit_json(titles: tuple[str, ...] = _REDDIT_TITLES) -> dict[str, Any]:
    """A canned reddit ``hot.json`` listing dict with one post per title."""
    children = [
        {
            "kind": "t3",
            "data": {
                "title": title,
                "score": 1_200 - 100 * i,
                "ups": 1_200 - 100 * i,
                "num_comments": 300 - 10 * i,
                "permalink": f"/r/CryptoCurrency/comments/unittest{i}/",
                "url": f"https://www.reddit.com/r/CryptoCurrency/comments/unittest{i}/",
                "created_utc": 1_751_500_000 + i * 600,
                "subreddit": "CryptoCurrency",
                "selftext": "",
                "stickied": False,
            },
        }
        for i, title in enumerate(titles)
    ]
    return {
        "kind": "Listing",
        "data": {"dist": len(children), "children": children, "after": None},
    }


_FNG_JSON: dict[str, Any] = {
    "name": "Fear and Greed Index",
    "data": [
        {
            "value": "27",
            "value_classification": "Fear",
            "timestamp": "1751673600",
            "time_until_update": "3600",
        }
    ],
    "metadata": {"error": None},
}


# --------------------------------------------------------------------------- #
# requests.get stub (URL-routed canned responses, records every call)          #
# --------------------------------------------------------------------------- #


class _StubResponse:
    """Minimal stand-in for ``requests.Response`` serving a canned payload."""

    def __init__(
        self,
        status_code: int = 200,
        text: str = "",
        json_data: Any = None,
    ) -> None:
        if json_data is not None and not text:
            text = json.dumps(json_data)
        self.status_code = int(status_code)
        self.text = text
        self.content = text.encode("utf-8")
        self.ok = 200 <= self.status_code < 300
        self.headers: dict[str, str] = {}
        self.encoding = "utf-8"
        self._json_data = json_data

    def json(self, **kwargs: Any) -> Any:
        """Return the canned JSON payload; ValueError when there is none."""
        if self._json_data is None:
            raise ValueError(f"No JSON in canned response (HTTP {self.status_code})")
        return self._json_data

    def raise_for_status(self) -> None:
        """Mirror ``requests.Response.raise_for_status`` for non-2xx codes."""
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code} (canned)")


class _RequestsRouter:
    """``requests.get`` replacement routing by URL substring.

    Unrouted URLs raise ``requests.ConnectionError`` — the fetchers must
    swallow that (contract: never raise), and it guarantees no test silently
    depends on an endpoint it did not stub.
    """

    def __init__(self, routes: dict[str, _StubResponse | Exception]) -> None:
        self.routes = dict(routes)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> _StubResponse:
        url = ""
        for arg in args:
            if isinstance(arg, str) and arg.startswith(("http://", "https://")):
                url = arg
                break
        if not url:
            url = str(kwargs.get("url", ""))
        self.calls.append((url, kwargs))
        for needle, outcome in self.routes.items():
            if needle in url:
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        raise requests.ConnectionError(f"no canned route for URL: {url!r}")

    def count(self, needle: str) -> int:
        """How many recorded calls targeted a URL containing ``needle``."""
        return sum(1 for url, _ in self.calls if needle in url)


def _install_requests(
    monkeypatch: pytest.MonkeyPatch,
    routes: dict[str, _StubResponse | Exception] | None = None,
) -> _RequestsRouter:
    """Replace ``requests.get`` with a recording URL router."""
    router = _RequestsRouter(routes or {})
    monkeypatch.setattr(requests, "get", router)
    return router


def _all_source_routes() -> dict[str, _StubResponse | Exception]:
    """Happy-path routes for all three public sources (fresh responses)."""
    return {
        "news.google.com": _StubResponse(text=_rss_xml()),
        "reddit.com": _StubResponse(json_data=_reddit_json()),
        "alternative.me": _StubResponse(json_data=_FNG_JSON),
    }


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
        self.prompts: list[str] = []

    def __call__(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        json_mode: bool = True,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> str:
        self.prompts.append(str(prompt))
        if self.error is not None:
            raise self.error
        return self.response


def _patch_chat(
    monkeypatch: pytest.MonkeyPatch,
    response: str = "",
    error: Exception | None = None,
) -> _ChatStub:
    """Replace ``OllamaClient.chat`` with a canned stub (records prompts)."""
    stub = _ChatStub(response=response, error=error)
    monkeypatch.setattr(OllamaClient, "chat", stub)
    return stub


def _sentiment_response(entries: list[dict[str, Any]]) -> str:
    """A SentimentAgent batch reply: ``{"results": [...]}`` JSON string."""
    return json.dumps({"results": entries})


def _coach_response(coins: list[dict[str, Any]]) -> str:
    """A Coach batch reply: ``{"coins": [...]}`` JSON string."""
    return json.dumps({"coins": coins})


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


def _symbol_activity_blob(symbol: str) -> str:
    """Every activity detail for one symbol as a single searchable string."""
    return json.dumps(
        [item for item in _bot_activity() if item["symbol"] == symbol],
        default=str,
    )


def _latest_sentiment_row(symbol: str) -> Any:
    """Most recent ``coin_sentiment`` row for ``symbol`` (None when absent)."""
    conn = get_conn()
    try:
        return conn.execute(
            "SELECT * FROM coin_sentiment WHERE symbol=? "
            "ORDER BY ts DESC, id DESC LIMIT 1",
            (symbol,),
        ).fetchone()
    finally:
        conn.close()


def _sentiment_count(symbol: str) -> int:
    """Number of persisted ``coin_sentiment`` rows for ``symbol``."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM coin_sentiment WHERE symbol=?", (symbol,)
        ).fetchone()
    finally:
        conn.close()
    return int(row["n"])


def _insert_sentiment(
    symbol: str,
    score: int,
    age_ms: int = 0,
    sentiment: str | None = None,
    confidence: int = 80,
) -> None:
    """Seed one ``coin_sentiment`` row aged ``age_ms`` into the past."""
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


def _insert_playbook(
    symbol: str,
    bench: int = 0,
    side_bias: str = "both",
    size_multiplier: float = 1.0,
    min_vote_override: int | None = None,
    reasoning: str = "seeded by test",
) -> None:
    """Seed one ``coin_playbook`` row directly (bypasses the Coach)."""
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO coin_playbook "
            "(symbol, updated_at, bench, side_bias, size_multiplier, "
            "min_vote_override, reasoning, stats) VALUES (?, ?, ?, ?, ?, ?, ?, '{}')",
            (
                symbol,
                utc_now_ms(),
                int(bench),
                side_bias,
                float(size_multiplier),
                min_vote_override,
                reasoning,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _playbook_row(symbol: str) -> Any:
    """The ``coin_playbook`` row for ``symbol`` (None when absent)."""
    conn = get_conn()
    try:
        return conn.execute(
            "SELECT * FROM coin_playbook WHERE symbol=?", (symbol,)
        ).fetchone()
    finally:
        conn.close()


def _persist_bot_config(watchlist: list[str]) -> None:
    """Persist a bot config (the Coach's watchlist source) without an engine."""
    config = BotConfig(
        watchlist=watchlist, source=SOURCE, timeframe=TIMEFRAME, use_ai=False
    )
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO account_state (key, value) "
            "VALUES ('bot_config', ?)",
            (json.dumps(config.model_dump()),),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_closed_positions(symbol: str, pnls: list[float]) -> None:
    """Seed closed ``paper_positions`` rows — the Coach's trade evidence.

    One long position per PnL value, closed within the last few hours (so no
    account-reset cutoff can exclude them).

    Args:
        symbol: The coin the trades belong to.
        pnls: Realized PnL per closed trade.
    """
    now = utc_now_ms()
    conn = get_conn()
    try:
        for i, pnl in enumerate(pnls):
            entry_price = 100.0
            exit_price = entry_price + float(pnl)
            opened = now - (i + 2) * _HOUR_MS
            closed = now - (i + 1) * _HOUR_MS
            conn.execute(
                "INSERT INTO paper_positions "
                "(symbol, source, timeframe, side, qty, entry_price, stop_loss, "
                "take_profit, opened_at, closed_at, exit_price, pnl, status, "
                "last_price) "
                "VALUES (?, ?, ?, 'long', 1.0, ?, NULL, NULL, ?, ?, ?, ?, "
                "'closed', ?)",
                (
                    symbol,
                    SOURCE,
                    TIMEFRAME,
                    entry_price,
                    opened,
                    closed,
                    exit_price,
                    float(pnl),
                    exit_price,
                ),
            )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Synthetic market data for the trader-integration tests (seeded RNG —         #
# identical to tests/test_auto_trader.py, whose vote sums are pinned there)    #
# --------------------------------------------------------------------------- #


def _make_ohlcv(close: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
    """Wrap a close/volume path into a canonical hourly UTC OHLCV frame.

    The frame ends at the most recent CLOSED hourly candle so the bot's
    closed-candle and data-freshness gates treat it as live data.
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
    warm-up, steady climb, decisive +2% breakout close on a 4x volume spike.
    Self-checked in ``test_trending_frame_votes_self_check``.
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


def _make_trader() -> tuple[PaperTradingEngine, AutoTrader]:
    """Build a fresh engine + trader pair against the per-test tmp database."""
    engine = PaperTradingEngine()
    return engine, AutoTrader(engine)


def _configure(trader: AutoTrader, **overrides: Any) -> BotConfig:
    """Apply the offline test baseline config plus per-test overrides."""
    updates: dict[str, Any] = {
        "watchlist": [TREND_SYMBOL],
        "source": SOURCE,
        "timeframe": TIMEFRAME,
        "use_ai": False,
        "min_vote": 2,
        "allow_short": False,
        # This module tests the intel (sentiment/playbook) integration; the
        # Improvement Pack v3 cost gate is disabled via its contract knob
        # (0 disables) — the low-ATR fixtures here would otherwise be
        # cost-gated before the behaviors under test run. The v3 gates have
        # their own binding suite in tests/test_improvements_traders.py.
        "cost_gate_multiple": 0.0,
    }
    updates.update(overrides)
    return trader.set_config(updates)


# --------------------------------------------------------------------------- #
# Autouse guards                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _fresh_news_cache() -> Iterator[None]:
    """Clear the news module's 10-min TTL cache around every test.

    Failure sentinels are cached too (by design), so without this a canned
    failure/success from one test would leak into the next.
    """
    news_mod.clear_cache()
    yield
    news_mod.clear_cache()


@pytest.fixture(autouse=True)
def _pin_contract_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every settings knob the sizing/risk assertions depend on.

    A local ``.env`` may override any of these; the integration tests do
    exact sizing math with the contract defaults, so pin them per test
    (monkeypatch restores).
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

    monkeypatch.setattr(data_service_mod, "update_symbol", _noop_update)
    if hasattr(auto_trader_mod, "update_symbol"):
        monkeypatch.setattr(auto_trader_mod, "update_symbol", _noop_update)


@pytest.fixture(autouse=True)
def _no_llm_unless_stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything calls the LLM outside an explicit chat stub.

    Sentiment/Coach tests overwrite this patch with ``_patch_chat`` (same
    function-scoped monkeypatch, so the later setattr wins); the trader
    integration tests run with ``use_ai=False`` and must never reach Ollama.
    """

    def _forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError(
            "OllamaClient.chat must not be called here — intel tests are "
            "offline and must stub the chat call explicitly."
        )

    monkeypatch.setattr(OllamaClient, "chat", _forbidden)
    monkeypatch.setattr(OllamaClient, "is_available", lambda self: True)
    monkeypatch.setattr(
        OllamaClient, "list_models", lambda self: [settings.ollama_model]
    )


# --------------------------------------------------------------------------- #
# COIN_NAMES sanity                                                            #
# --------------------------------------------------------------------------- #


def test_coin_names_covers_watchlist() -> None:
    """COIN_NAMES maps watchlist symbols to (full name, ticker) pairs."""
    assert tuple(COIN_NAMES["BTCUSDT"]) == ("Bitcoin", "BTC")
    assert tuple(COIN_NAMES["ETHUSDT"]) == ("Ethereum", "ETH")
    assert len(COIN_NAMES) >= 10  # at least the default 10-coin watchlist
    for symbol, pair in COIN_NAMES.items():
        assert symbol == symbol.upper()
        assert len(tuple(pair)) == 2
        assert all(isinstance(part, str) and part for part in pair)


# --------------------------------------------------------------------------- #
# fetch_google_news                                                            #
# --------------------------------------------------------------------------- #


def test_google_news_parses_normalized_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid RSS feed parses into normalized headline dicts."""
    _install_requests(
        monkeypatch, {"news.google.com": _StubResponse(text=_rss_xml())}
    )

    items = fetch_google_news("bitcoin parser check")

    assert isinstance(items, list) and len(items) == len(_RSS_TITLES)
    for item in items:
        assert isinstance(item, dict)
        assert {"title", "source", "url", "published"} <= set(item.keys())
    assert [item["title"] for item in items] == list(_RSS_TITLES)
    assert items[0]["source"] == "Example News"
    assert items[0]["url"] == "https://news.example.com/0"
    assert items[0]["published"]  # non-empty pubDate string


def test_google_news_respects_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """``limit`` caps the number of returned headlines."""
    _install_requests(
        monkeypatch, {"news.google.com": _StubResponse(text=_rss_xml())}
    )

    items = fetch_google_news("bitcoin limit check", limit=2)

    assert len(items) == 2
    assert [item["title"] for item in items] == list(_RSS_TITLES[:2])


def test_google_news_malformed_xml_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed XML degrades to [] — the fetcher never raises."""
    _install_requests(
        monkeypatch, {"news.google.com": _StubResponse(text=_MALFORMED_RSS)}
    )

    assert fetch_google_news("bitcoin malformed check") == []


def test_google_news_http_error_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An HTTP 500 (non-XML body) degrades to [] — the fetcher never raises."""
    _install_requests(
        monkeypatch,
        {
            "news.google.com": _StubResponse(
                status_code=500, text="Internal Server Error"
            )
        },
    )

    assert fetch_google_news("bitcoin http error check") == []


def test_google_news_network_error_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raised ConnectionError degrades to [] — the fetcher never raises."""
    _install_requests(
        monkeypatch,
        {"news.google.com": requests.ConnectionError("canned network outage")},
    )

    assert fetch_google_news("bitcoin network error check") == []


def test_google_news_ttl_cache_serves_second_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second identical call is served from cache — one requests.get only."""
    router = _install_requests(
        monkeypatch, {"news.google.com": _StubResponse(text=_rss_xml())}
    )

    first = fetch_google_news("bitcoin cache check")
    second = fetch_google_news("bitcoin cache check")

    assert first and second == first
    assert router.count("news.google.com") == 1


# --------------------------------------------------------------------------- #
# fetch_reddit_hot                                                             #
# --------------------------------------------------------------------------- #


def test_reddit_hot_parses_normalized_posts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid listing parses into normalized post dicts (titles preserved)."""
    _install_requests(
        monkeypatch, {"reddit.com": _StubResponse(json_data=_reddit_json())}
    )

    posts = fetch_reddit_hot("UnitTestParse")

    assert isinstance(posts, list) and len(posts) == len(_REDDIT_TITLES)
    for post in posts:
        assert isinstance(post, dict)
        assert "title" in post
    assert [post["title"] for post in posts] == list(_REDDIT_TITLES)


def test_reddit_hot_rate_limited_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 (no JSON body) degrades to [] — the fetcher never raises."""
    _install_requests(
        monkeypatch,
        {"reddit.com": _StubResponse(status_code=429, text="Too Many Requests")},
    )

    assert fetch_reddit_hot("UnitTestRateLimited") == []


def test_reddit_hot_network_error_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raised ConnectionError degrades to [] — the fetcher never raises."""
    _install_requests(
        monkeypatch,
        {"reddit.com": requests.ConnectionError("canned network outage")},
    )

    assert fetch_reddit_hot("UnitTestNetError") == []


# --------------------------------------------------------------------------- #
# fetch_fear_greed                                                             #
# --------------------------------------------------------------------------- #


def test_fear_greed_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any failure (network error here) degrades to None — never raises."""
    _install_requests(
        monkeypatch,
        {"alternative.me": requests.ConnectionError("canned network outage")},
    )

    assert fetch_fear_greed() is None


def test_fear_greed_parses_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The index parses to {value:int, classification:str} and is TTL-cached."""
    router = _install_requests(
        monkeypatch, {"alternative.me": _StubResponse(json_data=_FNG_JSON)}
    )

    first = fetch_fear_greed()
    second = fetch_fear_greed()

    assert first is not None
    assert first["value"] == 27 and isinstance(first["value"], int)
    assert first["classification"] == "Fear"
    assert second == first
    assert router.count("alternative.me") == 1  # second call came from cache


# --------------------------------------------------------------------------- #
# SentimentAgent                                                               #
# --------------------------------------------------------------------------- #


def test_sentiment_agent_persists_clamped_rows_and_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch run persists one clamped row per coin plus a GLOBAL row."""
    _install_requests(monkeypatch, _all_source_routes())
    stub = _patch_chat(
        monkeypatch,
        response=_sentiment_response(
            [
                {
                    "symbol": "BTCUSDT",
                    "sentiment": "bullish",
                    "score": 250,  # out of range → clamp to 100
                    "confidence": 170,  # out of range → clamp to 100
                    "summary": "moon mission (canned)",
                },
                {
                    "symbol": "ETHUSDT",
                    "sentiment": "bearish",
                    "score": -250,  # out of range → clamp to -100
                    "confidence": -5,  # out of range → clamp to 0
                    "summary": "doom spiral (canned)",
                },
            ]
        ),
    )

    agent = SentimentAgent()
    result = agent.run(["BTCUSDT", "ETHUSDT"])

    assert stub.prompts, "the agent must consult the (mocked) LLM"
    assert isinstance(result, dict)
    assert result["status"] == "ok"
    assert result["scored"] == 2
    assert result["failed_batches"] == 0

    btc = _latest_sentiment_row("BTCUSDT")
    assert btc is not None
    assert int(btc["score"]) == 100
    assert int(btc["confidence"]) == 100
    assert btc["sentiment"] == "bullish"
    assert "moon mission" in btc["summary"]
    headlines = json.loads(btc["headlines"])
    assert isinstance(headlines, list) and len(headlines) == len(_RSS_TITLES)

    eth = _latest_sentiment_row("ETHUSDT")
    assert eth is not None
    assert int(eth["score"]) == -100
    assert int(eth["confidence"]) == 0
    assert eth["sentiment"] == "bearish"

    assert _sentiment_count("BTCUSDT") == 1
    assert _sentiment_count("ETHUSDT") == 1

    global_row = _latest_sentiment_row(GLOBAL_SYMBOL)
    assert global_row is not None
    assert -100 <= int(global_row["score"]) <= 100
    assert 0 <= int(global_row["confidence"]) <= 100
    assert global_row["sentiment"] in ("bullish", "bearish", "neutral")

    # get_latest surfaces the same clamped values, JSON-safe.
    items = agent.get_latest("BTCUSDT")["items"]
    assert len(items) == 1 and items[0]["score"] == 100


def test_sentiment_agent_scores_in_batches_of_five(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seven coins are scored in two LLM batches (5 + 2), all rows persisted."""
    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "DOGEUSDT",
        "ADAUSDT",
    ]
    _install_requests(monkeypatch, _all_source_routes())
    # One canned reply listing every coin: each batch call picks out its own.
    stub = _patch_chat(
        monkeypatch,
        response=_sentiment_response(
            [
                {
                    "symbol": symbol,
                    "sentiment": "neutral",
                    "score": 10,
                    "confidence": 55,
                    "summary": f"canned batch entry for {symbol}",
                }
                for symbol in symbols
            ]
        ),
    )

    result = SentimentAgent().run(symbols)

    assert result["status"] == "ok"
    assert result["batches"] == 2
    assert len(stub.prompts) == 2
    assert result["scored"] == len(symbols)
    for symbol in symbols:
        assert _sentiment_count(symbol) == 1
    assert _latest_sentiment_row(GLOBAL_SYMBOL) is not None


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ("sorry, no JSON from me today ][", None),
        ("", OllamaError("model unreachable (synthetic)")),
    ],
    ids=["bad_json", "ollama_error"],
)
def test_sentiment_agent_skips_failed_batch_without_raising(
    monkeypatch: pytest.MonkeyPatch, response: str, error: Exception | None
) -> None:
    """A batch whose chat fails is skipped — no coin rows, no exception."""
    _install_requests(monkeypatch, _all_source_routes())
    _patch_chat(monkeypatch, response=response, error=error)

    agent = SentimentAgent()
    result = agent.run(["BTCUSDT", "ETHUSDT"])  # must not raise

    assert isinstance(result, dict)
    assert result["scored"] == 0
    assert result["failed_batches"] == 1
    assert _latest_sentiment_row("BTCUSDT") is None
    assert _latest_sentiment_row("ETHUSDT") is None


# --------------------------------------------------------------------------- #
# Coach                                                                        #
# --------------------------------------------------------------------------- #


def test_coach_no_change_with_fewer_than_three_trades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A coin with <3 closed trades gets NO playbook changes (no LLM review)."""
    _persist_bot_config([COACH_FEW])
    _seed_closed_positions(COACH_FEW, [5.0, -2.0])  # only 2 closed trades
    stub = _patch_chat(
        monkeypatch,
        response=_coach_response(
            [
                {
                    "symbol": COACH_FEW,
                    "bench": True,
                    "side_bias": "short_only",
                    "size_multiplier": 0.25,
                    "min_vote_override": 4,
                    "reasoning": "must never be applied — not enough evidence",
                }
            ]
        ),
    )

    coach = Coach()
    result = coach.run()  # must not raise

    assert isinstance(result, dict)
    assert result["insufficient"] == 1
    assert result["updated"] == 0
    assert stub.prompts == []  # too little evidence → no LLM call at all

    row = _playbook_row(COACH_FEW)
    if row is not None:  # an all-defaults row is also acceptable
        assert int(row["bench"]) == 0
        assert row["side_bias"] == "both"
        assert float(row["size_multiplier"]) == pytest.approx(1.0)
        assert row["min_vote_override"] is None
    assert _bot_activity("coach_tune") == []


def test_coach_clamps_llm_proposals_and_writes_coach_tune(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out-of-range proposals are clamped and every change logs coach_tune."""
    _persist_bot_config([COACH_HI, COACH_LO])
    # 20 closed trades: enough evidence for the v3 shrink-only coach gates
    # (COACH_SIZE_UP_MIN_TRADES=20, COACH_SIDE_BIAS_MIN_TRADES=10), so the
    # size-up to 1.5 and the long_only bias below stay permitted and this
    # test keeps pinning the pure bounds-clamping behavior. The evidence
    # gates themselves are covered in tests/test_improvements_risk_coach.py.
    _seed_closed_positions(COACH_HI, [12.0, -6.0, 9.0, -3.0] * 5)
    _seed_closed_positions(COACH_LO, [-8.0, -5.0, -7.0])  # 3 closed trades
    stub = _patch_chat(
        monkeypatch,
        response=_coach_response(
            [
                {
                    "symbol": COACH_HI,
                    "bench": False,
                    "side_bias": "long_only",
                    "size_multiplier": 9.0,  # above 1.5 → clamp to 1.5
                    "min_vote_override": 99,  # above 4 → clamp to 4
                    "reasoning": "wild upsize (must be clamped)",
                },
                {
                    "symbol": COACH_LO,
                    "bench": False,
                    "side_bias": "both",
                    "size_multiplier": 0.01,  # below 0.25 → clamp to 0.25
                    "min_vote_override": 0,  # below 2 → clamp to 2
                    "reasoning": "wild downsize (must be clamped)",
                },
            ]
        ),
    )

    coach = Coach()
    result = coach.run()

    assert stub.prompts, "the coach must consult the (mocked) LLM"
    assert result["status"] == "ok"
    assert result["updated"] == 2
    assert result["insufficient"] == 0

    hi = _playbook_row(COACH_HI)
    assert hi is not None
    assert float(hi["size_multiplier"]) == pytest.approx(1.5)
    assert int(hi["min_vote_override"]) == 4
    assert hi["side_bias"] == "long_only"
    assert int(hi["bench"]) == 0

    lo = _playbook_row(COACH_LO)
    assert lo is not None
    assert float(lo["size_multiplier"]) == pytest.approx(0.25)
    assert int(lo["min_vote_override"]) == 2

    # The traders' read path re-clamps and surfaces the same values.
    playbook = load_playbook_map()
    assert playbook[COACH_HI]["size_multiplier"] == pytest.approx(1.5)
    assert playbook[COACH_LO]["size_multiplier"] == pytest.approx(0.25)

    # Contract-mandated transparency: coach_tune rows with before/after.
    tunes = _bot_activity("coach_tune")
    tuned_symbols = {item["symbol"] for item in tunes}
    assert {COACH_HI, COACH_LO} <= tuned_symbols
    for item in tunes:
        assert {"before", "after", "reasoning"} <= set(item["detail"].keys())
    hi_tune = next(item for item in tunes if item["symbol"] == COACH_HI)
    assert hi_tune["detail"]["after"]["size_multiplier"] == pytest.approx(1.5)

    # get_playbook mirrors the persisted (clamped) state, JSON-safe.
    items = {entry["symbol"]: entry for entry in coach.get_playbook()}
    assert items[COACH_HI]["size_multiplier"] == pytest.approx(1.5)
    assert items[COACH_LO]["min_vote_override"] == 2


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ("no JSON, just vibes ][", None),
        ("", OllamaError("model unreachable (synthetic)")),
    ],
    ids=["bad_json", "ollama_error"],
)
def test_coach_llm_failure_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, response: str, error: Exception | None
) -> None:
    """An LLM failure aborts the batch quietly: playbook stays exactly as-is."""
    _persist_bot_config([COACH_HI])
    _seed_closed_positions(COACH_HI, [4.0, -2.0, 3.0])
    _insert_playbook(COACH_HI, size_multiplier=0.8, side_bias="long_only")
    _patch_chat(monkeypatch, response=response, error=error)

    coach = Coach()
    result = coach.run()  # must not raise

    assert isinstance(result, dict)
    assert result["updated"] == 0
    assert result["errors"] >= 1

    row = _playbook_row(COACH_HI)
    assert row is not None
    assert float(row["size_multiplier"]) == pytest.approx(0.8)
    assert row["side_bias"] == "long_only"
    assert _bot_activity("coach_tune") == []


# --------------------------------------------------------------------------- #
# Trader integration — playbook + sentiment gates inside AutoTrader            #
# --------------------------------------------------------------------------- #


def test_trending_frame_votes_self_check() -> None:
    """Pin the last-bar vote sum the integration tests rely on (+2 = entry)."""
    breakdown = VotingStrategy().vote_breakdown(add_features(_trending_df()))
    last = breakdown.iloc[-1]
    votes = {
        name: int(last[name]) for name in breakdown.columns if name != "ensemble"
    }
    assert sum(votes.values()) == TREND_VOTE_SUM, (
        f"trending frame must vote sum {TREND_VOTE_SUM} on its last bar, "
        f"got {votes}"
    )
    assert votes["trend_following"] == 1  # unambiguously long-side


def test_benched_coin_skipped_by_auto_trader() -> None:
    """A coach-benched coin is never entered; the skip cites coach_bench."""
    _seed_symbol(TREND_SYMBOL, _trending_df())
    _insert_playbook(TREND_SYMBOL, bench=1, reasoning="benched by test")

    engine, trader = _make_trader()
    _configure(trader)

    result = trader.run_cycle()

    assert isinstance(result, dict)
    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []
    assert "coach_bench" in _symbol_activity_blob(TREND_SYMBOL), (
        "the benched coin's skip must be logged with reason 'coach_bench'"
    )


def test_size_multiplier_scales_submitted_qty() -> None:
    """A 0.5 playbook size_multiplier halves the entered qty vs an untouched
    twin coin with identical candles (same cycle, same account)."""
    df = _trending_df()
    _seed_symbol(SIZE_FULL_SYMBOL, df)
    _seed_symbol(SIZE_HALF_SYMBOL, df)
    _insert_playbook(SIZE_HALF_SYMBOL, size_multiplier=0.5)

    engine, trader = _make_trader()
    _configure(trader, watchlist=[SIZE_FULL_SYMBOL, SIZE_HALF_SYMBOL])

    trader.run_cycle()

    positions = {str(p["symbol"]): p for p in engine.get_positions("open")}
    assert set(positions) == {SIZE_FULL_SYMBOL, SIZE_HALF_SYMBOL}

    qty_full = float(positions[SIZE_FULL_SYMBOL]["qty"])
    qty_half = float(positions[SIZE_HALF_SYMBOL]["qty"])
    assert qty_full > 0 and qty_half > 0
    # Tolerance covers the tiny equity drift between the two fills
    # (commission/slippage of the first entry shifts the second's caps).
    assert qty_half / qty_full == pytest.approx(0.5, rel=0.03)


def test_fresh_negative_sentiment_blocks_long_entry() -> None:
    """A fresh sentiment score of -60 blocks the long entry (news_negative)."""
    _seed_symbol(NEWS_SYMBOL, _trending_df())
    _insert_sentiment(NEWS_SYMBOL, score=-60, age_ms=60_000)  # 1 minute old

    engine, trader = _make_trader()
    _configure(trader, watchlist=[NEWS_SYMBOL])

    result = trader.run_cycle()

    assert isinstance(result, dict)
    assert engine.get_positions("open") == []
    assert _bot_activity("enter") == []
    assert "news_negative" in _symbol_activity_blob(NEWS_SYMBOL), (
        "the sentiment-gated skip must be logged with reason 'news_negative'"
    )


def test_stale_negative_sentiment_does_not_block_entry() -> None:
    """Sentiment older than 3h is ignored — the long entry goes through."""
    _seed_symbol(NEWS_SYMBOL, _trending_df())
    _insert_sentiment(NEWS_SYMBOL, score=-60, age_ms=4 * _HOUR_MS)  # stale

    engine, trader = _make_trader()
    _configure(trader, watchlist=[NEWS_SYMBOL])

    trader.run_cycle()

    positions = engine.get_positions("open")
    assert len(positions) == 1
    assert positions[0]["symbol"] == NEWS_SYMBOL
    assert positions[0]["side"] == "long"
    assert "news_negative" not in _symbol_activity_blob(NEWS_SYMBOL)
