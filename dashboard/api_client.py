"""HTTP client for the Trading AI Platform backend API.

The dashboard NEVER imports backend modules — all communication goes through
this thin ``requests`` wrapper. PAPER TRADING ONLY: this client only talks to
the local research API; it never places real orders and never touches private
exchange credentials.

Base URL comes from the ``API_URL`` environment variable (default
``http://127.0.0.1:8000``) so the same code works natively on Windows and
inside Docker (``API_URL=http://backend:8000``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

API_BASE_URL: str = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

#: Timeout (seconds) for fast GET endpoints.
GET_TIMEOUT: float = 10.0
#: Timeout (seconds) for long-running POSTs (AI analysis, backtests, data fetch).
LONG_POST_TIMEOUT: float = 300.0
#: Timeout (seconds) for quick paper-trading actions.
ACTION_TIMEOUT: float = 60.0
#: Timeout (seconds) for a full synchronous bot cycle (scans + LLM confirmations).
BOT_CYCLE_TIMEOUT: float = 600.0


class ApiError(Exception):
    """Raised for any failure talking to the backend API.

    Attributes:
        status_code: HTTP status code when the server responded with an error,
            otherwise ``None`` (network failure, timeout, bad JSON, ...).
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: float = GET_TIMEOUT,
) -> dict[str, Any]:
    """Perform one HTTP request and return the decoded JSON body.

    Args:
        method: HTTP verb (``"GET"`` / ``"POST"``).
        path: Route path starting with ``/``.
        params: Optional query parameters (``None`` values are dropped).
        json_body: Optional JSON request body (``None`` values are dropped).
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        ApiError: On connection failure, timeout, non-2xx status, or a
            non-JSON response body.
    """
    url = f"{API_BASE_URL}{path}"
    if params is not None:
        params = {k: v for k, v in params.items() if v is not None}
    if json_body is not None:
        json_body = {k: v for k, v in json_body.items() if v is not None}
    try:
        response = requests.request(
            method, url, params=params, json=json_body, timeout=timeout
        )
    except requests.RequestException as exc:
        logger.warning("API request failed: %s %s (%s)", method, url, exc)
        raise ApiError(f"Cannot reach API at {url}: {exc}") from exc

    if response.status_code >= 400:
        detail: str
        try:
            body = response.json()
            detail = str(body.get("detail", body))
        except ValueError:
            detail = response.text[:300]
        logger.warning(
            "API error %s on %s %s: %s", response.status_code, method, path, detail
        )
        raise ApiError(
            f"API returned {response.status_code} for {method} {path}: {detail}",
            status_code=response.status_code,
        )

    try:
        payload: dict[str, Any] = response.json()
    except ValueError as exc:
        raise ApiError(
            f"API returned non-JSON response for {method} {path}",
            status_code=response.status_code,
        ) from exc
    return payload


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """GET ``path`` with the standard fast timeout."""
    return _request("GET", path, params=params, timeout=GET_TIMEOUT)


def _post(
    path: str,
    json_body: dict[str, Any] | None = None,
    timeout: float = LONG_POST_TIMEOUT,
) -> dict[str, Any]:
    """POST ``json_body`` to ``path``."""
    return _request("POST", path, json_body=json_body, timeout=timeout)


# ---------------------------------------------------------------------------
# Health / models
# ---------------------------------------------------------------------------


def get_health() -> dict[str, Any]:
    """GET /health — API + Ollama + DB status."""
    return _get("/health")


def get_models() -> dict[str, Any]:
    """GET /api/models — configured and installed Ollama models."""
    return _get("/api/models")


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


def fetch_data(
    source: str, symbol: str, timeframe: str, lookback_days: int = 365
) -> dict[str, Any]:
    """POST /api/data/fetch — pull candles from the public exchange endpoint."""
    return _post(
        "/api/data/fetch",
        {
            "source": source,
            "symbol": symbol,
            "timeframe": timeframe,
            "lookback_days": lookback_days,
        },
    )


def get_ohlcv(
    source: str,
    symbol: str,
    timeframe: str,
    limit: int = 500,
    features: bool = False,
) -> dict[str, Any]:
    """GET /api/data/ohlcv — cached candles, optionally feature-enriched."""
    return _get(
        "/api/data/ohlcv",
        {
            "source": source,
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit,
            "features": "true" if features else "false",
        },
    )


def get_cached() -> dict[str, Any]:
    """GET /api/data/cached — inventory of locally cached (source, symbol, tf)."""
    return _get("/api/data/cached")


# ---------------------------------------------------------------------------
# AI analysis
# ---------------------------------------------------------------------------


def run_analysis(
    source: str, symbol: str, timeframe: str, model: str | None = None
) -> dict[str, Any]:
    """POST /api/analysis — run the local-LLM market analysis (long timeout)."""
    return _post(
        "/api/analysis",
        {"source": source, "symbol": symbol, "timeframe": timeframe, "model": model},
        timeout=LONG_POST_TIMEOUT,
    )


def get_analysis_history(
    symbol: str | None = None, limit: int = 50
) -> dict[str, Any]:
    """GET /api/analysis/history — previously persisted analyses."""
    return _get("/api/analysis/history", {"symbol": symbol, "limit": limit})


# ---------------------------------------------------------------------------
# Strategies / backtesting
# ---------------------------------------------------------------------------


def get_strategies() -> dict[str, Any]:
    """GET /api/strategies — registry of available strategies."""
    return _get("/api/strategies")


def run_backtest(
    source: str,
    symbol: str,
    timeframe: str,
    strategy: str,
    limit: int = 2000,
    params: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """POST /api/backtest — run one strategy backtest on cached data."""
    body: dict[str, Any] = {
        "source": source,
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy,
        "limit": limit,
    }
    if params:
        body["params"] = params
    if config:
        body["config"] = config
    return _post("/api/backtest", body, timeout=LONG_POST_TIMEOUT)


def run_walkforward(
    source: str,
    symbol: str,
    timeframe: str,
    strategy: str,
    n_splits: int = 4,
    train_ratio: float = 0.7,
    limit: int = 3000,
) -> dict[str, Any]:
    """POST /api/backtest/walkforward — rolling out-of-sample evaluation."""
    return _post(
        "/api/backtest/walkforward",
        {
            "source": source,
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
            "n_splits": n_splits,
            "train_ratio": train_ratio,
            "limit": limit,
        },
        timeout=LONG_POST_TIMEOUT,
    )


def get_backtest_runs(limit: int = 50) -> dict[str, Any]:
    """GET /api/backtest/runs — persisted backtest run summaries."""
    return _get("/api/backtest/runs", {"limit": limit})


# ---------------------------------------------------------------------------
# Paper trading (simulation only — no real orders, ever)
# ---------------------------------------------------------------------------


def submit_paper_order(
    symbol: str,
    side: str,
    order_type: str,
    qty: float | None = None,
    limit_price: float | None = None,
    stop_price: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    source: str = "binance",
    timeframe: str = "1h",
) -> dict[str, Any]:
    """POST /api/paper/orders — submit a SIMULATED order to the paper engine."""
    return _post(
        "/api/paper/orders",
        {
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "qty": qty,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "source": source,
            "timeframe": timeframe,
        },
        timeout=ACTION_TIMEOUT,
    )


def get_paper_orders(status: str | None = None) -> dict[str, Any]:
    """GET /api/paper/orders — simulated orders, optionally filtered by status."""
    return _get("/api/paper/orders", {"status": status})


def get_paper_positions(status: str = "open") -> dict[str, Any]:
    """GET /api/paper/positions — simulated positions."""
    return _get("/api/paper/positions", {"status": status})


def get_paper_trades(limit: int = 100) -> dict[str, Any]:
    """GET /api/paper/trades — completed simulated trades."""
    return _get("/api/paper/trades", {"limit": limit})


def get_paper_portfolio() -> dict[str, Any]:
    """GET /api/paper/portfolio — equity, PnL, drawdown, halt flags."""
    return _get("/api/paper/portfolio")


def get_paper_equity(limit: int = 1000) -> dict[str, Any]:
    """GET /api/paper/equity — equity snapshot history."""
    return _get("/api/paper/equity", {"limit": limit})


def paper_tick(
    symbol: str,
    source: str = "binance",
    timeframe: str = "1h",
    refresh: bool = False,
) -> dict[str, Any]:
    """POST /api/paper/tick — process latest candle through the paper engine."""
    return _post(
        "/api/paper/tick",
        {"symbol": symbol, "source": source, "timeframe": timeframe, "refresh": refresh},
        timeout=LONG_POST_TIMEOUT if refresh else ACTION_TIMEOUT,
    )


def close_paper_position(position_id: int) -> dict[str, Any]:
    """POST /api/paper/positions/{id}/close — market-close a simulated position."""
    return _post(f"/api/paper/positions/{position_id}/close", {}, timeout=ACTION_TIMEOUT)


def reset_paper() -> dict[str, Any]:
    """POST /api/paper/reset — wipe paper state, restore initial capital."""
    return _post("/api/paper/reset", {}, timeout=ACTION_TIMEOUT)


# ---------------------------------------------------------------------------
# AI auto-trader bot (paper only — trades the simulated account, never real)
# ---------------------------------------------------------------------------


def get_bot_status() -> dict[str, Any]:
    """GET /api/bot/status — running flag, cycle timestamps, equity, watchlist."""
    return _get("/api/bot/status")


def get_bot_config() -> dict[str, Any]:
    """GET /api/bot/config — current persisted BotConfig."""
    return _get("/api/bot/config")


def set_bot_config(updates: dict[str, Any]) -> dict[str, Any]:
    """POST /api/bot/config — apply a partial config update, return new config.

    Args:
        updates: Subset of BotConfig fields to change (e.g. ``{"watchlist":
            [...], "min_vote": 3}``). ``None`` values are dropped.

    Returns:
        The full updated BotConfig as a dict.
    """
    return _post("/api/bot/config", updates, timeout=ACTION_TIMEOUT)


def start_bot() -> dict[str, Any]:
    """POST /api/bot/start — enable the scheduled bot cycle; returns status()."""
    return _post("/api/bot/start", {}, timeout=ACTION_TIMEOUT)


def stop_bot() -> dict[str, Any]:
    """POST /api/bot/stop — disable the scheduled bot cycle; returns status()."""
    return _post("/api/bot/stop", {}, timeout=ACTION_TIMEOUT)


def run_bot_cycle() -> dict[str, Any]:
    """POST /api/bot/run-once — run one full cycle synchronously.

    May take several minutes when AI confirmation is enabled, hence the long
    ``BOT_CYCLE_TIMEOUT``.

    Returns:
        The cycle summary dict from ``AutoTrader.run_cycle()``.
    """
    return _post("/api/bot/run-once", {}, timeout=BOT_CYCLE_TIMEOUT)


def get_bot_activity(limit: int = 100) -> dict[str, Any]:
    """GET /api/bot/activity — latest bot decisions (newest first).

    Args:
        limit: Maximum number of activity rows to return.

    Returns:
        ``{"items": [{"ts": iso, "symbol": str, "action": str, "detail": dict}]}``.
    """
    return _get("/api/bot/activity", {"limit": limit})
