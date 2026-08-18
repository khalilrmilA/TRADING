"""Derivatives-positioning intel — objective crowd signals from public APIs.

PAPER TRADING ONLY. The :class:`DerivativesAgent` fetches **keyless public**
Binance USD-M futures statistics for the watchlist — funding rate (with its
percentile against the last ~30 days), 24h open-interest change, the
top-trader long/short position ratio, and the taker buy/sell ratio — and
persists one ``coin_derivatives`` row per coin per run.

Why these signals: unlike Telegram/Discord "signal groups" (whose calls are
unauditable and frequently adversarial), these are exchange-published
positioning facts. Funding-rate extremes and taker-flow imbalance have the
strongest practitioner evidence as crowd-positioning context; the ratios are
weaker and serve as context only. Nothing here gates trades — the snapshot is
surfaced to the analyst prompt via the auto-trader's news context and to the
dashboard, so the LLM (and the human) can weigh it against the technicals.

Design decisions (mirroring the sentiment agent's contract):

* **No LLM calls and no network in the trading cycle** — the agent fetches on
  the intel schedule and traders read the freshest DB row only.
* Failures are contained per endpoint AND per symbol: any HTTP/parse error
  logs a warning, leaves that field ``None`` (or skips the coin when nothing
  was fetched) and moves on — :meth:`DerivativesAgent.run` NEVER raises.
* The table is module-owned (``CREATE IF NOT EXISTS`` on first use, like the
  coach's playbook table) so the platform keeps working without a schema
  migration; rows older than 7 days are pruned each run.
* Geo note: fapi.binance.com serves no-auth data but blocks some regions
  (HTTP 451/403). A blocked network degrades to "no derivatives intel",
  never to an error.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from typing import Any

import requests

from backend.database.db import get_conn, init_db, utc_now_ms

logger = logging.getLogger(__name__)

__all__ = ["DerivativesAgent"]

_BASE = "https://fapi.binance.com"
_TIMEOUT = (5.0, 8.0)  # (connect, read) seconds per endpoint call
_RETENTION_MS = 7 * 24 * 3600 * 1000
#: Freshness horizon for reads — refreshes run every ~30 min with intel.
_FRESH_MS = 2 * 3600 * 1000

_DDL = """
CREATE TABLE IF NOT EXISTS coin_derivatives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    funding_rate REAL,
    funding_pctile REAL,
    oi_change_24h_pct REAL,
    top_long_short_ratio REAL,
    taker_buy_sell_ratio REAL,
    mark_price REAL,
    note TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_coin_derivatives_symbol
    ON coin_derivatives (symbol, ts DESC);
"""


def _get_json(session: requests.Session, path: str, params: dict[str, Any]) -> Any:
    """One GET against the public futures API; raises on HTTP/parse errors."""
    resp = session.get(f"{_BASE}{path}", params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


class DerivativesAgent:
    """Fetch + persist keyless Binance futures positioning for the watchlist."""

    def __init__(self) -> None:
        init_db()
        self._ensure_table()
        self._session = requests.Session()

    @staticmethod
    def _ensure_table() -> None:
        conn = get_conn()
        try:
            with conn:
                conn.executescript(_DDL)
        finally:
            conn.close()

    # ------------------------------------------------------------------ fetch
    def fetch_snapshot(self, symbol: str) -> dict[str, Any] | None:
        """Best-effort positioning snapshot for one symbol.

        Returns ``None`` when EVERY endpoint failed (symbol not on futures,
        network blocked, ...); otherwise a dict whose unavailable fields are
        ``None``.
        """
        funding_rate = mark_price = None
        try:
            premium = _get_json(
                self._session, "/fapi/v1/premiumIndex", {"symbol": symbol}
            )
            funding_rate = _safe_float(premium.get("lastFundingRate"))
            mark_price = _safe_float(premium.get("markPrice"))
        except Exception as exc:  # noqa: BLE001 — degrade per endpoint
            logger.warning("premiumIndex failed for %s: %s", symbol, exc)

        funding_pctile = None
        try:
            # ~30 days of 8h funding events.
            history = _get_json(
                self._session, "/fapi/v1/fundingRate", {"symbol": symbol, "limit": 90}
            )
            rates = [
                r for r in (_safe_float(h.get("fundingRate")) for h in history)
                if r is not None
            ]
            if funding_rate is not None and len(rates) >= 10:
                below = sum(1 for r in rates if r <= funding_rate)
                funding_pctile = round(100.0 * below / len(rates), 1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fundingRate history failed for %s: %s", symbol, exc)

        oi_change_24h_pct = None
        try:
            oi = _get_json(
                self._session,
                "/futures/data/openInterestHist",
                {"symbol": symbol, "period": "1h", "limit": 25},
            )
            if len(oi) >= 2:
                first = _safe_float(oi[0].get("sumOpenInterest"))
                last = _safe_float(oi[-1].get("sumOpenInterest"))
                if first and last and first > 0:
                    oi_change_24h_pct = round(100.0 * (last - first) / first, 2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("openInterestHist failed for %s: %s", symbol, exc)

        top_ratio = None
        try:
            rows = _get_json(
                self._session,
                "/futures/data/topLongShortPositionRatio",
                {"symbol": symbol, "period": "1h", "limit": 1},
            )
            if rows:
                top_ratio = _safe_float(rows[-1].get("longShortRatio"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("topLongShortPositionRatio failed for %s: %s", symbol, exc)

        taker_ratio = None
        try:
            rows = _get_json(
                self._session,
                "/futures/data/takerlongshortRatio",
                {"symbol": symbol, "period": "1h", "limit": 1},
            )
            if rows:
                taker_ratio = _safe_float(rows[-1].get("buySellRatio"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("takerlongshortRatio failed for %s: %s", symbol, exc)

        values = (funding_rate, funding_pctile, oi_change_24h_pct, top_ratio,
                  taker_ratio, mark_price)
        if all(v is None for v in values):
            return None
        return {
            "symbol": symbol,
            "funding_rate": funding_rate,
            "funding_pctile": funding_pctile,
            "oi_change_24h_pct": oi_change_24h_pct,
            "top_long_short_ratio": top_ratio,
            "taker_buy_sell_ratio": taker_ratio,
            "mark_price": mark_price,
            "note": self._note(funding_rate, funding_pctile, oi_change_24h_pct,
                               top_ratio, taker_ratio),
        }

    @staticmethod
    def _note(
        funding: float | None,
        pctile: float | None,
        oi_delta: float | None,
        top_ratio: float | None,
        taker: float | None,
    ) -> str:
        """Compact factual line for prompts/dashboards (no advice, facts only)."""
        parts: list[str] = []
        if funding is not None:
            line = f"funding {funding * 100:+.4f}%/8h"
            if pctile is not None:
                line += f" ({pctile:.0f}th pctile vs 30d)"
            parts.append(line)
        if oi_delta is not None:
            parts.append(f"open interest {oi_delta:+.1f}% over 24h")
        if top_ratio is not None:
            parts.append(f"top-trader long/short {top_ratio:.2f}")
        if taker is not None:
            parts.append(f"taker buy/sell {taker:.2f}")
        return "; ".join(parts)

    # -------------------------------------------------------------------- run
    def run(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch + persist snapshots for ``symbols``. Never raises."""
        scored = failed = 0
        now = utc_now_ms()
        for symbol in symbols:
            try:
                snap = self.fetch_snapshot(symbol)
            except Exception:  # noqa: BLE001 — belt and braces
                logger.exception("Derivatives snapshot blew up for %s", symbol)
                snap = None
            if snap is None:
                failed += 1
                continue
            try:
                conn = get_conn()
                try:
                    with conn:
                        conn.execute(
                            "INSERT INTO coin_derivatives (ts, symbol, funding_rate,"
                            " funding_pctile, oi_change_24h_pct,"
                            " top_long_short_ratio, taker_buy_sell_ratio,"
                            " mark_price, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                now, symbol, snap["funding_rate"],
                                snap["funding_pctile"], snap["oi_change_24h_pct"],
                                snap["top_long_short_ratio"],
                                snap["taker_buy_sell_ratio"], snap["mark_price"],
                                snap["note"],
                            ),
                        )
                finally:
                    conn.close()
                scored += 1
            except sqlite3.Error:
                logger.exception("Failed to persist derivatives row for %s", symbol)
                failed += 1
        self._prune(now)
        summary = {"requested": len(symbols), "scored": scored, "failed": failed}
        logger.info("Derivatives intel run: %s", summary)
        return summary

    @staticmethod
    def _prune(now_ms: int) -> None:
        try:
            conn = get_conn()
            try:
                with conn:
                    conn.execute(
                        "DELETE FROM coin_derivatives WHERE ts < ?",
                        (now_ms - _RETENTION_MS,),
                    )
            finally:
                conn.close()
        except sqlite3.Error:
            logger.exception("Derivatives retention pruning failed")

    # ------------------------------------------------------------------ reads
    @staticmethod
    def latest_map(max_age_ms: int = _FRESH_MS) -> dict[str, dict[str, Any]]:
        """Freshest row per symbol, empty dict on any failure (never raises)."""
        try:
            DerivativesAgent._ensure_table()
            conn = get_conn()
            try:
                rows = conn.execute(
                    "SELECT ts, symbol, funding_rate, funding_pctile,"
                    " oi_change_24h_pct, top_long_short_ratio,"
                    " taker_buy_sell_ratio, mark_price, note"
                    " FROM coin_derivatives WHERE ts >= ?"
                    " ORDER BY ts ASC",
                    (utc_now_ms() - max_age_ms,),
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.Error:
            logger.exception("Derivatives read failed")
            return {}
        out: dict[str, dict[str, Any]] = {}
        for row in rows:  # ascending ts — later rows overwrite older ones
            out[row["symbol"]] = dict(row)
        return out

    @staticmethod
    def latest(symbol: str, max_age_ms: int = _FRESH_MS) -> dict[str, Any] | None:
        """Freshest row for one symbol within ``max_age_ms``, else ``None``."""
        return DerivativesAgent.latest_map(max_age_ms).get(symbol)
