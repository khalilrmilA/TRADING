"""Data service: incremental cache updates, cache-first reads, background refresh.

PAPER TRADING ONLY. This service maintains the local SQLite candle cache from
*public* market-data endpoints (see :mod:`backend.data.fetchers`) and serves
canonical OHLCV frames to the rest of the platform. No real-money execution,
no private API keys.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pandas as pd

from backend.data import fetchers
from backend.database import db
from config.settings import settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

__all__ = ["update_symbol", "get_ohlcv", "start_scheduler", "stop_scheduler"]

#: Intraday timeframes fetch huge row counts over a full year — cap their
#: default lookback to keep updates fast and the cache reasonably sized.
_LOOKBACK_CAP_DAYS: dict[str, int] = {"1m": 7, "5m": 30}

_scheduler: "BackgroundScheduler | None" = None
_scheduler_lock = threading.Lock()
_db_ready = False


def _ensure_db() -> None:
    """Make sure the schema exists before touching the cache (idempotent)."""
    global _db_ready
    if not _db_ready:
        db.init_db()
        _db_ready = True


def update_symbol(
    source: str,
    symbol: str,
    timeframe: str,
    lookback_days: int = 365,
) -> int:
    """Incrementally update the candle cache for one (source, symbol, timeframe).

    Fetches from the last cached candle timestamp (re-fetching the final,
    possibly-partial candle) — or from ``lookback_days`` ago when nothing is
    cached — up to now, then upserts into SQLite.

    For very fine timeframes the lookback is clamped (1m → 7 days, 5m → 30
    days) to avoid pulling millions of rows.

    Args:
        source: ``"binance" | "bybit" | "yahoo"``.
        symbol: Source-native symbol.
        timeframe: One of ``1m, 5m, 15m, 1h, 4h, 1d``.
        lookback_days: Initial history depth when the cache is empty.

    Returns:
        Number of genuinely new rows added to the cache (replaced/refreshed
        candles are not counted).

    Raises:
        fetchers.DataFetchError: When the upstream fetch fails unrecoverably.
        ValueError: On an unknown source or timeframe.
    """
    _ensure_db()
    cap = _LOOKBACK_CAP_DAYS.get(timeframe)
    effective_lookback = lookback_days
    if cap is not None and lookback_days > cap:
        logger.info(
            "Clamping %s lookback from %d to %d days for %s:%s",
            timeframe, lookback_days, cap, source, symbol,
        )
        effective_lookback = cap

    now = datetime.now(timezone.utc)
    coverage = db.ohlcv_coverage(source, symbol, timeframe)
    if coverage is not None:
        # Start at the last cached candle so a previously-partial final candle
        # gets refreshed (upsert replaces it).
        start = min(coverage[1], now)
    else:
        start = now - timedelta(days=effective_lookback)

    frame = fetchers.fetch(source, symbol, timeframe, start=start, end=now)
    if frame.empty:
        logger.info("No new candles for %s:%s:%s", source, symbol, timeframe)
        return 0

    existing = db.load_ohlcv(source, symbol, timeframe, start=frame.index[0], end=frame.index[-1])
    new_rows = int((~frame.index.isin(existing.index)).sum())
    db.upsert_ohlcv(source, symbol, timeframe, frame)
    logger.info(
        "Updated %s:%s:%s — fetched %d candles, %d new",
        source, symbol, timeframe, len(frame), new_rows,
    )
    return new_rows


def get_ohlcv(
    source: str,
    symbol: str,
    timeframe: str,
    limit: int = 500,
    with_features: bool = False,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return cached candles, fetching from the source only when needed.

    Cache-first: reads the local SQLite cache; when the cache is empty (or
    ``refresh=True``) it runs :func:`update_symbol` first.

    Args:
        source: ``"binance" | "bybit" | "yahoo"``.
        symbol: Source-native symbol.
        timeframe: One of ``1m, 5m, 15m, 1h, 4h, 1d``.
        limit: Return at most the *limit* most recent candles (ascending).
        with_features: Enrich the frame via
            ``backend.indicators.features.add_features``.
        refresh: Force an incremental update before reading the cache.

    Returns:
        Canonical OHLCV DataFrame (feature-enriched when requested; possibly
        empty when no data is available at all).
    """
    _ensure_db()
    if refresh:
        update_symbol(source, symbol, timeframe)
    frame = db.load_ohlcv(source, symbol, timeframe, limit=limit)
    if frame.empty and not refresh:
        logger.info("Cache miss for %s:%s:%s — fetching from source", source, symbol, timeframe)
        update_symbol(source, symbol, timeframe)
        frame = db.load_ohlcv(source, symbol, timeframe, limit=limit)

    if with_features and not frame.empty:
        # Deferred import: this module must stay importable even while the
        # indicators module is not written/installed yet.
        from backend.indicators.features import add_features

        frame = add_features(frame)
    return frame


def _update_all_cached() -> None:
    """Scheduler job: refresh every (source, symbol, timeframe) in the cache.

    Each series is updated inside its own try/except so one bad symbol never
    kills the scheduler run.
    """
    try:
        items = db.list_cached()
    except Exception:
        logger.exception("Scheduled refresh could not list cached series")
        return
    if not items:
        logger.debug("Scheduled refresh: cache is empty, nothing to update")
        return

    total_new = 0
    failures = 0
    for item in items:
        try:
            total_new += update_symbol(item["source"], item["symbol"], item["timeframe"])
        except Exception:
            failures += 1
            logger.exception(
                "Scheduled update failed for %s:%s:%s",
                item.get("source"), item.get("symbol"), item.get("timeframe"),
            )
    logger.info(
        "Scheduled refresh finished: %d series, %d new rows, %d failures",
        len(items), total_new, failures,
    )


def start_scheduler() -> None:
    """Start the background data-refresh scheduler (idempotent).

    Re-updates everything reported by ``db.list_cached()`` every
    ``settings.data_update_interval_minutes`` minutes using an APScheduler
    ``BackgroundScheduler``. Safe to call when already running.
    """
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None:
            logger.debug("Data scheduler already running")
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError:
            logger.error("APScheduler is not installed — background data refresh disabled")
            return

        scheduler = BackgroundScheduler()
        scheduler.add_job(
            _update_all_cached,
            trigger="interval",
            minutes=settings.data_update_interval_minutes,
            id="ohlcv_refresh",
            coalesce=True,
            max_instances=1,
            misfire_grace_time=300,
        )
        scheduler.start()
        _scheduler = scheduler
        logger.info(
            "Data scheduler started (refresh every %d minutes)",
            settings.data_update_interval_minutes,
        )


def stop_scheduler() -> None:
    """Stop the background data-refresh scheduler (idempotent, never raises)."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            return
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            logger.exception("Error while stopping the data scheduler")
        finally:
            _scheduler = None
        logger.info("Data scheduler stopped")
