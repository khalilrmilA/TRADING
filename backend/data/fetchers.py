"""Public market-data fetchers for Binance, Bybit and Yahoo Finance.

PAPER TRADING ONLY. This module talks exclusively to *public* market-data
endpoints — no authentication, no private API keys, no order placement of any
kind. It exists purely to download historical/recent OHLCV candles for
research, backtesting and simulated trading.

Every fetcher returns the canonical OHLCV frame defined in CONTRACTS.md:

- Index: ``pd.DatetimeIndex``, UTC tz-aware, name ``"timestamp"``, ascending, unique.
- Columns: ``open, high, low, close, volume`` — all ``float64``.

Fetchers raise :class:`DataFetchError` on unrecoverable network/data failures
and ``ValueError`` on invalid arguments (unknown source/timeframe).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

__all__ = [
    "DataFetchError",
    "fetch",
    "fetch_binance",
    "fetch_bybit",
    "fetch_yahoo",
    "TIMEFRAME_MS",
]


class DataFetchError(Exception):
    """Raised when market data cannot be fetched after all retries."""


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BYBIT_KLINES_URL = "https://api.bybit.com/v5/market/kline"

OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]

#: Candle duration per supported timeframe, in milliseconds.
TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

_BINANCE_INTERVALS: dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d",
}
_BYBIT_INTERVALS: dict[str, str] = {
    "1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D",
}
# Yahoo has no native 4h interval — it is fetched as 60m and resampled locally.
_YAHOO_INTERVALS: dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "4h": "60m", "1d": "1d",
}
# Yahoo only serves intraday candles for a limited trailing window (days).
_YAHOO_MAX_LOOKBACK_DAYS: dict[str, int] = {"1m": 7, "5m": 59, "15m": 59, "60m": 729}

_CONNECT_TIMEOUT = 10.0  # seconds
_READ_TIMEOUT = 30.0  # seconds
_MAX_RETRIES = 3  # additional attempts after the first request
_BACKOFF_BASE = 1.0  # seconds → 1s, 2s, 4s
_PAGE_SIZE = 1000  # max klines per request on both Binance and Bybit
_MAX_PAGES = 2000  # hard safety cap on pagination loops
_PAGE_PAUSE_S = 0.1  # courtesy pause between paginated requests

_HEADERS = {"User-Agent": "trading-ai-platform/1.0 (paper-trading research; public data only)"}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _validate_timeframe(timeframe: str) -> None:
    """Raise ``ValueError`` if *timeframe* is not one of the supported keys."""
    if timeframe not in TIMEFRAME_MS:
        valid = ", ".join(sorted(TIMEFRAME_MS, key=lambda k: TIMEFRAME_MS[k]))
        raise ValueError(f"Unknown timeframe {timeframe!r}; expected one of: {valid}")


def _as_utc(dt: datetime | pd.Timestamp) -> datetime:
    """Coerce a datetime/Timestamp to a tz-aware UTC ``datetime`` (naive → UTC)."""
    ts = pd.Timestamp(dt)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _utc_ms(dt: datetime | pd.Timestamp) -> int:
    """Epoch milliseconds (UTC) for a datetime/Timestamp (naive treated as UTC)."""
    return int(pd.Timestamp(_as_utc(dt)).value // 1_000_000)


def _now_ms() -> int:
    """Current UTC time as epoch milliseconds."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _empty_frame() -> pd.DataFrame:
    """Return an empty canonical OHLCV frame."""
    frame = pd.DataFrame({col: pd.Series(dtype="float64") for col in OHLCV_COLUMNS})
    frame.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return frame


def _make_frame(records: list[tuple[int, float, float, float, float, float]]) -> pd.DataFrame:
    """Build the canonical OHLCV frame from ``(ts_ms, o, h, l, c, v)`` tuples.

    Sorts ascending and drops duplicate timestamps (keeps the last occurrence),
    which also normalises Bybit's newest-first ordering.
    """
    if not records:
        return _empty_frame()
    frame = pd.DataFrame(records, columns=["timestamp", *OHLCV_COLUMNS])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame[OHLCV_COLUMNS].astype("float64")


def _request_json(url: str, params: dict[str, Any]) -> Any:
    """GET *url* and return parsed JSON, retrying on 429/5xx/timeouts.

    Uses a 10s connect / 30s read timeout and up to ``_MAX_RETRIES`` retries
    with exponential backoff (1s, 2s, 4s). Non-retryable HTTP errors (4xx other
    than 429) raise :class:`DataFetchError` immediately.

    Args:
        url: Full endpoint URL.
        params: Query-string parameters.

    Returns:
        The decoded JSON payload (list or dict).

    Raises:
        DataFetchError: When the request keeps failing after all retries or
            fails with a non-retryable status code.
    """
    last_error = ""
    for attempt in range(_MAX_RETRIES + 1):
        if attempt:
            delay = _BACKOFF_BASE * (2 ** (attempt - 1))
            logger.warning(
                "Retrying GET %s in %.1fs (retry %d/%d) after: %s",
                url, delay, attempt, _MAX_RETRIES, last_error,
            )
            time.sleep(delay)
        try:
            response = requests.get(
                url,
                params=params,
                headers=_HEADERS,
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
        except requests.RequestException as exc:  # timeouts, connection errors, ...
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if response.status_code == 429 or response.status_code >= 500:
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            continue
        if response.status_code != 200:
            raise DataFetchError(
                f"GET {url} failed with HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            last_error = f"invalid JSON body: {exc}"
            continue
    raise DataFetchError(f"GET {url} failed after {_MAX_RETRIES + 1} attempts: {last_error}")


def _resolve_range(
    timeframe: str,
    start: datetime | pd.Timestamp | None,
    end: datetime | pd.Timestamp | None,
    limit: int | None,
) -> tuple[int, int]:
    """Resolve ``(start_ms, end_ms)`` for exchange fetchers.

    When *start* is ``None`` the window is derived from *limit* recent candles
    ending at *end* (or now).
    """
    end_ms = _utc_ms(end) if end is not None else _now_ms()
    if start is not None:
        start_ms = _utc_ms(start)
    else:
        n_candles = int(limit) if limit else _PAGE_SIZE
        start_ms = end_ms - n_candles * TIMEFRAME_MS[timeframe]
    return start_ms, end_ms


# --------------------------------------------------------------------------- #
# Fetchers                                                                     #
# --------------------------------------------------------------------------- #


def fetch_binance(
    symbol: str,
    timeframe: str,
    start: datetime | pd.Timestamp | None = None,
    end: datetime | pd.Timestamp | None = None,
    limit: int | None = 1000,
) -> pd.DataFrame:
    """Fetch spot klines from Binance's public REST API (no API key).

    Paginates ``GET /api/v3/klines`` (1000 candles per request) to honour any
    start/end window larger than a single page.

    Args:
        symbol: Binance-native symbol, e.g. ``"BTCUSDT"``.
        timeframe: One of ``1m, 5m, 15m, 1h, 4h, 1d``.
        start: Inclusive UTC start of the window. When ``None``, the most
            recent *limit* candles are fetched.
        end: Inclusive UTC end of the window (default: now).
        limit: Number of most-recent candles when *start* is ``None``. When
            *start* is provided the full range is fetched and *limit* is
            ignored (pagination handles size).

    Returns:
        Canonical OHLCV DataFrame (possibly empty).

    Raises:
        DataFetchError: On unrecoverable network or exchange errors.
        ValueError: On an unknown timeframe.
    """
    _validate_timeframe(timeframe)
    interval_ms = TIMEFRAME_MS[timeframe]
    start_ms, end_ms = _resolve_range(timeframe, start, end, limit)
    if start_ms >= end_ms:
        return _empty_frame()

    records: list[tuple[int, float, float, float, float, float]] = []
    cursor = start_ms
    for _page in range(_MAX_PAGES):
        payload = _request_json(
            BINANCE_KLINES_URL,
            {
                "symbol": symbol.upper(),
                "interval": _BINANCE_INTERVALS[timeframe],
                "startTime": cursor,
                "endTime": end_ms,
                "limit": _PAGE_SIZE,
            },
        )
        if isinstance(payload, dict):  # error payloads are dicts: {"code":..,"msg":..}
            raise DataFetchError(f"Binance error for {symbol}: {payload.get('msg', payload)}")
        if not payload:
            break
        try:
            records.extend(
                (int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
                for k in payload
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise DataFetchError(f"Malformed Binance kline payload for {symbol}: {exc}") from exc
        if len(payload) < _PAGE_SIZE:
            break
        cursor = int(payload[-1][0]) + interval_ms
        if cursor > end_ms:
            break
        time.sleep(_PAGE_PAUSE_S)
    else:
        logger.warning("Binance pagination for %s hit the %d-page safety cap", symbol, _MAX_PAGES)

    frame = _make_frame(records)
    if start is None and limit:
        frame = frame.tail(int(limit))
    logger.info("Fetched %d Binance candles for %s %s", len(frame), symbol, timeframe)
    return frame


def fetch_bybit(
    symbol: str,
    timeframe: str,
    start: datetime | pd.Timestamp | None = None,
    end: datetime | pd.Timestamp | None = None,
    limit: int | None = 1000,
) -> pd.DataFrame:
    """Fetch spot klines from Bybit's public REST API (no API key).

    Paginates ``GET /v5/market/kline`` (category=spot). Bybit returns candles
    **newest-first**, so pagination walks backwards from *end* and the result
    is re-sorted ascending.

    Args:
        symbol: Bybit-native symbol, e.g. ``"BTCUSDT"``.
        timeframe: One of ``1m, 5m, 15m, 1h, 4h, 1d``.
        start: Inclusive UTC start of the window. When ``None``, the most
            recent *limit* candles are fetched.
        end: Inclusive UTC end of the window (default: now).
        limit: Number of most-recent candles when *start* is ``None``; ignored
            as a cap when *start* is provided.

    Returns:
        Canonical OHLCV DataFrame (possibly empty).

    Raises:
        DataFetchError: On unrecoverable network or exchange errors.
        ValueError: On an unknown timeframe.
    """
    _validate_timeframe(timeframe)
    start_ms, end_ms = _resolve_range(timeframe, start, end, limit)
    if start_ms >= end_ms:
        return _empty_frame()

    records: list[tuple[int, float, float, float, float, float]] = []
    cursor_end = end_ms
    for _page in range(_MAX_PAGES):
        payload = _request_json(
            BYBIT_KLINES_URL,
            {
                "category": "spot",
                "symbol": symbol.upper(),
                "interval": _BYBIT_INTERVALS[timeframe],
                "start": start_ms,
                "end": cursor_end,
                "limit": _PAGE_SIZE,
            },
        )
        if not isinstance(payload, dict) or payload.get("retCode") != 0:
            message = (
                payload.get("retMsg", "unknown error")
                if isinstance(payload, dict)
                else str(payload)[:200]
            )
            raise DataFetchError(f"Bybit error for {symbol}: {message}")
        rows = (payload.get("result") or {}).get("list") or []
        if not rows:
            break
        try:
            # Bybit rows are newest-first: [startTimeMs, open, high, low, close, volume, turnover]
            records.extend(
                (int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
                for k in rows
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise DataFetchError(f"Malformed Bybit kline payload for {symbol}: {exc}") from exc
        oldest_ms = int(rows[-1][0])
        if len(rows) < _PAGE_SIZE or oldest_ms <= start_ms:
            break
        cursor_end = oldest_ms - 1
        time.sleep(_PAGE_PAUSE_S)
    else:
        logger.warning("Bybit pagination for %s hit the %d-page safety cap", symbol, _MAX_PAGES)

    frame = _make_frame(records)  # sorting ascending reverses Bybit's newest-first order
    if not frame.empty:
        lo = pd.Timestamp(start_ms, unit="ms", tz="UTC")
        hi = pd.Timestamp(end_ms, unit="ms", tz="UTC")
        frame = frame[(frame.index >= lo) & (frame.index <= hi)]
    if start is None and limit:
        frame = frame.tail(int(limit))
    logger.info("Fetched %d Bybit candles for %s %s", len(frame), symbol, timeframe)
    return frame


def fetch_yahoo(
    symbol: str,
    timeframe: str,
    start: datetime | pd.Timestamp | None = None,
    end: datetime | pd.Timestamp | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Fetch candles from Yahoo Finance via ``yfinance`` (public data only).

    Interval mapping: ``1h → "60m"``, ``1d → "1d"``; ``4h`` is fetched as 60m
    and resampled locally. Yahoo limits intraday history to a recent trailing
    window (about 7 days for 1m, 60 days for 5m/15m, 730 days for 60m) — the
    requested start is clamped to that window with a warning.

    Args:
        symbol: Yahoo-native ticker, e.g. ``"AAPL"`` or ``"BTC-USD"``.
        timeframe: One of ``1m, 5m, 15m, 1h, 4h, 1d``.
        start: Inclusive UTC start of the window. When ``None``, derived from
            *limit* recent candles, or 365 days when *limit* is also ``None``.
        end: Inclusive UTC end of the window (default: now).
        limit: Number of most-recent candles when *start* is ``None``; ignored
            as a cap when *start* is provided.

    Returns:
        Canonical OHLCV DataFrame (possibly empty, e.g. market closed or the
        requested window is entirely outside Yahoo's intraday retention).

    Raises:
        DataFetchError: When yfinance is unavailable or the download fails.
        ValueError: On an unknown timeframe.
    """
    _validate_timeframe(timeframe)
    try:
        import yfinance as yf  # deferred: keep this module importable without yfinance
    except ImportError as exc:  # pragma: no cover - environment issue
        raise DataFetchError(
            "yfinance is not installed — cannot fetch Yahoo data (pip install yfinance)"
        ) from exc

    interval = _YAHOO_INTERVALS[timeframe]
    now = datetime.now(timezone.utc)
    end_dt = _as_utc(end) if end is not None else now
    if start is not None:
        start_dt = _as_utc(start)
    elif limit:
        span = timedelta(milliseconds=TIMEFRAME_MS[timeframe] * int(limit))
        if timeframe == "1d":
            span = span * 1.5  # calendar buffer for weekends/holidays
        start_dt = end_dt - span
    else:
        start_dt = end_dt - timedelta(days=365)

    max_days = _YAHOO_MAX_LOOKBACK_DAYS.get(interval)
    if max_days is not None:
        floor_dt = now - timedelta(days=max_days)
        if start_dt < floor_dt:
            logger.warning(
                "Yahoo limits %s data to the last %d days — clamping start %s -> %s",
                interval, max_days, start_dt.isoformat(), floor_dt.isoformat(),
            )
            start_dt = floor_dt
    if start_dt >= end_dt:
        logger.warning(
            "Empty Yahoo window for %s %s after clamping (start=%s end=%s)",
            symbol, timeframe, start_dt.isoformat(), end_dt.isoformat(),
        )
        return _empty_frame()

    try:
        raw = yf.download(
            symbol,
            start=start_dt,
            end=end_dt,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as exc:
        raise DataFetchError(f"Yahoo download failed for {symbol}: {exc}") from exc

    if raw is None or raw.empty:
        logger.warning("Yahoo returned no data for %s (%s)", symbol, interval)
        return _empty_frame()

    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    missing = [col for col in ("open", "high", "low", "close") if col not in frame.columns]
    if missing:
        raise DataFetchError(f"Yahoo response for {symbol} is missing columns: {missing}")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame = frame[OHLCV_COLUMNS]

    index = pd.DatetimeIndex(frame.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    frame.index = index
    frame.index.name = "timestamp"
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.dropna(subset=["open", "high", "low", "close"]).astype("float64")

    if timeframe == "4h":
        frame = _resample_ohlcv(frame, "4h")

    lo = pd.Timestamp(start_dt)
    hi = pd.Timestamp(end_dt)
    frame = frame[(frame.index >= lo) & (frame.index <= hi)]
    if start is None and limit:
        frame = frame.tail(int(limit))
    logger.info("Fetched %d Yahoo candles for %s %s", len(frame), symbol, timeframe)
    return frame


def _resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a canonical OHLCV frame to a coarser bar size (e.g. 60m → 4h)."""
    if frame.empty:
        return frame
    resampled = frame.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    resampled.index.name = "timestamp"
    return resampled[OHLCV_COLUMNS].astype("float64")


def fetch(
    source: str,
    symbol: str,
    timeframe: str,
    start: datetime | pd.Timestamp | None = None,
    end: datetime | pd.Timestamp | None = None,
    limit: int | None = 1000,
) -> pd.DataFrame:
    """Dispatch to the fetcher for *source* (``binance`` | ``bybit`` | ``yahoo``).

    Args:
        source: Data source key.
        symbol: Source-native symbol (``BTCUSDT`` for exchanges, ``AAPL`` /
            ``BTC-USD`` for Yahoo).
        timeframe: One of ``1m, 5m, 15m, 1h, 4h, 1d``.
        start: Inclusive UTC start (``None`` → derive from *limit*).
        end: Inclusive UTC end (``None`` → now).
        limit: Number of most-recent candles when *start* is ``None``.

    Returns:
        Canonical OHLCV DataFrame.

    Raises:
        ValueError: On an unknown source or timeframe.
        DataFetchError: On unrecoverable fetch failures.
    """
    key = (source or "").strip().lower()
    if key == "binance":
        return fetch_binance(symbol, timeframe, start=start, end=end, limit=limit)
    if key == "bybit":
        return fetch_bybit(symbol, timeframe, start=start, end=end, limit=limit)
    if key == "yahoo":
        return fetch_yahoo(symbol, timeframe, start=start, end=end, limit=limit)
    raise ValueError(f"Unknown data source {source!r}; expected 'binance', 'bybit' or 'yahoo'")
