"""SQLite persistence layer.

One short-lived connection per call (WAL mode makes this cheap and safe across
the FastAPI worker threads). All timestamps are stored as epoch milliseconds UTC.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def get_conn() -> sqlite3.Connection:
    """Open a new connection with sane defaults (WAL, foreign keys, Row rows)."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db() -> None:
    """Create all tables. Idempotent."""
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_conn() as conn:
        conn.executescript(schema)
    logger.info("Database initialised at %s", settings.db_path)


def _to_ms(ts: datetime | pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.value // 1_000_000)


def _from_ms(ms: int) -> pd.Timestamp:
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def utc_now_ms() -> int:
    """Current UTC time as epoch milliseconds."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def upsert_ohlcv(source: str, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
    """Insert-or-replace candles from a canonical OHLCV frame. Returns rows written."""
    if df is None or df.empty:
        return 0
    rows = [
        (
            source,
            symbol,
            timeframe,
            _to_ms(idx),
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
            float(r["volume"]),
        )
        for idx, r in df.iterrows()
    ]
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO ohlcv "
            "(source, symbol, timeframe, timestamp, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    logger.debug("Upserted %d candles for %s:%s:%s", len(rows), source, symbol, timeframe)
    return len(rows)


def load_ohlcv(
    source: str,
    symbol: str,
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load cached candles as a canonical OHLCV frame (possibly empty).

    ``limit`` returns the most recent *n* rows, still sorted ascending.
    """
    query = (
        "SELECT timestamp, open, high, low, close, volume FROM ohlcv "
        "WHERE source = ? AND symbol = ? AND timeframe = ?"
    )
    params: list[object] = [source, symbol, timeframe]
    if start is not None:
        query += " AND timestamp >= ?"
        params.append(_to_ms(start))
    if end is not None:
        query += " AND timestamp <= ?"
        params.append(_to_ms(end))
    query += " ORDER BY timestamp DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()

    if not rows:
        empty = pd.DataFrame(columns=OHLCV_COLUMNS)
        empty.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return empty

    df = pd.DataFrame([dict(r) for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df[OHLCV_COLUMNS].astype("float64")


def ohlcv_coverage(
    source: str, symbol: str, timeframe: str
) -> tuple[datetime, datetime] | None:
    """Return (first, last) cached candle times (UTC), or None if nothing cached."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MIN(timestamp) AS lo, MAX(timestamp) AS hi FROM ohlcv "
            "WHERE source = ? AND symbol = ? AND timeframe = ?",
            (source, symbol, timeframe),
        ).fetchone()
    if row is None or row["lo"] is None:
        return None
    return _from_ms(row["lo"]).to_pydatetime(), _from_ms(row["hi"]).to_pydatetime()


def list_cached() -> list[dict]:
    """Summarise every (source, symbol, timeframe) present in the cache."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT source, symbol, timeframe, COUNT(*) AS rows_, "
            "MIN(timestamp) AS first_ts, MAX(timestamp) AS last_ts "
            "FROM ohlcv GROUP BY source, symbol, timeframe "
            "ORDER BY source, symbol, timeframe"
        ).fetchall()
    return [
        {
            "source": r["source"],
            "symbol": r["symbol"],
            "timeframe": r["timeframe"],
            "rows": r["rows_"],
            "first_ts": _from_ms(r["first_ts"]).isoformat(),
            "last_ts": _from_ms(r["last_ts"]).isoformat(),
        }
        for r in rows
    ]
