-- Trading AI Platform — SQLite schema. PAPER TRADING ONLY.
-- Timestamps are stored as INTEGER epoch milliseconds (UTC) unless noted.

CREATE TABLE IF NOT EXISTS ohlcv (
    source     TEXT    NOT NULL,
    symbol     TEXT    NOT NULL,
    timeframe  TEXT    NOT NULL,
    timestamp  INTEGER NOT NULL,           -- epoch ms UTC (candle open time)
    open       REAL    NOT NULL,
    high       REAL    NOT NULL,
    low        REAL    NOT NULL,
    close      REAL    NOT NULL,
    volume     REAL    NOT NULL,
    PRIMARY KEY (source, symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup
    ON ohlcv (source, symbol, timeframe, timestamp DESC);

CREATE TABLE IF NOT EXISTS ai_analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      INTEGER NOT NULL,      -- epoch ms UTC
    model           TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    sentiment       TEXT    NOT NULL CHECK (sentiment IN ('bullish','bearish','neutral')),
    confidence      INTEGER NOT NULL CHECK (confidence BETWEEN 0 AND 100),
    risk_commentary TEXT    NOT NULL DEFAULT '',
    key_indicators  TEXT    NOT NULL DEFAULT '[]',   -- JSON array
    reasoning       TEXT    NOT NULL DEFAULT '',
    raw_response    TEXT    NOT NULL DEFAULT '',
    -- Call telemetry (older DBs are migrated at runtime by the analyst):
    prompt_hash     TEXT    NOT NULL DEFAULT '',     -- sha256[:12] of system + "\n" + user prompt
    latency_ms      INTEGER,                         -- wall-clock chat latency (retries included)
    attempts        INTEGER,                         -- chat attempts actually made
    fallback_used   INTEGER NOT NULL DEFAULT 0,      -- 1 = fallback model answered
    repair_used     INTEGER NOT NULL DEFAULT 0       -- 1 = self-repair round-trip fired
);

CREATE INDEX IF NOT EXISTS idx_ai_analyses_symbol ON ai_analyses (symbol, created_at DESC);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   INTEGER NOT NULL,
    strategy     TEXT    NOT NULL,
    symbol       TEXT    NOT NULL,
    timeframe    TEXT    NOT NULL,
    params       TEXT    NOT NULL DEFAULT '{}',      -- JSON
    metrics      TEXT    NOT NULL DEFAULT '{}',      -- JSON
    equity_curve TEXT    NOT NULL DEFAULT '{}'       -- JSON {"timestamps": [...], "values": [...]}
);

CREATE TABLE IF NOT EXISTS paper_orders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  INTEGER NOT NULL,
    symbol      TEXT    NOT NULL,
    source      TEXT    NOT NULL DEFAULT 'binance',
    timeframe   TEXT    NOT NULL DEFAULT '1h',
    side        TEXT    NOT NULL CHECK (side IN ('buy','sell')),
    order_type  TEXT    NOT NULL CHECK (order_type IN ('market','limit','stop')),
    qty         REAL    NOT NULL,
    limit_price REAL,
    stop_price  REAL,
    stop_loss   REAL,
    take_profit REAL,
    status      TEXT    NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','filled','cancelled','rejected')),
    filled_at   INTEGER,
    fill_price  REAL,
    note        TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS paper_positions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    source      TEXT    NOT NULL DEFAULT 'binance',
    timeframe   TEXT    NOT NULL DEFAULT '1h',
    side        TEXT    NOT NULL CHECK (side IN ('long','short')),
    qty         REAL    NOT NULL,
    entry_price REAL    NOT NULL,
    stop_loss   REAL,
    take_profit REAL,
    opened_at   INTEGER NOT NULL,
    closed_at   INTEGER,
    exit_price  REAL,
    pnl         REAL,
    status      TEXT    NOT NULL DEFAULT 'open' CHECK (status IN ('open','closed')),
    last_price  REAL                                  -- latest mark-to-market price
);

CREATE INDEX IF NOT EXISTS idx_paper_positions_status ON paper_positions (status);

CREATE TABLE IF NOT EXISTS paper_trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    INTEGER REFERENCES paper_orders (id),
    position_id INTEGER REFERENCES paper_positions (id),
    symbol      TEXT    NOT NULL,
    side        TEXT    NOT NULL CHECK (side IN ('buy','sell')),
    qty         REAL    NOT NULL,
    price       REAL    NOT NULL,
    commission  REAL    NOT NULL DEFAULT 0,
    slippage    REAL    NOT NULL DEFAULT 0,
    executed_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             INTEGER NOT NULL,
    equity         REAL    NOT NULL,
    cash           REAL    NOT NULL,
    unrealized_pnl REAL    NOT NULL DEFAULT 0,
    realized_pnl_today REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_equity_snapshots_ts ON equity_snapshots (ts DESC);

-- Simple key/value store for account-level state (cash balance, peak equity, halts...)
CREATE TABLE IF NOT EXISTS account_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- AI auto-trader activity log (PAPER ONLY). Every decision the bot makes —
-- cycle boundaries, scans, entries, exits, skips, rejections, halts, errors —
-- is appended here with a JSON detail payload for full dashboard transparency.
CREATE TABLE IF NOT EXISTS bot_activity (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     INTEGER NOT NULL,                -- epoch ms UTC
    symbol TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL,                   -- cycle_start|scan|enter|exit|skip|reject|halt|error|cycle_end
    detail TEXT NOT NULL DEFAULT '{}'       -- JSON
);

CREATE INDEX IF NOT EXISTS idx_bot_activity_ts ON bot_activity (ts DESC);

-- News & community sentiment scored by the local LLM (PAPER ONLY — public,
-- keyless sources). One row per coin per SentimentAgent run, plus one
-- 'GLOBAL' row capturing the market-wide mood.
CREATE TABLE IF NOT EXISTS coin_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER NOT NULL, symbol TEXT NOT NULL,          -- 'GLOBAL' = market-wide mood
    sentiment TEXT NOT NULL CHECK (sentiment IN ('bullish','bearish','neutral')),
    score INTEGER NOT NULL,                             -- -100..100 (clamped)
    confidence INTEGER NOT NULL,                        -- 0..100
    summary TEXT NOT NULL DEFAULT '',
    headlines TEXT NOT NULL DEFAULT '[]',               -- JSON [{title,source,url,published}]
    model TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_coin_sentiment ON coin_sentiment (symbol, ts DESC);

-- Per-coin trading playbook adapted by the Coach agent from realised
-- paper-trade results (bench losers, bias direction, scale size).
CREATE TABLE IF NOT EXISTS coin_playbook (
    symbol TEXT PRIMARY KEY, updated_at INTEGER NOT NULL,
    bench INTEGER NOT NULL DEFAULT 0,                   -- 1 = don't trade this coin
    side_bias TEXT NOT NULL DEFAULT 'both' CHECK (side_bias IN ('both','long_only','short_only')),
    size_multiplier REAL NOT NULL DEFAULT 1.0,          -- clamp [0.25, 1.5]
    min_vote_override INTEGER,                          -- clamp [2,4] or NULL
    reasoning TEXT NOT NULL DEFAULT '',
    stats TEXT NOT NULL DEFAULT '{}'                    -- JSON evidence snapshot
);
