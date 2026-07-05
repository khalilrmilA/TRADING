# ARCHITECTURE CONTRACTS — trading_ai_platform

This document is the **binding interface specification** for all modules. Every module MUST
conform to these contracts exactly — other modules are being written against them concurrently.

**PAPER TRADING ONLY.** No real-money execution anywhere. No private exchange API keys.
Only public market-data endpoints are ever called. This is a research/backtest/paper platform.

## Runtime & conventions

- Python 3.14 on Windows (also runs in Docker with python:3.12-slim). Pure-Python + pandas/numpy only (no TA-Lib, no ccxt).
- Working directory for all entry points is the project root `trading_ai_platform/`.
  - Backend: `uvicorn backend.api.main:app --host 127.0.0.1 --port 8000`
  - Dashboard: `streamlit run dashboard/app.py`
  - Tests: `pytest` (pytest.ini sets `pythonpath = .`)
- Imports are absolute from the root: `from backend.indicators.technical import rsi`, `from config.settings import settings`.
- Every backend module uses `logging.getLogger(__name__)` — no `print()` in backend code.
- Full type hints. Google-style docstrings on public functions. f-strings.
- All timestamps are **UTC**. In DataFrames: tz-aware `pd.DatetimeIndex`. In SQLite: integer epoch **milliseconds**. In JSON API responses: ISO-8601 strings (`2026-07-04T12:00:00+00:00`).
- The dashboard NEVER imports backend modules — it talks to the API over HTTP only (decoupling + Docker).

## Canonical OHLCV DataFrame

Every function that passes candles uses this exact shape:

- Index: `pd.DatetimeIndex`, UTC tz-aware, name `"timestamp"`, ascending, unique.
- Columns: `open, high, low, close, volume` — all `float64`.
- Feature-enriched frames add the canonical feature columns below (same index).

Sources: `"binance" | "bybit" | "yahoo"`. Timeframes: `"1m" | "5m" | "15m" | "1h" | "4h" | "1d"`.
Symbols are stored/passed in the source's native format: Binance/Bybit `BTCUSDT`, Yahoo `AAPL` / `BTC-USD`.

## config/settings.py  (ALREADY WRITTEN — read it, do not modify)

`from config.settings import settings` — pydantic-settings singleton, loads `.env` from project root. Key fields:

```
settings.ollama_url: str = "http://localhost:11434"
settings.ollama_model: str = "qwen3:14b"            # best fit for 16GB-VRAM host GPU
settings.ollama_fallback_model: str = "deepseek-r1:32b"
settings.ollama_code_model: str = "qwen2.5-coder:32b"
settings.ollama_timeout: float = 180.0              # seconds
settings.ollama_retries: int = 3
settings.db_path: Path                               # <root>/data/trading.db (dir auto-created)
settings.api_host: str = "127.0.0.1"; settings.api_port: int = 8000
settings.api_url: str                                # "http://127.0.0.1:8000" (computed)
settings.initial_capital: float = 100_000.0
settings.commission_rate: float = 0.001              # 0.1% per fill
settings.slippage_rate: float = 0.0005               # 0.05% adverse per market/stop fill
settings.risk_per_trade: float = 0.01                # max 1% equity risk per trade
settings.max_open_positions: int = 5
settings.daily_loss_limit: float = 0.03              # halt new entries at -3% day PnL
settings.circuit_breaker_drawdown: float = 0.10      # halt at 10% drawdown from peak equity
settings.atr_stop_multiplier: float = 2.0
settings.atr_take_profit_multiplier: float = 3.0
settings.data_update_interval_minutes: int = 15
```

## backend/database/db.py  (ALREADY WRITTEN — read it, do not modify)

sqlite3 stdlib, WAL mode, one connection per call. Public API:

```
init_db() -> None                                    # executes schema.sql, idempotent
get_conn() -> sqlite3.Connection                     # row_factory=sqlite3.Row, FK on
upsert_ohlcv(source, symbol, timeframe, df) -> int   # INSERT OR REPLACE, returns rows written
load_ohlcv(source, symbol, timeframe, start=None, end=None, limit=None) -> pd.DataFrame
                                                     # canonical OHLCV frame (may be empty);
                                                     # limit returns the MOST RECENT n rows (ascending order)
ohlcv_coverage(source, symbol, timeframe) -> tuple[datetime, datetime] | None   # (first,last) UTC
list_cached() -> list[dict]                          # [{source,symbol,timeframe,rows,first_ts,last_ts}]
```

Tables (see schema.sql): `ohlcv`, `ai_analyses`, `backtest_runs`, `paper_orders`,
`paper_positions`, `paper_trades`, `equity_snapshots`, `account_state`.
`init_db()` is called by the API on startup and by the paper engine constructor.

## backend/data/  — owner: DATA agent

```
backend/data/fetchers.py
    fetch_binance(symbol, timeframe, start=None, end=None, limit=1000) -> pd.DataFrame
    fetch_bybit(symbol, timeframe, start=None, end=None, limit=1000) -> pd.DataFrame
    fetch_yahoo(symbol, timeframe, start=None, end=None, limit=None) -> pd.DataFrame
    fetch(source, symbol, timeframe, start=None, end=None, limit=1000) -> pd.DataFrame   # dispatcher
```
- Binance: public `GET https://api.binance.com/api/v3/klines` (interval map 1m,5m,15m,1h,4h,1d; paginate 1000/req).
- Bybit: public `GET https://api.bybit.com/v5/market/kline` (category=spot; interval map 1,5,15,60,240,D; paginate; **Bybit returns newest-first — reverse it**).
- Yahoo: `yfinance.download` (map 1h→"60m", 1d→"1d"; intraday limited to recent window by Yahoo — clamp and log).
- `requests` with 10s connect/30s read timeout, 3 retries w/ backoff on 429/5xx/timeouts. No API keys ever.
- All return the canonical frame; raise `DataFetchError(Exception)` (defined in fetchers.py) on unrecoverable failure.

```
backend/data/service.py
    update_symbol(source, symbol, timeframe, lookback_days=365) -> int
        # incremental: fetch from last cached ts (or lookback) to now, upsert, return new rows
    get_ohlcv(source, symbol, timeframe, limit=500, with_features=False, refresh=False) -> pd.DataFrame
        # cache-first; if empty or refresh=True → update_symbol first; features via add_features()
    start_scheduler() / stop_scheduler()
        # APScheduler BackgroundScheduler: every settings.data_update_interval_minutes,
        # re-update every (source,symbol,timeframe) already in cache. Called from API lifespan.
```

## backend/indicators/  — owner: INDICATORS agent

`backend/indicators/technical.py` — pure functions, pandas/numpy only, no lookahead (rolling windows only):

```
sma(s: pd.Series, window) -> pd.Series
ema(s, window) -> pd.Series                          # pandas ewm(span=window, adjust=False)
rsi(s, window=14) -> pd.Series                       # Wilder smoothing, 0..100
macd(s, fast=12, slow=26, signal=9) -> tuple[pd.Series, pd.Series, pd.Series]   # (macd, signal, hist)
bollinger(s, window=20, num_std=2.0) -> tuple[pd.Series, pd.Series, pd.Series]  # (upper, mid, lower)
atr(df, window=14) -> pd.Series                      # true range, Wilder smoothing
vwap(df) -> pd.Series                                # cumulative per UTC day (groupby index.date)
volume_profile(df, bins=24) -> pd.DataFrame          # columns: price_low, price_high, volume; index 0..bins-1
returns(s, periods=1) -> pd.Series                   # pct_change
volatility(s, window=20) -> pd.Series                # rolling std of 1-period returns, annualized × sqrt(365)
momentum(s, window=10) -> pd.Series                  # s / s.shift(window) - 1
```

`backend/indicators/features.py`:
```
add_features(df) -> pd.DataFrame    # copy of df + EXACTLY these columns:
# sma_20, sma_50, ema_12, ema_26, rsi_14, macd, macd_signal, macd_hist,
# bb_upper, bb_mid, bb_lower, atr_14, vwap, ret_1, ret_5, volatility_20, momentum_10
feature_summary(df) -> dict         # last-row snapshot for the AI prompt: numeric values (round 6,
                                    # None for NaN) + price, plus derived flags:
                                    # price_vs_sma20 ("above"/"below"), macd_state ("bullish"/"bearish"),
                                    # rsi_zone ("oversold"<30 /"neutral"/"overbought">70),
                                    # bb_position ("above_upper","upper_half","lower_half","below_lower")
```

## backend/ai/  — owner: AI agent

`backend/ai/ollama_client.py`:
```
class OllamaError(Exception)
class OllamaClient:
    def __init__(self, base_url=None, timeout=None, retries=None)   # defaults from settings
    def list_models(self) -> list[str]                # GET /api/tags, [] on failure (never raises)
    def is_available(self) -> bool
    def chat(self, prompt, system=None, model=None, json_mode=True, temperature=0.2) -> str
        # POST /api/chat, stream=False, options={"temperature":...}; format="json" when json_mode.
        # Retry settings.ollama_retries times with exponential backoff (1s,2s,4s).
        # If the primary model fails all retries (or is not in list_models()), try
        # settings.ollama_fallback_model once; then raise OllamaError.
        # ALWAYS strip <think>...</think> blocks (qwen3/deepseek-r1 emit them) before returning.
```

`backend/ai/analyst.py`:
```
class MarketAnalysis(pydantic.BaseModel):
    sentiment: Literal["bullish", "bearish", "neutral"]
    confidence: int                     # 0-100
    risk_commentary: str
    key_indicators: list[KeyIndicator]  # KeyIndicator: name:str, value:str, influence:str
    reasoning: str
    model_used: str = ""
    symbol: str = ""; timeframe: str = ""

analyze_market(symbol, timeframe, df, model=None) -> MarketAnalysis
    # df is feature-enriched. Build prompt from feature_summary(df) + last 20 closes.
    # System prompt: senior market analyst, research only, not financial advice,
    # respond ONLY with JSON matching the schema (schema included in prompt).
    # Parse via json.loads → MarketAnalysis.model_validate; on parse failure, one repair attempt:
    # re-ask the model to fix its own output into valid JSON; then raise OllamaError.
    # Clamp confidence to [0,100]. Persist every result to ai_analyses table.
get_analysis_history(symbol=None, limit=50) -> list[dict]
```

## backend/strategies/  — owner: STRATEGIES agent

`backend/strategies/base.py`:
```
class BaseStrategy(ABC):
    name: str = "base"                                # class attribute, unique key
    params: dict                                      # set in __init__(**params)
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """df: feature-enriched canonical frame. Returns a COPY with added column
        'signal' (int8: 1=long, -1=short, 0=flat). Signal is the TARGET STANCE for the
        NEXT bar (backtester enters at next bar open). NO LOOKAHEAD: row t uses data ≤ t.
        First rows may be 0 while indicators warm up (fillna(0))."""
```

Files & registry keys (all in `backend/strategies/`):
- `trend_following.py` → `"trend_following"`: long when ema_12 > ema_26 AND close > sma_50; short when inverse; else 0.
- `mean_reversion.py` → `"mean_reversion"`: long when close < bb_lower AND rsi_14 < 30; exit to 0 when close crosses bb_mid; short when close > bb_upper AND rsi_14 > 70.
- `breakout.py` → `"breakout"`: long on close > rolling 20-bar high of prior bars (shift(1)) with volume > 1.5× 20-bar avg volume; short on symmetric low breakdown; hold until opposite band of a 10-bar channel.
- `rsi_macd.py` → `"rsi_macd"`: long when macd crosses above macd_signal AND rsi_14 > 50; short when macd crosses below AND rsi_14 < 50; else hold previous stance (ffill) until opposite signal.
- `voting.py` → `"ensemble"`: `class VotingStrategy(BaseStrategy)` — instantiate the 4 above, sum their per-bar signals; ≥2 → 1, ≤-2 → -1, else 0. Also exposes `vote_breakdown(df) -> pd.DataFrame` (one column per strategy + 'ensemble').

`backend/strategies/__init__.py` (agent OVERWRITES the empty placeholder):
```
STRATEGIES: dict[str, type[BaseStrategy]]            # the 5 keys above
get_strategy(name, **params) -> BaseStrategy         # KeyError w/ helpful message on unknown
list_strategies() -> list[dict]                      # [{name, description, default_params}]
```

## backend/backtest/  — owner: BACKTEST agent

`backend/backtest/engine.py`:
```
@dataclass BacktestConfig:
    initial_capital, commission_rate, slippage_rate, risk_per_trade,
    atr_stop_multiplier, atr_take_profit_multiplier   # ALL default from settings
    allow_short: bool = True

@dataclass Trade:
    symbol, side ("long"/"short"), qty, entry_time, entry_price, exit_time, exit_price,
    commission, pnl, pnl_pct, exit_reason ("signal"/"stop_loss"/"take_profit"/"end_of_data")

@dataclass BacktestResult:
    equity_curve: pd.Series        # indexed like df, starts at initial_capital
    trades: list[Trade]
    metrics: dict                  # see metrics.py
    def to_dict(self) -> dict      # JSON-safe: equity_curve → {"timestamps": [iso...], "values": [...]},
                                   # trades → list[dict] with iso times, metrics as-is

class Backtester:
    def __init__(self, config: BacktestConfig | None = None)
    def run(self, df, strategy: BaseStrategy, symbol="") -> BacktestResult
```
Engine semantics (bar-by-bar, one position at a time per backtest):
- Signal at bar t → act at bar t+1 **open** (never fill on the signal bar).
- Entry fill = next open ± slippage (adverse: buy higher, sell lower). Commission = notional × commission_rate per fill (entry and exit).
- Position size: `qty = (equity * risk_per_trade) / (atr_stop_multiplier * atr_14[t])`, capped so notional ≤ 95% of equity. Skip entry if atr is NaN/0.
- Stop = entry ∓ atr_stop_multiplier × atr_14 ; TP = entry ± atr_take_profit_multiplier × atr_14 (sign by side).
- Intrabar: if a bar's high/low touches stop or TP, exit at that level ± slippage (check stop first — conservative). Signal flip → exit at next open, then open the new side.
- Equity curve marked-to-market on close each bar.

`backend/backtest/metrics.py`:
```
compute_metrics(equity_curve: pd.Series, trades: list[Trade], periods_per_year: int) -> dict
# keys EXACTLY: total_return, cagr, sharpe, sortino, max_drawdown, win_rate,
#               profit_factor, num_trades, avg_trade_pnl, avg_win, avg_loss, exposure  (floats; JSON-safe)
# sharpe/sortino from per-bar equity returns, annualized by periods_per_year, rf=0; 0.0 when undefined.
# max_drawdown as POSITIVE fraction (0.12 = -12%). profit_factor: gross_wins/gross_losses (cap 100.0 when no losses).
periods_per_year_for(timeframe: str) -> int   # 1m:525600, 5m:105120, 15m:35040, 1h:8760, 4h:2190, 1d:365
```

`backend/backtest/walk_forward.py`:
```
walk_forward(df, strategy_name, n_splits=4, train_ratio=0.7, config=None, params=None) -> dict
# Rolling windows: split df into n_splits contiguous folds; per fold, "train" segment is
# informational (strategies are rule-based) — evaluate on the TEST segment only, indicators
# recomputed per segment with add_features(). Returns:
# {"folds": [{"fold": i, "train_start": iso, "test_start": iso, "test_end": iso, "metrics": {...}}...],
#  "aggregate": {mean/std of test sharpe, total_return, max_drawdown, win_rate}}
```

## backend/paper_trading/  — owner: PAPER agent

**Simulation only.** Fills happen ONLY against cached candles from the local DB. State persists in SQLite (survives restart). Single class facade:

`backend/paper_trading/engine.py`:
```
class PaperTradingEngine:
    def __init__(self)                                # init_db(); loads open state from DB
    def submit_order(symbol, side, order_type, qty=None, limit_price=None, stop_price=None,
                     stop_loss=None, take_profit=None, source="binance", timeframe="1h") -> dict
        # side: "buy"|"sell"; order_type: "market"|"limit"|"stop"
        # qty=None → risk-based auto-size from ATR (risk.py). Returns order record dict.
        # Market orders fill immediately at latest cached close ± slippage; limit/stop go PENDING.
    def process_tick(symbol, source="binance", timeframe="1h", refresh=False) -> dict
        # Pull latest candle (data service; refresh → hit exchange public API first).
        # Fill pending limits/stops if candle range touches price; check open positions'
        # SL/TP; mark-to-market; snapshot equity. Returns {"filled": [...], "closed": [...], "equity": float}
    def close_position(position_id) -> dict          # market-close at latest cached price
    def get_portfolio() -> dict
        # {"equity","cash","unrealized_pnl","realized_pnl_today","daily_pnl_pct","open_positions": n,
        #  "peak_equity","drawdown","circuit_breaker_active","trading_halted","halt_reason"}
    def get_positions(status="open") -> list[dict]
    def get_orders(status=None) -> list[dict]
    def get_trades(limit=100) -> list[dict]
    def get_equity_history(limit=1000) -> list[dict]  # [{"ts": iso, "equity": float}]
    def reset() -> None                               # wipe paper tables, restore initial capital
```

`backend/paper_trading/risk.py`:
```
class RiskCheckResult(NamedTuple): allowed: bool; reason: str
class RiskManager:
    def check_order(portfolio: dict, open_positions: list) -> RiskCheckResult
        # Rejects when: open positions ≥ settings.max_open_positions;
        # daily realized+unrealized PnL ≤ -settings.daily_loss_limit × day-start equity;
        # drawdown from peak ≥ settings.circuit_breaker_drawdown (CIRCUIT BREAKER — also
        # blocks until manual reset via engine.reset()).
    def position_size(equity, atr_value, price) -> float
        # (equity × risk_per_trade) / (atr_stop_multiplier × atr) capped at 95% equity notional; 0 if atr invalid
```
Buy → long position; sell with no open long → short position (sell of an open long closes it FIFO).
Commission & slippage identical to backtester (settings rates). Every fill writes `paper_trades` +
updates `paper_positions`/`paper_orders`; every process_tick writes `equity_snapshots`.

## backend/api/  — owner: API agent

`backend/api/main.py`: FastAPI app, title "Trading AI Platform (Paper Only)". Lifespan: `init_db()`,
`start_scheduler()`; CORS allow-all (local tool). Global exception handler → `{"detail": str}` with 502 for
OllamaError, 502 for DataFetchError, 400 for ValueError/KeyError. Singleton `PaperTradingEngine` in module state.
`backend/api/schemas.py`: pydantic request/response models for everything below.

Routes (exact paths; request bodies are JSON):
```
GET  /health                     → {"status":"ok","ollama_available":bool,"db":"ok","paper_only":true}
GET  /api/models                 → {"configured":{"default","fallback","code"},"installed":[names],"ollama_available":bool}
POST /api/data/fetch             {source,symbol,timeframe,lookback_days:int=365} → {"rows_added":int,"coverage":{...}}
GET  /api/data/ohlcv             ?source&symbol&timeframe&limit=500&features=false
                                 → {"symbol","timeframe","rows":n,"data":[{"timestamp":iso,"open":...,...}]}
                                 (features=true adds feature columns to each record; NaN→null)
GET  /api/data/cached            → {"items": list_cached()}
POST /api/analysis               {source,symbol,timeframe,model?} → MarketAnalysis JSON (fetches+features data itself)
GET  /api/analysis/history       ?symbol&limit=50 → {"items":[...]}
GET  /api/strategies             → {"strategies": list_strategies()}
POST /api/backtest               {source,symbol,timeframe,strategy,limit:int=2000,params?:{},config?:{}}
                                 → BacktestResult.to_dict() + {"strategy","symbol","timeframe"}; persists to backtest_runs
POST /api/backtest/walkforward   {source,symbol,timeframe,strategy,n_splits=4,train_ratio=0.7,limit=3000} → walk_forward dict
GET  /api/backtest/runs          ?limit=50 → {"items":[{id,created_at,strategy,symbol,timeframe,metrics}]}
POST /api/paper/orders           {symbol,side,order_type,qty?,limit_price?,stop_price?,stop_loss?,take_profit?,source?,timeframe?}
                                 → order dict (400 + reason when RiskManager rejects)
GET  /api/paper/orders           ?status → {"items":[...]}
GET  /api/paper/positions        ?status=open → {"items":[...]}
GET  /api/paper/trades           ?limit=100 → {"items":[...]}
GET  /api/paper/portfolio        → get_portfolio()
GET  /api/paper/equity           ?limit=1000 → {"items":[...]}
POST /api/paper/tick             {symbol,source?,timeframe?,refresh?:bool} → process_tick result
POST /api/paper/positions/{id}/close → closed position dict
POST /api/paper/reset            → {"status":"reset"}
```

## dashboard/  — owner: DASHBOARD agent

`dashboard/app.py` (+ optional `dashboard/api_client.py`, `dashboard/components.py`). Streamlit, wide layout,
`st.tabs`: **Market** (candlestick w/ SMA/EMA/BB overlays + RSI & MACD subplots — plotly, volume bars),
**AI Analysis** (model dropdown from GET /api/models, run analysis button, sentiment badge, confidence gauge/progress,
key-indicators table, risk commentary, history), **Backtest** (strategy dropdown from /api/strategies, params, run →
equity curve + metrics cards + trades table; walk-forward section), **Paper Trading** (portfolio metric cards, order
form, positions/orders/trades tables, tick button, equity curve, reset w/ confirmation, red banner when
trading_halted), **Strategy Comparison** (run all 5 strategies on same data → grouped metric bar charts + table).
Sidebar: source/symbol/timeframe pickers, fetch-data button, API status indicator, "PAPER TRADING ONLY" caption.
API base URL from env `API_URL` default `http://127.0.0.1:8000`. All HTTP in try/except with `st.error` — the
dashboard must render even when the API is down. `@st.cache_data(ttl=30)` on GET helpers.

## tests/  — owner: TESTS agent

pytest; NO network, NO Ollama calls (mock `requests`/client methods with monkeypatch) — synthetic OHLCV via a
shared `tests/conftest.py` fixture `sample_df` (~300 rows, seeded RNG random walk + trend segments, 1h UTC index).
DB tests use `tmp_path` monkeypatching `settings.db_path` + fresh `init_db()`.
Files: `test_indicators.py` (SMA/EMA hand-checked values, RSI bounds 0-100, MACD relation, BB ordering upper≥mid≥lower,
ATR>0, no-lookahead: features on df[:100] equal features on full df for first 100 rows),
`test_strategies.py` (signal values ∈ {-1,0,1}, dtype, registry completeness, voting math),
`test_backtest.py` (deterministic uptrend → trend strategy profits; commissions reduce PnL; metrics keys exact;
max_drawdown positive fraction; walk-forward fold count),
`test_paper_trading.py` (market order fills w/ slippage; limit fills only when touched; SL closes position;
reset restores capital; NO network — pre-seed ohlcv table),
`test_risk.py` (rejects 6th position; daily-loss halt; circuit breaker; ATR sizing formula),
`test_database.py` (upsert idempotent; load roundtrip preserves values/index tz).

## backend/paper_trading/auto_trader.py — AI AUTO-TRADER (paper only) — owner: BOT agent

Autonomous simulated trader: scans a crypto watchlist every cycle, ranks opportunities by
strategy-ensemble votes, confirms top candidates with the local LLM, and trades a simulated
account through the existing PaperTradingEngine + RiskManager. **Never touches a real exchange.**

New table (append to schema.sql):
```
CREATE TABLE IF NOT EXISTS bot_activity (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     INTEGER NOT NULL,                -- epoch ms UTC
    symbol TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL,                   -- cycle_start|scan|enter|exit|skip|reject|halt|error|cycle_end
    detail TEXT NOT NULL DEFAULT '{}'       -- JSON
);
CREATE INDEX IF NOT EXISTS idx_bot_activity_ts ON bot_activity (ts DESC);
```

```
class BotConfig(pydantic.BaseModel):
    watchlist: list[str] = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
                            "DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT"]
    source: str = "binance"
    timeframe: str = "1h"
    interval_minutes: int = 15
    use_ai: bool = True                     # LLM confirmation gate for entries
    min_ai_confidence: int = 60
    max_ai_calls_per_cycle: int = 3
    allow_short: bool = False
    min_vote: int = 2                       # |sum of 4 strategy signals| to shortlist
    max_position_fraction: float = 0.35     # max notional per position vs equity
    running: bool = False

class AutoTrader:                            # singleton next to the engine in the API process
    def __init__(self, engine: PaperTradingEngine)
    def get_config() -> BotConfig            # persisted in account_state key 'bot_config' (JSON)
    def set_config(updates: dict) -> BotConfig
    def start() -> BotConfig                 # sets running=True (persisted)
    def stop() -> BotConfig
    def status() -> dict                     # {"running",last_cycle_ts,next_cycle_ts (iso|None),
                                             #  "last_cycle_summary",watchlist,equity,open_positions}
    def run_cycle() -> dict                  # full pass, thread-safe (threading.Lock, skip if already
                                             # running), NEVER raises — catches + logs per-symbol errors
```

run_cycle order (log every step to bot_activity):
1. `cycle_start`.
2. MANAGE — ALWAYS runs, even when `trading_halted` (process_tick is the only path that evaluates
   SL/TP on open positions, and a halt is exactly when protective stops must keep working;
   RiskManager allows risk-reducing exits during a halt): for each distinct open-position symbol →
   `engine.process_tick(refresh=True)`; recompute ensemble stance on featured data (VotingStrategy,
   CLOSED candles only — drop the still-forming last candle; skip the stance check when the data is
   stale, see step 4); if stance is OPPOSITE to the position side → `engine.close_position`, log
   `exit` with reason.
3. HALT GATE: if portfolio `trading_halted` → log `halt`, return summary (halts skip NEW entries
   only — steps 4-6 — never position management).
4. SCAN each watchlist symbol: `update_symbol` (lookback 30d), `get_ohlcv(limit=500, with_features=True)`;
   drop the still-forming (unclosed) last candle — signals act on CLOSED bars only, matching the
   backtester's "signal at bar t → act at t+1" semantics; skip (<100 rows) with log; skip
   `stale_data` with log when the last closed candle is older than ~2× the timeframe (+ grace) —
   never vote/size/fill on a frozen price after a data outage; vote_sum = last-bar sum over the 4
   individual strategies (`VotingStrategy.vote_breakdown`); log `scan` {vote_sum, momentum_10,
   rsi_14, price}.
5. Shortlist: no open position in symbol, |vote_sum| ≥ min_vote, long side only unless allow_short.
   Rank by |vote_sum| desc then |momentum_10| desc. Keep top `free_slots` (≤ max_ai_calls_per_cycle
   when use_ai).
6. Per candidate: if use_ai → `analyze_market` (OllamaError → log `error`, SKIP candidate —
   conservative); require sentiment matches direction AND confidence ≥ min_ai_confidence else log
   `skip`. Re-read the latest cached close immediately before sizing (the engine fills market
   orders at the submit-time cached close, which may have drifted since the scan). qty =
   min(RiskManager.position_size(...), max_position_fraction·equity/price, 0.95·cash/price); skip
   dust (< $1 notional). `engine.submit_order` market w/ ATR stop_loss & take_profit (price ∓/±
   atr_14 × settings multipliers). Log `enter` (or `reject` w/ risk reason).
7. Equity snapshot via engine; log `cycle_end` {scanned, entered, exited, skipped, equity}.

TRANSPARENCY (mandatory — the user watches every decision in the dashboard):
- `enter` detail: {side, qty, price, notional, stop_loss, take_profit, vote_sum,
  votes: {trend_following, mean_reversion, breakout, rsi_macd}, indicators: {rsi_14, macd_state,
  momentum_10, atr_14, price_vs_sma20, bb_position}, ai: {sentiment, confidence, reasoning,
  risk_commentary} | null, explanation: str}  — explanation is ONE plain-English sentence, e.g.
  "Bought 0.0004 BTC ($28.50): 3/4 strategies vote long, RSI 58 rising, AI bullish at 72% — stop $69,100, target $73,400".
- `exit` detail: {reason, pnl, pnl_pct, explanation}. `skip`/`reject` detail: {reason, explanation}.
- backend/ai/analyst.py: ADD `chart_context(df) -> dict` (do not alter existing functions' behavior):
  {swing_high_20, swing_low_20, dist_to_swing_high_pct, dist_to_swing_low_pct,
   trend_20: "up"|"down"|"sideways" (sign of sma_20 slope over last 10 bars, ±0.5% dead-zone),
   volume_trend: "rising"|"falling", last_5_closes: [floats]} — JSON-safe, NaN→None. Include it in
  the analyze_market prompt under "CHART STRUCTURE" so the model reads price structure
  (support/resistance distance, trend, volume) — existing MarketAnalysis schema unchanged.

API additions (main.py; schemas in schemas.py):
```
GET  /api/bot/status              → status()
GET  /api/bot/config              → BotConfig JSON
POST /api/bot/config  {subset}    → updated BotConfig JSON
POST /api/bot/start               → status()   (adds APScheduler job id 'bot_cycle', interval=
                                    config.interval_minutes, max_instances=1, coalesce=True, and
                                    kicks one immediate cycle in a daemon thread)
POST /api/bot/stop                → status()   (removes job)
POST /api/bot/run-once            → run_cycle() summary (sync; may take minutes with AI on)
GET  /api/bot/activity ?limit=100 → {"items":[{"ts":iso,"symbol","action","detail":dict}]}
```
On API startup: if persisted config.running → re-register the job (bot survives restarts).
The job's FIRST fire resumes the persisted cadence — `max(now, last_cycle_ts + interval)` — so a
restart with an overdue cycle fires immediately instead of silently waiting another full interval;
POST /api/bot/config re-adds the job only when `running`/`interval_minutes` actually changed (or
the job is missing), so unrelated config saves never reset the interval timer.

Dashboard: insert tab **AI Trader** between Paper Trading and Strategy Comparison. Status row
(running badge, equity, last/next cycle, open positions), Start / Stop / "Run cycle now" buttons
(run-once with 600s timeout + st.spinner), config expander (watchlist as comma-separated text_area,
timeframe/interval/min-confidence/use_ai/allow_short/min_vote inputs → POST /api/bot/config),
activity feed table (latest 100; ✅ enter, 🔴 exit, ⏭ skip, ⚠️ error/halt), caption
"The AI trades a simulated account — paper only, not financial advice."

Dashboard transparency additions (AI Trader tab):
- Top of tab: "📖 Trading Methodology" st.expander containing clear beginner-friendly markdown that
  documents the COMPLETE buy/sell methodology: each of the 4 strategies' exact entry/exit rules and
  the chart patterns they read (trend crossovers, Bollinger reversion, channel breakouts, RSI+MACD
  momentum), the ensemble voting gate, the AI confirmation step (what the model sees: indicators +
  chart structure; what it returns: sentiment/confidence/reasoning), ATR position sizing with the
  $-risk formula, stop-loss/take-profit placement, and every risk limit (1%/trade, 5 positions max,
  3% daily halt, 10% circuit breaker, 35% max per coin).
- Activity feed: `enter`/`exit` rows expandable (st.expander) showing the full detail: strategy
  votes table, indicator readout, the AI's full reasoning & risk commentary, and the plain-English
  explanation sentence. Recent trades table shows PnL per closed trade with its entry explanation.

tests/test_auto_trader.py: offline (use_ai=False or monkeypatched analyze_market); seed ohlcv with
synthetic trending candles for ≥2 symbols; assert: shortlist respects min_vote; position opened with
notional ≤ max_position_fraction × equity; AI confidence gate blocks low-confidence entries
(monkeypatch analyze_market to a fixed MarketAnalysis); halt path when circuit breaker active;
run_cycle never raises when a symbol's data update fails (monkeypatch update_symbol to raise).

## backend/paper_trading/scalper.py — FAST SCRIPT + AI TUNER (paper only) — owner: SCALPER agent

Two-tier design: a **fast mechanical scalper** (no LLM in the loop — trades every few minutes on
15m candles with fixed rules) + an **AI supervisor** that reviews its results hourly and re-tunes
its parameters within hard safety bounds. All orders still go through PaperTradingEngine +
RiskManager (global caps, daily-loss halt, circuit breaker gate every order). PAPER ONLY.

```
class ScalperParams(pydantic.BaseModel):        # persisted account_state key 'scalper_params'
    enabled: bool = False
    timeframe: str = "15m"                      # candles the rules read (closed candles only)
    interval_minutes: int = 2                   # how often the script runs
    tp_pct: float = 0.012                       # take-profit +1.2% (fee-aware: round trip ≈ 0.3%)
    sl_pct: float = 0.008                       # stop-loss −0.8%
    time_stop_bars: int = 16                    # exit at market after N bars if neither hit
    position_fraction: float = 0.06             # 6% of equity per scalp
    max_positions: int = 8                      # scalper's own sub-cap (global cap still applies)
    rsi_long_min: float = 50.0; rsi_long_max: float = 72.0
    allowed_sides: Literal["long","short","both"] = "both"
    cooldown_bars: int = 3                      # per-symbol wait after an exit (anti-churn)
    max_trades_per_day: int = 40                # hard anti-runaway guard
    disabled_symbols: list[str] = []            # AI tuner can bench losing coins

HARD_BOUNDS (module constant, enforced with clamping whatever the source of a change):
    tp_pct [0.006, 0.03], sl_pct [0.004, 0.02] and sl_pct < tp_pct, time_stop_bars [4, 64],
    position_fraction [0.02, 0.10], max_positions [1, 10], interval_minutes [1, 10],
    cooldown_bars [1, 20], max_trades_per_day [5, 100]. Stops can NEVER be disabled.

class Scalper:
    def __init__(self, engine: PaperTradingEngine)
    get_params()/set_params(updates: dict) -> ScalperParams    # clamp to HARD_BOUNDS, persist
    def run_tick() -> dict          # one fast pass; threading.Lock non-blocking; NEVER raises
    def stats(since_ms=None) -> dict  # {trades, wins, losses, win_rate, pnl, profit_factor,
                                      #  pnl_by_symbol: {..}, open: n, trades_today: n}
    def tune_with_ai() -> dict      # the AI supervisor pass (see below)
```

run_tick (log to bot_activity with action prefix `scalp_`: scalp_enter/scalp_exit/scalp_skip/scalp_tune/error):
1. If not enabled → return. Update 15m data for watchlist coins (bot config watchlist minus
   disabled_symbols); closed candles only (reuse the unclosed-candle + staleness guards).
2. MANAGE own positions: TP hit (close at +tp_pct), SL hit (−sl_pct), or age ≥ time_stop_bars →
   engine.close_position, log scalp_exit {reason, pnl, explanation}. (SL/TP here are engine-level
   stop_loss/take_profit set at entry — the tick just also enforces time-stop.)
3. ENTER: for each eligible coin without an open position and not in cooldown and under both caps
   and under max_trades_per_day: LONG when ema_12 > ema_26 AND rsi_14 in [rsi_long_min, rsi_long_max]
   AND close > vwap; SHORT (mirrored: rsi in [100-max, 100-min], close < vwap, ema down) when
   allowed_sides permits. qty from position_fraction like the bot (cash-capped, dust-guarded);
   stop_loss/take_profit at entry ∓/± sl_pct/tp_pct of fill. Log scalp_enter with votes-free detail:
   {side, qty, price, notional, stop_loss, take_profit, rsi_14, vwap_side, ema_state, explanation}.
Ownership separation: scalper position ids tracked in account_state 'scalper_position_ids' (JSON
list, pruned on close). AutoTrader._manage_positions MUST skip scalper-owned positions (no
ensemble-flip exits on them); Scalper manages ONLY its own. (SCALPER agent edits auto_trader.py
for this one skip-check.)

tune_with_ai (scheduled hourly when enabled; also POST endpoint):
- Build a compact report: current params + stats since last tune + per-symbol PnL + 5 worst/best
  recent trades. Prompt qwen3:14b (OllamaClient, json_mode): "You supervise this mechanical
  paper-trading script... propose parameter changes as JSON {changes: {param: value...},
  disabled_symbols: [...], reasoning: str} within these bounds: <HARD_BOUNDS>. Conservative,
  fee-aware (round trip ≈0.3%), prefer benching losing symbols over widening stops."
- Validate & CLAMP every proposed value to HARD_BOUNDS; ignore unknown params; apply via
  set_params; log scalp_tune {before, after, reasoning, model}. OllamaError → log error, change
  nothing. Return {applied: {...}, reasoning}.

API (main.py; schemas.py):
GET  /api/scalper/status      → {params, stats, running_job: bool, last_tune: {...}|null}
POST /api/scalper/start       → enable + APScheduler job 'scalper_tick' (interval_minutes,
                                max_instances=1, coalesce) + job 'scalper_tune' (60 min); status()
POST /api/scalper/stop        → disable + remove jobs; open scalp positions keep their SL/TP and
                                remain managed by a residual: keep 'scalper_tick' if positions open? NO —
                                simpler: on stop, market-close all scalper positions (log scalp_exit
                                reason="script_stopped"). Document this.
POST /api/scalper/params {subset} → clamped params
POST /api/scalper/tune        → tune_with_ai() (sync)
On API startup: re-register jobs when persisted params.enabled.

webui/index.html additions (owner: SCALPER-UI agent):
- New card "⚡ Fast script" above the activity feed: Enable/Disable button, one-line rule summary
  ("buys momentum on 15m candles · +1.2% target · −0.8% stop · exit after 4h"), stats row (today:
  trades / win rate / PnL), and "🤖 AI supervisor" box showing last tune time + the AI's reasoning
  + what it changed (before→after chips). Feed renders scalp_* rows with ⚡ prefix and the same
  expandable detail; scalp_tune rows show the reasoning inline.
- Feed/action emoji map extended: scalp_enter ⚡✅, scalp_exit ⚡🔴, scalp_tune 🛠️, scalp_skip (hidden
  unless "show scans").

tests/test_scalper.py: entry rule truth-table on crafted candles (long fires only when all three
conditions align; short mirrored; respects allowed_sides); TP/SL/time-stop exits; cooldown blocks
immediate re-entry; max_trades_per_day guard; set_params clamps out-of-bounds values; tune applies
a mocked AI response with out-of-range values → clamped, bad JSON → no change; AutoTrader manage
skips scalper-owned positions; all offline (mock OllamaClient.chat), tmp DB.

## backend/intel/ + backend/paper_trading/coach.py — AGENT TEAM (paper only) — owner: INTEL/COACH agents

Multi-agent layer on top of the traders: a **News & Community agent** (public sources, no API keys)
and a **Coach agent** (reviews every trade per coin and adapts a per-coin playbook). Both feed the
AutoTrader/Scalper. All LLM work through the existing OllamaClient. PAPER ONLY.

New tables (append to schema.sql):
```
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
CREATE TABLE IF NOT EXISTS coin_playbook (
    symbol TEXT PRIMARY KEY, updated_at INTEGER NOT NULL,
    bench INTEGER NOT NULL DEFAULT 0,                   -- 1 = don't trade this coin
    side_bias TEXT NOT NULL DEFAULT 'both' CHECK (side_bias IN ('both','long_only','short_only')),
    size_multiplier REAL NOT NULL DEFAULT 1.0,          -- clamp [0.25, 1.5]
    min_vote_override INTEGER,                          -- clamp [2,4] or NULL
    reasoning TEXT NOT NULL DEFAULT '',
    stats TEXT NOT NULL DEFAULT '{}'                    -- JSON evidence snapshot
);
```

`backend/intel/news.py` — public fetchers, requests only, 10s timeouts, NO keys, [] / None on any
failure (never raises), 10-min in-module cache:
```
COIN_NAMES: dict[str, tuple[str, str]]   # "BTCUSDT" -> ("Bitcoin","BTC") ... all 20 watchlist coins
fetch_google_news(query, limit=8) -> list[dict]   # news.google.com/rss/search?q=<query>+crypto (xml.etree)
fetch_reddit_hot(subreddit="CryptoCurrency", limit=25) -> list[dict]  # public .json, UA header
fetch_fear_greed() -> dict | None                 # api.alternative.me/fng/ {value:int, classification:str}
```

`backend/intel/sentiment.py`:
```
class SentimentAgent:
    def run(symbols: list[str]) -> dict
        # fear&greed + reddit + general news once; per-coin Google News by coin name;
        # LLM scores coins in BATCHES OF 5 per chat call (json array response) to stay fast;
        # clamp score/confidence; persist one row per coin + one 'GLOBAL' row; returns summary.
        # LLM/feed failures per batch → log, skip batch, continue (never raises).
    def get_latest(symbol=None) -> dict              # most recent row per symbol (+GLOBAL), JSON-safe
enabled/interval via account_state 'intel_config' {"enabled": bool, "interval_minutes": 30};
APScheduler job 'intel_refresh' when enabled; re-register on API startup.
```

`backend/paper_trading/coach.py`:
```
class Coach:
    def run() -> dict
        # per watchlist coin: last 20 closed trades (bot+scalper), win rate, pnl, exit reasons,
        # current playbook → LLM in batches of 5 → playbook updates
        # {bench, side_bias, size_multiplier, min_vote_override, reasoning}; CLAMP everything;
        # persist coin_playbook; log 'coach_tune' bot_activity rows {symbol, before, after, reasoning}.
        # A coin with <3 closed trades gets NO changes (not enough evidence — reset to defaults only
        # if previously benched >7 days). LLM failure → no change.
    def get_playbook() -> list[dict]
APScheduler job 'coach_run' every 120 min when intel enabled (shares intel_config.enabled).
```

Trader integration (small, surgical edits to auto_trader.py + scalper.py):
- Playbook: benched coin → skip (log skip reason "coach_bench"); side_bias filters direction;
  qty *= size_multiplier AFTER existing caps (then cash/dust guards); min_vote_override replaces
  config.min_vote for that coin.
- Sentiment gate (AutoTrader entries only): latest score ≤ -50 blocks LONG (skip "news_negative"),
  ≥ +50 blocks SHORT (skip "news_positive"); stale sentiment (>3h) = no gate.
- analyze_market prompt gains a "NEWS & SENTIMENT" block (score, summary, top 3 headlines,
  fear&greed) when fresh sentiment exists — passed in via an optional param from auto_trader,
  existing signature default None keeps all current tests green.

API: GET /api/intel/status → {enabled, last_refresh, fear_greed, global: {...}|null};
POST /api/intel/start | /api/intel/stop | /api/intel/refresh (sync);
GET /api/intel/sentiment?symbol= → {items:[...]}; GET /api/intel/playbook → {items:[...]};
POST /api/coach/run (sync).

webui/index.html: "🌍 Market intelligence" card in the left column under the equity card: Fear &
Greed value + label, global mood sentence, grid of per-coin sentiment chips (coin + colored score,
green≥+25 / red≤-25 / gray neutral), click a chip → expandable headlines + AI summary; "📒 Coach
playbook" list under it (only coins with non-default playbooks: bench/bias/size chips + reasoning
expander); Enable/Disable intelligence button; feed renders coach_tune 🎓 rows with reasoning and
skip reasons news_negative/news_positive/coach_bench like other skips. All fetches defensive
(404 → hide cards).

tests/test_intel.py: offline fixtures (canned RSS XML string, reddit JSON dict, fng JSON) via
monkeypatched requests.get; parser correctness; SentimentAgent batch prompt → mocked chat returns
array (values clamped, bad JSON → batch skipped, rows persisted); Coach clamps + <3-trades no-change
rule; playbook application: benched coin skipped by AutoTrader, size_multiplier scales qty,
sentiment -60 blocks a long entry (all with mocked LLM + tmp DB, reuse conftest patterns).

## SECOND JUDGE (Fin-R1) + SENTIMENT-BOOSTED RANKING — owner: JUDGE agent

Two upgrades to the AutoTrader's entry pipeline (scalper unaffected — it stays LLM-free):

BotConfig additions (backend/paper_trading/auto_trader.py, with persistence like existing fields):
```
use_second_judge: bool = True
second_judge_model: str = "mychen76/Fin-R1:Q5"     # finance-specialist reasoning model
judge_min_confidence: int = 55
sentiment_rank_weight: float = 0.5                  # clamp [0.0, 2.0] in set_config
```

1) SENTIMENT-BOOSTED RANKING (in _shortlist): candidates are ranked by
   `rank_score = |vote_sum| + sentiment_bonus`, tiebreak |momentum_10| (unchanged), where
   `sentiment_bonus = sentiment_rank_weight × direction × (score/100)` — direction +1 for long
   candidates, −1 for short; `score` from the freshest coin_sentiment row (reuse the existing
   _fresh_sentiment 3h rule); missing/stale sentiment → bonus 0. News NEVER qualifies a candidate
   (min_vote gate unchanged) — it only reorders qualified ones. Add `rank_score` and
   `sentiment_bonus` to the scan/enter activity detail.

2) SECOND JUDGE (in _enter_candidate, after the existing primary AI gate passes):
   - When use_second_judge: call the existing analyze_market with model=config.second_judge_model
     (same df/news_context). Require: same sentiment direction as the trade AND
     confidence ≥ judge_min_confidence — else skip with reason
     `judge_disagreement: <model> said <sentiment> at <conf>%` (logged like other skips).
   - Model-availability rule: if second_judge_model is NOT in OllamaClient().list_models()
     (match by prefix before ':' too), log ONE warning-level 'error' activity row per cycle and
     proceed WITHOUT the second judge (feature waits for the model pull to finish). If the model
     IS installed but the call raises OllamaError → conservative: skip the candidate (log 'error').
   - To avoid GPU model-swap thrash: evaluate candidates in TWO PASSES — primary gate for all
     candidates first (qwen3 stays loaded), then second-judge pass over the survivors (one swap).
   - enter detail: keep `ai` as the primary result (unchanged shape, UI back-compat) and ADD
     `ai_second: {sentiment, confidence, model} | null`.

webui/index.html (small): in the expanded enter detail, when ai_second exists render a second
aibox line "🎓 Second judge (<model>): <sentiment> · <confidence>% sure"; skip rows already render
judge_disagreement reasons via the generic path.

tests/test_second_judge.py: monkeypatch analyze_market where auto_trader looks it up, dispatching
on the model argument. Assert: both-agree → enter with ai_second populated; direction disagreement
→ skip w/ judge_disagreement reason; judge confidence < judge_min_confidence → skip; model missing
from mocked list_models → enters with warning row and ai_second null; OllamaError from judge →
skip w/ error row; sentiment_bonus reorders two equal-vote candidates (fresh sentiment) and stale
sentiment gives bonus 0; set_config clamps sentiment_rank_weight to [0, 2]. Offline, tmp DB.

## docker/ + scripts + README — owner: OPS agent

- `docker/Dockerfile.backend` (python:3.12-slim, `uvicorn backend.api.main:app --host 0.0.0.0`),
  `docker/Dockerfile.dashboard` (streamlit on 0.0.0.0:8501).
- `docker-compose.yml` (project root): services `backend` (8000:8000) + `dashboard` (8501:8501, env
  `API_URL=http://backend:8000`), shared volume `./data:/app/data`, env `OLLAMA_URL=http://host.docker.internal:11434`,
  `extra_hosts: ["host.docker.internal:host-gateway"]`. NO ollama container (runs on Windows host for GPU).
- `scripts/setup.bat` (venv + pip install + init db), `scripts/start_backend.bat`, `scripts/start_dashboard.bat`,
  `scripts/start_all.bat` — use `py` launcher, venv at `.venv`, cd to project root (`%~dp0..`).
- `README.md`: overview, PAPER-ONLY disclaimer, architecture diagram (ASCII), hardware/model guidance
  (RTX 5060 Ti 16GB → qwen3:14b default; qwen3:30b-a3b quality option; 32B dense models supported but CPU-offloaded),
  setup (Windows native + Docker), usage walkthrough, API table, ASCII dashboard mockup, troubleshooting.
