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

## Improvement Pack v3

Binding contracts for the v3 improvement wave: market-regime gating, cost gating, shadow
diagnostics, R-multiple logging, judge de-biasing, ATR-geometry scalping, shrink-only tuning,
portfolio heat / direction caps, drawdown de-risking, coach evidence gates, and an Ollama
watchdog. **PAPER ONLY — nothing here changes that.** No schema.sql changes in this pack: every
new action/reason fits the existing `bot_activity` shape and every new persisted blob lives in
`account_state`.

Conventions for this section:
- Detail-JSON shapes list REQUIRED keys. Extra keys are permitted; listed keys must be present
  with exactly these names. All numeric detail values go through the existing `_json_num`
  (round 6, NaN/inf → null).
- All `*_pct` values in the new gate/geometry math are **fractions** (0.006 = 0.6%), consistent
  with `settings.commission_rate` / `settings.slippage_rate` — NOT percentage points. Exception:
  `pct_change_7d` / `pct_change_30d` on `RegimeInfo` and `atr_pctile` in shadow flags are
  human-scale percentages (−100..100 / 0..100), because they feed prompts and dashboards.
- New reason strings are binding EXACT values (or exact prefixes where a `:`-suffixed
  explanation follows, matching the existing `stale_data: ...` convention).
- Private helper names below are RECOMMENDED; public functions, classes, fields, defaults,
  clamps, config/state keys, reason strings and detail shapes are BINDING.

### 1. backend/paper_trading/regime.py — NEW MODULE — owner: builder-regime

Pure, read-only market-regime utilities. Reads market data ONLY via
`backend.database.db.load_ohlcv` (cache-only — never fetches, never writes any table). Must not
import `engine`, `auto_trader` or `scalper` (both traders import THIS module). pandas/numpy +
`backend.indicators.technical` only. Every public function is cheap and NEVER raises (any
internal failure degrades to the conservative/neutral result and logs a warning).

```
# Module constants (binding values)
REF_TIMEFRAME = "1h"                    # source candles for the regime
RESAMPLE_RULE = "4h"                    # pandas resample rule applied to the 1h closes
REGIME_LOOKBACK_1H_BARS = 1500          # load_ohlcv limit (~62 days of 1h → ~375 4h bars)
MIN_4H_BARS = 220                       # fewer usable 4h closes → regime "neutral"
EMA_FAST = 50
EMA_SLOW = 200
SHADOW_VOL_MULT = 1.5                   # volume_ok threshold vs SMA20(volume)
SHADOW_ATR_WINDOW = 100                 # bars ranked for the ATR percentile
SHADOW_ATR_BAND = (20.0, 90.0)          # atr_in_band inclusive bounds
DEAD_ZONE_HOURS = (2, 5)                # dead_zone: 2 <= hour_utc < 5

@dataclass(frozen=True)
class RegimeInfo:
    regime: str                         # "uptrend" | "downtrend" | "neutral"
    close: float | None                 # last 4h close used (None when no data)
    ema50: float | None                 # EMA50 of 4h closes (None when < MIN_4H_BARS)
    ema200: float | None                # EMA200 of 4h closes (None when < MIN_4H_BARS)
    bars_used: int                      # usable 4h closes after resample+dropna
    pct_change_7d: float | None         # % change vs 168 1h bars ago (human %, round 4)
    pct_change_30d: float | None        # % change vs 720 1h bars ago (human %, round 4)

def get_regime(source: str, symbol: str, ref_timeframe: str = "1h") -> RegimeInfo
def regime_allows(side: str, symbol_regime: str, btc_regime: str) -> bool
def cost_gate(atr: float | None, price: float | None, commission_pct: float,
              slippage_pct: float, multiple: float) -> dict[str, Any]
def shadow_flags(df: pd.DataFrame, entry_ts_utc_hour: int) -> dict[str, Any]
```

`get_regime` computation (binding):
1. `df = load_ohlcv(source, symbol, ref_timeframe, limit=REGIME_LOOKBACK_1H_BARS)` — cache
   only. Empty → `RegimeInfo("neutral", None, None, None, 0, None, None)`.
2. `pct_change_7d` / `pct_change_30d` from the 1h closes: `(last / close[-1-N] - 1) * 100` with
   N=168 / N=720; `None` when the frame is too short (round 4).
3. `closes_4h = df["close"].resample(RESAMPLE_RULE, label="right", closed="right").last().dropna()`;
   `bars_used = len(closes_4h)`. No unclosed-candle drop (observation-grade EMA smoothing).
4. `bars_used < MIN_4H_BARS` → regime `"neutral"`, `ema50 = ema200 = None`,
   `close =` last 4h close. (Conservative by design: neutral blocks shorts, allows longs.)
5. Else `ema50/ema200 = technical.ema(closes_4h, 50/200).iloc[-1]`, `close = closes_4h.iloc[-1]`:
   `uptrend` iff `close > ema200 AND ema50 > ema200`; `downtrend` iff
   `close < ema200 AND ema50 < ema200`; else `neutral`.
Callers cache results per cycle/tick — `get_regime` itself does no caching.

`regime_allows` — the single shared gate predicate (both traders MUST use it):
```
both_down = (symbol_regime == "downtrend") and (btc_regime == "downtrend")
return both_down if side == "short" else not both_down       # side: "long" | "short"
```
i.e. SHORT allowed only when symbol AND BTCUSDT are both in a 4h downtrend; LONG blocked only in
that same double-downtrend state; every other combination allows longs and blocks shorts.

`cost_gate` (binding): `round_trip = 2.0 * (commission_pct + slippage_pct)`;
`needed_pct = max(0.0, multiple) * round_trip`; `expected_move_pct = atr / price` when both are
finite and > 0, else `None`. Returns
`{"passes": bool, "expected_move_pct": float | None, "needed_pct": float}`.
`multiple <= 0` → `passes=True, needed_pct=0.0` (gate disabled). Otherwise
`passes = expected_move_pct is not None and expected_move_pct >= needed_pct` (invalid ATR/price
fails conservatively).

`shadow_flags` (binding — PURE OBSERVATION, never gates, never raises). `df` is a
feature-enriched canonical frame (1h for the bot, 15m for the scalper);
`entry_ts_utc_hour` is `datetime.now(timezone.utc).hour` at entry time. Returns:
```
{
  "volume_ok":   bool | None,   # last volume >= SHADOW_VOL_MULT * SMA20(volume); None when
                                #   volume missing or < 20 rows
  "atr_pctile":  float | None,  # 100 * (# of window ATR values <= current) / n, window = last
                                #   min(100, available) finite atr_14 values INCL. current bar,
                                #   round 2; None when < 20 finite ATR values
  "atr_in_band": bool | None,   # 20.0 <= atr_pctile <= 90.0; None when atr_pctile is None
  "dead_zone":   bool           # 2 <= entry_ts_utc_hour < 5 (UTC)
}
```
Uses the `atr_14` column when present, else computes `technical.atr(df, 14)`.

### 2. config/settings.py additions — owner: builder-api

Append to `Settings` (all env-overridable like every existing field):
```
# --- Improvement pack v3 ---
heat_cap_fraction: float = 0.06          # portfolio stop-distance risk cap vs equity
max_same_direction: int = 5              # max open positions in one direction, platform-wide
ollama_app_path: str = "%LOCALAPPDATA%/Programs/Ollama/ollama app.exe"   # expandvars at USE time
ollama_watchdog_enabled: bool = True
```

### 3. backend/paper_trading/risk.py — owner: builder-risk

```
NO_STOP_RISK_FRACTION = 0.02             # stop-distance reference for positions without a stop

class RiskManager:                        # existing methods unchanged, plus:
    @staticmethod
    def portfolio_heat(open_positions: list[dict[str, Any]]) -> float
        # DOLLARS of stop-distance risk across OPEN positions:
        # per position: |entry_price - stop_loss| * qty when stop_loss is finite and the
        # distance > 0; else entry_price * qty * NO_STOP_RISK_FRACTION. Malformed rows
        # contribute 0. Never raises.

    def check_order(self, portfolio, open_positions, side=None, qty=None, symbol=None,
                    source=None, timeframe=None,
                    price: float | None = None,
                    stop_loss: float | None = None) -> RiskCheckResult
        # Two NEW optional params appended (existing callers unaffected). Two NEW checks
        # appended AFTER the existing four, both skipped for risk-reducing orders (existing
        # early return) and skipped when side is None:
        # 5) DIRECTION CAP: direction = "long" if side=="buy" else "short"; count open
        #    positions with that side (platform-wide, all sources/timeframes); count >=
        #    settings.max_same_direction → reject, reason EXACT prefix
        #    "direction_cap: {n} {direction} positions already open >= limit {max} — "
        #    "too much exposure in one direction".
        # 6) HEAT CAP: new_risk = |price - stop_loss| * qty when price & stop_loss finite and
        #    distance > 0; else price * qty * NO_STOP_RISK_FRACTION when price finite;
        #    else 0.0 (unknown context — degraded check on existing heat only).
        #    projected = portfolio_heat(open_positions) + new_risk. Reject when
        #    projected > settings.heat_cap_fraction * equity + 1e-9, reason EXACT prefix
        #    "heat_cap: total stop-distance risk would reach {projected/equity:.2%} of "
        #    "equity, above the {settings.heat_cap_fraction:.2%} cap — the account would "
        #    "lose too much if every stop was hit at once".

    @staticmethod
    def soft_daily_stop_active(equity: float, realized_pnl_today: float,
                               daily_limit_fraction: float) -> bool
        # equity > 0 and realized_pnl_today <= -0.8 * daily_limit_fraction * equity.
        # (Base is CURRENT equity — the closest thing available from these args.)

    @staticmethod
    def derisk_multiplier(equity: float, peak_equity: float, realized_pnl_today: float,
                          daily_limit_fraction: float) -> float
        # dd_pct = 100 * max(0, (peak_equity - equity) / peak_equity) when peak_equity > 0
        #          else 0.0
        # m = 0.8 ** int(dd_pct // 10)
        # if soft_daily_stop_active(...): m *= 0.5
        # return min(m, 1.0)              # always in (0, 1]; invalid inputs → 1.0
```
The engine's existing `check_order` call (no price/stop) gets the direction cap for free and the
heat cap in degraded form; FULL heat enforcement happens via the traders' pre-submit check
(sections 4 and 6). `NO_STOP_RISK_FRACTION` positions-without-stop rule applies on both sides of
the sum.

### 4. backend/paper_trading/auto_trader.py — owner: builder-trader

`BotConfig` new fields (persisted like all others; old persisted configs load fine via defaults):
```
regime_gate_enabled: bool = True
cost_gate_multiple: float = 3.0          # field_validator clamps to [0.0, 10.0]; 0 disables
```
Clamp validator mirrors `_clamp_sentiment_rank_weight` (clamp, don't reject; NaN → 0.0).

Cycle integration (binding behavior):
- **Per-cycle regime cache**: at most ONE `regime.get_regime(config.source, sym)` call per
  symbol per cycle, and BTCUSDT computed at most once per cycle and reused for every candidate
  (recommended: a `dict[str, RegimeInfo]` created in `_run_cycle_locked`, lazy-filled).
- **Pass-1 hard gates** — at the TOP of `_confirm_candidate`, in this order, ALL before any
  `analyze_market` call (GPU saved; ensemble votes already exist at this point, so include
  them):
  1. **Regime gate** (only when `config.regime_gate_enabled`): direction from `vote_sum`;
     blocked when `not regime.regime_allows(direction, symbol_regime, btc_regime)` →
     `counters["skipped"] += 1`, log `skip` with detail:
     ```
     {"reason": "regime_block", "side": "long"|"short",
      "symbol_regime": str, "btc_regime": str,
      "close": num|null, "ema50": num|null, "ema200": num|null,   # the SYMBOL's RegimeInfo
      "vote_sum": int, "votes": {strategy: int, ...},
      "explanation": <plain English, e.g. "Skipped a short in ETHUSDT: shorts are only
       allowed when both the coin and Bitcoin are in a 4h downtrend — ETHUSDT is neutral
       and BTC is uptrend.">}
     ```
     return False.
  2. **Cost gate** (only when `config.cost_gate_multiple > 0`): `regime.cost_gate(atr_14 of the
     candidate df's last row, last close, settings.commission_rate, settings.slippage_rate,
     config.cost_gate_multiple)`; not passing → `counters["skipped"] += 1`, log `skip` with:
     ```
     {"reason": "cost_gate", "expected_move_pct": num|null, "needed_pct": num,
      "explanation": <e.g. "Skipped BTCUSDT: the hourly range is 0.21% of price but fees +
       slippage need at least 0.90% of expected movement to be worth trading.">}
     ```
     return False.
  3. Existing news-sentiment veto, then the existing primary AI gate (both unchanged).
- **AI-offline cycle skip** (watchdog integration): before the pass-1 loop, when
  `config.use_ai` and the shortlist is non-empty and the persisted `account_state` key
  `'ollama_status'` exists with `available == False` (missing/malformed key = ONLINE), log ONE
  `skip` row (symbol `""`) and drop the whole shortlist
  (`counters["skipped"] += len(shortlist)`), touching no candidate:
  ```
  {"reason": "ai_offline", "since_ms": int|null, "skipped_candidates": int,
   "explanation": "Skipped all AI confirmations this cycle: the local AI (Ollama) is
    offline — the watchdog is trying to restart it. AI-gated entries stay off until it
    is back."}
  ```
- **Pre-submit risk check** (heat/direction, full context) in `_enter_candidate` AFTER
  stop/take-profit are computed, immediately BEFORE `engine.submit_order`:
  `self.risk.check_order(portfolio, self.engine.get_positions("open"), side=side, qty=qty,
  symbol=symbol, source=config.source, timeframe=config.timeframe, price=price,
  stop_loss=stop_loss)` — not allowed → `counters["rejected"] += 1`, log `reject` with
  `{"reason": <RiskCheckResult.reason verbatim>, "explanation": ...}`, return.
- **De-risk multiplier**: in `_enter_candidate`, as the FINAL sizing step (after the playbook
  multiplier and its cap re-application, immediately before the dust guard):
  `m = RiskManager.derisk_multiplier(equity, portfolio["peak_equity"],
  portfolio["realized_pnl_today"], settings.daily_loss_limit)`; `qty *= m`. The bot still
  enters during a soft daily stop — at the halved size.
- **`enter` detail additions** (required keys on every `enter` row):
  `"shadow_flags": regime.shadow_flags(candidate df, current UTC hour)` and
  `"derisk_multiplier": num` (1.0 when no de-risk). The `ai` payload gains
  `"opposing_case": str` (see section 5).
- **`exit` detail additions** (every bot close row, i.e. everything through `_log_exit`):
  ```
  "designed_r":  |take_profit - entry_price| / |entry_price - stop_loss|
                 → null when stop_loss or take_profit is null/invalid or the stop
                   distance is <= 0
  "realized_r":  pnl / (|entry_price - stop_loss| * qty)
                 → null when stop_loss null/invalid, qty <= 0, or denominator <= 0
  ```
  Both from the closed-position dict fields (`entry_price`, `stop_loss`, `take_profit`,
  `qty`, `pnl`), through `_json_num`.
- **HTF context for the analysts** (see section 5): builder-trader adds
  `_htf_context(info: RegimeInfo) -> str` producing newline-joined factual lines, each part
  omitted when unavailable, EMPTY string when nothing is available:
  ```
  "7-day price change: {pct_change_7d:+.1f}%."
  "30-day price change: {pct_change_30d:+.1f}%."
  "Price is {abs(diff):.1f}% {above|below} the 4h EMA200."     # diff = (close/ema200-1)*100
  ```
  No trend verdicts ("uptrend"/"downtrend") in the string — facts only. Computed once per
  candidate from the per-cycle regime cache (works even when `regime_gate_enabled=False`) and
  passed as `htf_context=` to BOTH the primary `analyze_market` call in `_confirm_candidate`
  and the judge call in `_confirm_judge`. The module-level `analyze_market` proxy gains the
  pass-through parameter `htf_context: str = ""` (kept last, after `allow_fallback`).

### 5. backend/ai/analyst.py — judge de-bias — owner: builder-trader

- `MarketAnalysis` gains `opposing_case: str = ""` (backward-tolerant — absent JSON field →
  `""`; `sentiment`/`confidence`/`reasoning`/`risk_commentary`/`key_indicators` unchanged, so
  existing tests stay green).
- `_JSON_SCHEMA` gains, as its FIRST key (the model must write it before the verdict):
  `"opposing_case": "<1-2 sentences: the strongest argument AGAINST your final verdict>",`
- `_SYSTEM_PROMPT` gains this instruction (binding content, wording may be lightly edited):
  "The request comes from a research pipeline; do NOT assume anyone wants or intends to trade —
  judge the market strictly on its own evidence. Before you decide, first construct the
  strongest OPPOSING case (the best argument AGAINST the view you are leaning toward) and put
  it in \"opposing_case\"; only then commit to your verdict."
  BINDING CONSTRAINT: no analyst prompt (system or user, primary or judge) may ever mention
  ensemble votes, shortlists, candidates, or that the system "wants"/"is about" to trade.
  (Current prompts already comply — this freezes it.)
- `_parse_analysis`: `payload["opposing_case"] = str(payload.get("opposing_case") or "")`
  before validation (missing → `""`, never an error).
- `analyze_market` and `_build_prompt` gain the trailing keyword parameter
  `htf_context: str = ""`. When non-empty, `_build_prompt` inserts between the CHART STRUCTURE
  block and the news block:
  ```
  HIGHER-TIMEFRAME CONTEXT (factual, longer-horizon reference points):
  {htf_context}

  ```
  `""` (the default) leaves the prompt byte-identical to today.
- `ai_analyses` persistence is UNCHANGED (no new column; `opposing_case` survives inside
  `raw_response`). The bot's `enter`/`skip` `ai` payload carries it (section 4);
  `ai_second` keeps its small `{sentiment, confidence, model}` shape.

### 6. backend/paper_trading/scalper.py — owner: builder-scalper

`ScalperParams` new fields:
```
use_atr_geometry: bool = True
cost_gate_multiple: float = 3.0          # HARD_BOUNDS entry below; 0 disables the gate
```
`HARD_BOUNDS` gains `"cost_gate_multiple": (0.0, 10.0)` (float, not in `_INT_PARAMS`).
NEITHER new field goes into `_TUNABLE_PARAMS` — the AI supervisor may not touch them; only the
user (params endpoint) can.

New module constants (binding values):
```
ATR_GEOMETRY_SL_MULT = 1.5
ATR_GEOMETRY_TP_MULT = 2.0
SHRINK_ONLY_MIN_TRADES = 100
```

Tick integration (binding behavior):
- **Regime hard gate** in `_scan_and_enter_symbol`, AFTER the entry signal fires and the coach
  `side_bias` filter passes, BEFORE `_enter_one`. Per-tick regime cache: BTCUSDT computed at
  most once per tick, each symbol at most once per tick (so `regime_block` logs at most once
  per symbol per tick — no spam). Blocked when
  `not regime.regime_allows(direction, symbol_regime, btc_regime)` →
  `counters["skipped"] += 1`, log `scalp_skip` with the SAME detail shape as the bot's
  `regime_block` (section 4) minus the `vote_sum`/`votes` keys. The predicate is shared —
  scalp longs are likewise blocked in a BTC+symbol double-downtrend.
- **Cost gate** immediately after the regime gate, using the 15m frame's last-row `atr_14` and
  `close`, `multiple=params.cost_gate_multiple`: not passing → `counters["skipped"] += 1`,
  `scalp_skip` with the bot's `cost_gate` detail shape.
- **Soft daily stop**: once per tick, before the entry loop (manage pass always runs): when
  `RiskManager.soft_daily_stop_active(equity, portfolio["realized_pnl_today"],
  settings.daily_loss_limit)` → NO new scalps this tick; log ONE `scalp_skip` row:
  ```
  {"reason": "soft_daily_stop", "realized_pnl_today": num, "daily_limit_pct": num,
   "explanation": "Skipped all new scalps: today's realized loss is at 80% of the daily
    loss limit — the scalper stands down while the slower bot may still trade at half size."}
  ```
  Tick summary `status` = `"soft_stop"`.
- **ATR geometry** in `_enter_one`: extract `atr_14` alongside the existing last-row values.
  When `params.use_atr_geometry` and `atr_14` is finite and > 0:
  ```
  sl_pct_eff = clamp(ATR_GEOMETRY_SL_MULT * atr_14 / price, HARD_BOUNDS["sl_pct"])   # [0.004, 0.02]
  tp_pct_eff = clamp(ATR_GEOMETRY_TP_MULT * sl_pct_eff,     HARD_BOUNDS["tp_pct"])   # [0.006, 0.03]
  ```
  (`price` = the same re-read sizing/fill-basis price the stops are placed from; the formula
  guarantees `sl_pct_eff < tp_pct_eff`.) These replace `params.sl_pct`/`params.tp_pct` for
  stop/target placement on THIS entry. Disabled or invalid ATR → fixed `params.tp_pct`/
  `params.sl_pct` (the fallback and still the tuner's object).
- **Pre-submit risk check** in `_enter_one` right before `engine.submit_order` (after the
  existing enabled/position re-checks): `self.risk.check_order(portfolio,
  self.engine.get_positions("open"), side=side, qty=qty, symbol=symbol, source=source,
  timeframe=params.timeframe, price=price, stop_loss=stop_loss)` — not allowed →
  `counters["skipped"] += 1`, `scalp_skip` with
  `{"reason": <RiskCheckResult.reason verbatim>, "explanation": ...}` (reason starts with
  `heat_cap:` or `direction_cap:`), return False. Engine `ValueError` rejections keep the
  existing `risk_rejected: {exc}` reason.
- **De-risk multiplier**: `qty *= RiskManager.derisk_multiplier(...)` (same args as the bot) as
  the final sizing step before the dust guard.
- **`scalp_enter` detail additions** (required): `"shadow_flags"` (from
  `regime.shadow_flags` on the 15m frame, current UTC hour), `"derisk_multiplier": num`, and
  `"atr_geometry": bool` (true when the geometry path priced the stops). The existing
  `tp_pct`/`sl_pct` keys carry the EFFECTIVE values used for this entry.
- **`scalp_exit` detail additions**: `designed_r` and `realized_r`, same formulas and
  null-safety as the bot (section 4), computed in `_log_scalp_exit` from the position dict.
- **AI-offline tune skip**: `tune_with_ai`, after the enabled check, reads the persisted
  `'ollama_status'` (missing/malformed = online); when `available == False` → change nothing,
  log ONE `scalp_skip` `{"reason": "ai_offline", "explanation": ...}` and return
  `{"applied": {}, "reasoning": "", "status": "ai_offline"}`.

**Shrink-only tuner** (in `tune_with_ai`, applied to the SANITIZED update dict before
`set_params`, comparing proposed vs current persisted values):
- While `self.stats()["trades"] < SHRINK_ONLY_MIN_TRADES` (closed scalp trades since the last
  account reset), a proposal may NEVER (a) raise `sl_pct` (widen the stop), (b) raise
  `max_positions`, (c) raise `position_fraction`, (d) raise `max_trades_per_day`. Each
  offending field is replaced by its current value and its name appended to a
  `shrink_only_clamped` list.
- **Martingale ban** (ALWAYS, regardless of trade count): when the proposal raises
  `position_fraction` AND the tune report's `stats_since_last_tune["pnl"] < 0`, the
  `position_fraction` change is dropped (current value kept) and `martingale_blocked = true`.
- The `scalp_tune` detail gains two required keys on every completed tune:
  `"shrink_only_clamped": [field, ...]` (possibly `[]`) and `"martingale_blocked": bool`.

### 7. backend/paper_trading/coach.py — shrink-only coach — owner: builder-coach

New module constants (binding values):
```
COACH_SIZE_UP_MIN_TRADES = 20            # evidence needed before size_multiplier may exceed 1.0
COACH_SIDE_BIAS_MIN_TRADES = 10          # evidence needed before side_bias may leave "both"
```
`_sanitize_proposal` gains the coin's evidence trade count and returns the clamp audit:
```
_sanitize_proposal(item: dict, current: dict, trades: int)
    -> tuple[dict, str, list[str]]       # (clamped entry, reasoning, shrink_only_clamped)
```
Rules applied AFTER the existing bounds clamps:
- `trades < COACH_SIZE_UP_MIN_TRADES` → `size_multiplier = min(value, 1.0)`; when this changed
  the outcome, append `"size_multiplier"` to the clamp list. At `trades >= 20` the existing
  `[0.25, 1.5]` clamp applies unchanged.
- `trades < COACH_SIDE_BIAS_MIN_TRADES` → resulting `side_bias` is forced to `"both"`
  (whatever the proposal or the stored row said); when this overrode a non-"both" value,
  append `"side_bias"`.
- Bench stays allowed at the existing `MIN_TRADES_FOR_CHANGE = 3` (unchanged), as is the
  7-day auto-unbench reset.
Every `coach_tune` detail gains the required key `"shrink_only_clamped": [field, ...]`
(possibly `[]`; the rule-based auto-reset rows log `[]`).

### 8. Ollama watchdog + health + web UI — owner: builder-api (main.py, settings.py, schemas.py, webui/index.html)

Constants in `backend/api/main.py` (binding):
```
OLLAMA_WATCHDOG_JOB_ID = "ollama_watchdog"
OLLAMA_STATUS_KEY = "ollama_status"           # account_state key — the cross-module contract
OLLAMA_FAIL_THRESHOLD = 3                     # consecutive failures before "offline"
OLLAMA_RESTART_COOLDOWN_MS = 600_000          # at most one restart attempt per 10 minutes
```
Persisted status shape (`account_state['ollama_status']`, written ONLY by the watchdog;
read by traders and `/health`):
```
{"available": bool, "since_ms": int, "restarts": int}
# since_ms  = epoch ms when the CURRENT available/unavailable state began
# restarts  = cumulative restart attempts (persisted, monotonically increasing)
```
Job: registered in `lifespan` (after the intel restore block) when
`settings.ollama_watchdog_enabled` — `add_job(_run_ollama_watchdog, trigger="interval",
seconds=60, id=OLLAMA_WATCHDOG_JOB_ID, replace_existing=True, coalesce=True, max_instances=1,
misfire_grace_time=30)` on the shared bot scheduler. `_run_ollama_watchdog` NEVER raises.
Module state (lock-guarded): consecutive-failure count + last-restart-attempt ms.

Watchdog pass (binding behavior):
1. `ok = OllamaClient().is_available()`.
2. `ok=True`: reset the failure counter; when the persisted status is missing or says
   unavailable, write `{"available": True, "since_ms": now, "restarts": <kept>}`. No activity
   row on recovery.
3. `ok=False`: increment the counter. Exactly on reaching `OLLAMA_FAIL_THRESHOLD` (state
   transition — NOT every minute): persist `{"available": False, "since_ms": now,
   "restarts": <kept>}` and write ONE `bot_activity` `error` row (symbol `""`; may reuse
   `AutoTrader._log`):
   ```
   {"where": "ollama_watchdog", "reason": "ai_offline", "consecutive_failures": 3,
    "restart_attempted": bool, "app_path": str,
    "explanation": "The local AI (Ollama) stopped answering — AI-confirmed entries are
     paused and the watchdog is trying to restart it."}
   ```
   At threshold AND on every later failing pass: attempt a restart when the last attempt is
   more than `OLLAMA_RESTART_COOLDOWN_MS` ago — `path = os.path.expandvars
   (settings.ollama_app_path)`; only when `os.path.isfile(path)`:
   `subprocess.Popen([path], shell=False, stdout=DEVNULL, stderr=DEVNULL)` in try/except
   (log, never raise); increment the persisted `restarts`; record the attempt time. A missing
   file (Docker/POSIX) logs a warning and skips the launch — never crashes.

`GET /health` (schemas.py `HealthResponse`): keeps the existing live `ollama_available` check
and gains `ollama_since_ms: int | None = None`, read from the persisted status (`None` when the
watchdog has never written it).

`webui/index.html`: add `<div id="bannerAI" class="banner"></div>` directly after
`<div id="bannerApi" ...>`. In the existing health-poll handler: when the API is reachable and
`health.ollama_available === false` → `className = "banner err"`, innerHTML EXACTLY:
`🔌 AI offline — the trading brain is unreachable; AI-gated trades are paused. Watchdog is trying to restart it.`
— else `className = "banner"` (auto-clears). No other styling (reuses the existing banner CSS).

Trader read rule (sections 4 and 6): traders read `account_state['ollama_status']` directly
via `PaperTradingEngine._state_get` (like the sentiment table reads); a MISSING or malformed
key means ONLINE — a watchdog that never ran must not block trading.

### 9. Activity reference — new/changed rows (all within existing `bot_activity` actions)

| action | reason | required detail keys |
|---|---|---|
| `skip` (bot) | `regime_block` | side, symbol_regime, btc_regime, close, ema50, ema200, vote_sum, votes, explanation |
| `scalp_skip` | `regime_block` | side, symbol_regime, btc_regime, close, ema50, ema200, explanation |
| `skip` / `scalp_skip` | `cost_gate` | expected_move_pct, needed_pct, explanation |
| `skip` (bot, ≤1/cycle) | `ai_offline` | since_ms, skipped_candidates, explanation |
| `scalp_skip` (tune) | `ai_offline` | explanation |
| `scalp_skip` (≤1/tick) | `soft_daily_stop` | realized_pnl_today, daily_limit_pct, explanation |
| `reject` (bot) | `heat_cap: ...` / `direction_cap: ...` (verbatim RiskCheckResult.reason) | reason, explanation |
| `scalp_skip` | `heat_cap: ...` / `direction_cap: ...` (verbatim) | reason, explanation |
| `enter` | — | + shadow_flags, derisk_multiplier; ai gains opposing_case |
| `scalp_enter` | — | + shadow_flags, derisk_multiplier, atr_geometry (tp_pct/sl_pct = effective) |
| `exit` / `scalp_exit` | — | + designed_r, realized_r |
| `scalp_tune` | — | + shrink_only_clamped, martingale_blocked |
| `coach_tune` | — | + shrink_only_clamped |
| `error` (watchdog, transition only) | `ai_offline` | where, reason, consecutive_failures, restart_attempted, app_path, explanation |

The web UI's generic skip/reason rendering path handles every new reason; no UI work is needed
beyond the section-8 banner.

### 10. File ownership (binding — edit ONLY your files; interfaces above are the seams)

| file | owner |
|---|---|
| `backend/paper_trading/regime.py` (new) | builder-regime |
| `backend/paper_trading/auto_trader.py`, `backend/ai/analyst.py` | builder-trader |
| `backend/paper_trading/scalper.py` | builder-scalper |
| `backend/paper_trading/risk.py` | builder-risk |
| `backend/paper_trading/coach.py` | builder-coach |
| `backend/api/main.py`, `backend/api/schemas.py`, `config/settings.py`, `webui/index.html` | builder-api |
| `tests/test_regime.py`, `tests/test_improvements_risk_coach.py` (new) | test-agent-1 |
| `tests/test_improvements_traders.py`, `tests/test_watchdog.py` (new) | test-agent-2 |
| `CONTRACTS.md` | spec agent ONLY |

No one touches `engine.py`, `schema.sql`, or any existing test file in this pack. Test scope:
`test_regime.py` = section-1 pure functions (synthetic frames, tmp DB for `get_regime`);
`test_improvements_risk_coach.py` = heat/direction caps, `derisk_multiplier`,
`soft_daily_stop_active`, coach shrink-only rules; `test_improvements_traders.py` = both
traders' regime/cost gates, ATR geometry, shrink-only tuner + martingale ban, R logging,
`htf_context`/`opposing_case` plumbing (monkeypatched `get_regime`/`analyze_market`);
`test_watchdog.py` = watchdog transitions, restart cooldown + missing-file guard (mocked
`Popen`/`is_available`), `ai_offline` cycle/tune skips, `/health` fields. All offline — no
network, no Ollama, tmp DB via the existing conftest patterns.

## Indicator Pack v4

Binding contracts for the v4 indicator expansion: ~20 new pure indicator functions in
`backend/indicators/technical.py`, the canonical `FEATURE_COLUMNS` list grows from 17 to **48**
(original 17 unchanged and FIRST), and `feature_summary` gains new qualitative flags plus a
compact nested `groups` dict for the analyst prompt. **PAPER ONLY — unchanged.** Pure
pandas/numpy only (no TA-Lib). The no-lookahead invariant is absolute: every new column at row
`t` uses data at rows `<= t` only, and the existing prefix-stability test semantics
(`add_features(df[:100]) == add_features(df)[:100]`) MUST hold over the FULL 48-column set.

Conventions for this section (all binding):
- **"Wilder smoothing"** always means `ewm(alpha=1/window, adjust=False).mean()` — identical to
  the existing `rsi`/`atr` implementations. **"SMA"/"rolling"** always means
  `rolling(window, min_periods=window)` — identical to the existing `sma`. **"EMA"** always
  means `ewm(span=window, adjust=False).mean()` — identical to the existing `ema`.
- All outputs are float64 Series (or tuples of Series) aligned to the input index, NaN during
  warm-up. No artificial NaN masking beyond what the formulas produce (consistent with
  `rsi`/`atr`: early ewm values exist but are unsettled — tests must not pin exact values in
  the first `window` bars of any Wilder-smoothed output).
- New functions reuse `_validate_window` / `_require_columns`, get Google-style docstrings and
  are appended to `__all__`. The existing 11 functions are byte-identical after this pack.
- `df` parameters are canonical OHLCV frames; `s` parameters are price Series (typically close).

### 1. backend/indicators/technical.py — NEW FUNCTIONS — owner: builder-technical

```
adx(df, window=14) -> tuple[pd.Series, pd.Series, pd.Series]        # (adx, di_plus, di_minus)
stochastic(df, k_window=14, d_window=3, smooth_k=3) -> tuple[pd.Series, pd.Series]   # (k, d)
stoch_rsi(s, rsi_window=14, stoch_window=14, k=3, d=3) -> tuple[pd.Series, pd.Series] # (k, d)
williams_r(df, window=14) -> pd.Series
cci(df, window=20) -> pd.Series
roc(s, window=10) -> pd.Series                                      # PERCENT
mfi(df, window=14) -> pd.Series
obv(df) -> pd.Series
cmf(df, window=20) -> pd.Series
adl(df) -> pd.Series
vwma(df, window=20) -> pd.Series
supertrend(df, window=10, multiplier=3.0) -> tuple[pd.Series, pd.Series]  # (line, direction)
psar(df, af_start=0.02, af_step=0.02, af_max=0.2) -> pd.Series
aroon(df, window=25) -> tuple[pd.Series, pd.Series]                 # (aroon_up, aroon_down)
donchian(df, window=20) -> tuple[pd.Series, pd.Series, pd.Series]   # (upper, mid, lower)
keltner(df, window=20, atr_window=10, multiplier=2.0) -> tuple[pd.Series, pd.Series, pd.Series]
hull_ma(s, window=20) -> pd.Series
trix(s, window=15) -> pd.Series                                     # PERCENT
ichimoku(df) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]   # (tenkan, kijun, senkou_a, senkou_b)
bollinger_pct_b(close, upper, lower) -> pd.Series                   # lives HERE (pinned below)
bollinger_bandwidth(upper, mid, lower) -> pd.Series                 # lives HERE (pinned below)
```

Canonical formulas (binding — variant named in each case):

- **adx** — Wilder throughout, consistent with existing `rsi`/`atr`.
  `up = high.diff()`, `down = -low.diff()`;
  `+DM = up where (up > down) & (up > 0) else 0`; `-DM = down where (down > up) & (down > 0) else 0`.
  `di_plus = 100 * wilder(+DM) / atr(df, window)`, `di_minus = 100 * wilder(-DM) / atr(df, window)`
  (reuse the existing `atr` — Wilder-smoothed TR; mask division by 0 → NaN).
  `dx = 100 * |di_plus - di_minus| / (di_plus + di_minus)` (denominator 0 → NaN);
  `adx = wilder(dx)`. All three in `[0, 100]`.
- **stochastic** — classic slow stochastic, SMA smoothing:
  `raw_k = 100 * (close - LL) / (HH - LL)` with `HH = high.rolling(k_window).max()`,
  `LL = low.rolling(k_window).min()` (min_periods=k_window; `HH == LL` → NaN);
  `k = sma(raw_k, smooth_k)`; `d = sma(k, d_window)`. Range `[0, 100]`.
- **stoch_rsi** — stochastic OF the existing Wilder `rsi`, pinned to the **0–100 scale**
  (TradingView plot scale, consistent with `stochastic` and the 20/80 zone flags):
  `r = rsi(s, rsi_window)`; `raw = 100 * (r - r.rolling(stoch_window).min()) /
  (r.rolling(stoch_window).max() - r.rolling(stoch_window).min())` (flat window → NaN);
  `k_line = sma(raw, k)`; `d_line = sma(k_line, d)`.
- **williams_r** — `-100 * (HH - close) / (HH - LL)` with HH/LL over `window` bars including
  the current bar (`HH == LL` → NaN). Range `[-100, 0]`.
- **cci** — `tp = (high + low + close) / 3`;
  `cci = (tp - sma(tp, window)) / (0.015 * MAD)` where `MAD` = rolling mean absolute deviation
  of `tp` about the rolling mean over `window` bars (`MAD == 0` → NaN). Implementation free
  (e.g. `rolling(window).apply(..., raw=True)`); formula binding.
- **roc** — `100 * (s / s.shift(window) - 1)` — human-scale **percent** (= `momentum * 100`).
- **mfi** — rolling-SUM variant (classic, NOT Wilder): `tp = (high+low+close)/3`;
  `rmf = tp * volume`; positive flow where `tp > tp.shift(1)`, negative where `tp < tp.shift(1)`,
  ties contribute to neither; `pos/neg = rolling(window, min_periods=window).sum()` of each;
  `mfi = 100 * pos / (pos + neg)`; pinned edge cases mirroring `rsi`: `neg == 0 & pos > 0` → 100,
  `pos == 0 & neg > 0` → 0, both 0 → 50. Range `[0, 100]`.
- **obv** — `(sign(close.diff()) * volume).fillna(0).cumsum()`; first bar contributes 0;
  unbounded, in volume units.
- **cmf** — Money Flow Multiplier `mfm = ((close - low) - (high - close)) / (high - low)`,
  pinned `mfm = 0.0` where `high == low`; `mfv = mfm * volume`;
  `cmf = rolling(window).sum()(mfv) / rolling(window).sum()(volume)` (volume sum 0 → NaN).
  Range `[-1, 1]`.
- **adl** — Accumulation/Distribution Line: `mfv.cumsum()` (same `mfm`, same `high == low` → 0
  rule). Unbounded.
- **vwma** — `rolling(window).sum()(close * volume) / rolling(window).sum()(volume)`
  (volume sum 0 → NaN).
- **supertrend** — Wilder ATR (reuse `atr(df, window)`), `hl2 = (high + low) / 2`;
  `upper_basic = hl2 + multiplier * atr`; `lower_basic = hl2 - multiplier * atr`.
  Rows `0 .. window-1` are NaN (line AND direction — pinned warm-up). Recursion starts at
  `t0 = window` (0-indexed) with `final_upper = upper_basic[t0]`, `final_lower =
  lower_basic[t0]`, `direction[t0] = +1.0 if close[t0] >= hl2[t0] else -1.0` (pinned seed;
  self-corrects within a few bars). For `t > t0`:
  `final_upper[t] = upper_basic[t] if (upper_basic[t] < final_upper[t-1] or close[t-1] >
  final_upper[t-1]) else final_upper[t-1]`; `final_lower[t]` mirrored;
  `direction[t] = +1 if close[t] > final_upper[t-1]; -1 if close[t] < final_lower[t-1];
  else direction[t-1]`. `line = final_lower where direction == +1 else final_upper`.
  `direction` is float `+1.0 / -1.0` (NaN in warm-up). O(n) Python loop acceptable.
- **psar** — classic Wilder parabolic SAR, iterative O(n) loop (explicitly acceptable).
  `psar[0] = NaN`; needs ≥ 2 bars (1-row frame → all-NaN). Initial trend at bar 1: up if
  `close[1] >= close[0]` else down; initial `sar = low[0]` (up) / `high[0]` (down); initial
  `ep = max(high[0], high[1])` (up) / `min(low[0], low[1])` (down); `af = af_start`.
  Per bar `t >= 2`: `sar[t] = sar[t-1] + af * (ep - sar[t-1])`; clamp — uptrend
  `sar[t] = min(sar[t], low[t-1], low[t-2])`, downtrend `sar[t] = max(sar[t], high[t-1],
  high[t-2])`. Reversal when the bar pierces the SAR (uptrend: `low[t] < sar[t]`; downtrend:
  `high[t] > sar[t]`): flip trend, `sar[t] = prior ep`, `ep = low[t]` / `high[t]`, `af =
  af_start`. Otherwise on a new extreme (`high[t] > ep` up / `low[t] < ep` down): `ep =
  extreme`, `af = min(af + af_step, af_max)`. Returns the SAR level in effect at each bar.
- **aroon** — over the last `window + 1` bars INCLUDING the current bar (needs `window + 1`
  rows; first `window` rows NaN):
  `aroon_up = 100 * (window - bars_since_highest_high) / window`; `aroon_down` mirrored with
  the lowest low. Tie rule (pinned): the MOST RECENT occurrence of the extreme wins (a flat
  window yields 100/100). Range `[0, 100]`.
- **donchian** — `upper = high.rolling(window).max()`, `lower = low.rolling(window).min()`,
  `mid = (upper + lower) / 2` — INCLUDES the current bar, no shift (consumers wanting prior-bar
  channels shift themselves, as the breakout strategy already does).
- **keltner** — `mid = ema(close, window)` (**EMA mid — pinned**); `upper/lower = mid ±
  multiplier * atr(df, atr_window)` (Wilder ATR).
- **hull_ma** — `HMA = wma(2 * wma(s, half) - wma(s, window), sqrt_w)` where `wma` is the
  linearly-weighted MA (weights `1..k`, most recent heaviest, `rolling(k, min_periods=k)`),
  `half = max(1, window // 2)` (floor), `sqrt_w = max(1, int(round(math.sqrt(window))))`
  (round — TradingView convention; for window=20 → half=10, sqrt_w=4).
- **trix** — triple EMA: `t3 = ema(ema(ema(s, window), window), window)`;
  `trix = 100 * (t3 / t3.shift(1) - 1)` — 1-bar **percent** rate of change.
- **ichimoku** — fixed classic constants, no parameters: `tenkan = (high.rolling(9).max() +
  low.rolling(9).min()) / 2`; `kijun = same with 26`; `senkou_a_raw = (tenkan + kijun) / 2`;
  `senkou_b_raw = (high.rolling(52).max() + low.rolling(52).min()) / 2`; displacement = 26.
  **PINNED LOOKAHEAD CONVENTION — "cloud in effect at t":** the returned spans are the raw
  spans shifted FORWARD by the displacement — `senkou_a = senkou_a_raw.shift(26)`,
  `senkou_b = senkou_b_raw.shift(26)` — i.e. row `t` holds the cloud a chartist actually sees
  above/below price at time `t`, computed exclusively from data `<= t-26`. Justification:
  (a) it is the only convention in which the canonical price-vs-cloud signal
  (`ichimoku_state`) means what every chart and textbook means by it; (b) the unshifted
  as-computed values at `t` are just an average of tenkan/kijun-family windows ending at `t` —
  redundant with columns we already store and NOT a "cloud" relative to current price;
  (c) `shift(+26)` is strictly backward-looking, so prefix-stability holds unchanged.
  Warm-up: tenkan NaN first 8 rows, kijun first 25, senkou_a first 51 (25+26), senkou_b first
  77 (51+26) — callers need ~80+ rows for a full set (bot uses limit=500: fine).
  **chikou is EXCLUDED** from this platform entirely — it is `close.shift(-26)` (the future by
  construction) and must never appear in any function, column or prompt.
- **bollinger_pct_b / bollinger_bandwidth** — PINNED DECISION: these live in `technical.py`
  (not features.py) because they are pure indicator math reusable by strategies; features.py
  stays a thin wiring layer. `bollinger_pct_b = (close - lower) / (upper - lower)` (band width
  0 → NaN; NOT clipped — values outside `[0, 1]` are meaningful band breaks).
  `bollinger_bandwidth = (upper - lower) / mid` (`mid == 0` → NaN).

### 2. backend/indicators/features.py — FEATURE_COLUMNS v4 — owner: builder-features

`FEATURE_COLUMNS` becomes EXACTLY this 48-name tuple, in EXACTLY this order — the original 17
unchanged and FIRST (backward compatibility: every existing consumer indexes by name and is
unaffected; anything indexing by position past the OHLCV+17 boundary was never contractual):

```
sma_20, sma_50, ema_12, ema_26, rsi_14, macd, macd_signal, macd_hist,
bb_upper, bb_mid, bb_lower, atr_14, vwap, ret_1, ret_5, volatility_20, momentum_10,
adx_14, di_plus_14, di_minus_14, stoch_k, stoch_d, stoch_rsi_k, williams_r_14,
cci_20, roc_10, mfi_14, obv, cmf_20, vwma_20, rel_volume_20,
supertrend_10_3, supertrend_dir, psar, aroon_up_25, aroon_down_25,
donchian_upper_20, donchian_lower_20, keltner_upper_20, keltner_lower_20,
bb_pct_b, bb_bandwidth, hull_20, trix_15,
ichimoku_tenkan, ichimoku_kijun, ichimoku_senkou_a, ichimoku_senkou_b
```

Wiring (all defaults; binding):
- `adx_14, di_plus_14, di_minus_14 = adx(df)`; `stoch_k, stoch_d = stochastic(df)`;
  `stoch_rsi_k = stoch_rsi(close)[0]`; `williams_r_14 = williams_r(df)`; `cci_20 = cci(df)`;
  `roc_10 = roc(close)`; `mfi_14 = mfi(df)`; `obv = obv(df)`; `cmf_20 = cmf(df)`;
  `vwma_20 = vwma(df)`; `supertrend_10_3, supertrend_dir = supertrend(df)`;
  `psar = psar(df)`; `aroon_up_25, aroon_down_25 = aroon(df)`;
  `donchian_upper_20, _, donchian_lower_20 = donchian(df)`;
  `keltner_upper_20, _, keltner_lower_20 = keltner(df)`; `hull_20 = hull_ma(close)`;
  `trix_15 = trix(close)`; `ichimoku_tenkan, ichimoku_kijun, ichimoku_senkou_a,
  ichimoku_senkou_b = ichimoku(df)` (the pinned shifted/cloud-in-effect spans).
- `rel_volume_20 = volume / sma(volume, 20)` (computed in features.py — a derivation, not a
  new technical function; SMA 0 or NaN → NaN).
- `bb_pct_b = bollinger_pct_b(close, bb_upper, bb_lower)` and `bb_bandwidth =
  bollinger_bandwidth(bb_upper, bb_mid, bb_lower)` — reuse the band columns already computed
  in `add_features` (do NOT recompute bollinger).
- Deliberately NOT feature columns (functions exist for strategies/ad-hoc analysis only, kept
  out of the frame to bound prompt/DB size): `stoch_rsi` d-line, `adl`, donchian `mid`,
  keltner `mid`.
- The empty-frame path extends unchanged: all 48 columns as empty float64 Series.

Test impact (binding): `tests/test_indicators.py` pins "exactly 17" in two places — its local
`FEATURE_COLUMNS` list and `test_add_features_exact_columns` — this contract SUPERSEDES those
pins; the Green-phase integrator updates them to the canonical 48-name list above (§5). The
`test_no_lookahead` prefix test MUST keep passing over all 48 columns as-is — psar/supertrend
recursions always start from bar 0, hence prefix-stable; ichimoku's forward shift is causal.

### 3. feature_summary v4 — owner: builder-features

- ALL 48 columns included as numbers via the existing `_json_safe_number` (round 6, NaN →
  None), plus `price`, exactly as today. The existing 4 flags (`price_vs_sma20`, `macd_state`,
  `rsi_zone`, `bb_position`) are unchanged.
- NEW derived qualitative flags (top-level flat keys; every flag is `None` when any required
  input is None/warm-up, matching the existing flags). Pinned thresholds:
  - `adx_trend`: `"strong"` when `adx_14 >= 25`; `"weak"` when `adx_14 < 20`; else `"moderate"`.
  - `di_state`: `"bullish"` when `di_plus_14 > di_minus_14` else `"bearish"` (strict `>`,
    matching `macd_state`).
  - `stoch_zone`: `"oversold"` when `stoch_k < 20`; `"overbought"` when `stoch_k > 80`;
    else `"neutral"`.
  - `williams_zone`: `"oversold"` when `williams_r_14 < -80`; `"overbought"` when
    `williams_r_14 > -20`; else `"neutral"`.
  - `cci_zone`: `"overbought"` when `cci_20 > 100`; `"oversold"` when `cci_20 < -100`;
    else `"neutral"`.
  - `mfi_zone`: `"oversold"` when `mfi_14 < 20`; `"overbought"` when `mfi_14 > 80`;
    else `"neutral"`.
  - `obv_trend`: `"rising"` when the last `obv` value > SMA20 of the `obv` column
    (`obv.rolling(20, min_periods=20).mean()` at the last row — computed on the fly in
    feature_summary, NOT a stored column) else `"falling"`; `None` when < 20 rows.
  - `supertrend_side`: `"bullish"` when `supertrend_dir > 0` else `"bearish"`.
  - `psar_side`: `"bullish"` when `price > psar` else `"bearish"`.
  - `aroon_state`: `"bullish"` when `aroon_up_25 > 70 and aroon_down_25 < 30`; `"bearish"`
    when `aroon_down_25 > 70 and aroon_up_25 < 30`; else `"neutral"`.
  - `ichimoku_state`: with `cloud_top = max(ichimoku_senkou_a, ichimoku_senkou_b)` and
    `cloud_bot = min(...)` (the pinned in-effect spans): `"above_cloud"` when
    `price > cloud_top`; `"below_cloud"` when `price < cloud_bot`; else `"in_cloud"`.
  - `squeeze_on`: boolean — `True` when `bb_upper < keltner_upper_20 AND bb_lower >
    keltner_lower_20` (Bollinger fully inside Keltner), else `False`; `None` on NaN inputs.
  - `donchian_position`: `"at_upper"` when `price >= donchian_upper_20`; `"at_lower"` when
    `price <= donchian_lower_20`; else `"upper_half"` when `price >= (donchian_upper_20 +
    donchian_lower_20) / 2` else `"lower_half"`.
- PROMPT-SIZE RULE (binding): `summary["groups"]` is a nested dict containing ONLY the
  qualitative flags (same values as the flat keys — a compact, family-grouped view the LLM can
  scan; the small duplication is accepted), with EXACTLY this membership:
  ```
  summary["groups"] = {
    "trend":      {"price_vs_sma20", "adx_trend", "di_state", "supertrend_side",
                   "psar_side", "aroon_state", "ichimoku_state"},
    "momentum":   {"macd_state", "rsi_zone", "stoch_zone", "williams_zone", "cci_zone"},
    "volume":     {"mfi_zone", "obv_trend"},
    "volatility": {"bb_position", "squeeze_on", "donchian_position"},
  }              # each set shown = the flat keys copied under that group, key: value
  ```
- `backend/ai/analyst.py` is NOT touched in this pack: `_build_prompt` keeps serializing the
  flat `feature_summary` dict as before and the new keys (≈31 numbers + 13 flags + `groups`)
  flow into the prompt automatically. The prompt grows by roughly 30 numeric keys — accepted.

### 4. Performance budget (binding)

`add_features` on a 2,000-row canonical frame must complete in under **~150 ms** (median of 5
runs, `time.perf_counter`, dev box). The O(n) Python loops in `psar` and `supertrend` are
explicitly acceptable within that budget; `cci`'s MAD and `hull_ma`'s WMA may use
`rolling(...).apply(..., raw=True)` (raw=True REQUIRED — no per-row pandas objects). No other
Python-level per-row iteration in `add_features`.

### 5. File ownership (binding — edit ONLY your files)

| file | owner |
|---|---|
| `backend/indicators/technical.py` | builder-technical |
| `backend/indicators/features.py` | builder-features |
| `tests/test_indicators_v4.py` (new) | test-agent |
| `tests/test_indicators.py` — ONLY the two exactly-17 pins (local `FEATURE_COLUMNS` list + `test_add_features_exact_columns`), updated to the 48-name canonical list | Green-phase integrator |
| `CONTRACTS.md` | spec agent ONLY |

No one touches `analyst.py`, any strategy file, `engine.py` or `schema.sql` in this pack.
`tests/test_indicators_v4.py` scope: range bounds (adx/di/stoch/stoch_rsi/mfi/aroon in
`[0, 100]`, williams in `[-100, 0]`, cmf in `[-1, 1]`), band ordering (donchian and keltner
upper ≥ mid ≥ lower), `supertrend_dir ∈ {+1.0, -1.0}` after warm-up, psar flips to the
opposite side of price after a reversal, `ichimoku` shift identity
(`senkou_a[t] == senkou_a_raw[t-26]`), the pinned edge cases (mfi 100/0/50, cmf `high==low`,
stochastic flat window → NaN), no-lookahead prefix stability over the full 48 columns, and a
performance smoke test asserting the §4 budget (generous CI margin: assert < 500 ms, log the
measured time). Offline, seeded synthetic frames via the existing conftest patterns.

## Trailing Stops ("let winners run", 2026-07-12)

Bot-only (scalper untouched — its fixed TP/SL + time-stop remain tuner territory).

- `BotConfig` gains `trailing_enabled: bool = True`, `trail_activate_pct: float = 0.02` (clamp [0.005, 0.2]), `trail_distance_pct: float = 0.015` (clamp [0.005, 0.1]); exposed via `BotConfigUpdate`.
- `PaperTradingEngine.update_protective_levels(position_id, stop_loss=None, clear_take_profit=False)` — TIGHTEN-ONLY: a long's stop may only move up, a short's only down; loosening proposals are silently ignored (`stop_moved` reports the outcome). `clear_take_profit=True` sets `take_profit` NULL. Raises ValueError on unknown/closed positions.
- `AutoTrader._update_trailing(config, position, df)` runs from `_manage_positions` for every surviving (non-flipped) bot position: activation at `gain >= trail_activate_pct` (close-based), stop = best CLOSE since entry ± `trail_distance_pct` (closes, not wicks), take-profit cleared on activation, `trail` activity row (reason `trailing_stop`, plain-English explanation with locked-in %) logged only when the stop actually moves. Never raises; scalper-owned ids never reach it.

## Scalper Research Mode (data collection, 2026-07-15)

`ScalperParams` gains `research_mode: bool = False` — USER-ONLY (never in `_TUNABLE_PARAMS`),
exposed via `ScalperParamsUpdate.research_mode`. PAPER TRADING ONLY. When True:

- **Watchlist**: every `bot_config.watchlist` coin is scanned — `disabled_symbols` and the
  coach's `bench` are both ignored (lists preserved, just not applied).
- **Entry gates bypassed**: soft daily stop, coach `side_bias`, regime gate and cost gate are
  all skipped — every fired `_entry_signal` goes straight to `_enter_one`. The mechanical
  signal itself (EMA12/26 trend + RSI band + VWAP side + `allowed_sides`) still decides
  direction; nothing is inverted or randomized.
- **AI tuner paused**: `tune_with_ai` returns `{"status": "research_mode", "applied": {}}`
  and logs a `scalp_skip` row (reason `research_mode`) — the config is frozen during the
  experiment.
- **Still enforced**: `HARD_BOUNDS` clamping, SL/TP/time-stop exits, per-symbol cooldown,
  `max_trades_per_day`, `max_positions`, RiskManager order-level checks and the engine's hard
  daily-loss halt (`DAILY_LOSS_LIMIT`, raised 0.03 → 0.20 in `.env` on 2026-07-15 as the
  research backstop). Stops can NEVER be disabled, research mode included.
