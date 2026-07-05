"""Streamlit dashboard for the Trading AI Platform.

PAPER TRADING ONLY — research platform, not financial advice. The dashboard
never imports backend modules and never places real orders: everything goes
through the local HTTP API via ``api_client`` (simulation-only endpoints).

Run from the project root with: ``streamlit run dashboard/app.py``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
import streamlit as st

try:  # package-style import (project root on sys.path)
    from dashboard import api_client as api
    from dashboard import components as ui
    from dashboard.api_client import ApiError
except ImportError:  # script-style import (streamlit puts dashboard/ on sys.path)
    import api_client as api  # type: ignore[no-redef]
    import components as ui  # type: ignore[no-redef]
    from api_client import ApiError  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

SOURCES = ["binance", "bybit", "yahoo"]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
FALLBACK_STRATEGIES = [
    "trend_following",
    "mean_reversion",
    "breakout",
    "rsi_macd",
    "ensemble",
]

st.set_page_config(
    page_title="Trading AI Platform — Paper Only",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached GET wrappers (ttl=30s; the sidebar Refresh button clears them all)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def cached_health() -> dict[str, Any]:
    """Cached GET /health."""
    return api.get_health()


@st.cache_data(ttl=30, show_spinner=False)
def cached_models() -> dict[str, Any]:
    """Cached GET /api/models."""
    return api.get_models()


@st.cache_data(ttl=30, show_spinner=False)
def cached_ohlcv(
    source: str, symbol: str, timeframe: str, limit: int, features: bool
) -> dict[str, Any]:
    """Cached GET /api/data/ohlcv."""
    return api.get_ohlcv(source, symbol, timeframe, limit=limit, features=features)


@st.cache_data(ttl=30, show_spinner=False)
def cached_cached_items() -> dict[str, Any]:
    """Cached GET /api/data/cached."""
    return api.get_cached()


@st.cache_data(ttl=30, show_spinner=False)
def cached_strategies() -> dict[str, Any]:
    """Cached GET /api/strategies."""
    return api.get_strategies()


@st.cache_data(ttl=30, show_spinner=False)
def cached_analysis_history(symbol: str | None, limit: int) -> dict[str, Any]:
    """Cached GET /api/analysis/history."""
    return api.get_analysis_history(symbol=symbol, limit=limit)


@st.cache_data(ttl=30, show_spinner=False)
def cached_backtest_runs(limit: int) -> dict[str, Any]:
    """Cached GET /api/backtest/runs."""
    return api.get_backtest_runs(limit=limit)


@st.cache_data(ttl=30, show_spinner=False)
def cached_paper_portfolio() -> dict[str, Any]:
    """Cached GET /api/paper/portfolio."""
    return api.get_paper_portfolio()


@st.cache_data(ttl=30, show_spinner=False)
def cached_paper_positions(status: str) -> dict[str, Any]:
    """Cached GET /api/paper/positions."""
    return api.get_paper_positions(status=status)


@st.cache_data(ttl=30, show_spinner=False)
def cached_paper_orders(status: str | None) -> dict[str, Any]:
    """Cached GET /api/paper/orders."""
    return api.get_paper_orders(status=status)


@st.cache_data(ttl=30, show_spinner=False)
def cached_paper_trades(limit: int) -> dict[str, Any]:
    """Cached GET /api/paper/trades."""
    return api.get_paper_trades(limit=limit)


@st.cache_data(ttl=30, show_spinner=False)
def cached_paper_equity(limit: int) -> dict[str, Any]:
    """Cached GET /api/paper/equity."""
    return api.get_paper_equity(limit=limit)


# --- AI Trader bot GETs use a shorter TTL: the bot acts between page loads ---


@st.cache_data(ttl=15, show_spinner=False)
def cached_bot_status() -> dict[str, Any]:
    """Cached GET /api/bot/status."""
    return api.get_bot_status()


@st.cache_data(ttl=15, show_spinner=False)
def cached_bot_config() -> dict[str, Any]:
    """Cached GET /api/bot/config."""
    return api.get_bot_config()


@st.cache_data(ttl=15, show_spinner=False)
def cached_bot_activity(limit: int) -> dict[str, Any]:
    """Cached GET /api/bot/activity."""
    return api.get_bot_activity(limit=limit)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def flash(message: str, kind: str = "success", key: str = "_flash") -> None:
    """Queue a message that survives the next ``st.rerun()``.

    Args:
        message: Text to display on the next run.
        kind: One of ``"success" | "warning" | "error"``.
        key: Session-state slot — use a distinct key per tab so a message
            queued by one tab is not consumed by another.
    """
    st.session_state[key] = (kind, message)


def show_flash(key: str = "_flash") -> None:
    """Display and clear a queued flash message, if any."""
    item = st.session_state.pop(key, None)
    if not item:
        return
    kind, message = item
    if kind == "error":
        st.error(message)
    elif kind == "warning":
        st.warning(message)
    else:
        st.success(message)


def none_if_zero(value: float) -> float | None:
    """Return None for non-positive numeric form inputs (meaning 'not set')."""
    return value if value and value > 0 else None


def strategy_names() -> list[str]:
    """Strategy registry names from the API, with a static fallback."""
    try:
        items = cached_strategies().get("strategies", [])
        names = [s.get("name", "") for s in items if s.get("name")]
        return names or list(FALLBACK_STRATEGIES)
    except ApiError:
        return list(FALLBACK_STRATEGIES)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[str, str, str, int]:
    """Render the sidebar; return (source, symbol, timeframe, lookback_days)."""
    with st.sidebar:
        st.title("Trading AI Platform")
        st.caption("PAPER TRADING ONLY — research platform, not financial advice")

        source = st.selectbox("Data source", SOURCES, index=0, key="sb_source")
        symbol = st.text_input("Symbol", value="BTCUSDT", key="sb_symbol").strip()
        timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=3, key="sb_timeframe")
        lookback_days = int(
            st.number_input(
                "Lookback (days)",
                min_value=1,
                max_value=1825,
                value=365,
                step=30,
                key="sb_lookback",
            )
        )

        if st.button("Fetch / Update Data", type="primary", key="sb_fetch"):
            with st.spinner(f"Fetching {symbol} {timeframe} from {source}..."):
                try:
                    result = api.fetch_data(source, symbol, timeframe, lookback_days)
                    st.cache_data.clear()
                    st.success(f"Added {result.get('rows_added', 0)} new candles.")
                except ApiError as exc:
                    st.error(f"Data fetch failed: {exc}")

        if st.button("Refresh (clear cache)", key="sb_refresh"):
            st.cache_data.clear()

        st.divider()

        # --- API health indicator -------------------------------------------
        try:
            health = cached_health()
            api_ok = health.get("status") == "ok"
            ollama_ok = bool(health.get("ollama_available"))
        except ApiError:
            api_ok = False
            ollama_ok = False
        st.markdown(
            f"{'🟢' if api_ok else '🔴'} **API** "
            f"{'online' if api_ok else 'offline'} — `{api.API_BASE_URL}`"
        )
        st.markdown(
            f"{'🟢' if ollama_ok else '🔴'} **Ollama** "
            f"{'available' if ollama_ok else 'unavailable'}"
        )

        # --- Model info -----------------------------------------------------
        if api_ok:
            try:
                models = cached_models()
                configured = models.get("configured", {})
                installed = models.get("installed", [])
                with st.expander("Model info"):
                    st.markdown(
                        f"**Default:** `{configured.get('default', '?')}`\n\n"
                        f"**Fallback:** `{configured.get('fallback', '?')}`\n\n"
                        f"**Code:** `{configured.get('code', '?')}`\n\n"
                        f"**Installed:** {len(installed)} model(s)"
                    )
                    for name in installed:
                        st.caption(f"• {name}")
            except ApiError as exc:
                st.caption(f"Model info unavailable: {exc}")

        st.divider()
        st.caption("PAPER TRADING ONLY — research platform, not financial advice")
    return source, symbol, timeframe, lookback_days


# ---------------------------------------------------------------------------
# Tab: Market
# ---------------------------------------------------------------------------


def render_market_tab(source: str, symbol: str, timeframe: str) -> None:
    """Candlestick chart with indicator overlays and subplots."""
    left, right = st.columns([3, 1])
    with right:
        limit = int(
            st.number_input(
                "Candles", min_value=50, max_value=5000, value=500, step=50,
                key="market_limit",
            )
        )
    with left:
        st.subheader(f"Market — {symbol} ({source}, {timeframe})")

    try:
        payload = cached_ohlcv(source, symbol, timeframe, limit, True)
    except ApiError as exc:
        st.error(f"Could not load market data: {exc}")
        return

    df = ui.ohlcv_records_to_df(payload.get("data", []))
    if df.empty:
        st.info(
            "No cached data for this selection yet — use **Fetch / Update Data** "
            "in the sidebar."
        )
        return

    last = df.iloc[-1]
    prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else float(last["close"])
    change = float(last["close"]) - prev_close
    change_pct = change / prev_close if prev_close else 0.0
    ui.metric_row(
        [
            ("Last Close", ui.fmt_num(last["close"]), f"{change:+,.2f} ({change_pct:+.2%})"),
            ("High", ui.fmt_num(last["high"]), None),
            ("Low", ui.fmt_num(last["low"]), None),
            ("Volume", ui.fmt_num(last["volume"], 0), None),
            ("Candles", ui.fmt_int(len(df)), None),
        ],
        per_row=5,
    )
    st.plotly_chart(
        ui.candlestick_figure(df, symbol, timeframe), use_container_width=True
    )

    with st.expander("Cached datasets"):
        try:
            items = cached_cached_items().get("items", [])
            if items:
                st.dataframe(pd.DataFrame(items), use_container_width=True)
            else:
                st.caption("Nothing cached yet.")
        except ApiError as exc:
            st.error(f"Could not list cached data: {exc}")


# ---------------------------------------------------------------------------
# Tab: AI Analysis
# ---------------------------------------------------------------------------


def render_ai_tab(source: str, symbol: str, timeframe: str) -> None:
    """Run and display local-LLM market analyses."""
    st.subheader(f"AI Analysis — {symbol} ({timeframe})")
    st.caption(
        "Runs a local Ollama model against the latest indicators. "
        "Research output only — NOT financial advice."
    )

    default_model = ""
    installed: list[str] = []
    try:
        models = cached_models()
        default_model = str(models.get("configured", {}).get("default", ""))
        installed = [str(m) for m in models.get("installed", [])]
    except ApiError as exc:
        st.error(f"Could not load model list: {exc}")

    options = installed or ([default_model] if default_model else [])
    if options:
        index = options.index(default_model) if default_model in options else 0
        model = st.selectbox("Model", options, index=index, key="ai_model")
    else:
        model = st.text_input(
            "Model (list unavailable — enter an Ollama model name)",
            value=default_model,
            key="ai_model_text",
        ).strip()

    if st.button("Run AI Analysis", type="primary", key="ai_run"):
        with st.spinner("Querying local LLM — this can take a few minutes..."):
            try:
                st.session_state["analysis"] = api.run_analysis(
                    source, symbol, timeframe, model or None
                )
            except ApiError as exc:
                st.error(f"Analysis failed: {exc}")

    analysis = st.session_state.get("analysis")
    if analysis:
        badge_col, gauge_col = st.columns([1, 3])
        with badge_col:
            ui.sentiment_badge(
                analysis.get("sentiment", "neutral"), analysis.get("confidence")
            )
            st.caption(
                f"model: `{analysis.get('model_used', '?')}` · "
                f"{analysis.get('symbol', '')} {analysis.get('timeframe', '')}"
            )
        with gauge_col:
            confidence = int(analysis.get("confidence", 0) or 0)
            confidence = max(0, min(100, confidence))
            st.progress(confidence / 100.0, text=f"Confidence: {confidence}%")

        indicators = analysis.get("key_indicators", [])
        if indicators:
            st.markdown("**Key indicators**")
            st.dataframe(pd.DataFrame(indicators), use_container_width=True)

        risk = analysis.get("risk_commentary")
        if risk:
            st.warning(f"**Risk commentary:** {risk}")
        reasoning = analysis.get("reasoning")
        if reasoning:
            with st.expander("Model reasoning", expanded=True):
                st.markdown(reasoning)

    st.divider()
    st.markdown("**Analysis history**")
    try:
        history = cached_analysis_history(symbol or None, 50).get("items", [])
        if history:
            st.dataframe(pd.DataFrame(history), use_container_width=True)
        else:
            st.caption("No analyses recorded yet for this symbol.")
    except ApiError as exc:
        st.error(f"Could not load analysis history: {exc}")


# ---------------------------------------------------------------------------
# Tab: Backtest
# ---------------------------------------------------------------------------


def render_backtest_tab(source: str, symbol: str, timeframe: str) -> None:
    """Single-strategy backtest with metrics, trades and walk-forward."""
    st.subheader(f"Backtest — {symbol} ({source}, {timeframe})")

    strategies_meta: list[dict[str, Any]] = []
    try:
        strategies_meta = cached_strategies().get("strategies", [])
    except ApiError as exc:
        st.error(f"Could not load strategies from API: {exc}")
    names = [s.get("name", "") for s in strategies_meta if s.get("name")] or list(
        FALLBACK_STRATEGIES
    )
    meta_by_name = {s.get("name"): s for s in strategies_meta}

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        strategy = st.selectbox("Strategy", names, key="bt_strategy")
        description = meta_by_name.get(strategy, {}).get("description", "")
        if description:
            st.caption(description)
    with col2:
        limit = int(
            st.number_input(
                "Candles", min_value=100, max_value=20000, value=2000, step=100,
                key="bt_limit",
            )
        )
    with col3:
        allow_short = st.checkbox("Allow short", value=True, key="bt_allow_short")

    default_params = meta_by_name.get(strategy, {}).get("default_params", {}) or {}
    params_text = st.text_area(
        "Strategy params (JSON)",
        value=json.dumps(default_params),
        height=70,
        key="bt_params",
    )
    params: dict[str, Any] = {}
    try:
        parsed = json.loads(params_text) if params_text.strip() else {}
        if isinstance(parsed, dict):
            params = parsed
        else:
            st.error("Params must be a JSON object — ignoring.")
    except json.JSONDecodeError as exc:
        st.error(f"Invalid params JSON ({exc}) — running with defaults.")

    if st.button("Run Backtest", type="primary", key="bt_run"):
        with st.spinner(f"Backtesting {strategy} on {symbol}..."):
            try:
                st.session_state["backtest_result"] = api.run_backtest(
                    source,
                    symbol,
                    timeframe,
                    strategy,
                    limit=limit,
                    params=params or None,
                    config={"allow_short": allow_short},
                )
            except ApiError as exc:
                st.error(f"Backtest failed: {exc}")

    result = st.session_state.get("backtest_result")
    if result:
        st.markdown(
            f"**{result.get('strategy', strategy)}** on "
            f"{result.get('symbol', symbol)} {result.get('timeframe', timeframe)}"
        )
        ui.backtest_metric_cards(result.get("metrics", {}))
        curve = result.get("equity_curve", {})
        st.plotly_chart(
            ui.equity_curve_figure(
                curve.get("timestamps", []), curve.get("values", [])
            ),
            use_container_width=True,
        )
        trades = result.get("trades", [])
        st.markdown(f"**Trades ({len(trades)})**")
        if trades:
            st.dataframe(pd.DataFrame(trades), use_container_width=True)
        else:
            st.caption("No trades were generated.")

    # --- Walk-forward section ------------------------------------------------
    st.divider()
    with st.expander("Walk-forward analysis"):
        wf1, wf2, wf3 = st.columns(3)
        with wf1:
            n_splits = int(
                st.number_input(
                    "Folds", min_value=2, max_value=10, value=4, key="wf_splits"
                )
            )
        with wf2:
            train_ratio = float(
                st.slider("Train ratio", 0.5, 0.9, 0.7, 0.05, key="wf_ratio")
            )
        with wf3:
            wf_limit = int(
                st.number_input(
                    "Candles", min_value=300, max_value=20000, value=3000, step=100,
                    key="wf_limit",
                )
            )
        if st.button("Run Walk-Forward", key="wf_run"):
            with st.spinner("Running walk-forward evaluation..."):
                try:
                    st.session_state["walkforward_result"] = api.run_walkforward(
                        source, symbol, timeframe, strategy,
                        n_splits=n_splits, train_ratio=train_ratio, limit=wf_limit,
                    )
                except ApiError as exc:
                    st.error(f"Walk-forward failed: {exc}")

        wf_result = st.session_state.get("walkforward_result")
        if wf_result:
            folds = wf_result.get("folds", [])
            if folds:
                folds_df = pd.json_normalize(folds)
                st.dataframe(folds_df, use_container_width=True)
            aggregate = wf_result.get("aggregate", {})
            if aggregate:
                st.markdown("**Aggregate (test segments)**")
                st.dataframe(
                    pd.DataFrame([aggregate]), use_container_width=True
                )

    with st.expander("Previous backtest runs"):
        try:
            runs = cached_backtest_runs(50).get("items", [])
            if runs:
                st.dataframe(pd.json_normalize(runs), use_container_width=True)
            else:
                st.caption("No stored runs yet.")
        except ApiError as exc:
            st.error(f"Could not load backtest runs: {exc}")


# ---------------------------------------------------------------------------
# Tab: Paper Trading
# ---------------------------------------------------------------------------


def render_paper_tab(source: str, symbol: str, timeframe: str) -> None:
    """Simulated portfolio: metrics, order form, tables, tick, reset."""
    st.subheader("Paper Trading (simulation only)")
    show_flash()

    portfolio: dict[str, Any] = {}
    try:
        portfolio = cached_paper_portfolio()
    except ApiError as exc:
        st.error(f"Could not load portfolio: {exc}")

    if portfolio:
        if portfolio.get("trading_halted"):
            st.error(
                f"🛑 TRADING HALTED — {portfolio.get('halt_reason') or 'risk limit hit'}. "
                "New entries are blocked; reset the paper account to resume."
            )
        ui.portfolio_metric_cards(portfolio)

    order_col, tick_col = st.columns([2, 1])

    # --- Order form -----------------------------------------------------------
    with order_col:
        st.markdown("**New simulated order**")
        with st.form("paper_order_form", clear_on_submit=False):
            f1, f2, f3 = st.columns(3)
            with f1:
                order_symbol = st.text_input("Symbol", value=symbol).strip()
                side = st.selectbox("Side", ["buy", "sell"])
            with f2:
                order_type = st.selectbox("Type", ["market", "limit", "stop"])
                qty = st.number_input(
                    "Qty (0 = auto risk-size)", min_value=0.0, value=0.0,
                    step=0.001, format="%.6f",
                )
            with f3:
                limit_price = st.number_input(
                    "Limit price (0 = n/a)", min_value=0.0, value=0.0, format="%.4f"
                )
                stop_price = st.number_input(
                    "Stop price (0 = n/a)", min_value=0.0, value=0.0, format="%.4f"
                )
            f4, f5 = st.columns(2)
            with f4:
                stop_loss = st.number_input(
                    "Stop loss (0 = auto/none)", min_value=0.0, value=0.0, format="%.4f"
                )
            with f5:
                take_profit = st.number_input(
                    "Take profit (0 = auto/none)", min_value=0.0, value=0.0,
                    format="%.4f",
                )
            submitted = st.form_submit_button("Submit paper order", type="primary")
        if submitted:
            try:
                order = api.submit_paper_order(
                    symbol=order_symbol,
                    side=side,
                    order_type=order_type,
                    qty=none_if_zero(qty),
                    limit_price=none_if_zero(limit_price),
                    stop_price=none_if_zero(stop_price),
                    stop_loss=none_if_zero(stop_loss),
                    take_profit=none_if_zero(take_profit),
                    source=source,
                    timeframe=timeframe,
                )
                flash(
                    f"Order accepted: {order.get('side', side)} "
                    f"{order.get('symbol', order_symbol)} "
                    f"({order.get('status', 'submitted')})."
                )
                st.cache_data.clear()
                st.rerun()
            except ApiError as exc:
                st.error(f"Order rejected: {exc}")

    # --- Tick -----------------------------------------------------------------
    with tick_col:
        st.markdown("**Process tick**")
        refresh = st.checkbox(
            "Refresh from exchange first", value=False, key="tick_refresh"
        )
        if st.button("Process Tick", key="tick_btn"):
            with st.spinner("Processing tick..."):
                try:
                    result = api.paper_tick(symbol, source, timeframe, refresh)
                    flash(
                        f"Tick processed — filled {len(result.get('filled', []))}, "
                        f"closed {len(result.get('closed', []))}, "
                        f"equity {ui.fmt_currency(result.get('equity'))}."
                    )
                    st.cache_data.clear()
                    st.rerun()
                except ApiError as exc:
                    st.error(f"Tick failed: {exc}")

    st.divider()

    # --- Positions -------------------------------------------------------------
    st.markdown("**Open positions**")
    positions: list[dict[str, Any]] = []
    try:
        positions = cached_paper_positions("open").get("items", [])
    except ApiError as exc:
        st.error(f"Could not load positions: {exc}")
    if positions:
        st.dataframe(pd.DataFrame(positions), use_container_width=True)
        pos_ids = [p.get("id") for p in positions if p.get("id") is not None]
        if pos_ids:
            c1, c2 = st.columns([3, 1])
            with c1:
                position_id = st.selectbox(
                    "Position to close", pos_ids, key="close_pos_id"
                )
            with c2:
                st.write("")
                if st.button("Close position", key="close_pos_btn"):
                    try:
                        api.close_paper_position(int(position_id))
                        flash(f"Position {position_id} closed.")
                        st.cache_data.clear()
                        st.rerun()
                    except ApiError as exc:
                        st.error(f"Close failed: {exc}")
    else:
        st.caption("No open positions.")

    # --- Orders / trades tables --------------------------------------------------
    orders_tab, trades_tab, equity_tab = st.tabs(["Orders", "Trades", "Equity curve"])
    with orders_tab:
        status = st.selectbox(
            "Status filter", ["all", "pending", "filled", "cancelled", "rejected"],
            key="orders_status",
        )
        try:
            orders = cached_paper_orders(None if status == "all" else status).get(
                "items", []
            )
            if orders:
                st.dataframe(pd.DataFrame(orders), use_container_width=True)
            else:
                st.caption("No orders.")
        except ApiError as exc:
            st.error(f"Could not load orders: {exc}")
    with trades_tab:
        try:
            trades = cached_paper_trades(100).get("items", [])
            if trades:
                st.dataframe(pd.DataFrame(trades), use_container_width=True)
            else:
                st.caption("No trades yet.")
        except ApiError as exc:
            st.error(f"Could not load trades: {exc}")
    with equity_tab:
        try:
            equity_items = cached_paper_equity(1000).get("items", [])
            if equity_items:
                st.plotly_chart(
                    ui.equity_curve_figure(
                        [e.get("ts") for e in equity_items],
                        [e.get("equity") for e in equity_items],
                        title="Paper Equity",
                    ),
                    use_container_width=True,
                )
            else:
                st.caption("No equity snapshots yet — process a tick first.")
        except ApiError as exc:
            st.error(f"Could not load equity history: {exc}")

    # --- Reset -------------------------------------------------------------------
    st.divider()
    st.markdown("**Danger zone**")
    confirm = st.checkbox(
        "I understand this wipes all paper positions, orders, trades and equity "
        "history and restores the initial capital.",
        key="reset_confirm",
    )
    if st.button("Reset Paper Account", disabled=not confirm, key="reset_btn"):
        try:
            api.reset_paper()
            flash("Paper account reset to initial capital.")
            st.cache_data.clear()
            st.rerun()
        except ApiError as exc:
            st.error(f"Reset failed: {exc}")


# ---------------------------------------------------------------------------
# Tab: AI Trader (autonomous paper bot)
# ---------------------------------------------------------------------------


def render_bot_tab() -> None:
    """Autonomous AI paper-trading bot: status, controls, config, activity."""
    st.subheader("AI Trader — autonomous simulated trading")
    st.caption("The AI trades a simulated account — paper only, not financial advice.")
    show_flash(key="_bot_flash")

    # --- Methodology (mandated transparency documentation) -------------------
    with st.expander("📖 Trading Methodology — how the bot buys and sells"):
        st.markdown(ui.TRADING_METHODOLOGY_MD)

    # --- Status row -----------------------------------------------------------
    status: dict[str, Any] = {}
    try:
        status = cached_bot_status()
    except ApiError as exc:
        st.error(f"Could not load bot status: {exc}")
    if status:
        ui.bot_status_row(status)

    # --- Controls ---------------------------------------------------------------
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("▶ Start bot", type="primary", key="bot_start"):
            try:
                api.start_bot()
                flash(
                    "Bot started — it will scan and trade the paper account on "
                    "schedule.",
                    key="_bot_flash",
                )
                st.cache_data.clear()
                st.rerun()
            except ApiError as exc:
                st.error(f"Start failed: {exc}")
    with b2:
        if st.button("⏹ Stop bot", key="bot_stop"):
            try:
                api.stop_bot()
                flash("Bot stopped — no further cycles will run.", key="_bot_flash")
                st.cache_data.clear()
                st.rerun()
            except ApiError as exc:
                st.error(f"Stop failed: {exc}")
    with b3:
        if st.button("⚡ Run cycle now", key="bot_run_once"):
            with st.spinner(
                "Running one full bot cycle — scans plus AI confirmations can "
                "take several minutes..."
            ):
                try:
                    st.session_state["bot_cycle_summary"] = api.run_bot_cycle()
                    flash("Cycle finished — see summary below.", key="_bot_flash")
                    st.cache_data.clear()
                    st.rerun()
                except ApiError as exc:
                    st.error(f"Cycle failed: {exc}")
    with b4:
        if st.button("🔄 Refresh", key="bot_refresh"):
            st.cache_data.clear()
            st.rerun()

    summary = st.session_state.get("bot_cycle_summary")
    if summary:
        with st.expander("Last manual cycle summary"):
            st.json(summary)

    # --- Configuration ----------------------------------------------------------
    with st.expander("⚙️ Bot configuration"):
        config: dict[str, Any] = {}
        try:
            config = cached_bot_config()
        except ApiError as exc:
            st.error(f"Could not load bot config: {exc}")
        current_tf = str(config.get("timeframe", "1h"))
        tf_index = TIMEFRAMES.index(current_tf) if current_tf in TIMEFRAMES else 3
        with st.form("bot_config_form"):
            watchlist_text = st.text_area(
                "Watchlist (comma-separated symbols)",
                value=", ".join(str(s) for s in config.get("watchlist", []) or []),
                height=70,
                key="bot_cfg_watchlist",
            )
            k1, k2, k3 = st.columns(3)
            with k1:
                timeframe_cfg = st.selectbox(
                    "Timeframe", TIMEFRAMES, index=tf_index, key="bot_cfg_timeframe"
                )
                interval_minutes = int(
                    st.number_input(
                        "Cycle interval (minutes)",
                        min_value=1,
                        max_value=1440,
                        value=int(config.get("interval_minutes", 15) or 15),
                        key="bot_cfg_interval",
                    )
                )
            with k2:
                min_vote = int(
                    st.number_input(
                        "Min strategy votes (of 4)",
                        min_value=1,
                        max_value=4,
                        value=int(config.get("min_vote", 2) or 2),
                        key="bot_cfg_min_vote",
                    )
                )
                min_ai_confidence = int(
                    st.slider(
                        "Min AI confidence (%)",
                        min_value=0,
                        max_value=100,
                        value=int(config.get("min_ai_confidence", 60) or 60),
                        key="bot_cfg_min_conf",
                    )
                )
            with k3:
                use_ai = st.checkbox(
                    "AI confirmation gate",
                    value=bool(config.get("use_ai", True)),
                    key="bot_cfg_use_ai",
                )
                allow_short = st.checkbox(
                    "Allow short entries",
                    value=bool(config.get("allow_short", False)),
                    key="bot_cfg_allow_short",
                )
            saved = st.form_submit_button("Save configuration", type="primary")
        if saved:
            watchlist = [
                part.strip().upper()
                for part in watchlist_text.split(",")
                if part.strip()
            ]
            if not watchlist:
                st.error("Watchlist cannot be empty.")
            else:
                try:
                    api.set_bot_config(
                        {
                            "watchlist": watchlist,
                            "timeframe": timeframe_cfg,
                            "interval_minutes": interval_minutes,
                            "min_vote": min_vote,
                            "min_ai_confidence": min_ai_confidence,
                            "use_ai": use_ai,
                            "allow_short": allow_short,
                        }
                    )
                    flash("Bot configuration saved.", key="_bot_flash")
                    st.cache_data.clear()
                    st.rerun()
                except ApiError as exc:
                    st.error(f"Config update failed: {exc}")

    # --- Activity feed ----------------------------------------------------------
    st.divider()
    head_col, filter_col = st.columns([3, 1])
    with head_col:
        st.markdown(
            "**Activity feed** — ✅ enter · 🔴 exit · ⏭ skip · ⚠️ error/halt "
            "(latest 100, newest first; click enter/exit rows for full detail)"
        )
    with filter_col:
        show_scans = st.checkbox("Show scan rows", value=True, key="bot_show_scans")
    items: list[dict[str, Any]] = []
    try:
        items = cached_bot_activity(100).get("items", [])
    except ApiError as exc:
        st.error(f"Could not load bot activity: {exc}")
    ui.bot_activity_feed(items, show_scans=show_scans)

    # --- Recent closed trades with their entry reasoning -------------------------
    st.divider()
    st.markdown("**Recent bot trades (closed)** — PnL with the entry explanation")
    ui.bot_trades_table(items)


# ---------------------------------------------------------------------------
# Tab: Strategy Comparison
# ---------------------------------------------------------------------------


def render_comparison_tab(source: str, symbol: str, timeframe: str) -> None:
    """Run every registered strategy on the same data and compare metrics."""
    st.subheader(f"Strategy Comparison — {symbol} ({source}, {timeframe})")

    c1, c2 = st.columns([1, 3])
    with c1:
        limit = int(
            st.number_input(
                "Candles", min_value=100, max_value=20000, value=2000, step=100,
                key="cmp_limit",
            )
        )
    with c2:
        st.caption(
            "Runs each registered strategy through POST /api/backtest on the same "
            "cached candles, then compares Sharpe, total return, max drawdown and "
            "win rate."
        )

    if st.button("Run All Strategies", type="primary", key="cmp_run"):
        names = strategy_names()
        progress = st.progress(0.0, text="Starting comparison...")
        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        for i, name in enumerate(names):
            progress.progress(i / len(names), text=f"Backtesting {name}...")
            try:
                result = api.run_backtest(
                    source, symbol, timeframe, name, limit=limit
                )
                rows.append({"strategy": name, **result.get("metrics", {})})
            except ApiError as exc:
                errors.append(f"{name}: {exc}")
        progress.progress(1.0, text="Comparison complete.")
        st.session_state["comparison"] = {"rows": rows, "errors": errors}

    comparison = st.session_state.get("comparison")
    if not comparison:
        st.info("Run the comparison to see grouped metric charts.")
        return

    for message in comparison.get("errors", []):
        st.error(f"Strategy failed: {message}")

    rows = comparison.get("rows", [])
    if not rows:
        st.warning("No strategy produced a result.")
        return

    metrics_df = pd.DataFrame(rows).set_index("strategy")
    st.plotly_chart(ui.comparison_figure(metrics_df), use_container_width=True)
    st.markdown("**Full metrics**")
    st.dataframe(metrics_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: sidebar + the six feature tabs."""
    source, symbol, timeframe, _lookback_days = render_sidebar()

    st.title("Trading AI Platform")
    st.caption("PAPER TRADING ONLY — research platform, not financial advice")

    market_tab, ai_tab, backtest_tab, paper_tab, bot_tab, comparison_tab = st.tabs(
        [
            "Market",
            "AI Analysis",
            "Backtest",
            "Paper Trading",
            "AI Trader",
            "Strategy Comparison",
        ]
    )
    with market_tab:
        render_market_tab(source, symbol, timeframe)
    with ai_tab:
        render_ai_tab(source, symbol, timeframe)
    with backtest_tab:
        render_backtest_tab(source, symbol, timeframe)
    with paper_tab:
        render_paper_tab(source, symbol, timeframe)
    with bot_tab:
        render_bot_tab()
    with comparison_tab:
        render_comparison_tab(source, symbol, timeframe)


main()
