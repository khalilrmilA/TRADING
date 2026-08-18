"""Reusable Streamlit / Plotly building blocks for the dashboard.

Pure presentation helpers — no HTTP, no backend imports. All figures degrade
gracefully when optional feature columns (sma_20, rsi_14, ...) are missing.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Consistent color vocabulary across all charts.
COLOR_UP = "#16a34a"
COLOR_DOWN = "#dc2626"
COLOR_NEUTRAL = "#6b7280"
COLOR_ACCENT = "#6366f1"
STRATEGY_COLORS: list[str] = [
    "#6366f1",  # indigo
    "#16a34a",  # green
    "#f59e0b",  # amber
    "#dc2626",  # red
    "#0ea5e9",  # sky
    "#a855f7",  # purple
    "#14b8a6",  # teal
]

_SENTIMENT_COLORS: dict[str, str] = {
    "bullish": COLOR_UP,
    "bearish": COLOR_DOWN,
    "neutral": COLOR_NEUTRAL,
}


# ---------------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------------


def ohlcv_records_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert API OHLCV records into a UTC-indexed DataFrame.

    Args:
        records: ``data`` list from ``GET /api/data/ohlcv`` — dicts with an
            ISO-8601 ``timestamp`` plus OHLCV (and optional feature) columns.

    Returns:
        DataFrame indexed by tz-aware UTC ``timestamp`` (ascending); empty
        frame with OHLCV columns when ``records`` is empty.
    """
    if not records:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    """True when a metric value is None/NaN and should render as an em dash."""
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def fmt_pct(value: Any, digits: int = 2) -> str:
    """Format a fraction (0.12) as a percentage string ('12.00%')."""
    if _is_missing(value):
        return "—"
    return f"{float(value):.{digits}%}"


def fmt_num(value: Any, digits: int = 2) -> str:
    """Format a plain number with fixed decimals."""
    if _is_missing(value):
        return "—"
    return f"{float(value):,.{digits}f}"


def fmt_currency(value: Any, digits: int = 2) -> str:
    """Format a dollar amount ('$1,234.56')."""
    if _is_missing(value):
        return "—"
    return f"${float(value):,.{digits}f}"


def fmt_int(value: Any) -> str:
    """Format an integer count."""
    if _is_missing(value):
        return "—"
    return f"{int(value):,}"


# ---------------------------------------------------------------------------
# Metric card rows
# ---------------------------------------------------------------------------


def metric_row(
    cards: Sequence[tuple[str, str, str | None]], per_row: int = 6
) -> None:
    """Render metric cards in rows of ``st.columns`` + ``st.metric``.

    Args:
        cards: Sequence of ``(label, value, delta)`` tuples; ``delta`` may be
            ``None`` for a plain card.
        per_row: Maximum cards per visual row.
    """
    for start in range(0, len(cards), per_row):
        chunk = cards[start : start + per_row]
        cols = st.columns(len(chunk))
        for col, (label, value, delta) in zip(cols, chunk):
            col.metric(label, value, delta)


def backtest_metric_cards(metrics: dict[str, Any]) -> None:
    """Render the canonical backtest metrics dict as metric cards.

    Args:
        metrics: ``metrics`` dict from a backtest result (keys per contract:
            total_return, cagr, sharpe, sortino, max_drawdown, win_rate,
            profit_factor, num_trades, avg_trade_pnl, avg_win, avg_loss,
            exposure).
    """
    specs: list[tuple[str, str, str]] = [
        ("Total Return", "total_return", "pct"),
        ("CAGR", "cagr", "pct"),
        ("Sharpe", "sharpe", "num"),
        ("Sortino", "sortino", "num"),
        ("Max Drawdown", "max_drawdown", "pct"),
        ("Win Rate", "win_rate", "pct"),
        ("Profit Factor", "profit_factor", "num"),
        ("Trades", "num_trades", "int"),
        ("Avg Trade PnL", "avg_trade_pnl", "cur"),
        ("Avg Win", "avg_win", "cur"),
        ("Avg Loss", "avg_loss", "cur"),
        ("Exposure", "exposure", "pct"),
    ]
    formatters = {"pct": fmt_pct, "num": fmt_num, "cur": fmt_currency, "int": fmt_int}
    cards: list[tuple[str, str, str | None]] = [
        (label, formatters[kind](metrics.get(key)), None)
        for label, key, kind in specs
        if key in metrics
    ]
    metric_row(cards, per_row=6)


def portfolio_metric_cards(portfolio: dict[str, Any]) -> None:
    """Render paper-trading portfolio state as metric cards.

    Args:
        portfolio: dict from ``GET /api/paper/portfolio``.
    """
    daily_pct = portfolio.get("daily_pnl_pct")
    cards: list[tuple[str, str, str | None]] = [
        (
            "Equity",
            fmt_currency(portfolio.get("equity")),
            None if _is_missing(daily_pct) else f"{float(daily_pct):+.2%} today",
        ),
        ("Cash", fmt_currency(portfolio.get("cash")), None),
        (
            "Unrealized PnL",
            fmt_currency(portfolio.get("unrealized_pnl")),
            None,
        ),
        (
            "Realized PnL (today)",
            fmt_currency(portfolio.get("realized_pnl_today")),
            None,
        ),
        ("Open Positions", fmt_int(portfolio.get("open_positions")), None),
        ("Peak Equity", fmt_currency(portfolio.get("peak_equity")), None),
        (
            "Drawdown",
            fmt_pct(portfolio.get("drawdown")),
            "circuit breaker!" if portfolio.get("circuit_breaker_active") else None,
        ),
    ]
    metric_row(cards, per_row=4)


# ---------------------------------------------------------------------------
# Badges
# ---------------------------------------------------------------------------


def sentiment_badge(sentiment: str, confidence: int | None = None) -> None:
    """Render a colored sentiment pill (bullish=green, bearish=red, neutral=gray).

    Args:
        sentiment: One of ``"bullish" | "bearish" | "neutral"`` (anything else
            renders gray).
        confidence: Optional 0-100 confidence to append to the label.
    """
    key = str(sentiment).strip().lower()
    color = _SENTIMENT_COLORS.get(key, COLOR_NEUTRAL)
    label = key.upper() if key else "UNKNOWN"
    if confidence is not None:
        label = f"{label} · {int(confidence)}%"
    st.markdown(
        f'<span style="background-color:{color};color:#ffffff;'
        f"padding:0.35em 1.0em;border-radius:1em;font-weight:700;"
        f'font-size:1.05em;letter-spacing:0.05em;">{label}</span>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def candlestick_figure(
    df: pd.DataFrame, symbol: str = "", timeframe: str = ""
) -> go.Figure:
    """Build the main market chart: candles + overlays + volume + RSI + MACD.

    Uses ``make_subplots(rows=4, shared_xaxes=True)``:
    row 1 candlesticks with SMA20/50, EMA12/26 and Bollinger-band fill;
    row 2 volume bars; row 3 RSI(14) with 30/70 guides; row 4 MACD.
    Overlay traces are only added when their columns exist in ``df``.

    Args:
        df: Canonical (optionally feature-enriched) OHLCV frame, UTC index.
        symbol: Symbol for the title.
        timeframe: Timeframe for the title.

    Returns:
        Configured plotly Figure (rangeslider disabled).
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.52, 0.14, 0.17, 0.17],
    )

    # --- Row 1: price -------------------------------------------------------
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color=COLOR_UP,
            decreasing_line_color=COLOR_DOWN,
        ),
        row=1,
        col=1,
    )
    # Bollinger band fill (upper first, lower fills to it).
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_upper"],
                name="BB upper",
                line=dict(color="rgba(99,102,241,0.45)", width=1),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_lower"],
                name="Bollinger (20, 2σ)",
                line=dict(color="rgba(99,102,241,0.45)", width=1),
                fill="tonexty",
                fillcolor="rgba(99,102,241,0.10)",
            ),
            row=1,
            col=1,
        )
    if "bb_mid" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_mid"],
                name="BB mid",
                line=dict(color="rgba(99,102,241,0.6)", width=1, dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    overlay_specs = [
        ("sma_20", "SMA 20", "#f59e0b", "solid"),
        ("sma_50", "SMA 50", "#0ea5e9", "solid"),
        ("ema_12", "EMA 12", "#a855f7", "dash"),
        ("ema_26", "EMA 26", "#14b8a6", "dash"),
    ]
    for column, name, color, dash in overlay_specs:
        if column in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[column],
                    name=name,
                    line=dict(color=color, width=1.3, dash=dash),
                ),
                row=1,
                col=1,
            )

    # --- Row 2: volume ------------------------------------------------------
    volume_colors = [
        COLOR_UP if c >= o else COLOR_DOWN for o, c in zip(df["open"], df["close"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            marker_color=volume_colors,
            marker_line_width=0,
            opacity=0.6,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # --- Row 3: RSI ---------------------------------------------------------
    if "rsi_14" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["rsi_14"],
                name="RSI 14",
                line=dict(color=COLOR_ACCENT, width=1.3),
            ),
            row=3,
            col=1,
        )
        fig.add_hline(
            y=70, line_dash="dot", line_color=COLOR_DOWN, opacity=0.5, row=3, col=1
        )
        fig.add_hline(
            y=30, line_dash="dot", line_color=COLOR_UP, opacity=0.5, row=3, col=1
        )
        fig.update_yaxes(range=[0, 100], row=3, col=1)

    # --- Row 4: MACD --------------------------------------------------------
    if "macd" in df.columns and "macd_signal" in df.columns:
        if "macd_hist" in df.columns:
            hist_colors = [
                COLOR_UP if (v is not None and not _is_missing(v) and v >= 0) else COLOR_DOWN
                for v in df["macd_hist"]
            ]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["macd_hist"],
                    name="MACD hist",
                    marker_color=hist_colors,
                    marker_line_width=0,
                    opacity=0.55,
                    showlegend=False,
                ),
                row=4,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["macd"],
                name="MACD",
                line=dict(color=COLOR_ACCENT, width=1.3),
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["macd_signal"],
                name="Signal",
                line=dict(color="#f59e0b", width=1.3),
            ),
            row=4,
            col=1,
        )

    title = " · ".join(part for part in (symbol, timeframe) if part)
    fig.update_layout(
        title=title or None,
        height=860,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        margin=dict(l=40, r=20, t=60, b=30),
        hovermode="x unified",
        bargap=0.05,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    return fig


def equity_curve_figure(
    timestamps: Sequence[Any],
    values: Sequence[float],
    title: str = "Equity Curve",
) -> go.Figure:
    """Build an equity-curve line chart with a start-capital reference line.

    Args:
        timestamps: ISO strings or datetimes (x axis).
        values: Equity values (y axis).
        title: Figure title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=list(values),
            name="Equity",
            mode="lines",
            line=dict(color=COLOR_ACCENT, width=2),
        )
    )
    if len(values) > 0:
        fig.add_hline(
            y=float(values[0]),
            line_dash="dot",
            line_color=COLOR_NEUTRAL,
            opacity=0.6,
            annotation_text="start",
        )
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=40, r=20, t=50, b=30),
        yaxis_title="Equity ($)",
        hovermode="x unified",
        showlegend=False,
    )
    return fig


def comparison_figure(
    metrics_df: pd.DataFrame,
    metric_keys: Sequence[str] = ("sharpe", "total_return", "max_drawdown", "win_rate"),
) -> go.Figure:
    """Build the strategy-comparison grouped bar panels (one panel per metric).

    Args:
        metrics_df: DataFrame indexed by strategy name with metric columns.
        metric_keys: Metric columns to plot (2x2 grid).

    Returns:
        Plotly Figure with one bar chart per metric; each strategy keeps the
        same color across panels.
    """
    titles = [key.replace("_", " ").title() for key in metric_keys]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles, vertical_spacing=0.16)
    strategies = [str(s) for s in metrics_df.index]
    colors = [
        STRATEGY_COLORS[i % len(STRATEGY_COLORS)] for i in range(len(strategies))
    ]
    percent_metrics = {"total_return", "max_drawdown", "win_rate", "exposure", "cagr"}
    for i, key in enumerate(metric_keys):
        row, col = i // 2 + 1, i % 2 + 1
        if key not in metrics_df.columns:
            continue
        values = pd.to_numeric(metrics_df[key], errors="coerce")
        if key in percent_metrics:
            text = [fmt_pct(v) for v in values]
        else:
            text = [fmt_num(v) for v in values]
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=values,
                marker_color=colors,
                text=text,
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
                name=key,
            ),
            row=row,
            col=col,
        )
        if key in percent_metrics:
            fig.update_yaxes(tickformat=".0%", row=row, col=col)
    fig.update_layout(
        height=680,
        margin=dict(l=40, r=20, t=60, b=40),
        title="Strategy Comparison",
    )
    return fig


# ---------------------------------------------------------------------------
# AI Trader bot presentation (paper only)
# ---------------------------------------------------------------------------

#: Emoji marker per bot_activity action (contract: ✅ enter, 🔴 exit, ⏭ skip,
#: ⚠️ error/halt; the rest are informational).
BOT_ACTION_EMOJI: dict[str, str] = {
    "cycle_start": "🔁",
    "cycle_end": "🏁",
    "scan": "🔍",
    "enter": "✅",
    "exit": "🔴",
    "skip": "⏭",
    "reject": "🚫",
    "halt": "⚠️",
    "error": "⚠️",
}

#: Complete beginner-friendly documentation of the bot's buy/sell methodology.
#: Rendered inside the "📖 Trading Methodology" expander on the AI Trader tab.
TRADING_METHODOLOGY_MD: str = """
**PAPER TRADING ONLY.** The bot trades a *simulated* account against real
market data. No real money is ever at risk — this is a learning and research
tool, not financial advice.

#### The pipeline at a glance

Every cycle (default: every 15 minutes) the bot runs the exact same checklist:

1. **Manage** — refresh every open position with the latest candle (stop-loss /
   take-profit checks happen here); if the strategy consensus has flipped
   *against* a position, close it at market.
2. **Scan** — pull fresh candles for every watchlist coin and compute all
   indicators.
3. **Vote** — four rule-based strategies each vote long / short / flat; a coin
   is shortlisted only when enough of them agree (default: at least 2 of 4).
4. **AI confirm** — the local LLM reads the indicators *and* the chart
   structure and must agree with the direction with enough confidence
   (default ≥ 60%).
5. **Size & enter** — position size comes from the ATR risk formula (about 1%
   of equity at risk per trade), with a stop-loss and take-profit attached at
   the moment of entry.

Every step is written to the activity feed below, and every trade carries a
one-sentence plain-English explanation.

---

#### 1 · Reading the chart — what each indicator means

These are the same lines drawn on the **Market** tab chart:

- **SMA 20 / SMA 50 (simple moving averages)** — the average close of the last
  20 / 50 candles. They smooth out the noise: price above a rising SMA means
  the market is trending up. The slower SMA 50 is the bot's
  "which way is the tide flowing" filter.
- **EMA 12 / EMA 26 (exponential moving averages)** — like SMAs but weighted
  toward recent candles, so they react faster. The fast EMA 12 crossing above
  the slow EMA 26 means momentum is shifting up (this crossover is also the
  heart of MACD).
- **Bollinger Bands (20, 2σ)** — the shaded envelope: 2 standard deviations
  above and below the 20-bar average. Price spends roughly 95% of its time
  inside; a close *outside* the band is a statistically stretched move that
  often snaps back toward the middle band.
- **RSI 14 (relative strength index)** — the 0–100 oscillator in the third
  chart panel. Above 70 = overbought (the rally may be exhausted), below 30 =
  oversold (the sell-off may be exhausted); above / below 50 = bullish /
  bearish momentum.
- **MACD (12, 26, 9)** — bottom panel: the gap between EMA 12 and EMA 26 (MACD
  line), its 9-bar average (signal line) and their difference (histogram
  bars). MACD crossing above its signal line = momentum turning up; the
  histogram flipping green/red shows the same thing at a glance.
- **Volume bars** — how much was traded in each candle. A breakout on heavy
  volume is far more trustworthy than one on thin volume.
- **ATR 14 (average true range)** — not drawn as a line but crucial: the
  average size of one candle's move. The bot uses it to measure "how big is a
  normal wiggle", so stops sit *outside* the noise and position sizes shrink
  automatically when the market turns wild.
- **Momentum 10** — the % change of price versus 10 candles ago; used to rank
  candidates that received the same number of votes.

#### 2 · The four strategies — exact buy & sell triggers

Each strategy looks at the latest closed candle and votes **+1 (long)**,
**−1 (short)** or **0 (flat)**:

- **Trend Following** — *buys* when EMA 12 > EMA 26 **and** close > SMA 50
  (fast momentum up inside an established uptrend); *shorts* when EMA 12 <
  EMA 26 and close < SMA 50; otherwise flat. Chart pattern: riding a trend —
  "the trend is your friend".
- **Mean Reversion** — *buys* when the close drops **below the lower Bollinger
  band** while RSI 14 < 30 (price panic-stretched *and* oversold); goes back
  to flat once the close crosses the middle band; *shorts* the mirror image
  (close above the upper band with RSI 14 > 70). Chart pattern: the
  rubber-band snap-back after an overdone move.
- **Breakout** — *buys* when the close pushes **above the highest high of the
  previous 20 candles** with volume > 1.5× its 20-bar average (a genuine break
  of resistance, confirmed by participation); *shorts* the symmetric breakdown
  below 20-bar support; holds until price reaches the opposite side of a
  10-bar channel. Chart pattern: tight consolidation → explosive escape.
- **RSI + MACD** — *buys* the moment the MACD line **crosses above its signal
  line** while RSI 14 > 50 (a momentum trigger with bullish confirmation);
  *shorts* when MACD crosses below its signal with RSI 14 < 50; keeps its
  stance until the opposite cross. Chart pattern: momentum turning over with
  confirmation.

#### 3 · The ensemble vote

The four votes are summed into a score from −4 to +4. A coin is shortlisted
only when **|score| ≥ min_vote** (default 2 — at least two independent methods
agree on direction), the bot holds no position in it, and the direction is
long (unless *Allow short* is enabled). Candidates are ranked by vote strength
first, momentum second, and only enough to fill the free position slots move
on. The same votes also manage exits: if the ensemble stance flips against an
open position, the bot closes it at market on the next cycle.

#### 4 · The AI confirmation gate

Before entering, the bot asks the local LLM for a second opinion. The model
receives the full indicator snapshot (RSI, MACD state, moving averages,
Bollinger position, ATR, volatility, recent returns) **plus the chart
structure**: the 20-bar swing high/low and how far price sits from each
(distance to resistance/support), whether the 20-bar trend is up / down /
sideways, whether volume is rising or falling, and the last 5 closes. It
answers with a **sentiment** (bullish / bearish / neutral), a **confidence**
(0–100), its **reasoning**, and a **risk commentary** — all shown in the
expandable feed row of every trade. The entry proceeds only when the sentiment
matches the trade direction **and** confidence ≥ **min_ai_confidence**
(default 60). If the model is unreachable or errors, the bot *skips* the
candidate — it never buys blind.

#### 5 · Position sizing — the ATR risk formula

The dollar amount a single trade can lose is capped at **1% of equity**:

```
qty = (equity × 1%) / (2 × ATR14)
```

then capped twice more: notional ≤ **35% of equity** per coin, and ≤ 95% of
available cash. Anything under $1 notional is skipped as dust.

**Worked example — a $100 account buying BTC at $70,000 with ATR14 = $500:**

1. Risk budget: 1% × $100 = **$1.00** maximum planned loss.
2. Stop distance: 2 × ATR = **$1,000** below entry.
3. Raw size: $1.00 ÷ $1,000 = **0.001 BTC** (a $70 position).
4. Per-coin cap: 35% × $100 = $35 → size reduced to **0.0005 BTC** ($35).
5. Result: stop-loss at $69,000 (max loss ≈ **$0.50**), take-profit at
   $71,500 (target gain ≈ **$0.75**).

Because size is *inversely* proportional to ATR, the bot automatically buys
less when the market is volatile and more when it is calm — every trade risks
roughly the same real amount.

#### 6 · Stop-loss & take-profit placement

- **Stop-loss:** entry − 2 × ATR for longs (entry + 2 × ATR for shorts) — far
  enough outside normal candle noise that random wiggles don't stop you out,
  close enough that a real reversal does.
- **Take-profit:** entry + 3 × ATR for longs (entry − 3 × ATR for shorts) — a
  1.5 : 1 reward-to-risk ratio, so the bot can be profitable even when fewer
  than half its trades win.
- Both levels are attached at entry and checked against every new candle's
  high/low. A position can also close early when the strategy consensus flips
  (step 3).

#### 7 · Hard risk limits (always on)

| Limit | Value | What it does |
|---|---|---|
| Risk per trade | 1% of equity | the ATR sizing formula above |
| Max open positions | 5 | prevents over-concentration |
| Max per coin | 35% of equity | one coin can never dominate the account |
| Daily loss halt | −3% in a day | blocks *new* entries until the next day |
| Circuit breaker | −10% from peak equity | halts ALL new entries until a manual reset |
| Simulated costs | 0.1% commission + 0.05% slippage per fill | keeps results honest |

#### 8 · Reading the activity feed

Every decision is logged: 🔁 cycle start · 🔍 scan (votes & indicators per
coin) · ✅ enter · 🔴 exit · ⏭ skip (filters or the AI said no) · 🚫 reject
(the risk manager said no) · ⚠️ error / halt · 🏁 cycle end. Click any
**enter** or **exit** row to see the full evidence: how each strategy voted,
the indicator readout, the AI's reasoning and risk notes, and the
plain-English explanation of the trade.
"""


def _fmt_bot_ts(value: Any) -> str:
    """Format an ISO timestamp (or None) as a short UTC 'YYYY-MM-DD HH:MM'."""
    if value in (None, ""):
        return "—"
    try:
        ts = pd.to_datetime(value, utc=True)
        return ts.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(value)


def _detail_dict(detail: Any) -> dict[str, Any]:
    """Coerce a bot_activity ``detail`` payload (dict or JSON string) to a dict."""
    if isinstance(detail, dict):
        return detail
    if isinstance(detail, str) and detail.strip():
        try:
            parsed = json.loads(detail)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            logger.debug("Unparseable bot activity detail: %s", detail[:200])
    return {}


def _stance_label(vote: Any) -> str:
    """Human label for a single strategy vote (+1/-1/0)."""
    try:
        value = int(vote)
    except (TypeError, ValueError):
        return str(vote)
    return {1: "long", -1: "short", 0: "flat"}.get(value, str(value))


def bot_status_row(status: dict[str, Any]) -> None:
    """Render the AI Trader status row: running badge + key metrics.

    Args:
        status: dict from ``GET /api/bot/status`` (running, equity,
            last_cycle_ts, next_cycle_ts, open_positions, watchlist,
            last_cycle_summary).
    """
    running = bool(status.get("running"))
    color = COLOR_UP if running else COLOR_NEUTRAL
    label = "🟢 RUNNING" if running else "⏸ STOPPED"
    st.markdown(
        f'<span style="background-color:{color};color:#ffffff;'
        f"padding:0.35em 1.0em;border-radius:1em;font-weight:700;"
        f'font-size:1.05em;letter-spacing:0.05em;">{label}</span>',
        unsafe_allow_html=True,
    )
    metric_row(
        [
            ("Equity", fmt_currency(status.get("equity")), None),
            ("Open Positions", fmt_int(status.get("open_positions")), None),
            ("Last Cycle (UTC)", _fmt_bot_ts(status.get("last_cycle_ts")), None),
            ("Next Cycle (UTC)", _fmt_bot_ts(status.get("next_cycle_ts")), None),
        ],
        per_row=4,
    )
    watchlist = status.get("watchlist") or []
    if watchlist:
        st.caption("Watchlist: " + ", ".join(str(s) for s in watchlist))
    summary = status.get("last_cycle_summary")
    if isinstance(summary, dict) and summary:
        st.caption(f"Last cycle: {_cycle_summary_text(summary)}")


def _cycle_summary_text(detail: dict[str, Any]) -> str:
    """Compact one-line text for a cycle summary / cycle_end detail dict."""
    parts: list[str] = []
    for key in ("scanned", "entered", "exited", "skipped"):
        if key in detail:
            parts.append(f"{key} {fmt_int(detail.get(key))}")
    if "equity" in detail:
        parts.append(f"equity {fmt_currency(detail.get('equity'))}")
    if parts:
        return " · ".join(parts)
    return json.dumps(detail, default=str)[:200]


def _bot_line_summary(action: str, detail: dict[str, Any]) -> str:
    """One-line summary text for a non-expandable activity row."""
    if action == "scan":
        parts: list[str] = []
        if "vote_sum" in detail:
            parts.append(f"votes {detail.get('vote_sum')}")
        if "rsi_14" in detail and not _is_missing(detail.get("rsi_14")):
            parts.append(f"RSI {fmt_num(detail.get('rsi_14'), 1)}")
        if "momentum_10" in detail and not _is_missing(detail.get("momentum_10")):
            parts.append(f"momentum {fmt_pct(detail.get('momentum_10'))}")
        if "price" in detail and not _is_missing(detail.get("price")):
            parts.append(f"price {fmt_num(detail.get('price'), 4)}")
        return " · ".join(parts) or json.dumps(detail, default=str)[:160]
    if action in ("skip", "reject", "halt", "error"):
        return str(
            detail.get("explanation")
            or detail.get("reason")
            or detail.get("error")
            or json.dumps(detail, default=str)[:160]
        )
    if action == "cycle_end":
        return _cycle_summary_text(detail)
    if detail:
        return json.dumps(detail, default=str)[:160]
    return ""


def _bot_enter_detail(detail: dict[str, Any]) -> None:
    """Render the full transparency block for an ``enter`` activity row."""
    metric_row(
        [
            ("Side", str(detail.get("side") or "—").upper(), None),
            ("Qty", fmt_num(detail.get("qty"), 6), None),
            ("Price", fmt_num(detail.get("price"), 4), None),
            ("Notional", fmt_currency(detail.get("notional")), None),
            ("Stop Loss", fmt_num(detail.get("stop_loss"), 4), None),
            ("Take Profit", fmt_num(detail.get("take_profit"), 4), None),
        ],
        per_row=6,
    )

    votes = detail.get("votes") or {}
    if votes:
        st.markdown(f"**Strategy votes** — sum {detail.get('vote_sum', '—')}")
        votes_df = pd.DataFrame(
            [
                {"strategy": name, "vote": vote, "stance": _stance_label(vote)}
                for name, vote in votes.items()
            ]
        )
        st.dataframe(votes_df, use_container_width=True)

    indicators = detail.get("indicators") or {}
    if indicators:
        st.markdown("**Indicator readout**")
        indicators_df = pd.DataFrame(
            [
                {
                    "indicator": name,
                    "value": fmt_num(value, 4)
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else str(value),
                }
                for name, value in indicators.items()
            ]
        )
        st.dataframe(indicators_df, use_container_width=True)

    ai = detail.get("ai")
    if isinstance(ai, dict) and ai:
        st.markdown("**AI confirmation**")
        sentiment_badge(str(ai.get("sentiment", "neutral")), ai.get("confidence"))
        reasoning = ai.get("reasoning")
        if reasoning:
            st.markdown(f"**Reasoning:** {reasoning}")
        risk = ai.get("risk_commentary")
        if risk:
            st.warning(f"**Risk commentary:** {risk}")
    else:
        st.caption("AI confirmation was disabled (or not used) for this entry.")

    explanation = detail.get("explanation")
    if explanation:
        st.info(f"💬 {explanation}")


def _bot_exit_detail(detail: dict[str, Any]) -> None:
    """Render the transparency block for an ``exit`` activity row."""
    metric_row(
        [
            ("Reason", str(detail.get("reason") or "—"), None),
            ("PnL", fmt_currency(detail.get("pnl")), None),
            ("PnL %", fmt_pct(detail.get("pnl_pct")), None),
        ],
        per_row=3,
    )
    explanation = detail.get("explanation")
    if explanation:
        st.info(f"💬 {explanation}")


def bot_activity_feed(items: list[dict[str, Any]], show_scans: bool = True) -> None:
    """Render the bot activity feed with expandable enter/exit rows.

    ``enter``/``exit`` rows render as ``st.expander`` blocks containing the
    full transparency detail (votes table, indicator readout, AI reasoning &
    risk commentary, explanation sentence); everything else renders as a
    single line with its emoji marker.

    Args:
        items: Activity records from ``GET /api/bot/activity`` (newest first):
            dicts with ``ts`` (iso), ``symbol``, ``action`` and ``detail``.
        show_scans: When False, ``scan`` rows are hidden to reduce noise.
    """
    if not items:
        st.caption("No bot activity yet — start the bot or run a cycle.")
        return
    for item in items:
        action = str(item.get("action", "")).lower()
        if action == "scan" and not show_scans:
            continue
        detail = _detail_dict(item.get("detail"))
        emoji = BOT_ACTION_EMOJI.get(action, "•")
        ts = _fmt_bot_ts(item.get("ts"))
        symbol = str(item.get("symbol") or "")
        if action in ("enter", "exit"):
            explanation = str(detail.get("explanation") or "").strip()
            title = f"{emoji} {ts} · {action.upper()} {symbol}"
            if explanation:
                snippet = (
                    explanation
                    if len(explanation) <= 140
                    else explanation[:137] + "..."
                )
                title = f"{title} — {snippet}"
            with st.expander(title):
                if action == "enter":
                    _bot_enter_detail(detail)
                else:
                    _bot_exit_detail(detail)
        else:
            summary = _bot_line_summary(action, detail)
            line = f"{emoji} `{ts}` · **{action}** {symbol}".rstrip()
            if summary:
                line = f"{line} — {summary}"
            st.markdown(line)


def bot_trades_table(items: list[dict[str, Any]]) -> None:
    """Render recent closed bot trades: exit PnL paired with entry explanation.

    Walks the activity feed newest-first; every ``exit`` row is matched with
    the most recent *earlier* ``enter`` row for the same symbol so the user
    sees each closed trade's PnL next to the reasoning that opened it.

    Args:
        items: Activity records from ``GET /api/bot/activity``.
    """
    ordered = sorted(items, key=lambda rec: str(rec.get("ts") or ""), reverse=True)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(ordered):
        if str(item.get("action", "")).lower() != "exit":
            continue
        detail = _detail_dict(item.get("detail"))
        symbol = str(item.get("symbol") or "")
        entry_explanation = ""
        for older in ordered[index + 1 :]:
            if (
                str(older.get("action", "")).lower() == "enter"
                and str(older.get("symbol") or "") == symbol
            ):
                entry_explanation = str(
                    _detail_dict(older.get("detail")).get("explanation") or ""
                )
                break
        rows.append(
            {
                "closed (UTC)": _fmt_bot_ts(item.get("ts")),
                "symbol": symbol,
                "pnl": fmt_currency(detail.get("pnl")),
                "pnl %": fmt_pct(detail.get("pnl_pct")),
                "exit reason": str(detail.get("reason") or ""),
                "exit explanation": str(detail.get("explanation") or ""),
                "entry explanation": entry_explanation,
            }
        )
    if not rows:
        st.caption("No closed bot trades in the recent activity window.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
