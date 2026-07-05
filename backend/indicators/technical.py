"""Pure-pandas technical indicators for the Trading AI Platform.

PAPER TRADING ONLY — these are research indicators; nothing here places orders.

Every function in this module is pure (no side effects, no I/O, no network)
and strictly free of lookahead bias: only ``rolling``, ``ewm``, ``shift``,
``cumsum``-per-group and similar backward-looking operations are used, so the
value at row ``t`` depends exclusively on data at rows ``<= t``.

Conventions (see CONTRACTS.md):
    * Input candle frames are the canonical OHLCV shape — UTC tz-aware
      ``pd.DatetimeIndex`` named ``"timestamp"``, ascending and unique, with
      float64 columns ``open, high, low, close, volume``.
    * RSI and ATR use Wilder smoothing: ``ewm(alpha=1/window, adjust=False)``.
    * VWAP resets at every UTC calendar-day boundary.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger",
    "atr",
    "vwap",
    "volume_profile",
    "returns",
    "volatility",
    "momentum",
]

_OHLC_COLUMNS = ("open", "high", "low", "close", "volume")


def _validate_window(window: int, name: str = "window") -> None:
    """Raise ``ValueError`` if ``window`` is not a positive integer.

    Args:
        window: The lookback length to validate.
        name: Parameter name used in the error message.
    """
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError(f"{name} must be a positive integer, got {window!r}")


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Raise ``ValueError`` if ``df`` is missing any of ``columns``.

    Args:
        df: Candidate OHLCV frame.
        columns: Column names that must be present.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def sma(s: pd.Series, window: int) -> pd.Series:
    """Simple moving average.

    Args:
        s: Input price series.
        window: Rolling window length in bars.

    Returns:
        Rolling mean of ``s``; the first ``window - 1`` values are NaN.
    """
    _validate_window(window)
    return s.rolling(window=window, min_periods=window).mean()


def ema(s: pd.Series, window: int) -> pd.Series:
    """Exponential moving average (``ewm(span=window, adjust=False)``).

    Args:
        s: Input price series.
        window: Span of the exponential window in bars.

    Returns:
        Exponentially weighted mean of ``s``.
    """
    _validate_window(window)
    return s.ewm(span=window, adjust=False).mean()


def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder smoothing, bounded to ``[0, 100]``.

    Average gains/losses use ``ewm(alpha=1/window, adjust=False)`` (Wilder).
    When average loss is zero and average gain is positive the RSI is 100;
    when both are zero (perfectly flat series) it is 50.

    Args:
        s: Input price series (typically close).
        window: Wilder smoothing period.

    Returns:
        RSI series in ``[0, 100]``; leading values are NaN during warm-up.
    """
    _validate_window(window)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 & avg_gain > 0 -> pure upside -> 100
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    # avg_loss == 0 & avg_gain == 0 -> flat -> neutral 50
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out.clip(lower=0.0, upper=100.0)


def macd(
    s: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence Divergence.

    Args:
        s: Input price series.
        fast: Span of the fast EMA.
        slow: Span of the slow EMA.
        signal: Span of the signal-line EMA of the MACD line.

    Returns:
        Tuple ``(macd_line, signal_line, histogram)`` where
        ``macd_line = ema(s, fast) - ema(s, slow)``,
        ``signal_line = ema(macd_line, signal)`` and
        ``histogram = macd_line - signal_line``.
    """
    _validate_window(fast, "fast")
    _validate_window(slow, "slow")
    _validate_window(signal, "signal")
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger(
    s: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands around a simple moving average.

    Args:
        s: Input price series.
        window: Rolling window for the middle band and standard deviation.
        num_std: Number of standard deviations for the upper/lower bands.

    Returns:
        Tuple ``(upper, mid, lower)`` where ``mid`` is the ``window``-bar SMA
        and the bands sit ``num_std`` rolling standard deviations away.
    """
    _validate_window(window)
    mid = sma(s, window)
    std = s.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range with Wilder smoothing.

    True range is ``max(high - low, |high - prev_close|, |low - prev_close|)``;
    on the first bar (no previous close) it degrades to ``high - low``. The
    average uses ``ewm(alpha=1/window, adjust=False)`` (Wilder).

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``, ``close``).
        window: Wilder smoothing period.

    Returns:
        ATR series aligned to ``df.index``.
    """
    _validate_window(window)
    _require_columns(df, ("high", "low", "close"))
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price, cumulative within each UTC calendar day.

    Typical price ``(high + low + close) / 3`` is volume-weighted and the
    running sums reset at every UTC day boundary (groupby on the normalized
    index date), so no bar ever sees data from a later bar or a later day.

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``, ``close``,
            ``volume``) with a UTC ``pd.DatetimeIndex``.

    Returns:
        VWAP series aligned to ``df.index``; NaN where the day's cumulative
        volume is still zero.
    """
    _require_columns(df, ("high", "low", "close", "volume"))
    if df.empty:
        return pd.Series(dtype="float64", index=df.index, name="vwap")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("vwap requires a pd.DatetimeIndex (UTC) index")

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    day = df.index.normalize()  # UTC calendar day (index is tz-aware UTC)
    cum_pv = pv.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum()
    out = cum_pv / cum_vol.mask(cum_vol == 0)
    out.name = "vwap"
    return out


def volume_profile(df: pd.DataFrame, bins: int = 24) -> pd.DataFrame:
    """Volume distribution across equal price bins for the given frame.

    The price range ``[low.min(), high.max()]`` is split into ``bins`` equal
    intervals; each candle's volume is assigned to the bin containing its
    typical price ``(high + low + close) / 3`` and summed per bin.

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``, ``close``,
            ``volume``).
        bins: Number of equal-width price bins.

    Returns:
        DataFrame with integer index ``0..bins-1`` and columns
        ``price_low``, ``price_high``, ``volume``. Empty (zero rows) when the
        input frame is empty or the price range is undefined.
    """
    _validate_window(bins, "bins")
    _require_columns(df, ("high", "low", "close", "volume"))

    empty = pd.DataFrame(
        {
            "price_low": pd.Series(dtype="float64"),
            "price_high": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )
    if df.empty:
        logger.warning("volume_profile called with an empty DataFrame")
        return empty

    lo = float(df["low"].min())
    hi = float(df["high"].max())
    if not np.isfinite(lo) or not np.isfinite(hi):
        logger.warning("volume_profile: non-finite price range (%s, %s)", lo, hi)
        return empty
    if hi <= lo:
        # Degenerate flat range — widen it symmetrically so binning is defined.
        pad = max(abs(lo) * 1e-9, 1e-9)
        lo, hi = lo - pad, hi + pad

    edges = np.linspace(lo, hi, bins + 1)
    typical = ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy(dtype="float64")
    volume = df["volume"].to_numpy(dtype="float64")

    idx = np.searchsorted(edges, typical, side="right") - 1
    idx = np.clip(idx, 0, bins - 1)

    valid = np.isfinite(typical) & np.isfinite(volume)
    binned = np.zeros(bins, dtype="float64")
    np.add.at(binned, idx[valid], volume[valid])

    return pd.DataFrame(
        {
            "price_low": edges[:-1],
            "price_high": edges[1:],
            "volume": binned,
        },
        index=pd.RangeIndex(bins),
    )


def returns(s: pd.Series, periods: int = 1) -> pd.Series:
    """Simple percentage returns over ``periods`` bars.

    Args:
        s: Input price series.
        periods: Number of bars between the two observations.

    Returns:
        ``s.pct_change(periods)``; leading values are NaN.
    """
    _validate_window(periods, "periods")
    return s.pct_change(periods=periods)


def volatility(s: pd.Series, window: int = 20) -> pd.Series:
    """Annualized rolling volatility of 1-period returns.

    Rolling standard deviation of 1-bar percentage returns, annualized by
    ``sqrt(365)`` (crypto trades every day).

    Args:
        s: Input price series.
        window: Rolling window length in bars.

    Returns:
        Annualized volatility series; leading values are NaN.
    """
    _validate_window(window)
    r = returns(s, periods=1)
    return r.rolling(window=window, min_periods=window).std() * np.sqrt(365.0)


def momentum(s: pd.Series, window: int = 10) -> pd.Series:
    """Rate-of-change momentum: ``s / s.shift(window) - 1``.

    Args:
        s: Input price series.
        window: Lookback in bars.

    Returns:
        Fractional change versus ``window`` bars ago; leading values are NaN.
    """
    _validate_window(window)
    return s / s.shift(window) - 1.0
