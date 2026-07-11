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
import math

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
    "adx",
    "stochastic",
    "stoch_rsi",
    "williams_r",
    "cci",
    "roc",
    "mfi",
    "obv",
    "cmf",
    "adl",
    "vwma",
    "supertrend",
    "psar",
    "aroon",
    "donchian",
    "keltner",
    "hull_ma",
    "trix",
    "ichimoku",
    "bollinger_pct_b",
    "bollinger_bandwidth",
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


# ---------------------------------------------------------------------------
# Indicator Pack v4 — see CONTRACTS.md "## Indicator Pack v4" (binding).
# All functions below are pure, never mutate their inputs, produce NaN during
# warm-up (never raise on short frames) and are strictly free of lookahead:
# the recursive indicators (supertrend, psar) start from bar 0 with pinned
# seeds, so recomputing on a truncated frame reproduces the retained rows.
# ---------------------------------------------------------------------------


def _wilder(s: pd.Series, window: int) -> pd.Series:
    """Wilder smoothing: ``ewm(alpha=1/window, adjust=False).mean()``.

    Identical to the smoothing used by :func:`rsi` and :func:`atr`.

    Args:
        s: Input series.
        window: Wilder smoothing period.

    Returns:
        Wilder-smoothed series aligned to ``s.index``.
    """
    return s.ewm(alpha=1.0 / window, adjust=False).mean()


def _wma(s: pd.Series, window: int) -> pd.Series:
    """Linearly-weighted moving average (weights ``1..window``, recent heaviest).

    Equivalent to ``rolling(window, min_periods=window)`` with weights
    ``1..window``; implemented with a vectorized sliding-window dot product
    (no per-row pandas objects). Windows containing NaN yield NaN.

    Args:
        s: Input series.
        window: Rolling window length in bars.

    Returns:
        Weighted moving average; the first ``window - 1`` values are NaN.
    """
    n = len(s)
    out = np.full(n, np.nan)
    if n >= window:
        weights = np.arange(1, window + 1, dtype="float64")
        windows = np.lib.stride_tricks.sliding_window_view(
            s.to_numpy(dtype="float64"), window
        )
        out[window - 1 :] = windows @ weights / weights.sum()
    return pd.Series(out, index=s.index)


def _money_flow_volume(df: pd.DataFrame) -> pd.Series:
    """Money Flow Volume: ``mfm * volume`` (shared by :func:`cmf`/:func:`adl`).

    Money Flow Multiplier ``mfm = ((close - low) - (high - close)) /
    (high - low)``, pinned to ``0.0`` where ``high == low``.

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``, ``close``,
            ``volume``).

    Returns:
        Money Flow Volume series aligned to ``df.index``.
    """
    _require_columns(df, ("high", "low", "close", "volume"))
    hl_range = df["high"] - df["low"]
    with np.errstate(divide="ignore", invalid="ignore"):
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range
    mfm = mfm.mask(hl_range == 0, 0.0)
    return mfm * df["volume"]


def adx(
    df: pd.DataFrame,
    window: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index with +DI/-DI (Wilder throughout).

    ``+DM = high.diff() where it exceeds both -low.diff() and 0, else 0``
    (``-DM`` mirrored); ``di = 100 * wilder(DM) / atr``;
    ``dx = 100 * |di+ - di-| / (di+ + di-)``; ``adx = wilder(dx)``.
    Divisions by zero yield NaN.

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``, ``close``).
        window: Wilder smoothing period.

    Returns:
        Tuple ``(adx, di_plus, di_minus)``, each in ``[0, 100]``; leading
        values are NaN/unsettled during warm-up.
    """
    _validate_window(window)
    _require_columns(df, ("high", "low", "close"))
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_ = atr(df, window)
    atr_masked = atr_.mask(atr_ == 0)
    di_plus = 100.0 * _wilder(plus_dm, window) / atr_masked
    di_minus = 100.0 * _wilder(minus_dm, window) / atr_masked
    di_sum = di_plus + di_minus
    dx = 100.0 * (di_plus - di_minus).abs() / di_sum.mask(di_sum == 0)
    adx_line = _wilder(dx, window)
    return adx_line, di_plus, di_minus


def stochastic(
    df: pd.DataFrame,
    k_window: int = 14,
    d_window: int = 3,
    smooth_k: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Slow stochastic oscillator (SMA smoothing).

    ``raw_k = 100 * (close - LL) / (HH - LL)`` over ``k_window`` bars
    (``HH == LL`` yields NaN); ``k = sma(raw_k, smooth_k)``;
    ``d = sma(k, d_window)``.

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``, ``close``).
        k_window: Lookback for the highest-high/lowest-low range.
        d_window: SMA period of the %D line.
        smooth_k: SMA period applied to raw %K.

    Returns:
        Tuple ``(k, d)``, each in ``[0, 100]``; NaN during warm-up.
    """
    _validate_window(k_window, "k_window")
    _validate_window(d_window, "d_window")
    _validate_window(smooth_k, "smooth_k")
    _require_columns(df, ("high", "low", "close"))
    hh = df["high"].rolling(window=k_window, min_periods=k_window).max()
    ll = df["low"].rolling(window=k_window, min_periods=k_window).min()
    rng = hh - ll
    raw_k = 100.0 * (df["close"] - ll) / rng.mask(rng == 0)
    k = sma(raw_k, smooth_k)
    d = sma(k, d_window)
    return k, d


def stoch_rsi(
    s: pd.Series,
    rsi_window: int = 14,
    stoch_window: int = 14,
    k: int = 3,
    d: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI on the 0-100 scale (stochastic of the Wilder RSI).

    ``raw = 100 * (rsi - min(rsi)) / (max(rsi) - min(rsi))`` over
    ``stoch_window`` bars (flat window yields NaN); ``k_line = sma(raw, k)``;
    ``d_line = sma(k_line, d)``.

    Args:
        s: Input price series (typically close).
        rsi_window: Wilder RSI period.
        stoch_window: Lookback for the RSI min/max range.
        k: SMA period applied to the raw stochastic.
        d: SMA period of the signal line.

    Returns:
        Tuple ``(k_line, d_line)``, each in ``[0, 100]``; NaN during warm-up.
    """
    _validate_window(rsi_window, "rsi_window")
    _validate_window(stoch_window, "stoch_window")
    _validate_window(k, "k")
    _validate_window(d, "d")
    r = rsi(s, rsi_window)
    r_min = r.rolling(window=stoch_window, min_periods=stoch_window).min()
    r_max = r.rolling(window=stoch_window, min_periods=stoch_window).max()
    rng = r_max - r_min
    raw = 100.0 * (r - r_min) / rng.mask(rng == 0)
    k_line = sma(raw, k)
    d_line = sma(k_line, d)
    return k_line, d_line


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Williams %R: ``-100 * (HH - close) / (HH - LL)``.

    Highest high / lowest low are taken over ``window`` bars including the
    current bar; ``HH == LL`` yields NaN.

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``, ``close``).
        window: Lookback in bars.

    Returns:
        Williams %R series in ``[-100, 0]``; NaN during warm-up.
    """
    _validate_window(window)
    _require_columns(df, ("high", "low", "close"))
    hh = df["high"].rolling(window=window, min_periods=window).max()
    ll = df["low"].rolling(window=window, min_periods=window).min()
    rng = hh - ll
    return -100.0 * (hh - df["close"]) / rng.mask(rng == 0)


def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Commodity Channel Index.

    ``tp = (high + low + close) / 3``;
    ``cci = (tp - sma(tp, window)) / (0.015 * MAD)`` where ``MAD`` is the
    rolling mean absolute deviation of ``tp`` about the rolling mean
    (``MAD == 0`` yields NaN). Implemented with a vectorized sliding window.

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``, ``close``).
        window: Rolling window length in bars.

    Returns:
        CCI series (unbounded, typically within ±200); NaN during warm-up.
    """
    _validate_window(window)
    _require_columns(df, ("high", "low", "close"))
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    n = len(tp)
    mad_arr = np.full(n, np.nan)
    if n >= window:
        windows = np.lib.stride_tricks.sliding_window_view(
            tp.to_numpy(dtype="float64"), window
        )
        window_mean = windows.mean(axis=1)
        mad_arr[window - 1 :] = np.abs(windows - window_mean[:, None]).mean(axis=1)
    mad = pd.Series(mad_arr, index=tp.index)
    return (tp - sma(tp, window)) / (0.015 * mad.mask(mad == 0))


def roc(s: pd.Series, window: int = 10) -> pd.Series:
    """Rate of Change in percent: ``100 * (s / s.shift(window) - 1)``.

    Human-scale percent variant of :func:`momentum`
    (``roc == momentum * 100``).

    Args:
        s: Input price series.
        window: Lookback in bars.

    Returns:
        Percent change versus ``window`` bars ago; leading values are NaN.
    """
    _validate_window(window)
    return 100.0 * (s / s.shift(window) - 1.0)


def mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Money Flow Index (classic rolling-sum variant, not Wilder).

    ``tp = (high + low + close) / 3``; raw money flow ``tp * volume`` counts
    as positive where ``tp > tp.shift(1)``, negative where ``tp <
    tp.shift(1)`` (ties contribute to neither); flows are summed over
    ``window`` bars and ``mfi = 100 * pos / (pos + neg)``. Pinned edge cases
    mirroring :func:`rsi`: no negative flow with positive flow present is
    100, the reverse is 0, both zero is 50.

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``, ``close``,
            ``volume``).
        window: Rolling-sum window length in bars.

    Returns:
        MFI series in ``[0, 100]``; NaN during warm-up.
    """
    _validate_window(window)
    _require_columns(df, ("high", "low", "close", "volume"))
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    rmf = tp * df["volume"]
    prev_tp = tp.shift(1)
    pos_flow = rmf.where(tp > prev_tp, 0.0)
    neg_flow = rmf.where(tp < prev_tp, 0.0)
    pos = pos_flow.rolling(window=window, min_periods=window).sum()
    neg = neg_flow.rolling(window=window, min_periods=window).sum()
    total = pos + neg
    out = 100.0 * pos / total.mask(total == 0)
    out = out.mask((neg == 0) & (pos > 0), 100.0)
    out = out.mask((pos == 0) & (neg > 0), 0.0)
    out = out.mask((pos == 0) & (neg == 0), 50.0)
    return out


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: ``(sign(close.diff()) * volume).fillna(0).cumsum()``.

    The first bar contributes 0. Unbounded, in volume units.

    Args:
        df: Canonical OHLCV frame (needs ``close``, ``volume``).

    Returns:
        Cumulative OBV series aligned to ``df.index``.
    """
    _require_columns(df, ("close", "volume"))
    direction = np.sign(df["close"].diff())
    return (direction * df["volume"]).fillna(0.0).cumsum()


def cmf(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Chaikin Money Flow.

    Money Flow Multiplier ``mfm = ((close - low) - (high - close)) /
    (high - low)`` (pinned to 0 where ``high == low``); ``mfv = mfm *
    volume``; ``cmf = rolling_sum(mfv) / rolling_sum(volume)`` (zero volume
    sum yields NaN).

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``, ``close``,
            ``volume``).
        window: Rolling-sum window length in bars.

    Returns:
        CMF series in ``[-1, 1]``; NaN during warm-up.
    """
    _validate_window(window)
    mfv = _money_flow_volume(df)
    mfv_sum = mfv.rolling(window=window, min_periods=window).sum()
    vol_sum = df["volume"].rolling(window=window, min_periods=window).sum()
    return mfv_sum / vol_sum.mask(vol_sum == 0)


def adl(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line: cumulative Money Flow Volume.

    Uses the same Money Flow Multiplier as :func:`cmf` (0 where
    ``high == low``). Unbounded, in volume units.

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``, ``close``,
            ``volume``).

    Returns:
        Cumulative ADL series aligned to ``df.index``.
    """
    return _money_flow_volume(df).cumsum()


def vwma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume-Weighted Moving Average.

    ``rolling_sum(close * volume) / rolling_sum(volume)`` (zero volume sum
    yields NaN).

    Args:
        df: Canonical OHLCV frame (needs ``close``, ``volume``).
        window: Rolling-sum window length in bars.

    Returns:
        VWMA series aligned to ``df.index``; NaN during warm-up.
    """
    _validate_window(window)
    _require_columns(df, ("close", "volume"))
    pv = df["close"] * df["volume"]
    pv_sum = pv.rolling(window=window, min_periods=window).sum()
    vol_sum = df["volume"].rolling(window=window, min_periods=window).sum()
    return pv_sum / vol_sum.mask(vol_sum == 0)


def supertrend(
    df: pd.DataFrame,
    window: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """Supertrend line and direction (Wilder ATR bands with band ratcheting).

    Basic bands sit ``multiplier * atr`` around ``hl2 = (high + low) / 2``.
    Final bands ratchet toward price (an upper band may only move down
    unless price closed above it, mirrored for the lower band); direction
    flips to +1 when close crosses above the prior final upper band and to
    -1 when it crosses below the prior final lower band. Rows ``0 ..
    window - 1`` are NaN (line and direction); the recursion seeds at row
    ``window`` with direction +1 when ``close >= hl2`` else -1.

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``, ``close``).
        window: Wilder ATR period.
        multiplier: Band width in ATR multiples.

    Returns:
        Tuple ``(line, direction)`` where ``line`` is the active band (lower
        band in uptrends, upper band in downtrends) and ``direction`` is
        float ``+1.0`` / ``-1.0`` (NaN during warm-up).
    """
    _validate_window(window)
    _require_columns(df, ("high", "low", "close"))
    n = len(df)
    line = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    if n > window:
        atr_np = atr(df, window).to_numpy(dtype="float64")
        hl2 = ((df["high"] + df["low"]) / 2.0).to_numpy(dtype="float64")
        close = df["close"].to_numpy(dtype="float64")
        upper_basic = hl2 + multiplier * atr_np
        lower_basic = hl2 - multiplier * atr_np
        t0 = window
        final_upper = upper_basic[t0]
        final_lower = lower_basic[t0]
        trend = 1.0 if close[t0] >= hl2[t0] else -1.0
        direction[t0] = trend
        line[t0] = final_lower if trend > 0 else final_upper
        for t in range(t0 + 1, n):
            prev_upper = final_upper
            prev_lower = final_lower
            if upper_basic[t] < prev_upper or close[t - 1] > prev_upper:
                final_upper = upper_basic[t]
            if lower_basic[t] > prev_lower or close[t - 1] < prev_lower:
                final_lower = lower_basic[t]
            if close[t] > prev_upper:
                trend = 1.0
            elif close[t] < prev_lower:
                trend = -1.0
            direction[t] = trend
            line[t] = final_lower if trend > 0 else final_upper
    return pd.Series(line, index=df.index), pd.Series(direction, index=df.index)


def psar(
    df: pd.DataFrame,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2,
) -> pd.Series:
    """Parabolic SAR (classic Wilder, iterative).

    ``sar[t] = sar[t-1] + af * (ep - sar[t-1])``, clamped to not enter the
    prior two bars' range (uptrend: at most the two prior lows; downtrend:
    at least the two prior highs). When the bar pierces the SAR the trend
    flips: the SAR jumps to the prior extreme point, the extreme point
    resets to the current bar's low/high and ``af`` resets to ``af_start``.
    Otherwise a new extreme advances ``ep`` and ``af`` by ``af_step`` up to
    ``af_max``. The first bar is NaN; frames with fewer than 2 rows return
    all-NaN. The initial trend at bar 1 is up when ``close[1] >= close[0]``.

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``, ``close``).
        af_start: Initial acceleration factor.
        af_step: Acceleration factor increment per new extreme.
        af_max: Acceleration factor cap.

    Returns:
        Series with the SAR level in effect at each bar.
    """
    _require_columns(df, ("high", "low", "close"))
    n = len(df)
    out = np.full(n, np.nan)
    if n >= 2:
        high = df["high"].to_numpy(dtype="float64")
        low = df["low"].to_numpy(dtype="float64")
        close = df["close"].to_numpy(dtype="float64")
        uptrend = close[1] >= close[0]
        sar = low[0] if uptrend else high[0]
        ep = max(high[0], high[1]) if uptrend else min(low[0], low[1])
        af = af_start
        out[1] = sar
        for t in range(2, n):
            sar = sar + af * (ep - sar)
            if uptrend:
                sar = min(sar, low[t - 1], low[t - 2])
                if low[t] < sar:  # bar pierced the SAR -> reverse down
                    uptrend = False
                    sar = ep
                    ep = low[t]
                    af = af_start
                elif high[t] > ep:  # new extreme in the uptrend
                    ep = high[t]
                    af = min(af + af_step, af_max)
            else:
                sar = max(sar, high[t - 1], high[t - 2])
                if high[t] > sar:  # bar pierced the SAR -> reverse up
                    uptrend = True
                    sar = ep
                    ep = high[t]
                    af = af_start
                elif low[t] < ep:  # new extreme in the downtrend
                    ep = low[t]
                    af = min(af + af_step, af_max)
            out[t] = sar
    return pd.Series(out, index=df.index)


def aroon(df: pd.DataFrame, window: int = 25) -> tuple[pd.Series, pd.Series]:
    """Aroon Up/Down over the last ``window + 1`` bars including the current.

    ``aroon_up = 100 * (window - bars_since_highest_high) / window``
    (``aroon_down`` mirrored with the lowest low). Ties resolve to the most
    recent occurrence of the extreme, so a flat window yields 100/100. The
    first ``window`` rows are NaN.

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``).
        window: Aroon period (the lookback spans ``window + 1`` bars).

    Returns:
        Tuple ``(aroon_up, aroon_down)``, each in ``[0, 100]``.
    """
    _validate_window(window)
    _require_columns(df, ("high", "low"))
    n = len(df)
    up = np.full(n, np.nan)
    down = np.full(n, np.nan)
    if n >= window + 1:
        high = df["high"].to_numpy(dtype="float64")
        low = df["low"].to_numpy(dtype="float64")
        # Reverse each window so position 0 is the current bar: argmax/argmin
        # then return bars-since-extreme with the most recent occurrence
        # winning ties.
        high_windows = np.lib.stride_tricks.sliding_window_view(high, window + 1)
        low_windows = np.lib.stride_tricks.sliding_window_view(low, window + 1)
        since_high = np.argmax(high_windows[:, ::-1], axis=1)
        since_low = np.argmin(low_windows[:, ::-1], axis=1)
        up[window:] = 100.0 * (window - since_high) / window
        down[window:] = 100.0 * (window - since_low) / window
    return pd.Series(up, index=df.index), pd.Series(down, index=df.index)


def donchian(
    df: pd.DataFrame,
    window: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian channel including the current bar (no shift).

    ``upper = rolling_max(high)``, ``lower = rolling_min(low)``,
    ``mid = (upper + lower) / 2``. Consumers wanting prior-bar channels
    shift themselves.

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``).
        window: Rolling window length in bars.

    Returns:
        Tuple ``(upper, mid, lower)``; NaN during warm-up.
    """
    _validate_window(window)
    _require_columns(df, ("high", "low"))
    upper = df["high"].rolling(window=window, min_periods=window).max()
    lower = df["low"].rolling(window=window, min_periods=window).min()
    mid = (upper + lower) / 2.0
    return upper, mid, lower


def keltner(
    df: pd.DataFrame,
    window: int = 20,
    atr_window: int = 10,
    multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner channel: EMA midline with Wilder-ATR bands.

    ``mid = ema(close, window)``; ``upper/lower = mid ± multiplier *
    atr(df, atr_window)``.

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``, ``close``).
        window: EMA span of the midline.
        atr_window: Wilder ATR period for the band width.
        multiplier: Band width in ATR multiples.

    Returns:
        Tuple ``(upper, mid, lower)`` aligned to ``df.index``.
    """
    _validate_window(window)
    _validate_window(atr_window, "atr_window")
    _require_columns(df, ("high", "low", "close"))
    mid = ema(df["close"], window)
    band = multiplier * atr(df, atr_window)
    return mid + band, mid, mid - band


def hull_ma(s: pd.Series, window: int = 20) -> pd.Series:
    """Hull Moving Average.

    ``HMA = wma(2 * wma(s, window // 2) - wma(s, window), round(sqrt(window)))``
    where ``wma`` is the linearly-weighted MA (weights ``1..k``, most recent
    heaviest); the half window floors at 1 and the square-root window rounds
    (TradingView convention; window=20 gives half=10, sqrt=4).

    Args:
        s: Input price series.
        window: HMA period.

    Returns:
        HMA series aligned to ``s.index``; NaN during warm-up.
    """
    _validate_window(window)
    half = max(1, window // 2)
    sqrt_w = max(1, int(round(math.sqrt(window))))
    return _wma(2.0 * _wma(s, half) - _wma(s, window), sqrt_w)


def trix(s: pd.Series, window: int = 15) -> pd.Series:
    """TRIX: 1-bar percent rate of change of a triple EMA.

    ``t3 = ema(ema(ema(s, window), window), window)``;
    ``trix = 100 * (t3 / t3.shift(1) - 1)``.

    Args:
        s: Input price series.
        window: EMA span applied three times.

    Returns:
        TRIX series in percent; the first value is NaN.
    """
    _validate_window(window)
    t3 = ema(ema(ema(s, window), window), window)
    return 100.0 * (t3 / t3.shift(1) - 1.0)


def ichimoku(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Ichimoku lines with the cloud-in-effect convention (no chikou).

    Classic fixed constants (9/26/52, displacement 26):
    ``tenkan = (rolling_max(high, 9) + rolling_min(low, 9)) / 2``;
    ``kijun`` is the same with 26; ``senkou_a = ((tenkan + kijun) / 2)
    .shift(26)`` and ``senkou_b = ((rolling_max(high, 52) +
    rolling_min(low, 52)) / 2).shift(26)`` — row ``t`` holds the cloud a
    chartist sees at time ``t``, computed exclusively from data at rows
    ``<= t - 26`` (strictly backward-looking). The chikou span is excluded
    from this platform entirely (it is the future by construction).

    Args:
        df: Canonical OHLCV frame (needs ``high``, ``low``).

    Returns:
        Tuple ``(tenkan, kijun, senkou_a, senkou_b)``; warm-up NaNs span the
        first 8 / 25 / 51 / 77 rows respectively.
    """
    _require_columns(df, ("high", "low"))
    high = df["high"]
    low = df["low"]
    tenkan = (
        high.rolling(window=9, min_periods=9).max()
        + low.rolling(window=9, min_periods=9).min()
    ) / 2.0
    kijun = (
        high.rolling(window=26, min_periods=26).max()
        + low.rolling(window=26, min_periods=26).min()
    ) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).shift(26)
    senkou_b = (
        (
            high.rolling(window=52, min_periods=52).max()
            + low.rolling(window=52, min_periods=52).min()
        )
        / 2.0
    ).shift(26)
    return tenkan, kijun, senkou_a, senkou_b


def bollinger_pct_b(
    close: pd.Series,
    upper: pd.Series,
    lower: pd.Series,
) -> pd.Series:
    """Bollinger %B: position of price within the bands.

    ``(close - lower) / (upper - lower)``; zero band width yields NaN. NOT
    clipped — values outside ``[0, 1]`` are meaningful band breaks.

    Args:
        close: Price series.
        upper: Upper Bollinger band.
        lower: Lower Bollinger band.

    Returns:
        %B series aligned to the inputs.
    """
    rng = upper - lower
    return (close - lower) / rng.mask(rng == 0)


def bollinger_bandwidth(
    upper: pd.Series,
    mid: pd.Series,
    lower: pd.Series,
) -> pd.Series:
    """Bollinger bandwidth: ``(upper - lower) / mid`` (zero mid yields NaN).

    Args:
        upper: Upper Bollinger band.
        mid: Middle band (SMA).
        lower: Lower Bollinger band.

    Returns:
        Bandwidth series aligned to the inputs.
    """
    return (upper - lower) / mid.mask(mid == 0)
