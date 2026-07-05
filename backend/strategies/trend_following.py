"""Trend-following strategy: EMA(12/26) alignment filtered by the SMA(50) trend.

PAPER TRADING ONLY — research signals, no real-money execution.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from backend.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """EMA(12/26) trend alignment with SMA(50) filter: long when ema_12 > ema_26 and close > sma_50; short on the inverse.

    Rules (per CONTRACTS.md):
        * Long (1): ``ema_12 > ema_26`` AND ``close > sma_50``.
        * Short (-1): ``ema_12 < ema_26`` AND ``close < sma_50``.
        * Otherwise flat (0). NaN warm-up rows are 0.
    """

    name: str = "trend_following"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend-following signals.

        Args:
            df: Feature-enriched canonical OHLCV frame (needs ``ema_12``,
                ``ema_26``, ``sma_50``, ``close``).

        Returns:
            Copy of ``df`` with the int8 ``'signal'`` column in {-1, 0, 1}.
        """
        self._require_columns(df, ("close", "ema_12", "ema_26", "sma_50"))

        long_cond = (df["ema_12"] > df["ema_26"]) & (df["close"] > df["sma_50"])
        short_cond = (df["ema_12"] < df["ema_26"]) & (df["close"] < df["sma_50"])
        # NaN comparisons are False, so warm-up rows fall through to 0.
        signal = np.where(long_cond, 1.0, np.where(short_cond, -1.0, 0.0))
        return self._attach_signal(df, signal)
