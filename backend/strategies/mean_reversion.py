"""Mean-reversion strategy: Bollinger-band extremes confirmed by RSI, exit at the mid band.

PAPER TRADING ONLY — research signals, no real-money execution.
"""

from __future__ import annotations

import logging
import math
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from backend.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Bollinger-band mean reversion with RSI confirmation: fade band extremes, exit when close crosses the mid band.

    Rules (per CONTRACTS.md):
        * Enter long (1): ``close < bb_lower`` AND ``rsi_14 < rsi_oversold`` (30).
        * Enter short (-1): ``close > bb_upper`` AND ``rsi_14 > rsi_overbought`` (70).
        * Exit to 0: a long is held until ``close`` crosses above ``bb_mid``;
          a short is held until ``close`` crosses below ``bb_mid``.
        * Warm-up rows (NaN indicators) are 0.

    The exit is stateful, so signals are produced with a simple forward loop
    over the rows (row ``t`` only uses data at ``t`` — no lookahead).
    """

    name: str = "mean_reversion"
    default_params: ClassVar[dict[str, Any]] = {
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate stateful mean-reversion signals.

        Args:
            df: Feature-enriched canonical OHLCV frame (needs ``close``,
                ``bb_upper``, ``bb_mid``, ``bb_lower``, ``rsi_14``).

        Returns:
            Copy of ``df`` with the int8 ``'signal'`` column in {-1, 0, 1}.
        """
        self._require_columns(df, ("close", "bb_upper", "bb_mid", "bb_lower", "rsi_14"))

        oversold = float(self.params["rsi_oversold"])
        overbought = float(self.params["rsi_overbought"])

        close = df["close"].to_numpy(dtype="float64")
        upper = df["bb_upper"].to_numpy(dtype="float64")
        mid = df["bb_mid"].to_numpy(dtype="float64")
        lower = df["bb_lower"].to_numpy(dtype="float64")
        rsi = df["rsi_14"].to_numpy(dtype="float64")

        n = len(df)
        signal = np.zeros(n, dtype="int8")
        state = 0  # current stance carried forward bar to bar
        for i in range(n):
            c, up, md, lo, r = close[i], upper[i], mid[i], lower[i], rsi[i]
            if math.isnan(c) or math.isnan(up) or math.isnan(md) or math.isnan(lo) or math.isnan(r):
                state = 0  # indicators not ready — stay/return flat
                continue
            # Exit first: close crossing the mid band flattens the position.
            if state == 1 and c > md:
                state = 0
            elif state == -1 and c < md:
                state = 0
            # Entries evaluated only when flat (including a just-exited bar).
            if state == 0:
                if c < lo and r < oversold:
                    state = 1
                elif c > up and r > overbought:
                    state = -1
            signal[i] = state

        return self._attach_signal(df, signal)
