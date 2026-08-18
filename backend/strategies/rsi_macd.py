"""RSI + MACD strategy: MACD signal-line crosses confirmed by RSI momentum.

PAPER TRADING ONLY — research signals, no real-money execution.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from backend.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class RsiMacdStrategy(BaseStrategy):
    """MACD signal-line crossover confirmed by RSI(14): hold each stance until the opposite crossover fires.

    Rules (per CONTRACTS.md):
        * Go long (1): ``macd`` crosses ABOVE ``macd_signal`` AND
          ``rsi_14 > rsi_threshold`` (50).
        * Go short (-1): ``macd`` crosses BELOW ``macd_signal`` AND
          ``rsi_14 < rsi_threshold`` (50).
        * Otherwise hold the previous stance (forward-fill of the last cross
          event) until the opposite signal; 0 before the first event.

    Implemented vectorized: cross events are marked as +/-1 and forward-filled.
    """

    name: str = "rsi_macd"
    default_params: ClassVar[dict[str, Any]] = {
        "rsi_threshold": 50.0,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI+MACD crossover signals with hold-until-opposite.

        Args:
            df: Feature-enriched canonical OHLCV frame (needs ``macd``,
                ``macd_signal``, ``rsi_14``).

        Returns:
            Copy of ``df`` with the int8 ``'signal'`` column in {-1, 0, 1}.
        """
        self._require_columns(df, ("macd", "macd_signal", "rsi_14"))

        threshold = float(self.params["rsi_threshold"])

        macd = df["macd"]
        macd_sig = df["macd_signal"]
        rsi = df["rsi_14"]

        # Cross detection: strict inequality now, opposite-or-equal on the
        # prior bar. NaN comparisons are False, so warm-up rows never fire.
        cross_up = (macd > macd_sig) & (macd.shift(1) <= macd_sig.shift(1))
        cross_down = (macd < macd_sig) & (macd.shift(1) >= macd_sig.shift(1))

        long_event = cross_up & (rsi > threshold)
        short_event = cross_down & (rsi < threshold)

        # Mark events, then forward-fill so a stance persists until the
        # opposite event; rows before the first event stay 0.
        events = pd.Series(np.nan, index=df.index, dtype="float64")
        events[long_event] = 1.0
        events[short_event] = -1.0
        signal = events.ffill().fillna(0.0)

        return self._attach_signal(df, signal)
