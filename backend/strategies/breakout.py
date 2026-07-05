"""Breakout strategy: Donchian-channel breakouts with a volume filter and channel exits.

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


class BreakoutStrategy(BaseStrategy):
    """Volume-confirmed 20-bar channel breakout with a 10-bar opposite-channel exit.

    Rules (per CONTRACTS.md):
        * Enter long (1): ``close`` breaks above the rolling 20-bar high of
          PRIOR bars (``high.rolling(20).max().shift(1)``) with
          ``volume > 1.5 ×`` the prior 20-bar average volume.
        * Enter short (-1): symmetric breakdown below the prior 20-bar low
          with the same volume confirmation.
        * Exit to 0: a long is held until ``close`` falls below the prior
          10-bar low; a short until ``close`` rises above the prior 10-bar
          high (opposite band of the 10-bar channel).
        * Warm-up rows (NaN channels) are 0.

    The hold-until-exit logic is stateful, so signals are produced with a
    simple forward loop (row ``t`` only uses data ≤ ``t`` — no lookahead).
    """

    name: str = "breakout"
    default_params: ClassVar[dict[str, Any]] = {
        "entry_window": 20,
        "exit_window": 10,
        "volume_mult": 1.5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate stateful breakout signals.

        Args:
            df: Canonical OHLCV frame (needs ``high``, ``low``, ``close``,
                ``volume``; the channels are computed internally).

        Returns:
            Copy of ``df`` with the int8 ``'signal'`` column in {-1, 0, 1}.
        """
        self._require_columns(df, ("high", "low", "close", "volume"))

        entry_window = int(self.params["entry_window"])
        exit_window = int(self.params["exit_window"])
        volume_mult = float(self.params["volume_mult"])

        # Channels use PRIOR bars only (shift(1)) — no lookahead and the
        # breakout bar itself does not contaminate its own trigger level.
        entry_high = (
            df["high"].rolling(entry_window, min_periods=entry_window).max().shift(1)
        )
        entry_low = (
            df["low"].rolling(entry_window, min_periods=entry_window).min().shift(1)
        )
        exit_high = (
            df["high"].rolling(exit_window, min_periods=exit_window).max().shift(1)
        )
        exit_low = (
            df["low"].rolling(exit_window, min_periods=exit_window).min().shift(1)
        )
        avg_volume = (
            df["volume"].rolling(entry_window, min_periods=entry_window).mean().shift(1)
        )

        close = df["close"].to_numpy(dtype="float64")
        volume = df["volume"].to_numpy(dtype="float64")
        e_hi = entry_high.to_numpy(dtype="float64")
        e_lo = entry_low.to_numpy(dtype="float64")
        x_hi = exit_high.to_numpy(dtype="float64")
        x_lo = exit_low.to_numpy(dtype="float64")
        a_vol = avg_volume.to_numpy(dtype="float64")

        n = len(df)
        signal = np.zeros(n, dtype="int8")
        state = 0  # current stance carried forward bar to bar
        for i in range(n):
            c = close[i]
            # Exit first: close breaching the opposite 10-bar band flattens.
            if state == 1 and not math.isnan(x_lo[i]) and c < x_lo[i]:
                state = 0
            elif state == -1 and not math.isnan(x_hi[i]) and c > x_hi[i]:
                state = 0
            # Entries evaluated only when flat (including a just-exited bar).
            if state == 0 and not (
                math.isnan(e_hi[i]) or math.isnan(e_lo[i]) or math.isnan(a_vol[i])
            ):
                volume_ok = volume[i] > volume_mult * a_vol[i]
                if volume_ok:
                    if c > e_hi[i]:
                        state = 1
                    elif c < e_lo[i]:
                        state = -1
            signal[i] = state

        return self._attach_signal(df, signal)
