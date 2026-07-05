"""Ensemble strategy: majority vote across the four core strategies.

PAPER TRADING ONLY — research signals, no real-money execution.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from backend.strategies.base import BaseStrategy
from backend.strategies.breakout import BreakoutStrategy
from backend.strategies.mean_reversion import MeanReversionStrategy
from backend.strategies.rsi_macd import RsiMacdStrategy
from backend.strategies.trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)


class VotingStrategy(BaseStrategy):
    """Majority-vote ensemble of trend_following, mean_reversion, breakout and rsi_macd.

    Rules (per CONTRACTS.md):
        * Each member strategy votes -1/0/1 per bar; votes are summed.
        * Sum >= ``threshold`` (2) → 1; sum <= ``-threshold`` → -1; else 0.

    Parameters:
        threshold: Minimum absolute vote sum required for a stance (default 2).
        <member name>: Optional dict of params forwarded to that member
            strategy, e.g. ``VotingStrategy(breakout={"volume_mult": 2.0})``.
    """

    name: str = "ensemble"
    default_params: ClassVar[dict[str, Any]] = {
        "threshold": 2,
    }

    _member_classes: ClassVar[tuple[type[BaseStrategy], ...]] = (
        TrendFollowingStrategy,
        MeanReversionStrategy,
        BreakoutStrategy,
        RsiMacdStrategy,
    )

    def __init__(self, **params: Any) -> None:
        """Instantiate the four member strategies.

        Args:
            **params: ``threshold`` and/or per-member param dicts keyed by
                member strategy name (see class docstring).
        """
        super().__init__(**params)
        self.strategies: list[BaseStrategy] = []
        for cls in self._member_classes:
            member_params = self.params.get(cls.name)
            if isinstance(member_params, dict):
                self.strategies.append(cls(**member_params))
            else:
                self.strategies.append(cls())

    def vote_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-strategy votes and the resulting ensemble signal.

        Args:
            df: Feature-enriched canonical OHLCV frame.

        Returns:
            DataFrame indexed like ``df`` with one int8 column per member
            strategy (``trend_following``, ``mean_reversion``, ``breakout``,
            ``rsi_macd``) plus the int8 ``'ensemble'`` column.
        """
        votes: dict[str, pd.Series] = {}
        for strategy in self.strategies:
            votes[strategy.name] = strategy.generate_signals(df)["signal"]

        breakdown = pd.DataFrame(votes, index=df.index)
        threshold = int(self.params["threshold"])
        total = breakdown.sum(axis=1)
        breakdown["ensemble"] = np.where(
            total >= threshold, 1, np.where(total <= -threshold, -1, 0)
        ).astype("int8")
        return breakdown

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate majority-vote ensemble signals.

        Args:
            df: Feature-enriched canonical OHLCV frame (must satisfy every
                member strategy's column requirements).

        Returns:
            Copy of ``df`` with the int8 ``'signal'`` column in {-1, 0, 1}.
        """
        breakdown = self.vote_breakdown(df)
        return self._attach_signal(df, breakdown["ensemble"])
