"""Abstract base class shared by all trading strategies.

PAPER TRADING ONLY — strategies emit research signals for backtesting and
simulated trading. Nothing in this module places real orders.

Every strategy consumes the canonical feature-enriched OHLCV DataFrame (see
CONTRACTS.md) and returns a *copy* of it with one added column:

    ``signal`` (int8): 1 = long, -1 = short, 0 = flat.

The signal at row ``t`` is the target stance for the *next* bar — the
backtester acts on it at bar ``t+1`` open. No lookahead: row ``t`` may only
use data at or before ``t``. Warm-up rows where indicators are NaN produce 0.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Base class for rule-based trading strategies.

    Attributes:
        name: Unique registry key for the strategy (class attribute).
        default_params: Class-level defaults merged with ``__init__`` kwargs.
        params: Effective parameters for this instance (defaults + overrides).
    """

    name: str = "base"
    default_params: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params: Any) -> None:
        """Initialize the strategy.

        Args:
            **params: Optional overrides for :attr:`default_params`. Unknown
                keys are stored but ignored by the built-in rules.
        """
        self.params: dict[str, Any] = {**self.default_params, **params}
        logger.debug("Initialized strategy %r with params=%s", self.name, self.params)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, params={self.params!r})"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals for a feature-enriched OHLCV frame.

        Args:
            df: Canonical feature-enriched OHLCV DataFrame (UTC tz-aware
                ``DatetimeIndex`` named ``"timestamp"``, ascending, unique).

        Returns:
            A COPY of ``df`` with an added ``'signal'`` column
            (``int8``: 1 = long, -1 = short, 0 = flat). The signal is the
            TARGET STANCE for the NEXT bar (the backtester enters at the next
            bar's open). NO LOOKAHEAD: row ``t`` uses data ≤ ``t``. First
            rows may be 0 while indicators warm up (``fillna(0)``).
        """

    # ------------------------------------------------------------------ #
    # Shared helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
        """Validate that ``df`` contains every column in ``columns``.

        Args:
            df: Input DataFrame.
            columns: Column names the strategy depends on.

        Raises:
            ValueError: If any required column is missing (with a hint to run
                ``backend.indicators.features.add_features`` first).
        """
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns {missing}; "
                "pass a feature-enriched frame (see "
                "backend.indicators.features.add_features)."
            )

    def _attach_signal(
        self, df: pd.DataFrame, signal: pd.Series | np.ndarray
    ) -> pd.DataFrame:
        """Return a copy of ``df`` with a sanitized int8 ``'signal'`` column.

        NaNs become 0 and values are clipped to {-1, 0, 1}.

        Args:
            df: Input DataFrame (never mutated).
            signal: Per-row signal values aligned with ``df``.

        Returns:
            Copy of ``df`` with the added ``'signal'`` column.
        """
        out = df.copy()
        series = pd.Series(np.asarray(signal, dtype="float64"), index=out.index)
        out["signal"] = series.fillna(0.0).clip(-1, 1).astype("int8")
        logger.debug(
            "%s: generated %d signals (long=%d, short=%d)",
            self.name,
            len(out),
            int((out["signal"] == 1).sum()),
            int((out["signal"] == -1).sum()),
        )
        return out
