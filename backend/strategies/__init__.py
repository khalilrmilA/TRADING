"""Strategy registry for the Trading AI Platform.

PAPER TRADING ONLY — strategies emit research signals for backtesting and
simulated trading; nothing here places real orders.

Public API (per CONTRACTS.md):
    * ``STRATEGIES``: registry mapping strategy name → strategy class.
    * ``get_strategy(name, **params)``: instantiate a strategy by name.
    * ``list_strategies()``: metadata for every registered strategy.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.strategies.base import BaseStrategy
from backend.strategies.breakout import BreakoutStrategy
from backend.strategies.mean_reversion import MeanReversionStrategy
from backend.strategies.rsi_macd import RsiMacdStrategy
from backend.strategies.trend_following import TrendFollowingStrategy
from backend.strategies.voting import VotingStrategy

logger = logging.getLogger(__name__)

STRATEGIES: dict[str, type[BaseStrategy]] = {
    TrendFollowingStrategy.name: TrendFollowingStrategy,
    MeanReversionStrategy.name: MeanReversionStrategy,
    BreakoutStrategy.name: BreakoutStrategy,
    RsiMacdStrategy.name: RsiMacdStrategy,
    VotingStrategy.name: VotingStrategy,
}

__all__ = [
    "STRATEGIES",
    "BaseStrategy",
    "BreakoutStrategy",
    "MeanReversionStrategy",
    "RsiMacdStrategy",
    "TrendFollowingStrategy",
    "VotingStrategy",
    "get_strategy",
    "list_strategies",
]


def get_strategy(name: str, **params: Any) -> BaseStrategy:
    """Instantiate a registered strategy by name.

    Args:
        name: Registry key, e.g. ``"trend_following"`` or ``"ensemble"``.
        **params: Parameter overrides forwarded to the strategy constructor.

    Returns:
        A configured :class:`BaseStrategy` instance.

    Raises:
        KeyError: If ``name`` is not a registered strategy (the message lists
            all valid names).
    """
    try:
        cls = STRATEGIES[name]
    except KeyError:
        valid = ", ".join(sorted(STRATEGIES))
        raise KeyError(
            f"Unknown strategy {name!r}. Valid strategies: {valid}"
        ) from None
    logger.debug("Instantiating strategy %r with params=%s", name, params)
    return cls(**params)


def _description(cls: type[BaseStrategy]) -> str:
    """Return the first line of a strategy class docstring (or '')."""
    doc = (cls.__doc__ or "").strip()
    return doc.splitlines()[0].strip() if doc else ""


def list_strategies() -> list[dict[str, Any]]:
    """List metadata for every registered strategy.

    Returns:
        One dict per strategy with keys ``name`` (registry key),
        ``description`` (first docstring line) and ``default_params``
        (copy of the class-level defaults).
    """
    return [
        {
            "name": name,
            "description": _description(cls),
            "default_params": dict(cls.default_params),
        }
        for name, cls in STRATEGIES.items()
    ]
