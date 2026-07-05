"""Shared pytest fixtures for the Trading AI Platform test suite.

PAPER TRADING ONLY — these tests never touch a network socket, never call
Ollama and never place orders anywhere. All market data is synthetic or
pre-seeded into a throwaway SQLite database under ``tmp_path``.

Key fixtures
------------
sample_df
    ~300 rows of a seeded random-walk 1h UTC canonical OHLCV frame with an
    uptrend segment followed by a downtrend segment.
_fresh_db (autouse, function-scoped)
    Points ``settings.db_path`` at ``tmp_path/test.db`` and initialises the
    schema so every test gets a brand-new database. ``db.py`` reads
    ``settings.db_path`` on every call, so runtime patching is sufficient.
_no_network (autouse)
    Hard-blocks outbound socket connections for the duration of each test.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

# settings must be imported (and later patched) before any lazy consumer runs.
from config.settings import settings
from backend.database.db import init_db


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Forbid any outbound network connection during tests."""

    def _blocked(*args: object, **kwargs: object) -> None:
        raise RuntimeError(
            "Network access is forbidden in the test suite (offline tests only)."
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    yield


@pytest.fixture(autouse=True)
def _contract_capital(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the paper-account capital to the contract default (100,000).

    A local ``.env`` may override ``INITIAL_CAPITAL`` to any value; the tests
    size explicit-qty orders (1-2 units at ~100) assuming the documented
    default, so buying-power checks would otherwise make results depend on
    the developer's environment.
    """
    monkeypatch.setattr(settings, "initial_capital", 100_000.0)
    yield


@pytest.fixture(autouse=True)
def _fresh_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Give every test its own empty SQLite database.

    Function-scoped on purpose: state from one test must never leak into the
    next. ``monkeypatch`` restores the original ``settings.db_path`` on
    teardown.
    """
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(settings, "db_path", db_file)
    init_db()
    yield db_file


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Synthetic canonical OHLCV frame: ~300 hourly UTC candles.

    A seeded (``default_rng(42)``) random walk with a positive-drift first
    half (uptrend) and a negative-drift second half (downtrend). All prices
    stay comfortably positive and all volumes are strictly positive.
    """
    n = 300
    rng = np.random.default_rng(42)

    drift = np.concatenate([np.full(n // 2, 0.4), np.full(n - n // 2, -0.3)])
    steps = drift + rng.normal(0.0, 0.8, n)
    close = 100.0 + np.cumsum(steps)

    open_ = np.empty(n, dtype=np.float64)
    open_[0] = 100.0
    open_[1:] = close[:-1]

    spread = np.abs(rng.normal(0.0, 0.5, n)) + 0.05
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(100.0, 1_000.0, n)

    # ms resolution matches the SQLite epoch-milliseconds canonical storage,
    # so DB roundtrips compare exactly.
    index = pd.date_range(
        "2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp"
    ).as_unit("ms")
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    ).astype("float64")

    assert (df["low"] > 0).all()
    assert (df["volume"] > 0).all()
    return df
