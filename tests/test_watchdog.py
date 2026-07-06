"""Improvement Pack v3 acceptance tests for the Ollama watchdog (paper only).

Covers the CONTRACTS.md "## Improvement Pack v3" section 8 behaviours in
``backend/api/main.py``:

* State transitions — ``is_available()`` sequence T,T,F,F,F flips the
  persisted ``account_state['ollama_status']`` to unavailable EXACTLY at the
  third consecutive failure, writing ONE ``bot_activity`` ``error`` row and
  attempting ONE restart; a later T resets both the failure counter and the
  persisted status (silently — no activity row on recovery).
* Restart guard — at most one launch attempt per cooldown window; later
  failing passes log NO additional error rows.
* Missing-file guard — a nonexistent app path skips the launch (no ``Popen``)
  without crashing and without counting a restart.
* ``GET /health`` — gains ``ollama_since_ms`` from the persisted status
  (``None`` when the watchdog never wrote it).

Everything runs offline: NO network (conftest blocks sockets), NO Ollama
(``OllamaClient.is_available`` is stubbed at the class level), NO real
process launches (``subprocess.Popen`` is stubbed and the app path points
into ``tmp_path``). PAPER TRADING ONLY — the watchdog only keeps the local
AI alive.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import backend.api.main as main_mod
from backend.ai.ollama_client import OllamaClient
from backend.database.db import get_conn
from config.settings import settings

#: account_state key the watchdog persists (cross-module contract).
STATUS_KEY = "ollama_status"

#: Required keys of the watchdog's ``error`` activity row (CONTRACTS.md §8/§9).
ERROR_DETAIL_KEYS = {
    "where",
    "reason",
    "consecutive_failures",
    "restart_attempted",
    "app_path",
    "explanation",
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _read_status() -> dict[str, Any] | None:
    """Read + JSON-decode ``account_state['ollama_status']`` (None when absent)."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT value FROM account_state WHERE key=?", (STATUS_KEY,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    try:
        value = json.loads(row["value"])
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _write_status(value: dict[str, Any]) -> None:
    """Persist a status dict under the contract key."""
    conn = get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO account_state (key, value) VALUES (?, ?)",
            (STATUS_KEY, json.dumps(value)),
        )
        conn.commit()
    finally:
        conn.close()


def _error_rows() -> list[dict[str, Any]]:
    """Watchdog ``error`` activity rows (oldest first), decoding detail JSON."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT ts, symbol, action, detail FROM bot_activity "
            "WHERE action='error' ORDER BY id ASC"
        ).fetchall()
    finally:
        conn.close()
    items: list[dict[str, Any]] = []
    for row in rows:
        try:
            detail = json.loads(row["detail"] or "{}")
        except (TypeError, ValueError):
            detail = {}
        if not isinstance(detail, dict):
            detail = {}
        if detail.get("where") != "ollama_watchdog":
            continue
        items.append({"symbol": row["symbol"], "detail": detail})
    return items


def _activity_count() -> int:
    """Total number of ``bot_activity`` rows (any action)."""
    conn = get_conn()
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM bot_activity").fetchone()
        return int(row["n"])
    finally:
        conn.close()


class _AvailabilityStub:
    """Scripted ``OllamaClient.is_available`` replacement.

    Installed as a class attribute; a non-function callable is not a
    descriptor, so the client instance is NOT prepended to the call.
    """

    def __init__(self, results: list[bool | Exception]) -> None:
        self.results = list(results)
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        if not self.results:
            raise AssertionError("unexpected extra Ollama liveness check")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return bool(result)


def _patch_availability(
    monkeypatch: pytest.MonkeyPatch, results: list[bool | Exception]
) -> _AvailabilityStub:
    """Script the next ``is_available()`` results at the class level."""
    stub = _AvailabilityStub(results)
    monkeypatch.setattr(OllamaClient, "is_available", stub)
    return stub


class _PopenStub:
    """Recording ``subprocess.Popen`` replacement — never launches anything."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, args: Any, **kwargs: Any) -> object:
        self.calls.append((args, dict(kwargs)))
        return object()


@pytest.fixture(autouse=True)
def _popen_guard(monkeypatch: pytest.MonkeyPatch) -> _PopenStub:
    """Stub ``subprocess.Popen`` everywhere the watchdog could resolve it."""
    stub = _PopenStub()
    monkeypatch.setattr(subprocess, "Popen", stub)
    if hasattr(main_mod, "Popen"):  # covers a `from subprocess import Popen`
        monkeypatch.setattr(main_mod, "Popen", stub)
    return stub


@pytest.fixture(autouse=True)
def _fresh_watchdog_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Isolate the watchdog's module state and app path per test.

    The consecutive-failure counter and last-restart timestamp live in
    module globals (lock-guarded, per contract); reset them so one test's
    failures/cooldown can never leak into the next. ``raising=False`` keeps
    the fixture harmless if the private names ever change — the tests then
    still normalise the counter behaviourally via a leading ``True`` pass.
    The app path defaults to a nonexistent file so no test can ever launch a
    real process even if the Popen stub were bypassed.
    """
    monkeypatch.setattr(main_mod, "_ollama_fail_count", 0, raising=False)
    monkeypatch.setattr(main_mod, "_ollama_last_restart_ms", 0, raising=False)
    monkeypatch.setattr(
        settings, "ollama_app_path", str(tmp_path / "missing" / "ollama app.exe")
    )


def _install_app_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Create a fake Ollama app file and point the settings path at it."""
    app = tmp_path / "Ollama" / "ollama app.exe"
    app.parent.mkdir(parents=True, exist_ok=True)
    app.write_bytes(b"")
    monkeypatch.setattr(settings, "ollama_app_path", str(app))
    return str(app)


# --------------------------------------------------------------------------- #
# Binding constants                                                            #
# --------------------------------------------------------------------------- #


def test_watchdog_binding_constants() -> None:
    """The section-8 constants are binding EXACT values."""
    assert main_mod.OLLAMA_WATCHDOG_JOB_ID == "ollama_watchdog"
    assert main_mod.OLLAMA_STATUS_KEY == "ollama_status"
    assert main_mod.OLLAMA_FAIL_THRESHOLD == 3
    assert main_mod.OLLAMA_RESTART_COOLDOWN_MS == 600_000
    assert isinstance(settings.ollama_watchdog_enabled, bool)
    assert isinstance(settings.ollama_app_path, str) and settings.ollama_app_path


# --------------------------------------------------------------------------- #
# State transitions                                                            #
# --------------------------------------------------------------------------- #


def test_sequence_ttfff_one_error_row_and_one_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T,T,F,F,F → offline exactly at the 3rd F: one error row, one restart."""
    app_path = _install_app_file(tmp_path, monkeypatch)
    popen: _PopenStub = _PopenStub()
    monkeypatch.setattr(subprocess, "Popen", popen)
    stub = _patch_availability(monkeypatch, [True, True, False, False, False])

    # --- two healthy passes: online, nothing logged ------------------------
    main_mod._run_ollama_watchdog()
    main_mod._run_ollama_watchdog()
    status = _read_status()
    assert status is None or status.get("available") is True
    assert _error_rows() == []
    assert popen.calls == []

    # --- two failures: still below the threshold ---------------------------
    main_mod._run_ollama_watchdog()
    main_mod._run_ollama_watchdog()
    status = _read_status()
    assert status is None or status.get("available") is True  # not flipped yet
    assert _error_rows() == []
    assert popen.calls == []

    # --- third consecutive failure: THE transition -------------------------
    main_mod._run_ollama_watchdog()
    status = _read_status()
    assert status is not None
    assert status["available"] is False
    assert isinstance(status["since_ms"], int) and status["since_ms"] > 0
    assert int(status["restarts"]) == 1  # exactly one attempt so far

    errors = _error_rows()
    assert len(errors) == 1  # transition-only logging
    assert errors[0]["symbol"] == ""
    detail = errors[0]["detail"]
    assert ERROR_DETAIL_KEYS <= set(detail.keys())
    assert detail["where"] == "ollama_watchdog"
    assert detail["reason"] == "ai_offline"
    assert int(detail["consecutive_failures"]) == 3
    assert detail["restart_attempted"] is True
    assert detail["app_path"] == app_path
    assert isinstance(detail["explanation"], str) and detail["explanation"]

    # Contract launch shape: Popen([path], shell=False, stdout/stderr DEVNULL).
    assert len(popen.calls) == 1
    args, kwargs = popen.calls[0]
    assert list(args) == [app_path]
    assert kwargs.get("shell") is False
    assert kwargs.get("stdout") == subprocess.DEVNULL
    assert kwargs.get("stderr") == subprocess.DEVNULL
    assert stub.calls == 5


def test_later_failures_respect_cooldown_and_log_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After the transition, failing passes within the cooldown do nothing new."""
    _install_app_file(tmp_path, monkeypatch)
    popen: _PopenStub = _PopenStub()
    monkeypatch.setattr(subprocess, "Popen", popen)
    _patch_availability(monkeypatch, [False] * 5)

    for _ in range(5):
        main_mod._run_ollama_watchdog()

    # One transition, one restart: passes 4 and 5 are inside the 10-minute
    # cooldown (they ran milliseconds after the attempt) and log nothing.
    assert len(_error_rows()) == 1
    assert len(popen.calls) == 1
    status = _read_status()
    assert status is not None and status["available"] is False
    assert int(status["restarts"]) == 1


def test_later_failures_retry_restart_once_cooldown_elapses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every later failing pass retries the restart when the cooldown is over."""
    _install_app_file(tmp_path, monkeypatch)
    popen: _PopenStub = _PopenStub()
    monkeypatch.setattr(subprocess, "Popen", popen)
    # A negative cooldown makes "last attempt more than cooldown ago" always
    # true, so every failing pass at/after the threshold may attempt.
    monkeypatch.setattr(main_mod, "OLLAMA_RESTART_COOLDOWN_MS", -1)
    _patch_availability(monkeypatch, [False] * 5)

    for _ in range(5):
        main_mod._run_ollama_watchdog()

    assert len(_error_rows()) == 1  # STILL only the transition row
    assert len(popen.calls) == 3  # passes 3, 4 and 5 each attempted
    status = _read_status()
    assert status is not None and status["available"] is False
    assert int(status["restarts"]) == 3


def test_missing_app_file_skips_launch_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonexistent app path (Docker/POSIX) logs the row but never launches."""
    popen: _PopenStub = _PopenStub()
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(main_mod, "OLLAMA_RESTART_COOLDOWN_MS", -1)
    _patch_availability(monkeypatch, [False, False, False])

    for _ in range(3):
        main_mod._run_ollama_watchdog()  # must not raise

    assert popen.calls == []  # the launch was skipped
    status = _read_status()
    assert status is not None
    assert status["available"] is False
    assert int(status["restarts"]) == 0  # no attempt happened
    errors = _error_rows()
    assert len(errors) == 1  # the offline transition is still logged
    detail = errors[0]["detail"]
    assert detail["restart_attempted"] is False
    assert isinstance(detail["app_path"], str) and detail["app_path"]


def test_recovery_resets_counter_and_status_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F,F,F → offline; T resets everything; F,F stays quiet; a 3rd F re-trips."""
    _patch_availability(
        monkeypatch, [False, False, False, True, False, False, False]
    )

    for _ in range(3):
        main_mod._run_ollama_watchdog()
    status = _read_status()
    assert status is not None and status["available"] is False
    offline_since = int(status["since_ms"])
    assert len(_error_rows()) == 1
    rows_before_recovery = _activity_count()

    # --- recovery: status flips back, counter resets, NO activity row ------
    main_mod._run_ollama_watchdog()
    status = _read_status()
    assert status is not None
    assert status["available"] is True
    assert int(status["since_ms"]) >= offline_since  # a NEW state period
    assert _activity_count() == rows_before_recovery  # silent recovery

    # --- the counter really reset: two failures stay below the threshold ---
    main_mod._run_ollama_watchdog()
    main_mod._run_ollama_watchdog()
    status = _read_status()
    assert status is not None and status["available"] is True
    assert len(_error_rows()) == 1

    # --- and a third consecutive failure trips a fresh transition ----------
    main_mod._run_ollama_watchdog()
    status = _read_status()
    assert status is not None and status["available"] is False
    assert len(_error_rows()) == 2


def test_watchdog_never_raises_when_the_check_explodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exploding availability check counts as a failure, never as a crash."""
    _patch_availability(
        monkeypatch,
        [RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")],
    )

    for _ in range(3):
        main_mod._run_ollama_watchdog()  # must not raise

    status = _read_status()
    assert status is not None and status["available"] is False
    assert len(_error_rows()) == 1


# --------------------------------------------------------------------------- #
# /health                                                                      #
# --------------------------------------------------------------------------- #


def test_health_reports_ollama_since_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    """/health keeps the live check and adds ollama_since_ms from the status."""
    assert any(
        getattr(route, "path", "") == "/health" for route in main_mod.app.routes
    )
    monkeypatch.setattr(OllamaClient, "is_available", lambda self: False)

    # Watchdog never ran on this fresh account → null since_ms.
    response = main_mod.health()
    payload = response.model_dump()
    assert payload["status"] == "ok"
    assert payload["paper_only"] is True
    assert payload["db"] == "ok"
    assert payload["ollama_available"] is False
    assert payload["ollama_since_ms"] is None

    # Once the watchdog persisted a status, /health surfaces its since_ms.
    _write_status({"available": False, "since_ms": 1712345678901, "restarts": 4})
    payload = main_mod.health().model_dump()
    assert payload["ollama_since_ms"] == 1712345678901
