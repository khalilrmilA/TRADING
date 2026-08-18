"""FastAPI application for the Trading AI Platform.

PAPER TRADING ONLY — this API performs research, backtesting and simulated
paper trading against locally cached public market data. It never places real
orders and never uses private exchange API keys.

Run from the project root::

    uvicorn backend.api.main:app --host 127.0.0.1 --port 8000

Design notes:
    * Heavy modules (data service, AI analyst, strategies, backtester, paper
      engine) are imported lazily inside route functions so importing this
      module stays fast and never touches the network or the database.
    * The :class:`PaperTradingEngine` is a lazily created module-level
      singleton (see :func:`get_engine`).
    * Domain exceptions map to HTTP statuses via a global handler:
      ``OllamaError``/``DataFetchError`` → 502, ``ValueError``/``KeyError``
      → 400, everything else → 500, always as ``{"detail": str}``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import subprocess
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.schemas import (
    AnalysisRequest,
    BacktestRequest,
    BotConfigUpdate,
    ConfiguredModels,
    CoverageInfo,
    FetchDataRequest,
    FetchDataResponse,
    HealthResponse,
    ModelsResponse,
    OHLCVResponse,
    PaperOrderRequest,
    ScalperParamsUpdate,
    StatusResponse,
    TickRequest,
    WalkForwardRequest,
)
from config.settings import settings

if TYPE_CHECKING:  # imported for type checkers only — never at runtime
    import pandas as pd
    from apscheduler.schedulers.background import BackgroundScheduler

    from backend.paper_trading.auto_trader import AutoTrader
    from backend.paper_trading.engine import PaperTradingEngine
    from backend.paper_trading.scalper import Scalper

logger = logging.getLogger(__name__)

MIN_BACKTEST_ROWS = 100

# ---------------------------------------------------------------------------
# Paper-trading engine singleton (lazy — import never touches the DB)
# ---------------------------------------------------------------------------

_engine: PaperTradingEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> PaperTradingEngine:
    """Return the module-level :class:`PaperTradingEngine`, creating it on first use.

    Returns:
        The shared paper-trading engine instance (simulation only).
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from backend.paper_trading.engine import PaperTradingEngine

                _engine = PaperTradingEngine()
                logger.info("PaperTradingEngine initialised (paper trading only)")
    return _engine


# ---------------------------------------------------------------------------
# AI auto-trader singleton + its APScheduler job (paper trading only)
# ---------------------------------------------------------------------------

_auto_trader: AutoTrader | None = None
_auto_trader_lock = threading.Lock()
_bot_scheduler: BackgroundScheduler | None = None
_bot_scheduler_lock = threading.Lock()

BOT_JOB_ID = "bot_cycle"


def get_auto_trader() -> AutoTrader:
    """Return the module-level :class:`AutoTrader`, creating it on first use.

    Returns:
        The shared auto-trader bound to the shared paper-trading engine.
    """
    global _auto_trader
    if _auto_trader is None:
        with _auto_trader_lock:
            if _auto_trader is None:
                from backend.paper_trading.auto_trader import AutoTrader

                _auto_trader = AutoTrader(get_engine())
                logger.info("AutoTrader initialised (paper trading only)")
    return _auto_trader


def _run_bot_cycle() -> None:
    """Scheduler target: run one auto-trader cycle (``run_cycle`` never raises).

    Gated on the PERSISTED running flag: if the config says the bot is stopped
    (e.g. ``POST /api/paper/reset`` wiped ``bot_config`` back to defaults
    while this job was still registered) the stale job removes itself instead
    of trading an account whose status reports ``running=False``.
    """
    trader = get_auto_trader()
    try:
        running = trader.get_config().running
    except Exception:
        logger.exception("Could not read the bot config — skipping this cycle")
        return
    if not running:
        logger.warning(
            "Bot cycle job fired but the persisted config says running=False — "
            "removing the stale job (no trading)"
        )
        _remove_bot_job()
        return
    trader.run_cycle()


def _bot_job_next_run(interval_minutes: int) -> datetime:
    """First fire time for the bot job, resuming the persisted cadence.

    ``add_job`` with an interval trigger schedules the first run at
    ``now + interval`` — so every restart (and every job re-registration)
    would silently push the next cycle back with no catch-up run. Resume
    from the persisted last cycle instead: ``max(now, last_cycle + interval)``
    — an overdue cycle fires immediately.

    Args:
        interval_minutes: Minutes between scheduled cycles.

    Returns:
        The UTC time the job should first fire.
    """
    now = datetime.now(timezone.utc)
    try:
        last_iso = get_auto_trader().status().get("last_cycle_ts")
        if last_iso:
            due = datetime.fromisoformat(last_iso) + timedelta(minutes=int(interval_minutes))
            return max(now, due)
    except Exception:
        logger.exception("Could not derive the bot job's first run from the last cycle")
    return now + timedelta(minutes=int(interval_minutes))


def _bot_job_exists() -> bool:
    """True when the ``bot_cycle`` job is currently registered."""
    with _bot_scheduler_lock:
        if _bot_scheduler is None:
            return False
        try:
            return _bot_scheduler.get_job(BOT_JOB_ID) is not None
        except Exception:
            return False


def _scheduler_locked() -> BackgroundScheduler:
    """Return the shared background scheduler, starting it on first use.

    Shared by the auto-trader (``bot_cycle``) and the scalper
    (``scalper_tick`` / ``scalper_tune``) jobs. The caller MUST already hold
    ``_bot_scheduler_lock``.

    Returns:
        The running APScheduler ``BackgroundScheduler`` instance.
    """
    global _bot_scheduler
    if _bot_scheduler is None:
        from apscheduler.schedulers.background import BackgroundScheduler

        _bot_scheduler = BackgroundScheduler()
        _bot_scheduler.start()
        logger.info("Background job scheduler started")
    return _bot_scheduler


def _ensure_bot_job(interval_minutes: int) -> None:
    """Create or replace the ``bot_cycle`` interval job on the bot scheduler.

    Args:
        interval_minutes: Minutes between scheduled cycles.
    """
    next_run_time = _bot_job_next_run(int(interval_minutes))
    with _bot_scheduler_lock:
        _scheduler_locked().add_job(
            _run_bot_cycle,
            trigger="interval",
            minutes=int(interval_minutes),
            id=BOT_JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=300,
            next_run_time=next_run_time,
        )
        logger.info(
            "Bot job '%s' scheduled every %d minutes (first run %s)",
            BOT_JOB_ID,
            interval_minutes,
            next_run_time.isoformat(),
        )


def _remove_bot_job() -> None:
    """Remove the ``bot_cycle`` job if present (idempotent, never raises)."""
    with _bot_scheduler_lock:
        if _bot_scheduler is None:
            return
        try:
            _bot_scheduler.remove_job(BOT_JOB_ID)
            logger.info("Bot job '%s' removed", BOT_JOB_ID)
        except Exception:
            logger.debug("Bot job '%s' was not registered", BOT_JOB_ID)


def _shutdown_bot_scheduler() -> None:
    """Stop the bot scheduler on application shutdown (idempotent)."""
    global _bot_scheduler
    with _bot_scheduler_lock:
        if _bot_scheduler is None:
            return
        try:
            _bot_scheduler.shutdown(wait=False)
            logger.info("Bot scheduler stopped")
        except Exception:
            logger.exception("Error while stopping the bot scheduler")
        finally:
            _bot_scheduler = None


def _bot_status() -> dict[str, Any]:
    """Auto-trader status, with ``next_cycle_ts`` from the live scheduler job."""
    status = get_auto_trader().status()
    with _bot_scheduler_lock:
        job = None
        if _bot_scheduler is not None:
            try:
                job = _bot_scheduler.get_job(BOT_JOB_ID)
            except Exception:
                job = None
    if job is not None and getattr(job, "next_run_time", None) is not None:
        status["next_cycle_ts"] = job.next_run_time.astimezone(timezone.utc).isoformat()
    return status


# ---------------------------------------------------------------------------
# Fast scalper singleton + its APScheduler jobs (paper trading only)
# ---------------------------------------------------------------------------

_scalper: Scalper | None = None
_scalper_singleton_lock = threading.Lock()

SCALPER_TICK_JOB_ID = "scalper_tick"
SCALPER_TUNE_JOB_ID = "scalper_tune"
SCALPER_TUNE_INTERVAL_MINUTES = 60


def get_scalper() -> Scalper:
    """Return the module-level :class:`Scalper`, creating it on first use.

    Returns:
        The shared scalper bound to the shared paper-trading engine.
    """
    global _scalper
    if _scalper is None:
        with _scalper_singleton_lock:
            if _scalper is None:
                from backend.paper_trading.scalper import Scalper

                _scalper = Scalper(get_engine())
                logger.info("Scalper initialised (paper trading only)")
    return _scalper


def _run_scalper_tick() -> None:
    """Scheduler target: one fast scalper pass (``run_tick`` never raises).

    ``run_tick`` itself no-ops when the persisted params say disabled; a
    stale job (e.g. left behind by ``POST /api/paper/reset`` wiping
    ``scalper_params``) additionally removes itself.
    """
    scalper = get_scalper()
    try:
        enabled = scalper.get_params().enabled
    except Exception:
        logger.exception("Could not read the scalper params — skipping this tick")
        return
    if not enabled:
        logger.warning(
            "Scalper tick job fired but the persisted params say enabled=False — "
            "removing the stale jobs"
        )
        _remove_scalper_jobs()
        return
    scalper.run_tick()


def _run_scalper_tune() -> None:
    """Scheduler target: hourly AI supervisor pass (failures only logged).

    When the tuner changes ``interval_minutes`` the tick job is re-registered
    so the new cadence takes effect without a restart.
    """
    try:
        scalper = get_scalper()
        result = scalper.tune_with_ai()
        if "interval_minutes" in (result.get("applied") or {}):
            params = scalper.get_params()
            if params.enabled:
                _ensure_scalper_jobs(params.interval_minutes)
    except Exception:
        logger.exception("Scheduled scalper tune failed")


def _ensure_scalper_jobs(interval_minutes: int) -> None:
    """Create or replace the ``scalper_tick`` + hourly ``scalper_tune`` jobs.

    Args:
        interval_minutes: Minutes between scalper ticks (already clamped to
            the scalper's HARD_BOUNDS by ``Scalper.set_params``).
    """
    with _bot_scheduler_lock:
        scheduler = _scheduler_locked()
        scheduler.add_job(
            _run_scalper_tick,
            trigger="interval",
            minutes=int(interval_minutes),
            id=SCALPER_TICK_JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=60,
        )
        scheduler.add_job(
            _run_scalper_tune,
            trigger="interval",
            minutes=SCALPER_TUNE_INTERVAL_MINUTES,
            id=SCALPER_TUNE_JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=600,
        )
        logger.info(
            "Scalper jobs scheduled: '%s' every %d min, '%s' every %d min",
            SCALPER_TICK_JOB_ID,
            interval_minutes,
            SCALPER_TUNE_JOB_ID,
            SCALPER_TUNE_INTERVAL_MINUTES,
        )


def _remove_scalper_jobs() -> None:
    """Remove both scalper jobs if present (idempotent, never raises)."""
    with _bot_scheduler_lock:
        if _bot_scheduler is None:
            return
        for job_id in (SCALPER_TICK_JOB_ID, SCALPER_TUNE_JOB_ID):
            try:
                _bot_scheduler.remove_job(job_id)
                logger.info("Scalper job '%s' removed", job_id)
            except Exception:
                logger.debug("Scalper job '%s' was not registered", job_id)


def _scalper_job_exists() -> bool:
    """True when the ``scalper_tick`` job is currently registered."""
    with _bot_scheduler_lock:
        if _bot_scheduler is None:
            return False
        try:
            return _bot_scheduler.get_job(SCALPER_TICK_JOB_ID) is not None
        except Exception:
            return False


def _scalper_status() -> dict[str, Any]:
    """Scalper status payload: params, stats, live job flag, last AI tune."""
    scalper = get_scalper()
    return {
        "params": scalper.get_params().model_dump(),
        "stats": scalper.stats(),
        "running_job": _scalper_job_exists(),
        "last_tune": scalper.get_last_tune(),
    }


# ---------------------------------------------------------------------------
# Intel (news/sentiment) + Coach singletons and their APScheduler jobs
# (paper trading only). The intel module is late-imported everywhere so the
# platform keeps working when backend/intel is absent or broken.
# ---------------------------------------------------------------------------

_coach: Any = None
_coach_singleton_lock = threading.Lock()

INTEL_JOB_ID = "intel_refresh"
COACH_JOB_ID = "coach_run"
INTEL_DEFAULT_INTERVAL_MINUTES = 30
COACH_RUN_INTERVAL_MINUTES = 120
_INTEL_CONFIG_KEY = "intel_config"


def get_coach() -> Any:
    """Return the module-level ``Coach``, creating it on first use.

    Returns:
        The shared coach agent (paper trading only).
    """
    global _coach
    if _coach is None:
        with _coach_singleton_lock:
            if _coach is None:
                from backend.paper_trading.coach import Coach

                _coach = Coach()
                logger.info("Coach initialised (paper trading only)")
    return _coach


def _intel_config() -> dict[str, Any]:
    """The persisted intel config (``account_state['intel_config']``).

    Prefers the intel module's own ``get_intel_config`` helper; when the
    module is absent or broken it falls back to a direct ``account_state``
    read of the same contract-fixed key, so this always works. Shape:
    ``{"enabled": bool, "interval_minutes": int}`` with defaults ``False`` /
    ``INTEL_DEFAULT_INTERVAL_MINUTES``; the interval used for the scheduler
    job is clamped into [1, 1440] minutes. Never raises.
    """
    enabled = False
    interval = INTEL_DEFAULT_INTERVAL_MINUTES
    raw: Any = None
    try:
        from backend.intel.sentiment import get_intel_config

        raw = get_intel_config()
    except Exception:
        logger.debug("intel module unavailable — reading intel_config directly")
        try:
            from backend.database.db import get_conn
            from backend.paper_trading.engine import PaperTradingEngine

            conn = get_conn()
            try:
                raw = PaperTradingEngine._state_get(conn, _INTEL_CONFIG_KEY, None)
            finally:
                conn.close()
        except Exception:
            logger.exception("Could not read the intel config — using defaults")
    if isinstance(raw, dict):
        enabled = bool(raw.get("enabled"))
        try:
            interval = int(raw.get("interval_minutes", interval))
        except (TypeError, ValueError):
            interval = INTEL_DEFAULT_INTERVAL_MINUTES
    interval = min(max(interval, 1), 1440)
    return {"enabled": enabled, "interval_minutes": interval}


def _set_intel_enabled(enabled: bool) -> dict[str, Any]:
    """Persist the intel enabled flag (interval preserved) and return the config.

    Prefers the intel module's own ``set_intel_config`` helper, falling back
    to a direct ``account_state`` write of the same key when the module is
    absent or broken — enabling/disabling must work either way.
    """
    try:
        from backend.intel.sentiment import set_intel_config

        set_intel_config({"enabled": bool(enabled)})
        return _intel_config()
    except Exception:
        logger.debug("intel module unavailable — writing intel_config directly")
    config = _intel_config()
    config["enabled"] = bool(enabled)
    from backend.database.db import get_conn
    from backend.paper_trading.engine import PaperTradingEngine

    conn = get_conn()
    try:
        with conn:
            PaperTradingEngine._state_set(conn, _INTEL_CONFIG_KEY, config)
    finally:
        conn.close()
    logger.info("Intel config updated: %s", config)
    return config


def _run_intel_refresh() -> None:
    """Scheduler target: one sentiment refresh over the bot watchlist.

    Gated on the PERSISTED enabled flag (a reset wipes ``intel_config``, so a
    stale job removes itself instead of refreshing for a disabled layer).
    Failures are logged, never raised into the scheduler.
    """
    config = _intel_config()
    if not config["enabled"]:
        logger.warning(
            "Intel refresh job fired but the persisted config says "
            "enabled=False — removing the stale intel/coach jobs"
        )
        _remove_intel_jobs()
        return
    try:
        from backend.intel.sentiment import SentimentAgent

        symbols = get_auto_trader().get_config().watchlist
        SentimentAgent().run(symbols)
    except Exception:
        logger.exception("Scheduled intel refresh failed")
    try:
        # Derivatives positioning rides the same schedule; its failure must
        # never take the news-sentiment refresh down with it (and vice versa).
        from backend.intel.derivatives import DerivativesAgent

        symbols = get_auto_trader().get_config().watchlist
        DerivativesAgent().run(symbols)
    except Exception:
        logger.exception("Scheduled derivatives refresh failed")


def _run_coach() -> None:
    """Scheduler target: one coach review pass (``Coach.run`` never raises).

    Shares the intel enabled flag (contract) and the same stale-job
    self-removal behaviour as :func:`_run_intel_refresh`.
    """
    config = _intel_config()
    if not config["enabled"]:
        logger.warning(
            "Coach job fired but the persisted intel config says "
            "enabled=False — removing the stale intel/coach jobs"
        )
        _remove_intel_jobs()
        return
    try:
        get_coach().run()
    except Exception:
        logger.exception("Scheduled coach run failed")


def _ensure_intel_jobs(interval_minutes: int) -> None:
    """Create or replace the ``intel_refresh`` + ``coach_run`` interval jobs.

    Args:
        interval_minutes: Minutes between sentiment refreshes (the coach
            always runs every ``COACH_RUN_INTERVAL_MINUTES`` minutes).
    """
    with _bot_scheduler_lock:
        scheduler = _scheduler_locked()
        scheduler.add_job(
            _run_intel_refresh,
            trigger="interval",
            minutes=int(interval_minutes),
            id=INTEL_JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=300,
        )
        scheduler.add_job(
            _run_coach,
            trigger="interval",
            minutes=COACH_RUN_INTERVAL_MINUTES,
            id=COACH_JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=900,
        )
        logger.info(
            "Intel jobs scheduled: '%s' every %d min, '%s' every %d min",
            INTEL_JOB_ID,
            interval_minutes,
            COACH_JOB_ID,
            COACH_RUN_INTERVAL_MINUTES,
        )


def _remove_intel_jobs() -> None:
    """Remove both intel/coach jobs if present (idempotent, never raises)."""
    with _bot_scheduler_lock:
        if _bot_scheduler is None:
            return
        for job_id in (INTEL_JOB_ID, COACH_JOB_ID):
            try:
                _bot_scheduler.remove_job(job_id)
                logger.info("Intel job '%s' removed", job_id)
            except Exception:
                logger.debug("Intel job '%s' was not registered", job_id)


def _clear_coach_playbook() -> None:
    """Wipe every ``coin_playbook`` row (best-effort, never raises).

    Called by ``POST /api/paper/reset``: ``engine.reset()`` only clears the
    engine-owned tables, but the playbook's benches/side-biases/multipliers
    were learned from the wiped account's trades and the traders apply them
    unconditionally every cycle/tick — while the reset also removes the
    ``coach_run`` job that could otherwise ever amend them. The table may not
    exist yet (intel/coach layer never installed) — that is fine.
    """
    try:
        from backend.database.db import get_conn

        conn = get_conn()
        try:
            with conn:
                conn.execute("DELETE FROM coin_playbook")
        finally:
            conn.close()
        logger.info("Coach playbook cleared by the paper account reset")
    except Exception as exc:
        logger.debug("coin_playbook not cleared (%s) — table may not exist", exc)


def _intel_job_exists() -> bool:
    """True when the ``intel_refresh`` job is currently registered."""
    with _bot_scheduler_lock:
        if _bot_scheduler is None:
            return False
        try:
            return _bot_scheduler.get_job(INTEL_JOB_ID) is not None
        except Exception:
            return False


def _sentiment_row_to_dict(row: Any) -> dict[str, Any]:
    """Convert one ``coin_sentiment`` row to a JSON-safe dict."""
    try:
        headlines = json.loads(row["headlines"] or "[]")
    except (TypeError, json.JSONDecodeError):
        headlines = []
    return {
        "ts": _ms_to_iso(int(row["ts"])) if row["ts"] is not None else None,
        "symbol": row["symbol"],
        "sentiment": row["sentiment"],
        "score": int(row["score"] or 0),
        "confidence": int(row["confidence"] or 0),
        "summary": row["summary"] or "",
        "headlines": headlines if isinstance(headlines, list) else [],
        "model": row["model"] or "",
    }


def _sentiment_items(symbol: str | None, limit: int) -> list[dict[str, Any]]:
    """Read ``coin_sentiment`` defensively: latest per symbol, or one symbol's
    recent history. ``[]`` on any failure (e.g. the table does not exist yet).
    """
    from backend.database.db import get_conn

    columns = "ts, symbol, sentiment, score, confidence, summary, headlines, model"
    try:
        conn = get_conn()
        try:
            if symbol:
                rows = conn.execute(
                    f"SELECT {columns} FROM coin_sentiment WHERE symbol = ? "
                    "ORDER BY ts DESC, id DESC LIMIT ?",
                    (symbol.strip().upper(), int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {columns} FROM coin_sentiment WHERE id IN "
                    "(SELECT MAX(id) FROM coin_sentiment GROUP BY symbol) "
                    "ORDER BY symbol ASC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        logger.debug("coin_sentiment unavailable (%s) — empty items", exc)
        return []
    return [_sentiment_row_to_dict(row) for row in rows]


def _intel_status() -> dict[str, Any]:
    """Intel status payload: enabled flag, last refresh, fear & greed, global mood."""
    config = _intel_config()
    payload: dict[str, Any] = {
        "enabled": config["enabled"],
        "interval_minutes": config["interval_minutes"],
        "running_job": _intel_job_exists(),
        "last_refresh": None,
        "fear_greed": None,
        "global": None,
    }
    try:
        # Prefer the intel module's persisted last-refresh stamp.
        from backend.intel.sentiment import get_last_refresh

        payload["last_refresh"] = get_last_refresh()
    except Exception:
        logger.debug("intel module unavailable — deriving last_refresh from rows")
    if payload["last_refresh"] is None:
        try:
            from backend.database.db import get_conn

            conn = get_conn()
            try:
                last = conn.execute("SELECT MAX(ts) AS ts FROM coin_sentiment").fetchone()
                if last is not None and last["ts"] is not None:
                    payload["last_refresh"] = _ms_to_iso(int(last["ts"]))
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("coin_sentiment unavailable (%s) — no last_refresh", exc)
    global_rows = _sentiment_items("GLOBAL", 1)
    payload["global"] = global_rows[0] if global_rows else None
    try:
        # Late-bound intel import (contract) — keyless public endpoint,
        # in-module cached, returns None on failure.
        from backend.intel.news import fetch_fear_greed

        payload["fear_greed"] = fetch_fear_greed()
    except Exception:
        logger.debug("Fear & greed unavailable for the intel status")
    return payload


# ---------------------------------------------------------------------------
# Ollama watchdog — monitors the local LLM server, restarts the desktop app
# when it goes down, and persists the availability state that the traders
# and /health read. PAPER TRADING ONLY — this only keeps the local AI alive.
# ---------------------------------------------------------------------------

OLLAMA_WATCHDOG_JOB_ID = "ollama_watchdog"
OLLAMA_STATUS_KEY = "ollama_status"  # account_state key — the cross-module contract
OLLAMA_FAIL_THRESHOLD = 3  # consecutive failures before "offline"
OLLAMA_RESTART_COOLDOWN_MS = 600_000  # at most one restart attempt per 10 minutes

# Lock-guarded module state: consecutive-failure count + last-restart-attempt ms.
_watchdog_lock = threading.Lock()
_ollama_fail_count = 0
_ollama_last_restart_ms = 0


def _read_ollama_status() -> dict[str, Any] | None:
    """The watchdog's persisted status (``account_state['ollama_status']``).

    Returns:
        The persisted ``{"available", "since_ms", "restarts"}`` dict, or
        ``None`` when the watchdog has never written it or the key is
        unreadable/malformed. Per contract, a missing status means ONLINE —
        a watchdog that never ran must not block anything. Never raises.
    """
    try:
        from backend.database.db import get_conn
        from backend.paper_trading.engine import PaperTradingEngine

        conn = get_conn()
        try:
            raw = PaperTradingEngine._state_get(conn, OLLAMA_STATUS_KEY, None)
        finally:
            conn.close()
        return raw if isinstance(raw, dict) else None
    except Exception:
        logger.debug("Could not read %r — treating Ollama as online", OLLAMA_STATUS_KEY)
        return None


def _write_ollama_status(available: bool, since_ms: int, restarts: int) -> None:
    """Persist the Ollama status to ``account_state`` (best-effort, never raises).

    Args:
        available: Whether the local Ollama server currently answers.
        since_ms: Epoch ms (UTC) when the CURRENT state began.
        restarts: Cumulative restart attempts (monotonically increasing).
    """
    status = {"available": bool(available), "since_ms": int(since_ms), "restarts": int(restarts)}
    try:
        from backend.database.db import get_conn
        from backend.paper_trading.engine import PaperTradingEngine

        conn = get_conn()
        try:
            with conn:
                PaperTradingEngine._state_set(conn, OLLAMA_STATUS_KEY, status)
        finally:
            conn.close()
        logger.info("Ollama status persisted: %s", status)
    except Exception:
        logger.exception("Could not persist the Ollama status %s", status)


def _launch_ollama_app(path: str) -> bool:
    """Try to launch the Ollama desktop app at ``path`` (never raises).

    A missing file (e.g. Docker/POSIX host) logs a warning and skips the
    launch. Popen failures are logged and still counted as an attempt.

    Args:
        path: Filesystem path of the app, env-vars already expanded.

    Returns:
        True when a launch was attempted (the file exists), else False.
    """
    if not os.path.isfile(path):
        logger.warning(
            "Ollama app not found at %r — restart skipped (set OLLAMA_APP_PATH "
            "if it is installed somewhere else)",
            path,
        )
        return False
    try:
        subprocess.Popen(  # noqa: S603 — fixed local app path, no shell
            [path], shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        logger.info("Ollama restart attempted via %r", path)
    except Exception:
        logger.exception("Failed to launch the Ollama app at %r", path)
    return True


def _run_ollama_watchdog() -> None:
    """Scheduler target: one Ollama liveness pass every 60 seconds.

    State-transition-only logging: exactly when the consecutive-failure count
    reaches ``OLLAMA_FAIL_THRESHOLD`` the persisted status flips to
    unavailable and ONE ``bot_activity`` ``error`` row is written — later
    failing passes only retry the guarded restart (at most one attempt per
    ``OLLAMA_RESTART_COOLDOWN_MS``). Recovery flips the status back silently.
    NEVER raises.
    """
    global _ollama_fail_count, _ollama_last_restart_ms
    try:
        ok = False
        try:
            from backend.ai.ollama_client import OllamaClient

            ok = OllamaClient().is_available()
        except Exception as exc:
            logger.warning("Ollama availability check failed: %s", exc)

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        status = _read_ollama_status()
        restarts = 0
        if status is not None:
            try:
                restarts = max(0, int(status.get("restarts") or 0))
            except (TypeError, ValueError):
                restarts = 0

        if ok:
            with _watchdog_lock:
                _ollama_fail_count = 0
            # Transition-only write: flip the persisted state back to
            # available (restart counter kept). No activity row on recovery.
            if status is None or not status.get("available"):
                _write_ollama_status(True, now_ms, restarts)
                logger.info("Ollama is answering again — AI-gated trading resumes")
            return

        with _watchdog_lock:
            _ollama_fail_count += 1
            failures = _ollama_fail_count
            cooldown_over = now_ms - _ollama_last_restart_ms > OLLAMA_RESTART_COOLDOWN_MS
        if failures < OLLAMA_FAIL_THRESHOLD:
            return

        # At threshold AND on every later failing pass: guarded restart —
        # at most one attempt per cooldown window.
        app_path = os.path.expandvars(settings.ollama_app_path)
        restart_attempted = False
        if cooldown_over:
            restart_attempted = _launch_ollama_app(app_path)
            if restart_attempted:
                restarts += 1
                with _watchdog_lock:
                    _ollama_last_restart_ms = now_ms

        if failures == OLLAMA_FAIL_THRESHOLD:
            # The state transition: persist "offline" and log ONE error row.
            _write_ollama_status(False, now_ms, restarts)
            try:
                get_auto_trader()._log(
                    "error",
                    symbol="",
                    detail={
                        "where": "ollama_watchdog",
                        "reason": "ai_offline",
                        "consecutive_failures": failures,
                        "restart_attempted": restart_attempted,
                        "app_path": app_path,
                        "explanation": (
                            "The local AI (Ollama) stopped answering — AI-confirmed "
                            "entries are paused and the watchdog is trying to "
                            "restart it."
                        ),
                    },
                )
            except Exception:
                logger.exception("Could not write the ollama_watchdog error activity row")
        elif restart_attempted:
            # Later failing pass: keep the state's since_ms, bump restarts.
            since_ms = now_ms
            if status is not None:
                try:
                    since_ms = int(status.get("since_ms") or now_ms)
                except (TypeError, ValueError):
                    since_ms = now_ms
            _write_ollama_status(False, since_ms, restarts)
    except Exception:  # noqa: BLE001 — a watchdog must never raise into the scheduler
        logger.exception("Ollama watchdog pass failed")


def _ensure_ollama_watchdog_job() -> None:
    """Register the 60-second ``ollama_watchdog`` job on the bot scheduler."""
    with _bot_scheduler_lock:
        _scheduler_locked().add_job(
            _run_ollama_watchdog,
            trigger="interval",
            seconds=60,
            id=OLLAMA_WATCHDOG_JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=30,
        )
        logger.info("Ollama watchdog job '%s' scheduled every 60 s", OLLAMA_WATCHDOG_JOB_ID)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> float | None:
    """Coerce a cell value to a JSON-safe float, mapping NaN/inf/None to None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a canonical (optionally feature-enriched) OHLCV frame to JSON records.

    Args:
        df: DataFrame with a UTC tz-aware DatetimeIndex and float columns.

    Returns:
        One dict per row with an ISO-8601 ``timestamp`` plus every column,
        NaN/inf sanitised to ``None``.
    """
    records: list[dict[str, Any]] = []
    columns = list(df.columns)
    for ts, row in df.iterrows():
        record: dict[str, Any] = {"timestamp": ts.isoformat()}
        for col in columns:
            record[col] = _json_safe(row[col])
        records.append(record)
    return records


def _ms_to_iso(ms: int | None) -> str | None:
    """Convert epoch milliseconds (UTC) to an ISO-8601 string, or None."""
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def _exc_detail(exc: Exception) -> str:
    """Human-readable message for an exception (KeyError strips its quoting)."""
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc) or type(exc).__name__


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise the database and background data scheduler; stop on shutdown."""
    logger.info("Starting Trading AI Platform API — PAPER TRADING ONLY")
    from backend.database.db import init_db

    init_db()

    scheduler_started = False
    try:
        from backend.data.service import start_scheduler

        start_scheduler()
        scheduler_started = True
        logger.info("Data update scheduler started")
    except Exception:
        logger.exception("Could not start the data update scheduler; API continues without it")

    # The auto-trader survives restarts: when the persisted config says the
    # bot was running, re-register its periodic cycle job.
    try:
        bot_config = get_auto_trader().get_config()
        if bot_config.running:
            _ensure_bot_job(bot_config.interval_minutes)
            logger.info(
                "Auto-trader was running before restart — cycle job re-registered "
                "(every %d min, paper only)",
                bot_config.interval_minutes,
            )
    except Exception:
        logger.exception("Could not restore the auto-trader schedule; bot job not registered")

    # The fast scalper also survives restarts: when the persisted params say
    # it was enabled, re-register its tick + hourly AI-tune jobs.
    try:
        scalper_params = get_scalper().get_params()
        if scalper_params.enabled:
            _ensure_scalper_jobs(scalper_params.interval_minutes)
            logger.info(
                "Scalper was enabled before restart — tick/tune jobs "
                "re-registered (tick every %d min, paper only)",
                scalper_params.interval_minutes,
            )
    except Exception:
        logger.exception("Could not restore the scalper schedule; scalper jobs not registered")

    # The intel/coach layer also survives restarts: when the persisted
    # intel_config says it was enabled, re-register the sentiment-refresh and
    # coach-review jobs.
    try:
        intel_config = _intel_config()
        if intel_config["enabled"]:
            _ensure_intel_jobs(intel_config["interval_minutes"])
            logger.info(
                "Intel was enabled before restart — refresh/coach jobs "
                "re-registered (refresh every %d min, coach every %d min)",
                intel_config["interval_minutes"],
                COACH_RUN_INTERVAL_MINUTES,
            )
    except Exception:
        logger.exception("Could not restore the intel schedule; intel/coach jobs not registered")

    # Ollama watchdog: keep the local AI alive (60 s liveness checks, guarded
    # restarts). Registered whenever enabled in settings — no persisted flag.
    if settings.ollama_watchdog_enabled:
        try:
            _ensure_ollama_watchdog_job()
        except Exception:
            logger.exception("Could not start the Ollama watchdog; API continues without it")

    try:
        yield
    finally:
        _shutdown_bot_scheduler()
        if scheduler_started:
            try:
                from backend.data.service import stop_scheduler

                stop_scheduler()
                logger.info("Data update scheduler stopped")
            except Exception:
                logger.exception("Error while stopping the data update scheduler")
        logger.info("Trading AI Platform API shut down")


app = FastAPI(
    title="Trading AI Platform (Paper Only)",
    description=(
        "Local AI-powered trading research platform. PAPER TRADING ONLY — "
        "no real-money execution, no private exchange API keys; public "
        "market-data endpoints only."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Simple web UI (static single-page app served at /ui, redirect from /) ---
from config.settings import PROJECT_ROOT

_WEBUI_DIR = PROJECT_ROOT / "webui"
if _WEBUI_DIR.is_dir():
    from fastapi.responses import RedirectResponse
    from fastapi.staticfiles import StaticFiles

    app.mount("/ui", StaticFiles(directory=str(_WEBUI_DIR), html=True), name="webui")

    @app.get("/", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/")


# ---------------------------------------------------------------------------
# Exception handling — always {"detail": str}
# ---------------------------------------------------------------------------


async def _bad_request_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map ValueError/KeyError to HTTP 400."""
    logger.warning("Bad request on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=400, content={"detail": _exc_detail(exc)})


async def _generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global fallback: OllamaError/DataFetchError → 502, else 500.

    The domain exception classes are owned by other modules and matched by
    class name (via the exception's MRO) so this module never has to import
    them at startup.
    """
    mro_names = {cls.__name__ for cls in type(exc).__mro__}
    if "OllamaError" in mro_names or "DataFetchError" in mro_names:
        logger.error("Upstream service error on %s: %s", request.url.path, exc)
        return JSONResponse(status_code=502, content={"detail": _exc_detail(exc)})
    if isinstance(exc, (ValueError, KeyError)):
        logger.warning("Bad request on %s: %s", request.url.path, exc)
        return JSONResponse(status_code=400, content={"detail": _exc_detail(exc)})
    logger.error("Unhandled error on %s", request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": _exc_detail(exc)})


app.add_exception_handler(ValueError, _bad_request_handler)
app.add_exception_handler(KeyError, _bad_request_handler)
app.add_exception_handler(Exception, _generic_exception_handler)


# ---------------------------------------------------------------------------
# Health & models
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe: API status, Ollama availability and DB reachability."""
    ollama_available = False
    try:
        from backend.ai.ollama_client import OllamaClient

        ollama_available = OllamaClient().is_available()
    except Exception as exc:
        logger.warning("Ollama availability check failed: %s", exc)

    db_status = "ok"
    try:
        from backend.database.db import get_conn

        conn = get_conn()
        try:
            conn.execute("SELECT 1")
        finally:
            conn.close()
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        db_status = "error"

    ollama_since_ms: int | None = None
    status = _read_ollama_status()
    if status is not None and status.get("since_ms") is not None:
        try:
            ollama_since_ms = int(status["since_ms"])
        except (TypeError, ValueError):
            ollama_since_ms = None

    return HealthResponse(
        status="ok",
        ollama_available=ollama_available,
        db=db_status,
        paper_only=True,
        ollama_since_ms=ollama_since_ms,
    )


@app.get("/api/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    """Configured Ollama models plus the models actually installed locally."""
    installed: list[str] = []
    ollama_available = False
    try:
        from backend.ai.ollama_client import OllamaClient

        client = OllamaClient()
        installed = client.list_models()
        ollama_available = client.is_available()
    except Exception as exc:
        logger.warning("Could not query Ollama for installed models: %s", exc)

    return ModelsResponse(
        configured=ConfiguredModels(
            default=settings.ollama_model,
            fallback=settings.ollama_fallback_model,
            code=settings.ollama_code_model,
        ),
        installed=installed,
        ollama_available=ollama_available,
    )


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


@app.post("/api/data/fetch", response_model=FetchDataResponse)
def fetch_data(req: FetchDataRequest) -> FetchDataResponse:
    """Incrementally fetch public market data into the local cache."""
    from backend.data.service import update_symbol
    from backend.database.db import ohlcv_coverage

    rows_added = update_symbol(req.source, req.symbol, req.timeframe, lookback_days=req.lookback_days)
    coverage = ohlcv_coverage(req.source, req.symbol, req.timeframe)
    coverage_info = (
        CoverageInfo(first=coverage[0].isoformat(), last=coverage[1].isoformat())
        if coverage is not None
        else CoverageInfo()
    )
    return FetchDataResponse(rows_added=int(rows_added), coverage=coverage_info)


@app.get("/api/data/ohlcv", response_model=OHLCVResponse)
def get_data_ohlcv(
    symbol: str = Query(..., min_length=1, description="Symbol in the source's native format."),
    source: str = Query("binance"),
    timeframe: str = Query("1h"),
    limit: int = Query(500, ge=1),
    features: bool = Query(False, description="True → include the canonical feature columns."),
) -> OHLCVResponse:
    """Cached candles (cache-first; auto-fetches when the cache is empty)."""
    from backend.data.service import get_ohlcv

    df = get_ohlcv(source, symbol, timeframe, limit=limit, with_features=features)
    records = _df_to_records(df)
    return OHLCVResponse(symbol=symbol, timeframe=timeframe, rows=len(records), data=records)


@app.get("/api/data/cached")
def get_data_cached() -> dict[str, Any]:
    """Every (source, symbol, timeframe) currently in the local cache."""
    from backend.database.db import list_cached

    return {"items": list_cached()}


# ---------------------------------------------------------------------------
# AI analysis
# ---------------------------------------------------------------------------


@app.post("/api/analysis")
def run_analysis(req: AnalysisRequest) -> dict[str, Any]:
    """Run the local-LLM market analyst on feature-enriched cached data."""
    from backend.ai.analyst import analyze_market
    from backend.data.service import get_ohlcv

    df = get_ohlcv(req.source, req.symbol, req.timeframe, limit=500, with_features=True)
    if df.empty:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No cached data for {req.source}:{req.symbol}:{req.timeframe}. "
                "POST /api/data/fetch first to download data."
            ),
        )
    analysis = analyze_market(req.symbol, req.timeframe, df, model=req.model)
    return analysis.model_dump()


@app.get("/api/analysis/history")
def analysis_history(
    symbol: str | None = Query(None),
    limit: int = Query(50, ge=1),
) -> dict[str, Any]:
    """Previously persisted AI analyses, newest first."""
    from backend.ai.analyst import get_analysis_history

    return {"items": get_analysis_history(symbol=symbol, limit=limit)}


# ---------------------------------------------------------------------------
# Strategies & backtesting
# ---------------------------------------------------------------------------


@app.get("/api/strategies")
def get_strategies() -> dict[str, Any]:
    """List all registered strategies with descriptions and default params."""
    from backend.strategies import list_strategies

    return {"strategies": list_strategies()}


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest) -> dict[str, Any]:
    """Backtest a strategy on cached candles and persist the run."""
    from backend.backtest.engine import BacktestConfig, Backtester
    from backend.database.db import get_conn, load_ohlcv, utc_now_ms
    from backend.indicators.features import add_features
    from backend.strategies import get_strategy

    df = load_ohlcv(req.source, req.symbol, req.timeframe, limit=req.limit)
    if len(df) < MIN_BACKTEST_ROWS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {len(df)} cached candles for {req.source}:{req.symbol}:{req.timeframe}; "
                f"at least {MIN_BACKTEST_ROWS} are required. "
                "POST /api/data/fetch first to download data."
            ),
        )
    df = add_features(df)
    strategy = get_strategy(req.strategy, **(req.params or {}))

    valid_fields = {f.name for f in dataclasses.fields(BacktestConfig)}
    overrides = {k: v for k, v in (req.config or {}).items() if k in valid_fields}
    ignored = set(req.config or {}) - valid_fields
    if ignored:
        logger.warning("Ignoring unknown backtest config keys: %s", sorted(ignored))
    config = BacktestConfig(**overrides)

    result = Backtester(config).run(df, strategy, symbol=req.symbol)
    payload = result.to_dict()
    payload["strategy"] = req.strategy
    payload["symbol"] = req.symbol
    payload["timeframe"] = req.timeframe

    try:
        conn = get_conn()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO backtest_runs "
                    "(created_at, strategy, symbol, timeframe, params, metrics, equity_curve) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        utc_now_ms(),
                        req.strategy,
                        req.symbol,
                        req.timeframe,
                        json.dumps(req.params or {}),
                        json.dumps(payload.get("metrics", {})),
                        json.dumps(payload.get("equity_curve", {})),
                    ),
                )
        finally:
            conn.close()
    except Exception:
        logger.exception("Failed to persist backtest run for %s/%s", req.strategy, req.symbol)

    return payload


@app.post("/api/backtest/walkforward")
def run_walkforward(req: WalkForwardRequest) -> dict[str, Any]:
    """Walk-forward evaluation of a strategy on cached candles."""
    from backend.backtest.walk_forward import walk_forward
    from backend.database.db import load_ohlcv

    df = load_ohlcv(req.source, req.symbol, req.timeframe, limit=req.limit)
    if len(df) < MIN_BACKTEST_ROWS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {len(df)} cached candles for {req.source}:{req.symbol}:{req.timeframe}; "
                f"at least {MIN_BACKTEST_ROWS} are required. "
                "POST /api/data/fetch first to download data."
            ),
        )
    return walk_forward(df, req.strategy, n_splits=req.n_splits, train_ratio=req.train_ratio)


@app.get("/api/backtest/runs")
def get_backtest_runs(limit: int = Query(50, ge=1)) -> dict[str, Any]:
    """Previously persisted backtest runs, newest first."""
    from backend.database.db import get_conn

    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id, created_at, strategy, symbol, timeframe, metrics "
            "FROM backtest_runs ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    items: list[dict[str, Any]] = []
    for row in rows:
        try:
            metrics = json.loads(row["metrics"])
        except (TypeError, json.JSONDecodeError):
            metrics = {}
        items.append(
            {
                "id": row["id"],
                "created_at": _ms_to_iso(row["created_at"]),
                "strategy": row["strategy"],
                "symbol": row["symbol"],
                "timeframe": row["timeframe"],
                "metrics": metrics,
            }
        )
    return {"items": items}


# ---------------------------------------------------------------------------
# Paper trading (simulation only — fills against locally cached candles)
# ---------------------------------------------------------------------------


@app.post("/api/paper/orders")
def submit_paper_order(req: PaperOrderRequest) -> dict[str, Any]:
    """Submit a simulated order; 400 with the reason when risk checks reject it."""
    order = get_engine().submit_order(
        symbol=req.symbol,
        side=req.side,
        order_type=req.order_type,
        qty=req.qty,
        limit_price=req.limit_price,
        stop_price=req.stop_price,
        stop_loss=req.stop_loss,
        take_profit=req.take_profit,
        source=req.source,
        timeframe=req.timeframe,
    )
    if isinstance(order, dict) and order.get("status") == "rejected":
        reason = order.get("note") or order.get("reason") or "Order rejected by risk manager"
        raise HTTPException(status_code=400, detail=str(reason))
    return order


@app.get("/api/paper/orders")
def get_paper_orders(status: str | None = Query(None)) -> dict[str, Any]:
    """Simulated orders, optionally filtered by status."""
    return {"items": get_engine().get_orders(status=status)}


@app.get("/api/paper/positions")
def get_paper_positions(status: str = Query("open")) -> dict[str, Any]:
    """Simulated positions filtered by status (default: open)."""
    return {"items": get_engine().get_positions(status=status)}


@app.get("/api/paper/trades")
def get_paper_trades(limit: int = Query(100, ge=1)) -> dict[str, Any]:
    """Most recent simulated fills."""
    return {"items": get_engine().get_trades(limit=limit)}


@app.get("/api/paper/portfolio")
def get_paper_portfolio() -> dict[str, Any]:
    """Current simulated portfolio snapshot (equity, PnL, risk halts...)."""
    portfolio = get_engine().get_portfolio()
    portfolio["max_open_positions"] = settings.max_open_positions
    return portfolio


@app.get("/api/paper/equity")
def get_paper_equity(limit: int = Query(1000, ge=1)) -> dict[str, Any]:
    """Equity-curve snapshots of the simulated account."""
    return {"items": get_engine().get_equity_history(limit=limit)}


@app.post("/api/paper/tick")
def paper_tick(req: TickRequest) -> dict[str, Any]:
    """Process the latest candle: fill pending orders, check SL/TP, snapshot equity."""
    return get_engine().process_tick(
        req.symbol, source=req.source, timeframe=req.timeframe, refresh=req.refresh
    )


@app.post("/api/paper/positions/{position_id}/close")
def close_paper_position(
    position_id: int = Path(..., ge=1, description="ID of the open position to close."),
) -> dict[str, Any]:
    """Market-close an open simulated position at the latest cached price."""
    return get_engine().close_position(position_id)


@app.post("/api/paper/reset", response_model=StatusResponse)
def reset_paper_account() -> StatusResponse:
    """Reset the paper account's MONEY while preserving the user's settings.

    ``engine.reset()`` wipes the account (cash, positions, orders, trades,
    equity history) and all of ``account_state`` — but the user's
    configuration must survive a money reset, so ``bot_config``,
    ``scalper_params`` and ``intel_config`` are snapshotted first and
    re-persisted afterwards with their run flags forced off
    (``running=False`` / ``enabled=False``): a fresh account never trades
    or re-tunes unattended until explicitly started again. Scheduler jobs
    are removed to match, and the coach's ``coin_playbook`` rows are cleared
    — they were learned from the wiped account's trades. The watchdog's
    ``ollama_status`` is likewise snapshotted and re-persisted: a money
    reset must not make an offline Ollama look online (missing key = ONLINE
    per contract) nor zero the monotonic ``restarts`` counter.
    """
    trader = get_auto_trader()
    scalper = get_scalper()
    bot_cfg = trader.get_config().model_dump()
    scalp_cfg = scalper.get_params().model_dump()
    intel_cfg = _intel_config()
    ollama_status = _read_ollama_status()

    get_engine().reset()
    _remove_bot_job()
    _remove_scalper_jobs()
    _remove_intel_jobs()
    _clear_coach_playbook()

    # Restore configuration with every run flag off.
    try:
        bot_cfg["running"] = False
        trader.set_config(bot_cfg)
        scalp_cfg["enabled"] = False
        scalper.set_params(scalp_cfg)
    except Exception:
        logger.exception("Could not restore bot/scalper config after reset — defaults apply")
    try:
        from backend.intel.sentiment import set_intel_config

        set_intel_config({**intel_cfg, "enabled": False})
    except Exception:
        logger.debug("intel module unavailable — restoring intel_config directly")
        try:
            from backend.database.db import get_conn
            from backend.paper_trading.engine import PaperTradingEngine

            conn = get_conn()
            try:
                with conn:
                    PaperTradingEngine._state_set(
                        conn, _INTEL_CONFIG_KEY, {**intel_cfg, "enabled": False}
                    )
            finally:
                conn.close()
        except Exception:
            logger.exception("Could not restore intel config after reset — defaults apply")

    # Restore the watchdog's persisted Ollama status: it is INFRASTRUCTURE
    # state, not account state. Wiping it would make an offline Ollama look
    # ONLINE (missing key = online per contract) until the watchdog's next
    # persisting pass — up to the whole restart cooldown — re-arming the
    # per-candidate timeout storm the ai_offline gates exist to prevent, and
    # would zero the contractually monotonic `restarts` counter.
    if ollama_status is not None:
        try:
            _write_ollama_status(
                bool(ollama_status.get("available")),
                int(ollama_status.get("since_ms") or 0),
                int(ollama_status.get("restarts") or 0),
            )
        except (TypeError, ValueError):
            logger.warning(
                "Malformed ollama_status snapshot %r — not restored after reset",
                ollama_status,
            )

    logger.info(
        "Paper account reset to initial capital — user configuration preserved, "
        "run flags forced off (bot/scalper/intel jobs removed, playbook cleared)"
    )
    return StatusResponse(status="reset")


# ---------------------------------------------------------------------------
# AI auto-trader (simulation only — trades the paper account autonomously)
# ---------------------------------------------------------------------------


@app.get("/api/bot/status")
def bot_status() -> dict[str, Any]:
    """Auto-trader status: running flag, cycle times, watchlist, equity."""
    return _bot_status()


@app.get("/api/bot/config")
def bot_get_config() -> dict[str, Any]:
    """Current persisted auto-trader configuration."""
    return get_auto_trader().get_config().model_dump()


@app.post("/api/bot/config")
def bot_set_config(req: BotConfigUpdate) -> dict[str, Any]:
    """Apply a partial config update; keeps the scheduler job in sync."""
    trader = get_auto_trader()
    old = trader.get_config()
    config = trader.set_config(req.model_dump(exclude_unset=True))
    # Keep the APScheduler job consistent with the persisted flag/interval —
    # but only re-add it when the schedule actually changed (or the job is
    # missing): re-adding resets the interval timer, so unrelated config
    # saves would otherwise push the next cycle back indefinitely.
    if config.running:
        if (
            not old.running
            or old.interval_minutes != config.interval_minutes
            or not _bot_job_exists()
        ):
            _ensure_bot_job(config.interval_minutes)
    else:
        _remove_bot_job()
    return config.model_dump()


@app.post("/api/bot/start")
def bot_start() -> dict[str, Any]:
    """Start the auto-trader: schedule periodic cycles and kick one now.

    Registers the ``bot_cycle`` interval job (``max_instances=1``,
    ``coalesce=True``) and launches one immediate cycle in a daemon thread —
    the AutoTrader's internal lock guarantees cycles never overlap.
    """
    trader = get_auto_trader()
    config = trader.start()
    _ensure_bot_job(config.interval_minutes)
    threading.Thread(
        target=trader.run_cycle, name="bot-cycle-kickoff", daemon=True
    ).start()
    logger.info("Auto-trader started (PAPER TRADING ONLY)")
    return _bot_status()


@app.post("/api/bot/stop")
def bot_stop() -> dict[str, Any]:
    """Stop the auto-trader and remove its scheduled cycle job."""
    get_auto_trader().stop()
    _remove_bot_job()
    logger.info("Auto-trader stopped")
    return _bot_status()


@app.post("/api/bot/run-once")
def bot_run_once() -> dict[str, Any]:
    """Run one full bot cycle synchronously (may take minutes with AI on)."""
    return get_auto_trader().run_cycle()


@app.get("/api/bot/activity")
def bot_activity(limit: int = Query(100, ge=1)) -> dict[str, Any]:
    """Most recent auto-trader activity rows (full decision transparency)."""
    return {"items": get_auto_trader().get_activity(limit=limit)}


# ---------------------------------------------------------------------------
# Fast scalper (simulation only — mechanical script + hourly AI supervisor)
# ---------------------------------------------------------------------------


@app.get("/api/scalper/status")
def scalper_status() -> dict[str, Any]:
    """Scalper status: params, closed-trade stats, job flag, last AI tune."""
    return _scalper_status()


@app.post("/api/scalper/start")
def scalper_start() -> dict[str, Any]:
    """Enable the scalper and schedule its tick + hourly AI-tune jobs.

    Registers ``scalper_tick`` (every ``params.interval_minutes``,
    ``max_instances=1``, ``coalesce=True``) and ``scalper_tune`` (hourly).
    """
    params = get_scalper().start()
    _ensure_scalper_jobs(params.interval_minutes)
    logger.info("Scalper started (PAPER TRADING ONLY)")
    return _scalper_status()


@app.post("/api/scalper/stop")
def scalper_stop() -> dict[str, Any]:
    """Disable the scalper, remove its jobs and market-close its positions.

    Documented contract behaviour: a stopped script no longer runs its
    time-stop, so every scalper-owned open position is market-closed
    (``scalp_exit`` reason ``script_stopped``) instead of being left behind.
    """
    _remove_scalper_jobs()
    result = get_scalper().stop()
    logger.info(
        "Scalper stopped — %d open scalp position(s) closed (paper only)",
        int(result.get("closed_positions") or 0),
    )
    return _scalper_status()


@app.post("/api/scalper/params")
def scalper_set_params(req: ScalperParamsUpdate) -> dict[str, Any]:
    """Apply a partial params update (clamped to HARD_BOUNDS); sync the job.

    Re-registers the tick job only when the script is enabled and the
    interval actually changed (or the job is missing) — re-adding resets the
    interval timer, so unrelated saves must not push the next tick back.
    """
    scalper = get_scalper()
    old = scalper.get_params()
    params = scalper.set_params(req.model_dump(exclude_unset=True))
    if params.enabled and (
        old.interval_minutes != params.interval_minutes or not _scalper_job_exists()
    ):
        _ensure_scalper_jobs(params.interval_minutes)
    return params.model_dump()


@app.post("/api/scalper/tune")
def scalper_tune() -> dict[str, Any]:
    """Run one AI supervisor pass synchronously (may take a while).

    Every proposed value is clamped to HARD_BOUNDS; Ollama failures or bad
    JSON change nothing. When the tuner changes ``interval_minutes`` the
    running tick job is re-registered on the new cadence.
    """
    result = get_scalper().tune_with_ai()
    if "interval_minutes" in (result.get("applied") or {}):
        params = get_scalper().get_params()
        if params.enabled and _scalper_job_exists():
            _ensure_scalper_jobs(params.interval_minutes)
    return result


# ---------------------------------------------------------------------------
# Market intelligence (news/community sentiment) + Coach (per-coin playbook)
# — public keyless sources + local LLM only; paper trading only.
# ---------------------------------------------------------------------------


@app.get("/api/intel/status")
def intel_status() -> dict[str, Any]:
    """Intel status: enabled flag, last refresh, fear & greed, global mood."""
    return _intel_status()


@app.post("/api/intel/start")
def intel_start() -> dict[str, Any]:
    """Enable the intel layer and schedule its refresh + coach jobs.

    Registers ``intel_refresh`` (every ``intel_config.interval_minutes``,
    default 30) and ``coach_run`` (every 120 minutes), both with
    ``max_instances=1`` and ``coalesce=True``, and kicks one immediate
    sentiment refresh in a daemon thread so the dashboard fills up quickly.
    """
    config = _set_intel_enabled(True)
    _ensure_intel_jobs(config["interval_minutes"])
    threading.Thread(
        target=_run_intel_refresh, name="intel-refresh-kickoff", daemon=True
    ).start()
    logger.info("Intel layer started (public keyless sources only)")
    return _intel_status()


@app.post("/api/intel/stop")
def intel_stop() -> dict[str, Any]:
    """Disable the intel layer and remove its refresh + coach jobs."""
    _set_intel_enabled(False)
    _remove_intel_jobs()
    logger.info("Intel layer stopped")
    return _intel_status()


@app.post("/api/intel/refresh")
def intel_refresh() -> dict[str, Any]:
    """Run one sentiment refresh over the bot watchlist synchronously.

    502 when the intel module is not installed/importable — the rest of the
    platform keeps working without it.
    """
    try:
        from backend.intel.sentiment import SentimentAgent
    except Exception as exc:
        logger.exception("Intel module unavailable")
        raise HTTPException(
            status_code=502, detail=f"intel module unavailable: {exc}"
        ) from exc
    symbols = get_auto_trader().get_config().watchlist
    return SentimentAgent().run(symbols)


@app.get("/api/intel/sentiment")
def intel_sentiment(
    symbol: str | None = Query(None, description="One symbol's recent history; omit for the latest row per symbol."),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    """Persisted coin sentiment: latest per symbol, or one symbol's history."""
    return {"items": _sentiment_items(symbol, limit)}


@app.get("/api/intel/playbook")
def intel_playbook() -> dict[str, Any]:
    """The coach's per-coin playbook rows (bench/bias/size/vote + reasoning)."""
    return {"items": get_coach().get_playbook()}


@app.post("/api/intel/derivatives/refresh")
def intel_derivatives_refresh() -> dict[str, Any]:
    """Fetch keyless Binance futures positioning for the watchlist now.

    502 when the intel module is not installed/importable; a geo-blocked or
    offline network simply yields ``failed`` counts (the run never raises).
    """
    try:
        from backend.intel.derivatives import DerivativesAgent
    except Exception as exc:
        logger.exception("Derivatives intel module unavailable")
        raise HTTPException(
            status_code=502, detail=f"derivatives module unavailable: {exc}"
        ) from exc
    symbols = get_auto_trader().get_config().watchlist
    return DerivativesAgent().run(symbols)


@app.get("/api/intel/derivatives")
def intel_derivatives() -> dict[str, Any]:
    """Freshest persisted derivatives-positioning row per symbol (< 2h old)."""
    try:
        from backend.intel.derivatives import DerivativesAgent
    except Exception as exc:
        logger.exception("Derivatives intel module unavailable")
        raise HTTPException(
            status_code=502, detail=f"derivatives module unavailable: {exc}"
        ) from exc
    return {"items": DerivativesAgent.latest_map()}


@app.post("/api/coach/run")
def coach_run() -> dict[str, Any]:
    """Run one coach review pass synchronously (may take minutes with the LLM).

    Every proposed value is clamped; coins with fewer than 3 closed trades
    are never changed; LLM failures change nothing (``Coach.run`` never
    raises).
    """
    return get_coach().run()
