"""Coach agent — reviews per-coin results and adapts a per-coin playbook. PAPER ONLY.

The :class:`Coach` is the second member of the agent team layered on top of
the traders (see CONTRACTS.md, section ``backend/intel/ +
backend/paper_trading/coach.py``). Every run it reviews, per watchlist coin,
the most recent closed trades made by BOTH traders (AutoTrader + Scalper),
asks the local LLM — in batches of 5 coins per chat call — for conservative
per-coin adjustments, and persists them to the ``coin_playbook`` table:

* ``bench``             — 1 = the traders must not trade this coin at all;
* ``side_bias``         — ``both`` / ``long_only`` / ``short_only``;
* ``size_multiplier``   — scales the traders' position size, clamped to
  :data:`SIZE_MULTIPLIER_BOUNDS`;
* ``min_vote_override`` — per-coin replacement for the AutoTrader's
  ``min_vote``, clamped to :data:`MIN_VOTE_OVERRIDE_BOUNDS` (or NULL).

Safety rules (all contract-mandated):

* every proposed value is CLAMPED whatever the model says;
* a coin with fewer than :data:`MIN_TRADES_FOR_CHANGE` closed trades gets NO
  changes (not enough evidence) — except that a coin benched for more than
  :data:`BENCH_RESET_DAYS` days is reset to defaults so it can earn its way
  back into the rotation;
* SHRINK-ONLY evidence gates (Improvement Pack v3): below
  :data:`COACH_SIZE_UP_MIN_TRADES` closed trades ``size_multiplier`` is
  capped at 1.0 (the coach may shrink a coin but not size it up), and below
  :data:`COACH_SIDE_BIAS_MIN_TRADES` closed trades ``side_bias`` is forced
  back to ``both`` whatever the proposal or the stored row said; every
  override is recorded in the ``coach_tune`` detail under
  ``shrink_only_clamped``;
* an LLM failure (Ollama error, unusable JSON) changes NOTHING;
* every applied change writes a ``coach_tune`` row to ``bot_activity`` with
  the full before/after playbook and the model's reasoning.

The traders read the playbook through :func:`load_playbook_map` — one query
per cycle/tick, ``{}`` on ANY failure (including a missing table), so the
platform keeps working when this layer is absent or broken. All LLM work goes
through the existing :class:`OllamaClient`. PAPER TRADING ONLY — nothing here
can ever place a real order or touch an API key.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from backend.ai.ollama_client import OllamaClient, OllamaError
from backend.database.db import get_conn, init_db, utc_now_ms
from backend.paper_trading.auto_trader import (
    _CONFIG_KEY as _BOT_CONFIG_KEY,
    BotConfig,
    _json_num,
    _ms_to_iso,
)
from backend.paper_trading.engine import (
    LAST_RESET_MS_KEY as _LAST_RESET_KEY,
    PaperTradingEngine,
)
from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard bounds & defaults — enforced with clamping whatever the source of a
# change (LLM proposal, drifted DB row). Contract-mandated values.
# ---------------------------------------------------------------------------

#: Allowed range of the per-coin position-size multiplier.
SIZE_MULTIPLIER_BOUNDS: tuple[float, float] = (0.25, 1.5)
#: Allowed range of the per-coin ``min_vote`` override (or None = no override).
MIN_VOTE_OVERRIDE_BOUNDS: tuple[int, int] = (2, 4)
#: Valid ``side_bias`` values.
SIDE_BIASES: tuple[str, ...] = ("both", "long_only", "short_only")
#: A coin with fewer closed trades than this gets NO playbook changes.
MIN_TRADES_FOR_CHANGE = 3
#: A benched coin with too few trades is reset to defaults after this long.
BENCH_RESET_DAYS = 7
#: Evidence needed before ``size_multiplier`` may exceed 1.0 (shrink-only).
COACH_SIZE_UP_MIN_TRADES = 20
#: Evidence needed before ``side_bias`` may leave ``"both"`` (shrink-only).
COACH_SIDE_BIAS_MIN_TRADES = 10

#: Closed trades per coin fed to the LLM (most recent first).
_TRADES_PER_COIN = 20
#: Coins reviewed per LLM chat call (contract: batches of 5).
_BATCH_SIZE = 5
#: Cap on stored/logged model reasoning text.
_MAX_REASONING_CHARS = 600
#: Recent ``exit``/``scalp_exit`` activity rows scanned for exit reasons.
_EXIT_ROWS_SCANNED = 2000

#: Exact contract DDL — idempotent, executed defensively by the Coach so the
#: playbook works even before ``schema.sql`` ships the same table.
_PLAYBOOK_DDL = """
CREATE TABLE IF NOT EXISTS coin_playbook (
    symbol TEXT PRIMARY KEY, updated_at INTEGER NOT NULL,
    bench INTEGER NOT NULL DEFAULT 0,
    side_bias TEXT NOT NULL DEFAULT 'both' CHECK (side_bias IN ('both','long_only','short_only')),
    size_multiplier REAL NOT NULL DEFAULT 1.0,
    min_vote_override INTEGER,
    reasoning TEXT NOT NULL DEFAULT '',
    stats TEXT NOT NULL DEFAULT '{}'
);
"""

#: The tunable playbook fields and their neutral defaults.
_DEFAULT_ENTRY: dict[str, Any] = {
    "bench": False,
    "side_bias": "both",
    "size_multiplier": 1.0,
    "min_vote_override": None,
}
#: The fields compared to decide whether a proposal changes anything.
_TUNABLE_FIELDS = ("bench", "side_bias", "size_multiplier", "min_vote_override")

#: Process-wide review guard (module-level, like the SentimentAgent's): the
#: scheduled ``coach_run`` job and the sync ``POST /api/coach/run`` endpoint
#: may fire concurrently, and two overlapping reviews would double the LLM
#: load and race each other's playbook writes.
_RUN_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Clamping helpers (every read/write path funnels through these)
# ---------------------------------------------------------------------------


def _coerce_bench(value: Any, fallback: bool) -> bool:
    """Coerce a bench flag from bool/int/str; unparseable → ``fallback``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "bench", "benched"):
            return True
        if lowered in ("false", "0", "no", ""):
            return False
    logger.warning("Unusable bench value %r — keeping %r", value, fallback)
    return fallback


def _clamp_side_bias(value: Any, fallback: str = "both") -> str:
    """Validate a side bias; common shorthands mapped, anything else → fallback."""
    bias = str(value or "").strip().lower()
    if bias in SIDE_BIASES:
        return bias
    if bias == "long":
        return "long_only"
    if bias == "short":
        return "short_only"
    if bias:
        logger.warning("Unusable side_bias %r — keeping %r", value, fallback)
    return fallback if fallback in SIDE_BIASES else "both"


def _clamp_multiplier(value: Any, fallback: float = 1.0) -> float:
    """Clamp a size multiplier into :data:`SIZE_MULTIPLIER_BOUNDS`."""
    low, high = SIZE_MULTIPLIER_BOUNDS
    try:
        number = float(value)
    except (TypeError, ValueError):
        logger.warning("Unusable size_multiplier %r — keeping %r", value, fallback)
        number = float(fallback)
    if not math.isfinite(number):
        number = float(fallback)
    return round(min(max(number, low), high), 4)


def _clamp_min_vote(value: Any, fallback: int | None = None) -> int | None:
    """Clamp a min-vote override into :data:`MIN_VOTE_OVERRIDE_BOUNDS`.

    ``None`` means "no override"; unparseable values keep ``fallback``.
    """
    if value is None:
        return None
    low, high = MIN_VOTE_OVERRIDE_BOUNDS
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        logger.warning("Unusable min_vote_override %r — keeping %r", value, fallback)
        return fallback
    return min(max(number, low), high)


def _clamped_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the four tunable fields of ``raw``, clamped to their bounds."""
    return {
        "bench": _coerce_bench(raw.get("bench"), bool(_DEFAULT_ENTRY["bench"])),
        "side_bias": _clamp_side_bias(raw.get("side_bias"), "both"),
        "size_multiplier": _clamp_multiplier(raw.get("size_multiplier"), 1.0),
        "min_vote_override": _clamp_min_vote(raw.get("min_vote_override"), None),
    }


def _playbook_bounds_text() -> str:
    """Human/LLM-readable rendering of the playbook bounds for the prompt."""
    return (
        "bench: true|false; side_bias: 'both'|'long_only'|'short_only'; "
        f"size_multiplier within [{SIZE_MULTIPLIER_BOUNDS[0]}, "
        f"{SIZE_MULTIPLIER_BOUNDS[1]}]; min_vote_override: null or an integer "
        f"within [{MIN_VOTE_OVERRIDE_BOUNDS[0]}, {MIN_VOTE_OVERRIDE_BOUNDS[1]}]"
    )


# ---------------------------------------------------------------------------
# Playbook access for the traders (one query per cycle/tick, never raises)
# ---------------------------------------------------------------------------


def load_playbook_map() -> dict[str, dict[str, Any]]:
    """Read the whole ``coin_playbook`` table keyed by symbol. Never raises.

    This is the traders' single read path (AutoTrader once per cycle, Scalper
    once per tick — contract: one query, not one per symbol). Every value is
    re-clamped on read so a drifted row can never smuggle an out-of-bounds
    multiplier into position sizing. Any failure — including the table not
    existing yet because the intel/coach layer was never installed — returns
    ``{}`` so the traders keep their original behaviour.

    Returns:
        ``{symbol: {"bench": bool, "side_bias": str, "size_multiplier": float,
        "min_vote_override": int | None, "reasoning": str,
        "updated_at": int_ms}}`` (possibly empty).
    """
    try:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT symbol, updated_at, bench, side_bias, size_multiplier, "
                "min_vote_override, reasoning FROM coin_playbook"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.debug("coin_playbook unavailable (%s) — empty playbook", exc)
        return {}

    playbook: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = str(row["symbol"] or "").strip().upper()
        if not symbol:
            continue
        entry = _clamped_entry(dict(row))
        entry["reasoning"] = str(row["reasoning"] or "")
        try:
            entry["updated_at"] = int(row["updated_at"])
        except (TypeError, ValueError):
            entry["updated_at"] = 0
        playbook[symbol] = entry
    return playbook


# ---------------------------------------------------------------------------
# Coach
# ---------------------------------------------------------------------------


class Coach:
    """Per-coin performance coach for the paper traders (LLM-supervised).

    Designed as a singleton living next to the engine/AutoTrader/Scalper in
    the API process. :meth:`run` NEVER raises — every batch/coin failure is
    caught, logged and written to ``bot_activity``. PAPER TRADING ONLY.
    """

    def __init__(self) -> None:
        """Ensure the schema (including ``coin_playbook``) exists."""
        init_db()  # idempotent — guarantees bot_activity/account_state exist
        try:
            with self._connect() as conn:
                conn.executescript(_PLAYBOOK_DDL)
        except sqlite3.Error:
            logger.exception("Could not ensure the coin_playbook table exists")
        logger.info("Coach ready (PAPER TRADING ONLY — no real orders, ever)")

    # ------------------------------------------------------------------ #
    # The review pass                                                      #
    # ------------------------------------------------------------------ #

    def run(self) -> dict[str, Any]:
        """Review every watchlist coin and adapt its playbook. NEVER raises.

        Contract behaviour: coins with fewer than
        :data:`MIN_TRADES_FOR_CHANGE` closed trades get no changes (a coin
        benched for more than :data:`BENCH_RESET_DAYS` days is reset to
        defaults); the rest are reviewed by the LLM in batches of
        :data:`_BATCH_SIZE`; every value is clamped; LLM failures change
        nothing; every applied change writes a ``coach_tune`` activity row.

        A non-blocking process-wide lock guards the pass: when a review is
        already in flight (the scheduled job overlapping the sync endpoint)
        the call returns a ``{"status": "busy"}`` summary instead of running
        a second review in parallel.

        Returns:
            Summary dict: ``status``, ``reviewed``, ``updated``,
            ``unchanged``, ``insufficient``, ``unbenched``, ``batches``,
            ``errors``, ``model``, ``finished_at`` (+``error`` on a crash);
            or ``{"status": "busy", "reason": ...}`` when skipped.
        """
        if not _RUN_LOCK.acquire(blocking=False):
            logger.info("Coach run skipped — a review pass is already running")
            return {"status": "busy", "reason": "a coach review is already running"}
        try:
            return self._run_inner()
        except Exception as exc:  # noqa: BLE001 — run must never raise
            logger.exception("Coach run crashed — recorded and swallowed")
            self._log("error", detail={"where": "coach_run", "reason": str(exc)})
            return {
                "status": "error",
                "error": str(exc),
                "reviewed": 0,
                "updated": 0,
                "unchanged": 0,
                "insufficient": 0,
                "unbenched": 0,
                "batches": 0,
                "errors": 1,
                "model": settings.ollama_model,
                "finished_at": _ms_to_iso(utc_now_ms()),
            }
        finally:
            _RUN_LOCK.release()

    def _run_inner(self) -> dict[str, Any]:
        """The actual review pass (see :meth:`run`)."""
        config = self._bot_config()
        watchlist = list(config.watchlist)
        playbook = load_playbook_map()
        evidence = self._gather_evidence(watchlist)
        model = settings.ollama_model

        counters = {
            "reviewed": 0,
            "updated": 0,
            "unchanged": 0,
            "insufficient": 0,
            "unbenched": 0,
            "batches": 0,
            "errors": 0,
        }

        # --- evidence gate: <3 closed trades → no changes -------------------
        eligible: list[str] = []
        now_ms = utc_now_ms()
        for symbol in watchlist:
            coin = evidence.get(symbol) or self._empty_evidence(symbol)
            if coin["trades"] >= MIN_TRADES_FOR_CHANGE:
                eligible.append(symbol)
                continue
            counters["insufficient"] += 1
            current = playbook.get(symbol)
            if (
                current is not None
                and current.get("bench")
                and now_ms - int(current.get("updated_at") or 0)
                > BENCH_RESET_DAYS * 86_400_000
            ):
                # Benched >7 days with too little fresh evidence to judge —
                # reset to defaults so the coin can earn its way back in.
                reasoning = (
                    f"Auto-reset to defaults: benched more than "
                    f"{BENCH_RESET_DAYS} days with fewer than "
                    f"{MIN_TRADES_FOR_CHANGE} closed trades to re-evaluate."
                )
                self._apply_update(
                    symbol, current, dict(_DEFAULT_ENTRY), reasoning, coin, model=""
                )
                counters["unbenched"] += 1
                counters["updated"] += 1

        # --- LLM review in batches of 5 --------------------------------------
        client = OllamaClient()
        for start in range(0, len(eligible), _BATCH_SIZE):
            batch = eligible[start : start + _BATCH_SIZE]
            counters["batches"] += 1
            try:
                proposals = self._review_batch(client, model, batch, evidence, playbook)
            except (OllamaError, ValueError) as exc:
                # Contract: LLM failure → NO change for the whole batch.
                counters["errors"] += 1
                logger.warning("Coach batch %s failed — no changes: %s", batch, exc)
                self._log(
                    "error",
                    detail={
                        "where": "coach_batch",
                        "symbols": batch,
                        "reason": str(exc),
                        "explanation": (
                            "The coach could not review "
                            f"{', '.join(batch)} (LLM failure) — their "
                            "playbooks were left unchanged."
                        ),
                    },
                )
                continue

            for symbol in batch:
                counters["reviewed"] += 1
                current = playbook.get(symbol) or {
                    **_DEFAULT_ENTRY,
                    "reasoning": "",
                    "updated_at": 0,
                }
                proposal = proposals.get(symbol)
                if proposal is None:
                    counters["unchanged"] += 1
                    continue
                coin = evidence.get(symbol) or self._empty_evidence(symbol)
                new_entry, reasoning, shrink_clamped = self._sanitize_proposal(
                    proposal, current, int(coin["trades"])
                )
                if all(new_entry[f] == current.get(f) for f in _TUNABLE_FIELDS):
                    counters["unchanged"] += 1
                    continue
                try:
                    self._apply_update(
                        symbol,
                        current,
                        new_entry,
                        reasoning,
                        coin,
                        model=model,
                        shrink_only_clamped=shrink_clamped,
                    )
                except sqlite3.Error as exc:
                    counters["errors"] += 1
                    logger.exception("Could not persist the playbook for %s", symbol)
                    self._log(
                        "error",
                        symbol=symbol,
                        detail={"where": "coach_persist", "reason": str(exc)},
                    )
                    continue
                playbook[symbol] = {
                    **new_entry,
                    "reasoning": reasoning,
                    "updated_at": utc_now_ms(),
                }
                counters["updated"] += 1

        summary: dict[str, Any] = {
            "status": "ok",
            **counters,
            "model": model,
            "finished_at": _ms_to_iso(utc_now_ms()),
        }
        logger.info(
            "Coach run finished: reviewed=%d updated=%d unchanged=%d "
            "insufficient=%d unbenched=%d batches=%d errors=%d",
            counters["reviewed"],
            counters["updated"],
            counters["unchanged"],
            counters["insufficient"],
            counters["unbenched"],
            counters["batches"],
            counters["errors"],
        )
        return summary

    # ------------------------------------------------------------------ #
    # Public playbook read (API)                                           #
    # ------------------------------------------------------------------ #

    def get_playbook(self) -> list[dict[str, Any]]:
        """Every persisted playbook row, JSON-safe, ordered by symbol.

        Returns:
            list of ``{"symbol", "updated_at" (ISO), "bench" (bool),
            "side_bias", "size_multiplier", "min_vote_override",
            "reasoning", "stats" (dict)}`` — ``[]`` on any failure.
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT symbol, updated_at, bench, side_bias, size_multiplier, "
                    "min_vote_override, reasoning, stats FROM coin_playbook "
                    "ORDER BY symbol ASC"
                ).fetchall()
        except sqlite3.Error as exc:
            logger.debug("coin_playbook unavailable (%s) — empty list", exc)
            return []

        items: list[dict[str, Any]] = []
        for row in rows:
            entry = _clamped_entry(dict(row))
            try:
                stats = json.loads(row["stats"] or "{}")
            except (TypeError, json.JSONDecodeError):
                stats = {}
            try:
                updated_ms: int | None = int(row["updated_at"])
            except (TypeError, ValueError):
                updated_ms = None
            items.append(
                {
                    "symbol": str(row["symbol"] or ""),
                    "updated_at": _ms_to_iso(updated_ms),
                    **entry,
                    "reasoning": str(row["reasoning"] or ""),
                    "stats": stats if isinstance(stats, dict) else {},
                }
            )
        return items

    # ------------------------------------------------------------------ #
    # Evidence gathering                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _empty_evidence(symbol: str) -> dict[str, Any]:
        """Zeroed evidence record for a coin with no closed trades."""
        return {
            "symbol": symbol,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "pnl": 0.0,
            "avg_pnl": 0.0,
            "long": {"trades": 0, "pnl": 0.0},
            "short": {"trades": 0, "pnl": 0.0},
            "exit_reasons": {},
            "last_trade_at": None,
        }

    def _gather_evidence(self, watchlist: list[str]) -> dict[str, dict[str, Any]]:
        """Per-coin closed-trade evidence (bot + scalper, post-reset only).

        Reads the last :data:`_TRADES_PER_COIN` closed ``paper_positions``
        rows per watchlist coin (both traders share that ledger) plus the
        exit-reason distribution from the ``exit``/``scalp_exit`` activity
        rows. Trades from before the last account reset are ignored — they
        belong to a different account.

        Args:
            watchlist: Symbols to gather evidence for.

        Returns:
            ``{symbol: evidence dict}`` (see :meth:`_empty_evidence`).
        """
        if not watchlist:
            return {}
        with self._connect() as conn:
            # Coerce the reset stamp ONCE, before any query string is built:
            # appending a placeholder and only then discovering the value is
            # unusable would leave a dangling '?' with no bound argument.
            reset_ms = PaperTradingEngine._state_get(conn, _LAST_RESET_KEY, None)
            reset_cutoff: int | None = None
            if reset_ms is not None:
                try:
                    reset_cutoff = int(reset_ms)
                except (TypeError, ValueError):
                    logger.warning("Ignoring malformed %s value", _LAST_RESET_KEY)
            placeholders = ",".join("?" for _ in watchlist)
            pos_query = (
                "SELECT symbol, side, pnl, closed_at FROM paper_positions "
                f"WHERE status='closed' AND symbol IN ({placeholders})"
            )
            pos_args: list[Any] = list(watchlist)
            if reset_cutoff is not None:
                pos_query += " AND closed_at >= ?"
                pos_args.append(reset_cutoff)
            pos_query += " ORDER BY closed_at DESC, id DESC"
            pos_rows = conn.execute(pos_query, pos_args).fetchall()

            exit_query = (
                "SELECT symbol, detail FROM bot_activity "
                "WHERE action IN ('exit','scalp_exit')"
            )
            exit_args: list[Any] = []
            if reset_cutoff is not None:
                exit_query += " AND ts >= ?"
                exit_args.append(reset_cutoff)
            exit_query += " ORDER BY id DESC LIMIT ?"
            exit_args.append(_EXIT_ROWS_SCANNED)
            exit_rows = conn.execute(exit_query, exit_args).fetchall()

        evidence: dict[str, dict[str, Any]] = {}
        counted: dict[str, int] = {}
        for row in pos_rows:
            symbol = str(row["symbol"] or "")
            if counted.get(symbol, 0) >= _TRADES_PER_COIN:
                continue
            counted[symbol] = counted.get(symbol, 0) + 1
            coin = evidence.setdefault(symbol, self._empty_evidence(symbol))
            pnl = float(row["pnl"] or 0.0)
            side = "short" if str(row["side"] or "long") == "short" else "long"
            coin["trades"] += 1
            coin["pnl"] += pnl
            coin[side]["trades"] += 1
            coin[side]["pnl"] += pnl
            if pnl > 0:
                coin["wins"] += 1
            elif pnl < 0:
                coin["losses"] += 1
            if coin["last_trade_at"] is None:
                try:
                    coin["last_trade_at"] = _ms_to_iso(int(row["closed_at"]))
                except (TypeError, ValueError):
                    coin["last_trade_at"] = None

        reason_counts: dict[str, int] = {}
        for row in exit_rows:
            symbol = str(row["symbol"] or "")
            coin = evidence.get(symbol)
            if coin is None or reason_counts.get(symbol, 0) >= _TRADES_PER_COIN:
                continue
            try:
                detail = json.loads(row["detail"] or "{}")
            except (TypeError, json.JSONDecodeError):
                detail = {}
            reason = str(detail.get("reason") or "") if isinstance(detail, dict) else ""
            if not reason:
                continue
            reason_counts[symbol] = reason_counts.get(symbol, 0) + 1
            coin["exit_reasons"][reason] = coin["exit_reasons"].get(reason, 0) + 1

        for coin in evidence.values():
            trades = coin["trades"]
            coin["win_rate"] = round(coin["wins"] / trades, 4) if trades else 0.0
            coin["pnl"] = _json_num(coin["pnl"]) or 0.0
            coin["avg_pnl"] = _json_num(coin["pnl"] / trades) if trades else 0.0
            for side in ("long", "short"):
                coin[side]["pnl"] = _json_num(coin[side]["pnl"]) or 0.0
        return evidence

    # ------------------------------------------------------------------ #
    # LLM batch review                                                     #
    # ------------------------------------------------------------------ #

    def _review_batch(
        self,
        client: OllamaClient,
        model: str,
        batch: list[str],
        evidence: dict[str, dict[str, Any]],
        playbook: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Ask the LLM to review one batch of coins; parse its proposals.

        Args:
            client: Shared Ollama client.
            model: Model name used for the chat call.
            batch: Up to :data:`_BATCH_SIZE` symbols with enough evidence.
            evidence: Per-coin evidence from :meth:`_gather_evidence`.
            playbook: Current playbook map (clamped).

        Returns:
            ``{symbol: raw proposal dict}`` for the batch symbols the model
            answered for (values NOT yet clamped — see
            :meth:`_sanitize_proposal`).

        Raises:
            OllamaError: When the model is unreachable.
            ValueError: When the reply contains no usable coin array.
        """
        report = []
        for symbol in batch:
            current = playbook.get(symbol) or {**_DEFAULT_ENTRY, "reasoning": ""}
            report.append(
                {
                    "symbol": symbol,
                    "results": evidence.get(symbol) or self._empty_evidence(symbol),
                    "current_playbook": {
                        field: current.get(field) for field in _TUNABLE_FIELDS
                    },
                }
            )
        prompt = (
            "PER-COIN PAPER-TRADING RESULTS to review (closed trades since "
            f"the last account reset, capped at the {_TRADES_PER_COIN} most "
            "recent per coin; PnL in account currency):\n"
            f"{json.dumps(report, indent=2, default=str)}\n\n"
            "Rules:\n"
            "- Be conservative: change a coin's playbook only when its "
            "results clearly justify it; keeping the current values is a "
            "perfectly good answer.\n"
            "- bench=true only for consistently losing coins; prefer a "
            "reduced size_multiplier or a side_bias before benching.\n"
            "- side_bias when one direction clearly loses and the other "
            "does not.\n"
            "- size_multiplier below 1.0 de-risks a shaky coin; above 1.0 "
            "only for consistent winners.\n"
            "- min_vote_override raises the entry bar for a coin (more "
            "strategy agreement required); null keeps the bot's default.\n"
            "- Evidence gates (enforced): size_multiplier above 1.0 needs at "
            f"least {COACH_SIZE_UP_MIN_TRADES} closed trades and a side_bias "
            f"other than 'both' needs at least {COACH_SIDE_BIAS_MIN_TRADES}; "
            "under-evidenced proposals are clamped back.\n\n"
            f"HARD BOUNDS (out-of-range values are clamped): "
            f"{_playbook_bounds_text()}\n\n"
            "Respond ONLY with JSON exactly in this shape, including EVERY "
            "reviewed coin exactly once:\n"
            '{"coins": [{"symbol": "BTCUSDT", "bench": false, '
            '"side_bias": "both", "size_multiplier": 1.0, '
            '"min_vote_override": null, '
            '"reasoning": "one short sentence"}]}'
        )
        system = (
            "You are the performance COACH of a PAPER-TRADING bot — simulated "
            "money only, research, never real orders. You review each coin's "
            "recent closed trades (made by a slow ensemble trader and a fast "
            "scalper sharing one simulated account) and adapt a small "
            "per-coin playbook: bench losing coins, bias the tradeable side, "
            "scale position size, or raise the entry-vote bar. Be "
            "conservative and evidence-driven. Respond ONLY with JSON. "
            "Not financial advice."
        )
        response = client.chat(
            prompt, system=system, model=model, json_mode=True, temperature=0.2
        )

        payload: Any = json.loads(response)  # JSONDecodeError is a ValueError
        items: Any = None
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            for key in ("coins", "items", "playbooks", "updates"):
                if isinstance(payload.get(key), list):
                    items = payload[key]
                    break
            else:
                items = next(
                    (v for v in payload.values() if isinstance(v, list)), None
                )
        if not isinstance(items, list):
            raise ValueError("model reply contains no coin array")

        batch_set = set(batch)
        proposals: dict[str, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "").strip().upper()
            if symbol in batch_set and symbol not in proposals:
                proposals[symbol] = item
        return proposals

    @staticmethod
    def _sanitize_proposal(
        item: dict[str, Any], current: dict[str, Any], trades: int
    ) -> tuple[dict[str, Any], str, list[str]]:
        """Clamp one raw LLM proposal against the coin's current playbook.

        Fields the model omitted keep their current values; an explicit
        ``min_vote_override: null`` clears the override.

        AFTER the bounds clamps, the shrink-only evidence gates apply
        (contract, Improvement Pack v3): with fewer than
        :data:`COACH_SIZE_UP_MIN_TRADES` closed trades the resulting
        ``size_multiplier`` is capped at 1.0, and with fewer than
        :data:`COACH_SIDE_BIAS_MIN_TRADES` closed trades the resulting
        ``side_bias`` is forced to ``"both"`` — whatever the proposal or the
        stored row said. Benching needs no extra evidence beyond the
        :data:`MIN_TRADES_FOR_CHANGE` review gate.

        Args:
            item: Raw proposal dict from the model.
            current: The coin's current (clamped) playbook entry.
            trades: The coin's closed-trade evidence count.

        Returns:
            ``(clamped entry dict, reasoning string, shrink_only_clamped)``
            where ``shrink_only_clamped`` lists the fields the evidence
            gates overrode (possibly empty).
        """
        cur_bench = bool(current.get("bench"))
        cur_bias = _clamp_side_bias(current.get("side_bias"), "both")
        cur_mult = _clamp_multiplier(current.get("size_multiplier"), 1.0)
        cur_vote = _clamp_min_vote(current.get("min_vote_override"), None)

        if "min_vote_override" in item:
            raw_vote = item["min_vote_override"]
            new_vote = None if raw_vote is None else _clamp_min_vote(raw_vote, cur_vote)
        else:
            new_vote = cur_vote
        new_entry = {
            "bench": _coerce_bench(item.get("bench", cur_bench), cur_bench),
            "side_bias": _clamp_side_bias(item.get("side_bias", cur_bias), cur_bias),
            "size_multiplier": _clamp_multiplier(
                item.get("size_multiplier", cur_mult), cur_mult
            ),
            "min_vote_override": new_vote,
        }

        # Shrink-only evidence gates — applied after the bounds clamps.
        shrink_only_clamped: list[str] = []
        if trades < COACH_SIZE_UP_MIN_TRADES and new_entry["size_multiplier"] > 1.0:
            new_entry["size_multiplier"] = 1.0
            shrink_only_clamped.append("size_multiplier")
        if trades < COACH_SIDE_BIAS_MIN_TRADES and new_entry["side_bias"] != "both":
            new_entry["side_bias"] = "both"
            shrink_only_clamped.append("side_bias")

        reasoning = str(item.get("reasoning") or "")[:_MAX_REASONING_CHARS]
        return new_entry, reasoning, shrink_only_clamped

    # ------------------------------------------------------------------ #
    # Persistence & logging                                                #
    # ------------------------------------------------------------------ #

    def _apply_update(
        self,
        symbol: str,
        before: dict[str, Any],
        after: dict[str, Any],
        reasoning: str,
        stats: dict[str, Any],
        model: str,
        shrink_only_clamped: list[str] | None = None,
    ) -> None:
        """Persist one playbook change and log the mandated ``coach_tune`` row.

        Args:
            symbol: The coin being re-tuned.
            before: Current (clamped) playbook entry.
            after: New (clamped) playbook entry.
            reasoning: The model's (or auto-reset) reasoning text.
            stats: Evidence snapshot stored alongside the row.
            model: Model name (empty string for rule-based auto-resets).
            shrink_only_clamped: Fields the shrink-only evidence gates
                overrode (``None``/omitted → ``[]``, e.g. for the
                rule-based auto-reset rows).
        """
        clamped_fields = list(shrink_only_clamped or [])
        after = _clamped_entry(after)  # belt & braces on the write path
        before_public = {field: before.get(field) for field in _TUNABLE_FIELDS}
        after_public = {field: after.get(field) for field in _TUNABLE_FIELDS}
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO coin_playbook "
                "(symbol, updated_at, bench, side_bias, size_multiplier, "
                "min_vote_override, reasoning, stats) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    symbol,
                    utc_now_ms(),
                    1 if after["bench"] else 0,
                    after["side_bias"],
                    float(after["size_multiplier"]),
                    after["min_vote_override"],
                    reasoning,
                    json.dumps(stats, default=str),
                ),
            )
        changes_txt = ", ".join(
            f"{field}: {before_public.get(field)} -> {after_public[field]}"
            for field in _TUNABLE_FIELDS
            if before_public.get(field) != after_public[field]
        )
        clamp_txt = ""
        if clamped_fields:
            clamp_txt = (
                f" Shrink-only rule kept {' and '.join(clamped_fields)} "
                "conservative: not enough closed trades yet for this coin to "
                "size up or pick one trading side."
            )
        self._log(
            "coach_tune",
            symbol=symbol,
            detail={
                "symbol": symbol,
                "before": before_public,
                "after": after_public,
                "reasoning": reasoning,
                "model": model,
                "shrink_only_clamped": clamped_fields,
                "explanation": (
                    f"Coach re-tuned {symbol} ({changes_txt or 'no field changes'})."
                    f"{clamp_txt}"
                ),
            },
        )
        logger.info("Coach updated the %s playbook: %s", symbol, changes_txt)

    def _log(
        self, action: str, symbol: str = "", detail: dict[str, Any] | None = None
    ) -> None:
        """Append a row to ``bot_activity`` (best-effort — never raises)."""
        try:
            payload = json.dumps(detail or {}, default=str)
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO bot_activity (ts, symbol, action, detail) "
                    "VALUES (?, ?, ?, ?)",
                    (utc_now_ms(), symbol, action, payload),
                )
        except Exception:  # noqa: BLE001 — logging must never break a run
            logger.exception(
                "Failed to write bot_activity row (action=%s, symbol=%s)",
                action,
                symbol,
            )

    # ------------------------------------------------------------------ #
    # Misc helpers                                                         #
    # ------------------------------------------------------------------ #

    def _bot_config(self) -> BotConfig:
        """The AutoTrader's persisted config (watchlist provider)."""
        with self._connect() as conn:
            raw = PaperTradingEngine._state_get(conn, _BOT_CONFIG_KEY, None)
        if raw is None:
            return BotConfig()
        try:
            return BotConfig.model_validate(raw)
        except Exception:  # noqa: BLE001 — ValidationError et al.
            logger.warning("Persisted bot_config is invalid — using defaults")
            return BotConfig()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection that commits on success and always closes."""
        conn = get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
