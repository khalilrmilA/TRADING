"""News & community sentiment agent — batch LLM scoring of watchlist coins.

PAPER TRADING ONLY. The :class:`SentimentAgent` gathers public, keyless
intelligence (Google News headlines per coin, hot reddit titles, the Fear &
Greed index — see :mod:`backend.intel.news`), asks the local Ollama model to
score coins in **batches of 5 per chat call** (one JSON array response per
batch, to stay fast on a local GPU) and persists one ``coin_sentiment`` row
per coin plus one market-wide ``'GLOBAL'`` row per run.

Design decisions (documented per contract):

* **GLOBAL row is DERIVED, not a separate LLM call**: its score blends the
  mean of this run's per-coin LLM scores with the Fear & Greed index mapped
  onto [-100, 100] (``(value - 50) * 2``), averaging whichever of the two is
  available. This costs zero extra LLM latency and still produces a global
  mood even when every batch fails (Fear & Greed alone) or the index is down
  (coin scores alone). When neither exists the GLOBAL row is skipped.
* Failures are contained per batch: an Ollama error or malformed JSON logs a
  warning, skips that batch's coins and moves on — :meth:`SentimentAgent.run`
  NEVER raises.
* ``score`` is clamped to [-100, 100] and ``confidence`` to [0, 100] whatever
  the model returns; an invalid ``sentiment`` string is re-derived from the
  score sign (>= +20 bullish, <= -20 bearish, else neutral).

Scheduling state lives in ``account_state`` under ``'intel_config'``
(``{"enabled": bool, "interval_minutes": 30}`` — helpers
:func:`get_intel_config` / :func:`set_intel_config` are exported for the API,
which owns the APScheduler job ``'intel_refresh'``). The last successful run
timestamp is persisted under ``'intel_last_refresh'`` (epoch ms) and exposed
via :func:`get_last_refresh`.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from backend.ai.ollama_client import OllamaClient, OllamaError
from backend.database.db import get_conn, init_db, utc_now_ms
from backend.intel.news import (
    COIN_NAMES,
    fetch_fear_greed,
    fetch_google_news,
    fetch_reddit_hot,
)
from config.settings import settings

logger = logging.getLogger(__name__)

__all__ = [
    "GLOBAL_SYMBOL",
    "INTEL_CONFIG_KEY",
    "SentimentAgent",
    "get_intel_config",
    "get_last_refresh",
    "set_intel_config",
]

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

#: Symbol of the market-wide mood row in ``coin_sentiment``.
GLOBAL_SYMBOL = "GLOBAL"

#: account_state key for the intel scheduling config (contract-mandated).
INTEL_CONFIG_KEY = "intel_config"
#: account_state key for the last successful refresh (epoch ms UTC).
_LAST_REFRESH_KEY = "intel_last_refresh"

#: Contract defaults for ``'intel_config'``.
_DEFAULT_INTEL_CONFIG: dict[str, Any] = {"enabled": False, "interval_minutes": 30}
#: Sanity bounds for the refresh cadence (minutes).
_INTERVAL_BOUNDS = (1, 1440)

#: Coins scored per LLM chat call (contract: batches of 5).
_BATCH_SIZE = 5
#: Headlines fetched per coin and shown per coin in the prompt.
_COIN_HEADLINES = 8
#: Reddit titles included in each batch prompt's market context.
_REDDIT_TITLES_IN_PROMPT = 15
#: |score| >= this maps to bullish/bearish when re-deriving a sentiment.
_SCORE_SENTIMENT_THRESHOLD = 20
#: GLOBAL row: |score| >= this reads as bullish/bearish (blended, softer).
_GLOBAL_SENTIMENT_THRESHOLD = 15
#: GLOBAL confidence when only the Fear & Greed index was available.
_GLOBAL_FALLBACK_CONFIDENCE = 40

#: ``coin_sentiment`` rows older than this many days are pruned after each
#: run. Every refresh INSERTs one row per coin + GLOBAL (each with a headline
#: JSON blob); without pruning the table grows without bound and every
#: latest-row query degrades linearly. Anything beyond the 3h freshness gate
#: plus a short UI history is dead weight.
_RETENTION_DAYS = 7

_VALID_SENTIMENTS = ("bullish", "bearish", "neutral")

#: Process-wide refresh guard. Module-level ON PURPOSE: every caller in the
#: API (the scheduler job, the sync ``POST /api/intel/refresh`` endpoint and
#: the ``intel_start`` kickoff thread) constructs its own agent instance, so
#: an instance lock could never detect an overlap between them.
_RUN_LOCK = threading.Lock()

#: Quote suffixes stripped for prompt display of unknown symbols.
_QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "USD")

_JSON_SCHEMA = """{
  "results": [
    {
      "symbol": "<exact symbol string from the request>",
      "sentiment": "bullish" | "bearish" | "neutral",
      "score": <integer -100 (very bearish news) .. 100 (very bullish news)>,
      "confidence": <integer 0-100>,
      "summary": "<one sentence explaining the score>"
    }
  ]
}"""

_SYSTEM_PROMPT = (
    "You are a crypto news and community sentiment analyst at a quantitative "
    "RESEARCH firm. This platform is a research and PAPER-TRADING simulation "
    "only: your output is NOT financial advice and never places a real "
    "order.\n\n"
    "Score each requested coin using ONLY the supplied headlines and market "
    "context. Confidence must reflect how much relevant evidence you actually "
    "saw — few or no coin-specific headlines means low confidence and a "
    "near-neutral score.\n\n"
    "Respond ONLY with a single valid JSON object — no prose, no markdown "
    "fences, no extra keys — matching exactly this schema:\n"
    f"{_JSON_SCHEMA}"
)


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #


def _ms_to_iso(ms: int | None) -> str | None:
    """Convert epoch milliseconds to an ISO-8601 UTC string (None-safe)."""
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    """Coerce ``value`` to an int clamped into [lo, hi]; ``default`` on junk."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(lo, min(hi, int(round(number))))


def _coin_display(symbol: str) -> tuple[str, str]:
    """(full name, ticker) for a symbol, falling back to its base asset.

    Args:
        symbol: Exchange symbol, e.g. ``"BTCUSDT"``.

    Returns:
        The :data:`COIN_NAMES` entry when known, otherwise the base asset
        twice (``"FOOUSDT"`` → ``("FOO", "FOO")``).
    """
    known = COIN_NAMES.get(symbol)
    if known is not None:
        return known
    base = symbol
    for suffix in _QUOTE_SUFFIXES:
        if symbol.endswith(suffix) and len(symbol) > len(suffix):
            base = symbol[: -len(suffix)].rstrip("-/")
            break
    return (base, base)


def _sentiment_from_score(score: int, threshold: int) -> str:
    """Map a numeric score onto a sentiment label using ``threshold``."""
    if score >= threshold:
        return "bullish"
    if score <= -threshold:
        return "bearish"
    return "neutral"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
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


def _state_get(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    """Read a JSON-encoded value from ``account_state``."""
    row = conn.execute(
        "SELECT value FROM account_state WHERE key=?", (key,)
    ).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except (TypeError, json.JSONDecodeError):
        logger.warning("Corrupt account_state value for key %r", key)
        return default


def _state_set(conn: sqlite3.Connection, key: str, value: Any) -> None:
    """Write a JSON-encoded value to ``account_state``."""
    conn.execute(
        "INSERT OR REPLACE INTO account_state (key, value) VALUES (?, ?)",
        (key, json.dumps(value)),
    )


# --------------------------------------------------------------------------- #
# intel_config helpers (exported for the API scheduler)                        #
# --------------------------------------------------------------------------- #


def get_intel_config() -> dict[str, Any]:
    """Load the persisted intel scheduling config (defaults when absent).

    Never raises: a missing table/row or corrupt value degrades to the
    contract defaults ``{"enabled": False, "interval_minutes": 30}``.

    Returns:
        ``{"enabled": bool, "interval_minutes": int}``.
    """
    raw: Any = None
    try:
        with _connect() as conn:
            raw = _state_get(conn, INTEL_CONFIG_KEY, None)
    except Exception:  # noqa: BLE001 — config reads must be safe everywhere
        logger.exception("Could not read intel_config — using defaults")
    config = dict(_DEFAULT_INTEL_CONFIG)
    if isinstance(raw, dict):
        config["enabled"] = bool(raw.get("enabled", config["enabled"]))
        config["interval_minutes"] = _clamp_int(
            raw.get("interval_minutes", config["interval_minutes"]),
            *_INTERVAL_BOUNDS,
            default=int(_DEFAULT_INTEL_CONFIG["interval_minutes"]),
        )
    return config


def set_intel_config(updates: dict[str, Any]) -> dict[str, Any]:
    """Apply a partial update to the persisted intel config.

    Unknown keys are ignored with a warning; ``enabled`` is coerced with
    ``bool()`` and ``interval_minutes`` is clamped into [1, 1440] (junk
    values fall back to the current setting).

    Args:
        updates: Subset of ``{"enabled", "interval_minutes"}`` to change.

    Returns:
        The updated, persisted config dict.
    """
    config = get_intel_config()
    unknown = sorted(set(updates) - set(_DEFAULT_INTEL_CONFIG))
    if unknown:
        logger.warning("Ignoring unknown intel config keys: %s", unknown)
    if "enabled" in updates:
        config["enabled"] = bool(updates["enabled"])
    if "interval_minutes" in updates:
        config["interval_minutes"] = _clamp_int(
            updates["interval_minutes"],
            *_INTERVAL_BOUNDS,
            default=int(config["interval_minutes"]),
        )
    init_db()  # idempotent — guarantees account_state exists
    with _connect() as conn:
        _state_set(conn, INTEL_CONFIG_KEY, config)
    logger.info(
        "Intel config updated (enabled=%s, every %d min)",
        config["enabled"],
        config["interval_minutes"],
    )
    return config


def get_last_refresh() -> str | None:
    """Timestamp of the last successful sentiment refresh (ISO-8601 or None).

    Never raises — any storage problem reads as "never refreshed".
    """
    try:
        with _connect() as conn:
            ms = _state_get(conn, _LAST_REFRESH_KEY, None)
        return _ms_to_iso(int(ms)) if ms is not None else None
    except Exception:  # noqa: BLE001 — status reads must be safe everywhere
        logger.exception("Could not read intel_last_refresh")
        return None


# --------------------------------------------------------------------------- #
# SentimentAgent                                                               #
# --------------------------------------------------------------------------- #


class SentimentAgent:
    """Scores watchlist coins from public news/community feeds via the LLM.

    Thread-safe: :meth:`run` uses a non-blocking PROCESS-WIDE lock
    (:data:`_RUN_LOCK` — shared across all instances, since each caller
    builds its own agent) and returns a ``busy`` summary instead of
    overlapping a refresh already in flight (the scheduler job and the sync
    API endpoint may fire concurrently).
    """

    def __init__(self, client: OllamaClient | None = None) -> None:
        """Prepare the agent and make sure the schema exists.

        Args:
            client: Ollama client to use; a default one is created when
                omitted (tests inject a mock here).
        """
        init_db()  # idempotent — guarantees coin_sentiment exists
        self._client = client or OllamaClient()
        logger.info("SentimentAgent ready (public keyless sources, PAPER ONLY)")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self, symbols: list[str]) -> dict[str, Any]:
        """Refresh sentiment for ``symbols`` (+ one GLOBAL row). NEVER raises.

        Gathers the shared market context once (Fear & Greed, hot reddit
        titles, general market headlines), fetches per-coin Google News
        headlines, scores the coins in batches of :data:`_BATCH_SIZE` per
        LLM call, persists one ``coin_sentiment`` row per successfully
        scored coin and finally writes the derived ``'GLOBAL'`` row (see the
        module docstring for the derivation).

        Args:
            symbols: Exchange symbols to score, e.g. ``["BTCUSDT", ...]``.

        Returns:
            Summary dict: ``status`` (``ok|busy|empty|error``), ``requested``,
            ``scored``, ``batches``, ``failed_batches``, ``global`` (persisted
            GLOBAL row dict or None), ``fear_greed``, ``model`` and
            ``finished_at``.
        """
        if not _RUN_LOCK.acquire(blocking=False):
            logger.info("Sentiment run skipped — a refresh is already running")
            return {"status": "busy", "reason": "a sentiment refresh is already running"}
        try:
            try:
                return self._run_locked(symbols)
            except Exception as exc:  # noqa: BLE001 — run must never raise
                logger.exception("Sentiment run crashed — recorded and swallowed")
                return {
                    "status": "error",
                    "error": str(exc),
                    "requested": len(symbols or []),
                    "scored": 0,
                    "batches": 0,
                    "failed_batches": 0,
                    "global": None,
                    "fear_greed": None,
                    "model": settings.ollama_model,
                    "finished_at": _ms_to_iso(utc_now_ms()),
                }
        finally:
            _RUN_LOCK.release()

    def get_latest(self, symbol: str | None = None) -> dict[str, Any]:
        """Most recent sentiment row per symbol (including ``'GLOBAL'``).

        Args:
            symbol: Restrict to one symbol (case-insensitive); ``None``
                returns the latest row of every symbol ever scored.

        Returns:
            ``{"items": [...]}`` — each item JSON-safe with keys ``symbol``,
            ``ts`` (ISO-8601), ``sentiment``, ``score``, ``confidence``,
            ``summary``, ``headlines`` (list of dicts) and ``model``.
        """
        query = (
            "SELECT * FROM coin_sentiment cs WHERE cs.id = ("
            "  SELECT c2.id FROM coin_sentiment c2 WHERE c2.symbol = cs.symbol"
            "  ORDER BY c2.ts DESC, c2.id DESC LIMIT 1)"
        )
        params: list[Any] = []
        if symbol is not None:
            query += " AND cs.symbol = ?"
            params.append(str(symbol).strip().upper())
        query += " ORDER BY cs.symbol"
        with _connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return {"items": [self._row_to_item(row) for row in rows]}

    # ------------------------------------------------------------------ #
    # Run internals                                                        #
    # ------------------------------------------------------------------ #

    def _run_locked(self, symbols: list[str]) -> dict[str, Any]:
        """The actual refresh body; lock already held."""
        requested: list[str] = []
        for item in symbols or []:
            cleaned = str(item).strip().upper()
            if cleaned and cleaned != GLOBAL_SYMBOL and cleaned not in requested:
                requested.append(cleaned)
        model = settings.ollama_model
        if not requested:
            logger.warning("Sentiment run called with no symbols — nothing to do")
            return {
                "status": "empty",
                "requested": 0,
                "scored": 0,
                "batches": 0,
                "failed_batches": 0,
                "global": None,
                "fear_greed": fetch_fear_greed(),
                "model": model,
                "finished_at": _ms_to_iso(utc_now_ms()),
            }

        logger.info("Sentiment refresh started for %d symbols", len(requested))

        # Shared market context — fetched ONCE per run (all cached 10 min).
        fear_greed = fetch_fear_greed()
        reddit_posts = fetch_reddit_hot()
        market_news = fetch_google_news("crypto market", limit=_COIN_HEADLINES)

        # Per-coin headlines by coin name (e.g. "Bitcoin BTC" -> "+crypto").
        headlines_by_symbol: dict[str, list[dict[str, Any]]] = {}
        for sym in requested:
            name, ticker = _coin_display(sym)
            query = name if name == ticker else f"{name} {ticker}"
            headlines_by_symbol[sym] = fetch_google_news(query, limit=_COIN_HEADLINES)

        context_block = self._context_block(fear_greed, reddit_posts, market_news)

        batches = [
            requested[i : i + _BATCH_SIZE]
            for i in range(0, len(requested), _BATCH_SIZE)
        ]
        scored: list[dict[str, Any]] = []
        failed_batches = 0
        for batch in batches:
            results = self._score_batch(
                batch, headlines_by_symbol, context_block, model
            )
            if results is None:
                failed_batches += 1
                continue
            scored.extend(results)

        global_row = self._persist_global(
            scored, fear_greed, market_news, model if scored else "derived"
        )

        self._prune_old_rows()

        try:
            with _connect() as conn:
                _state_set(conn, _LAST_REFRESH_KEY, utc_now_ms())
        except Exception:  # noqa: BLE001 — bookkeeping is best-effort
            logger.exception("Could not persist intel_last_refresh")

        logger.info(
            "Sentiment refresh finished: %d/%d coins scored, %d/%d batches failed",
            len(scored),
            len(requested),
            failed_batches,
            len(batches),
        )
        return {
            "status": "ok",
            "requested": len(requested),
            "scored": len(scored),
            "batches": len(batches),
            "failed_batches": failed_batches,
            "global": global_row,
            "fear_greed": fear_greed,
            "model": model,
            "finished_at": _ms_to_iso(utc_now_ms()),
        }

    def _context_block(
        self,
        fear_greed: dict[str, Any] | None,
        reddit_posts: list[dict[str, Any]],
        market_news: list[dict[str, Any]],
    ) -> str:
        """Render the shared market-context section of every batch prompt."""
        lines: list[str] = ["MARKET CONTEXT"]
        if fear_greed is not None:
            lines.append(
                f"Crypto Fear & Greed index: {fear_greed['value']} "
                f"({fear_greed['classification']})"
            )
        else:
            lines.append("Crypto Fear & Greed index: unavailable")
        if reddit_posts:
            lines.append("Hot r/CryptoCurrency post titles:")
            lines.extend(
                f"- {post['title']}"
                for post in reddit_posts[:_REDDIT_TITLES_IN_PROMPT]
            )
        else:
            lines.append("Hot r/CryptoCurrency post titles: unavailable")
        if market_news:
            lines.append("General crypto market headlines:")
            lines.extend(self._headline_line(h) for h in market_news)
        else:
            lines.append("General crypto market headlines: unavailable")
        return "\n".join(lines)

    @staticmethod
    def _headline_line(headline: dict[str, Any]) -> str:
        """One compact prompt line for a headline dict."""
        title = str(headline.get("title") or "").strip()
        source = str(headline.get("source") or "").strip()
        return f"- {title} ({source})" if source else f"- {title}"

    def _batch_prompt(
        self,
        batch: list[str],
        headlines_by_symbol: dict[str, list[dict[str, Any]]],
        context_block: str,
    ) -> str:
        """Build the user prompt scoring one batch of up to 5 coins."""
        lines: list[str] = [context_block, ""]
        lines.append(
            "COINS TO SCORE — return exactly one results entry per coin, "
            f"using these exact symbol strings: {', '.join(batch)}"
        )
        for sym in batch:
            name, ticker = _coin_display(sym)
            lines.append(f"\n### {sym} — {name} ({ticker})")
            headlines = headlines_by_symbol.get(sym) or []
            if headlines:
                lines.extend(self._headline_line(h) for h in headlines)
            else:
                lines.append(
                    "(no recent coin-specific headlines found — score from the "
                    "market context only, with low confidence)"
                )
        lines.append(
            "\nRespond ONLY with the JSON object described in the system prompt."
        )
        return "\n".join(lines)

    def _score_batch(
        self,
        batch: list[str],
        headlines_by_symbol: dict[str, list[dict[str, Any]]],
        context_block: str,
        model: str,
    ) -> list[dict[str, Any]] | None:
        """Score one batch via a single LLM call and persist its rows.

        Args:
            batch: Up to :data:`_BATCH_SIZE` symbols.
            headlines_by_symbol: Per-coin headline lists (prompt + storage).
            context_block: Shared market-context prompt section.
            model: Model name recorded on the persisted rows.

        Returns:
            The list of persisted per-coin dicts, or ``None`` when the whole
            batch failed (LLM error / unusable JSON) and was skipped.
        """
        prompt = self._batch_prompt(batch, headlines_by_symbol, context_block)
        try:
            raw = self._client.chat(prompt, system=_SYSTEM_PROMPT, json_mode=True)
            payload = json.loads(raw)
        except OllamaError as exc:
            logger.warning("Sentiment batch %s failed (LLM): %s", batch, exc)
            return None
        except (TypeError, json.JSONDecodeError) as exc:
            logger.warning("Sentiment batch %s returned bad JSON: %s", batch, exc)
            return None

        entries = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            logger.warning(
                "Sentiment batch %s response has no 'results' array — skipped", batch
            )
            return None

        # Accept exact symbols plus ticker/name aliases (models drop suffixes).
        aliases: dict[str, str] = {}
        for sym in batch:
            name, ticker = _coin_display(sym)
            aliases[sym] = sym
            aliases.setdefault(ticker.upper(), sym)
            aliases.setdefault(name.upper(), sym)

        by_symbol: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            sym = aliases.get(str(entry.get("symbol") or "").strip().upper())
            if sym is not None and sym not in by_symbol:
                by_symbol[sym] = entry

        results: list[dict[str, Any]] = []
        ts = utc_now_ms()
        with _connect() as conn:
            for sym in batch:
                entry = by_symbol.get(sym)
                if entry is None:
                    logger.warning(
                        "Sentiment batch response missing %s — coin skipped", sym
                    )
                    continue
                record = self._normalise_entry(sym, entry)
                record["headlines"] = headlines_by_symbol.get(sym) or []
                record["model"] = model
                record["ts"] = ts
                self._insert_row(conn, record)
                results.append(record)
        return results

    @staticmethod
    def _normalise_entry(symbol: str, entry: dict[str, Any]) -> dict[str, Any]:
        """Clamp/repair one model result entry into a persistable record.

        ``score`` → [-100, 100] and ``confidence`` → [0, 100] (junk values
        become 0 / 50 respectively). An invalid ``sentiment`` label is
        re-derived from the score sign; a missing score with a valid label
        gets a representative score (±50 / 0) so the two never contradict.
        """
        raw_sentiment = str(entry.get("sentiment") or "").strip().lower()
        has_valid_sentiment = raw_sentiment in _VALID_SENTIMENTS

        default_score = 0
        if has_valid_sentiment:
            default_score = {"bullish": 50, "bearish": -50, "neutral": 0}[raw_sentiment]
        score = _clamp_int(entry.get("score"), -100, 100, default=default_score)
        sentiment = (
            raw_sentiment
            if has_valid_sentiment
            else _sentiment_from_score(score, _SCORE_SENTIMENT_THRESHOLD)
        )
        confidence = _clamp_int(entry.get("confidence"), 0, 100, default=50)
        summary = str(entry.get("summary") or "").strip()
        return {
            "symbol": symbol,
            "sentiment": sentiment,
            "score": score,
            "confidence": confidence,
            "summary": summary,
        }

    def _persist_global(
        self,
        scored: list[dict[str, Any]],
        fear_greed: dict[str, Any] | None,
        market_news: list[dict[str, Any]],
        model: str,
    ) -> dict[str, Any] | None:
        """Derive and persist the market-wide ``'GLOBAL'`` row (best-effort).

        Derivation (documented choice — no extra LLM call): the score is the
        mean of (a) the average per-coin LLM score of this run and (b) the
        Fear & Greed index mapped onto [-100, 100]; whichever of the two is
        available carries the value alone. No coins scored AND no index →
        no GLOBAL row.

        Returns:
            The persisted GLOBAL record dict, or ``None`` when skipped/failed.
        """
        components: list[float] = []
        if scored:
            components.append(sum(r["score"] for r in scored) / len(scored))
        if fear_greed is not None:
            components.append((float(fear_greed["value"]) - 50.0) * 2.0)
        if not components:
            logger.warning(
                "No coin scores and no Fear & Greed reading — GLOBAL row skipped"
            )
            return None

        score = _clamp_int(sum(components) / len(components), -100, 100, default=0)
        sentiment = _sentiment_from_score(score, _GLOBAL_SENTIMENT_THRESHOLD)
        if scored:
            confidence = _clamp_int(
                sum(r["confidence"] for r in scored) / len(scored),
                0,
                100,
                default=_GLOBAL_FALLBACK_CONFIDENCE,
            )
        else:
            confidence = _GLOBAL_FALLBACK_CONFIDENCE

        parts: list[str] = []
        if fear_greed is not None:
            parts.append(
                f"Fear & Greed at {fear_greed['value']} "
                f"({fear_greed['classification']})"
            )
        if scored:
            n_bull = sum(1 for r in scored if r["sentiment"] == "bullish")
            n_bear = sum(1 for r in scored if r["sentiment"] == "bearish")
            parts.append(
                f"{n_bull} of {len(scored)} scored coins bullish, {n_bear} bearish"
            )
        summary = (
            f"{'; '.join(parts)} — overall {sentiment} market mood."
            if parts
            else f"Overall {sentiment} market mood."
        )

        record = {
            "symbol": GLOBAL_SYMBOL,
            "sentiment": sentiment,
            "score": score,
            "confidence": confidence,
            "summary": summary,
            "headlines": market_news,
            "model": model,
            "ts": utc_now_ms(),
        }
        try:
            with _connect() as conn:
                self._insert_row(conn, record)
        except Exception:  # noqa: BLE001 — GLOBAL row is best-effort
            logger.exception("Could not persist the GLOBAL sentiment row")
            return None
        return record

    # ------------------------------------------------------------------ #
    # Persistence helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prune_old_rows() -> None:
        """Delete ``coin_sentiment`` rows older than :data:`_RETENTION_DAYS`.

        Called after each refresh so the table stays bounded (roughly
        ``retention_days * runs_per_day * (coins + 1)`` rows) instead of
        growing forever. Best-effort: any failure is logged and swallowed —
        pruning must never break a refresh.
        """
        cutoff = utc_now_ms() - _RETENTION_DAYS * 86_400_000
        try:
            with _connect() as conn:
                deleted = conn.execute(
                    "DELETE FROM coin_sentiment WHERE ts < ?", (cutoff,)
                ).rowcount
            if deleted > 0:
                logger.info(
                    "Pruned %d coin_sentiment rows older than %d days",
                    deleted,
                    _RETENTION_DAYS,
                )
        except Exception:  # noqa: BLE001 — pruning is best-effort
            logger.exception("Could not prune old coin_sentiment rows")

    @staticmethod
    def _insert_row(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
        """Insert one ``coin_sentiment`` row from a normalised record."""
        conn.execute(
            "INSERT INTO coin_sentiment "
            "(ts, symbol, sentiment, score, confidence, summary, headlines, model) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(record["ts"]),
                str(record["symbol"]),
                str(record["sentiment"]),
                int(record["score"]),
                int(record["confidence"]),
                str(record["summary"]),
                json.dumps(record["headlines"], default=str),
                str(record["model"]),
            ),
        )

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a ``coin_sentiment`` DB row into a JSON-safe dict."""
        try:
            headlines = json.loads(row["headlines"])
        except (TypeError, json.JSONDecodeError):
            headlines = []
        if not isinstance(headlines, list):
            headlines = []
        return {
            "symbol": row["symbol"],
            "ts": _ms_to_iso(int(row["ts"])),
            "sentiment": row["sentiment"],
            "score": int(row["score"]),
            "confidence": int(row["confidence"]),
            "summary": row["summary"],
            "headlines": headlines,
            "model": row["model"],
        }
