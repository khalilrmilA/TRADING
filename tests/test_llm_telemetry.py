"""LLM call telemetry + prompt versioning tests.

PAPER TRADING ONLY — verifies that every persisted AI verdict is reproducible
and health-measurable:

    * ``OllamaClient.chat`` records ``last_call_stats`` (latency, attempts,
      fallback use, server-reported token/duration counters) without changing
      its signature, return type or retry/fallback behaviour.
    * ``analyze_market`` persists ``prompt_hash`` / ``latency_ms`` /
      ``attempts`` / ``fallback_used`` / ``repair_used`` to ``ai_analyses``,
      migrating older databases at runtime via ``ALTER TABLE ADD COLUMN``.
    * ``scripts/calibration_report.py`` reports fallback share, repair rate
      and average latency per model — and tolerates DBs without the columns.

Test hygiene (mirrors the rest of the suite):
    * NO network — the conftest blocks sockets; the client tests monkeypatch
      ``requests.get``/``requests.post`` inside ``backend.ai.ollama_client``.
    * NO Ollama — analyst tests patch ``OllamaClient.chat`` at the CLASS
      level (as a real function, so the stub binds like the original method
      and can set ``last_model_used`` / ``last_call_stats`` on the instance).
    * Fresh throwaway SQLite DB per test (conftest ``_fresh_db``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import requests

from backend.ai import analyst
from backend.ai.analyst import analyze_market
from backend.ai.ollama_client import OllamaClient
from backend.database.db import get_conn
from config.settings import settings

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_TELEMETRY_COLUMNS = ("prompt_hash", "latency_ms", "attempts", "fallback_used", "repair_used")

#: The ``ai_analyses`` DDL exactly as it was BEFORE the telemetry columns.
_LEGACY_AI_ANALYSES_DDL = """
CREATE TABLE ai_analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      INTEGER NOT NULL,
    model           TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    sentiment       TEXT    NOT NULL CHECK (sentiment IN ('bullish','bearish','neutral')),
    confidence      INTEGER NOT NULL CHECK (confidence BETWEEN 0 AND 100),
    risk_commentary TEXT    NOT NULL DEFAULT '',
    key_indicators  TEXT    NOT NULL DEFAULT '[]',
    reasoning       TEXT    NOT NULL DEFAULT '',
    raw_response    TEXT    NOT NULL DEFAULT ''
)
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _verdict(sentiment: str = "bullish", confidence: int = 72) -> str:
    """A well-formed analyst reply matching the frozen JSON schema."""
    return json.dumps(
        {
            "opposing_case": "Momentum could stall right at the swing high.",
            "sentiment": sentiment,
            "confidence": confidence,
            "risk_commentary": "Volatility is elevated near resistance.",
            "key_indicators": [
                {"name": "rsi_14", "value": "55", "influence": "supportive"}
            ],
            "reasoning": "Trend and momentum align on this timeframe.",
        }
    )


class _ChatScript:
    """Scripted replies + per-call stats for the class-level ``chat`` patch."""

    def __init__(
        self,
        responses: list[str],
        stats: dict[str, Any] | None = None,
        model: str = "stub-model",
    ) -> None:
        self.responses = list(responses)
        self.stats = dict(stats) if stats is not None else None
        self.model = model
        self.calls: list[dict[str, str]] = []


def _patch_chat(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[str],
    stats: dict[str, Any] | None = None,
    model: str = "stub-model",
) -> _ChatScript:
    """Replace ``OllamaClient.chat`` at the CLASS level with a recording stub.

    Installed as a plain function (a descriptor), so the stub receives the
    client instance and can mirror the real ``chat`` by setting
    ``last_model_used`` and ``last_call_stats`` on it. Replies are consumed
    front-to-back; the last one repeats.
    """
    script = _ChatScript(responses, stats=stats, model=model)

    def _chat(
        self: OllamaClient,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        json_mode: bool = True,
        temperature: float = 0.2,
        allow_fallback: bool = True,
    ) -> str:
        script.calls.append({"prompt": str(prompt), "system": str(system or "")})
        self.last_model_used = script.model
        self.last_call_stats = dict(script.stats) if script.stats is not None else None
        if len(script.responses) > 1:
            return script.responses.pop(0)
        return script.responses[0]

    monkeypatch.setattr(OllamaClient, "chat", _chat)
    return script


def _last_row() -> sqlite3.Row:
    """Newest ``ai_analyses`` row (all columns)."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM ai_analyses ORDER BY id DESC LIMIT 1"
        ).fetchone()


def _all_hashes() -> list[str]:
    """All persisted ``prompt_hash`` values, oldest first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT prompt_hash FROM ai_analyses ORDER BY id ASC"
        ).fetchall()
    return [r["prompt_hash"] for r in rows]


def _expected_hash(system_prompt: str, user_prompt: str) -> str:
    """The documented fingerprint: sha256(system + "\\n" + user)[:12]."""
    joined = f"{system_prompt}\n{user_prompt}".encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:12]


def _downgrade_ai_analyses() -> None:
    """Recreate ``ai_analyses`` with its pre-telemetry shape (older DBs)."""
    with get_conn() as conn:
        conn.execute("DROP TABLE ai_analyses")
        conn.execute(_LEGACY_AI_ANALYSES_DDL)


def _load_report_module():
    """Import ``scripts/calibration_report.py`` from its file path."""
    spec = importlib.util.spec_from_file_location(
        "calibration_report_under_test",
        _PROJECT_ROOT / "scripts" / "calibration_report.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    """Minimal ``requests.Response`` stand-in for the client-level tests."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.ok = True

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _patch_http(
    monkeypatch: pytest.MonkeyPatch,
    installed: list[str],
    post_fn,
) -> None:
    """Stub the two HTTP entry points the client uses (tags + chat)."""

    def _get(url: str, timeout: object = None) -> _FakeResponse:
        assert url.endswith("/api/tags")
        return _FakeResponse({"models": [{"name": name} for name in installed]})

    monkeypatch.setattr("backend.ai.ollama_client.requests.get", _get)
    monkeypatch.setattr("backend.ai.ollama_client.requests.post", post_fn)
    monkeypatch.setattr("backend.ai.ollama_client.time.sleep", lambda _s: None)


# --------------------------------------------------------------------------- #
# OllamaClient.chat — last_call_stats (requests mocked, no sockets)           #
# --------------------------------------------------------------------------- #


def test_chat_records_stats_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean first-attempt reply records latency/attempts and eval counters."""
    monkeypatch.setattr(settings, "ollama_fallback_model", "backup:1b")

    def _post(url: str, json: dict | None = None, timeout: object = None) -> _FakeResponse:
        return _FakeResponse(
            {
                "message": {"content": '{"answer": "ok"}'},
                "eval_count": 42,
                "prompt_eval_count": 7,
                "eval_duration": 111_000_000,
                "total_duration": 222_000_000,
            }
        )

    _patch_http(monkeypatch, ["primary:1b", "backup:1b"], _post)
    client = OllamaClient(base_url="http://localhost:11434", timeout=5.0, retries=1)

    reply = client.chat("hello", system="sys", model="primary:1b")

    assert reply == '{"answer": "ok"}'
    assert client.last_model_used == "primary:1b"
    stats = client.last_call_stats
    assert stats is not None
    assert isinstance(stats["latency_ms"], int) and stats["latency_ms"] >= 0
    assert stats["attempts"] == 1
    assert stats["fallback_used"] is False
    assert stats["eval_count"] == 42
    assert stats["prompt_eval_count"] == 7
    assert stats["eval_duration"] == 111_000_000
    assert stats["total_duration"] == 222_000_000


def test_chat_records_retries_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Primary exhausts its attempts, fallback answers: counters reflect it."""
    monkeypatch.setattr(settings, "ollama_fallback_model", "backup:1b")

    def _post(url: str, json: dict | None = None, timeout: object = None) -> _FakeResponse:
        if json["model"] == "primary:1b":
            raise requests.RequestException("connection refused (synthetic)")
        return _FakeResponse({"message": {"content": '{"answer": "fallback"}'}})

    _patch_http(monkeypatch, ["primary:1b", "backup:1b"], _post)
    client = OllamaClient(base_url="http://localhost:11434", timeout=5.0, retries=1)

    reply = client.chat("hello", model="primary:1b")

    assert reply == '{"answer": "fallback"}'
    assert client.last_model_used == "backup:1b"
    stats = client.last_call_stats
    assert stats["attempts"] == 3  # 2 primary (1 + retries=1) + 1 fallback
    assert stats["fallback_used"] is True
    # The fallback reply carried no counters — none must leak into the stats.
    assert "eval_count" not in stats


# --------------------------------------------------------------------------- #
# analyze_market — persisted telemetry (OllamaClient.chat mocked at the class) #
# --------------------------------------------------------------------------- #


def test_persists_telemetry_columns(
    monkeypatch: pytest.MonkeyPatch, sample_df
) -> None:
    """The new ai_analyses columns are written from the client call stats."""
    script = _patch_chat(
        monkeypatch,
        [_verdict()],
        stats={"latency_ms": 1234, "attempts": 2, "fallback_used": True},
        model="backup:1b",
    )

    result = analyze_market("BTCUSDT", "1h", sample_df)

    assert result.sentiment == "bullish"
    row = _last_row()
    assert row["latency_ms"] == 1234
    assert row["attempts"] == 2
    assert row["fallback_used"] == 1
    assert row["repair_used"] == 0
    assert row["model"] == "backup:1b"
    expected = _expected_hash(analyst._SYSTEM_PROMPT, script.calls[0]["prompt"])
    assert row["prompt_hash"] == expected
    assert len(row["prompt_hash"]) == 12
    assert set(row["prompt_hash"]) <= set("0123456789abcdef")


def test_prompt_hash_stable_for_identical_inputs(
    monkeypatch: pytest.MonkeyPatch, sample_df
) -> None:
    """Identical symbol/timeframe/frame → identical prompt_hash."""
    _patch_chat(monkeypatch, [_verdict()])

    analyze_market("BTCUSDT", "1h", sample_df)
    analyze_market("BTCUSDT", "1h", sample_df)

    hashes = _all_hashes()
    assert len(hashes) == 2
    assert hashes[0] == hashes[1]


def test_prompt_hash_changes_when_prompt_changes(
    monkeypatch: pytest.MonkeyPatch, sample_df
) -> None:
    """Any prompt change (here: an htf_context block) changes the hash."""
    _patch_chat(monkeypatch, [_verdict()])

    analyze_market("BTCUSDT", "1h", sample_df)
    analyze_market(
        "BTCUSDT", "1h", sample_df,
        htf_context="Price is 2% above the 4h EMA200.",
    )

    hashes = _all_hashes()
    assert len(hashes) == 2
    assert hashes[0] != hashes[1]


def test_missing_call_stats_degrade_to_nulls(
    monkeypatch: pytest.MonkeyPatch, sample_df
) -> None:
    """A chat without stats (legacy stub) persists NULLs, never an error."""
    _patch_chat(monkeypatch, [_verdict()], stats=None)

    analyze_market("BTCUSDT", "1h", sample_df)

    row = _last_row()
    assert row["latency_ms"] is None
    assert row["attempts"] is None
    assert row["fallback_used"] == 0
    assert row["repair_used"] == 0
    assert len(row["prompt_hash"]) == 12  # the fingerprint never degrades


def test_repair_round_trip_sets_repair_used(
    monkeypatch: pytest.MonkeyPatch, sample_df
) -> None:
    """A malformed first reply fires the self-repair path: repair_used=1 and
    latency/attempts accumulate over both round-trips."""
    script = _patch_chat(
        monkeypatch,
        ["this is not JSON ][", _verdict(sentiment="bearish", confidence=61)],
        stats={"latency_ms": 100, "attempts": 1, "fallback_used": False},
    )

    result = analyze_market("BTCUSDT", "1h", sample_df)

    assert result.sentiment == "bearish"
    assert len(script.calls) == 2  # original + repair round-trip
    assert "previous reply" in script.calls[1]["prompt"].lower()
    row = _last_row()
    assert row["repair_used"] == 1
    assert row["latency_ms"] == 200  # both calls
    assert row["attempts"] == 2
    assert row["fallback_used"] == 0
    # The fingerprint stays that of the ORIGINAL analysis prompt.
    expected = _expected_hash(analyst._SYSTEM_PROMPT, script.calls[0]["prompt"])
    assert row["prompt_hash"] == expected


def test_runtime_migration_adds_columns_to_old_db(
    monkeypatch: pytest.MonkeyPatch, sample_df
) -> None:
    """Persisting into a pre-telemetry DB ALTERs the missing columns in."""
    _downgrade_ai_analyses()
    with get_conn() as conn:
        before = {r[1] for r in conn.execute("PRAGMA table_info(ai_analyses)")}
    assert not (set(_TELEMETRY_COLUMNS) & before)

    _patch_chat(monkeypatch, [_verdict()], stats={"latency_ms": 55, "attempts": 1})
    analyze_market("BTCUSDT", "1h", sample_df)

    with get_conn() as conn:
        after = {r[1] for r in conn.execute("PRAGMA table_info(ai_analyses)")}
    assert set(_TELEMETRY_COLUMNS) <= after
    row = _last_row()
    assert row["latency_ms"] == 55
    assert len(row["prompt_hash"]) == 12


# --------------------------------------------------------------------------- #
# scripts/calibration_report.py — model health telemetry                      #
# --------------------------------------------------------------------------- #


def test_report_tolerates_db_without_telemetry_columns(capsys) -> None:
    """An old DB (no telemetry columns) still builds and prints a report."""
    _downgrade_ai_analyses()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO ai_analyses "
            "(created_at, model, symbol, timeframe, sentiment, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (1_700_000_000_000, "qwen3:14b", "BTCUSDT", "1h", "bullish", 70),
        )

    report_mod = _load_report_module()
    rep = report_mod.build_report(settings.db_path)

    health = rep["model_health"]
    assert len(health) == 1
    assert health[0]["model"] == "qwen3:14b"
    assert health[0]["calls"] == 1
    assert "fallback_share" not in health[0]  # columns absent → no telemetry

    report_mod.print_report(rep)  # must not raise on the old shape
    out = capsys.readouterr().out
    assert "Model health" in out
    assert "qwen3:14b" in out


def test_report_includes_telemetry_on_new_db(
    monkeypatch: pytest.MonkeyPatch, sample_df, capsys
) -> None:
    """A telemetry-aware DB yields fallback share, repair rate and latency."""
    _patch_chat(
        monkeypatch,
        [_verdict()],
        stats={"latency_ms": 100, "attempts": 1, "fallback_used": True},
        model="backup:1b",
    )
    analyze_market("BTCUSDT", "1h", sample_df)
    analyze_market("ETHUSDT", "1h", sample_df)

    report_mod = _load_report_module()
    rep = report_mod.build_report(settings.db_path)

    health = {m["model"]: m for m in rep["model_health"]}
    entry = health["backup:1b"]
    assert entry["calls"] == 2
    assert entry["fallback_share"] == 100.0
    assert entry["repair_rate"] == 0.0
    assert entry["avg_latency_ms"] == 100

    report_mod.print_report(rep)
    out = capsys.readouterr().out
    assert "fallback%=100.0" in out
    assert "avg_latency_ms=100" in out
