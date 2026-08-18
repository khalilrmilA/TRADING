"""AI calibration report — joins AI entry verdicts to trade outcomes.

Answers the question the platform currently cannot: does the AI gate add value,
and does higher AI confidence actually mean better trades?

Reads (read-only) from the bot_activity audit log and paper_positions:
  * ``enter`` rows carry the primary AI payload (sentiment/confidence) and the
    second-judge payload; ``exit`` rows carry pnl / realized_r. There is no
    shared id, so enter->exit pairing is FIFO per symbol by timestamp (the
    AutoTrader holds at most one position per symbol).
  * ``skip`` rows with ai_gate / judge_disagreement reasons carry the rejected
    verdicts, so the gate's veto behaviour is measurable too.
  * ``ai_analyses`` gives per-model call volume — including silent
    fallback-model substitution (the ``model`` column records who really
    answered).

Usage:
    python scripts/calibration_report.py [--db PATH] [--json PATH]

Stdlib only — safe to run with any Python >= 3.10, no venv required.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "trading.db"

CONFIDENCE_BUCKETS = [(0, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]


def _iso(ms: int | None) -> str:
    if not ms:
        return "?"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _load_activity(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT ts, symbol, action, detail FROM bot_activity ORDER BY ts ASC, id ASC"
    ).fetchall()
    out = []
    for r in rows:
        try:
            detail = json.loads(r["detail"] or "{}")
        except (TypeError, ValueError):
            detail = {}
        out.append({"ts": r["ts"], "symbol": r["symbol"], "action": r["action"], "detail": detail})
    return out


def pair_trades(activity: list[dict]) -> list[dict]:
    """FIFO-match each AutoTrader ``enter`` to the next ``exit`` on the same symbol."""
    open_by_symbol: dict[str, deque] = defaultdict(deque)
    trades: list[dict] = []
    for row in activity:
        if row["action"] == "enter":
            open_by_symbol[row["symbol"]].append(row)
        elif row["action"] == "exit":
            if open_by_symbol[row["symbol"]]:
                enter = open_by_symbol[row["symbol"]].popleft()
                d_in, d_out = enter["detail"], row["detail"]
                trades.append(
                    {
                        "symbol": row["symbol"],
                        "entered": enter["ts"],
                        "exited": row["ts"],
                        "side": d_in.get("side"),
                        "ai": d_in.get("ai"),
                        "ai_second": d_in.get("ai_second"),
                        "vote_sum": d_in.get("vote_sum"),
                        "reason": d_out.get("reason"),
                        "pnl": d_out.get("pnl"),
                        "pnl_pct": d_out.get("pnl_pct"),
                        "realized_r": d_out.get("realized_r"),
                        "designed_r": d_out.get("designed_r"),
                    }
                )
            else:
                trades.append({"symbol": row["symbol"], "unmatched_exit": True, "exited": row["ts"]})
    # Entries still open (never exited) stay useful for funnel counts but not outcomes.
    still_open = sum(len(q) for q in open_by_symbol.values())
    return trades, still_open


def _bucket(conf: int | None) -> str:
    if conf is None:
        return "no-AI"
    for lo, hi in CONFIDENCE_BUCKETS:
        if lo <= conf < hi:
            return f"{lo}-{min(hi - 1, 100)}"
    return "invalid"


def _agg(trades: list[dict], key_fn) -> dict[str, dict]:
    groups: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "wins": 0, "pnl": 0.0, "r_sum": 0.0, "r_n": 0}
    )
    for t in trades:
        if t.get("unmatched_exit") or t.get("pnl") is None:
            continue
        g = groups[key_fn(t)]
        g["n"] += 1
        g["wins"] += 1 if t["pnl"] > 0 else 0
        g["pnl"] += t["pnl"]
        if t.get("realized_r") is not None:
            g["r_sum"] += t["realized_r"]
            g["r_n"] += 1
    for g in groups.values():
        g["win_rate"] = round(100 * g["wins"] / g["n"], 1) if g["n"] else None
        g["avg_r"] = round(g["r_sum"] / g["r_n"], 3) if g["r_n"] else None
        g["pnl"] = round(g["pnl"], 2)
        del g["r_sum"], g["r_n"]
    return dict(groups)


def skip_funnel(activity: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in activity:
        if row["action"] != "skip":
            continue
        reason = str(row["detail"].get("reason", "unknown"))
        counts[reason.split(":", 1)[0].strip()] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def model_health(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT model, COUNT(*) AS n, AVG(confidence) AS avg_conf,
                  SUM(CASE WHEN sentiment='neutral' THEN 1 ELSE 0 END) AS neutral,
                  SUM(CASE WHEN sentiment='bullish' THEN 1 ELSE 0 END) AS bullish,
                  SUM(CASE WHEN sentiment='bearish' THEN 1 ELSE 0 END) AS bearish,
                  MIN(created_at) AS first_ts, MAX(created_at) AS last_ts
           FROM ai_analyses GROUP BY model ORDER BY n DESC"""
    ).fetchall()
    return [
        {
            "model": r["model"],
            "calls": r["n"],
            "avg_confidence": round(r["avg_conf"], 1) if r["avg_conf"] is not None else None,
            "bullish": r["bullish"],
            "bearish": r["bearish"],
            "neutral": r["neutral"],
            "first": _iso(r["first_ts"]),
            "last": _iso(r["last_ts"]),
        }
        for r in rows
    ]


def scalper_summary(activity: list[dict]) -> dict:
    exits = [r["detail"] for r in activity if r["action"] == "scalp_exit"]
    if not exits:
        return {"n": 0}
    by_reason: dict[str, dict] = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    total = {"n": 0, "wins": 0, "pnl": 0.0}
    for d in exits:
        pnl = d.get("pnl") or 0.0
        total["n"] += 1
        total["wins"] += 1 if pnl > 0 else 0
        total["pnl"] += pnl
        g = by_reason[str(d.get("reason", "unknown"))]
        g["n"] += 1
        g["pnl"] += pnl
    total["pnl"] = round(total["pnl"], 2)
    total["win_rate"] = round(100 * total["wins"] / total["n"], 1)
    return {**total, "by_reason": {k: {"n": v["n"], "pnl": round(v["pnl"], 2)} for k, v in by_reason.items()}}


def build_report(db_path: Path) -> dict:
    conn = _connect(db_path)
    try:
        activity = _load_activity(conn)
        trades, still_open = pair_trades(activity)
        closed = [t for t in trades if not t.get("unmatched_exit") and t.get("pnl") is not None]

        judged = [t for t in closed if t.get("ai_second")]
        unjudged = [t for t in closed if t.get("ai") and not t.get("ai_second")]

        positions = conn.execute(
            "SELECT COUNT(*) AS n, SUM(pnl) AS pnl FROM paper_positions WHERE status='closed'"
        ).fetchone()

        report = {
            "db": str(db_path),
            "generated_utc": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "inventory": {
                "activity_rows": len(activity),
                "auto_entries": sum(1 for r in activity if r["action"] == "enter"),
                "auto_exits": sum(1 for r in activity if r["action"] == "exit"),
                "closed_pairs": len(closed),
                "open_unmatched_entries": still_open,
                "paper_positions_closed": positions["n"],
                "paper_positions_pnl": round(positions["pnl"], 2) if positions["pnl"] else 0.0,
                "window": f"{_iso(activity[0]['ts'])} .. {_iso(activity[-1]['ts'])}" if activity else "empty",
            },
            "gate_funnel": skip_funnel(activity),
            "calibration_by_confidence": _agg(
                closed, lambda t: _bucket((t.get("ai") or {}).get("confidence"))
            ),
            "by_side": _agg(closed, lambda t: t.get("side") or "?"),
            "by_exit_reason": _agg(closed, lambda t: t.get("reason") or "?"),
            "judge": {
                "judged": _agg(judged, lambda t: "judged"),
                "unjudged_ai_only": _agg(unjudged, lambda t: "unjudged"),
            },
            "model_health": model_health(conn),
            "scalper": scalper_summary(activity),
        }
        return report
    finally:
        conn.close()


def print_report(rep: dict) -> None:
    inv = rep["inventory"]
    print(f"# AI Calibration Report — {rep['generated_utc']} UTC")
    print(f"DB: {rep['db']}")
    print(f"\n## Inventory\n"
          f"activity rows: {inv['activity_rows']} | auto entries: {inv['auto_entries']} | "
          f"auto exits: {inv['auto_exits']} | closed enter->exit pairs: {inv['closed_pairs']} | "
          f"still open: {inv['open_unmatched_entries']}\n"
          f"paper_positions closed: {inv['paper_positions_closed']} "
          f"(pnl {inv['paper_positions_pnl']}) | window: {inv['window']}")

    print("\n## Gate funnel (skip reasons)")
    if rep["gate_funnel"]:
        for reason, n in rep["gate_funnel"].items():
            print(f"  {reason:<24} {n}")
    else:
        print("  (no skip rows yet)")

    def _table(title: str, groups: dict) -> None:
        print(f"\n## {title}")
        if not groups:
            print("  (no closed trades yet)")
            return
        print(f"  {'group':<12} {'n':>4} {'win%':>6} {'avg R':>7} {'pnl':>10}")
        for k in sorted(groups):
            g = groups[k]
            print(f"  {k:<12} {g['n']:>4} {str(g['win_rate']):>6} {str(g['avg_r']):>7} {g['pnl']:>10}")

    _table("Calibration by AI confidence", rep["calibration_by_confidence"])
    _table("By side", rep["by_side"])
    _table("By exit reason", rep["by_exit_reason"])
    _table("Second judge — judged entries", rep["judge"]["judged"])
    _table("Second judge — unjudged (AI-only) entries", rep["judge"]["unjudged_ai_only"])

    print("\n## Model health (ai_analyses — who actually answered)")
    if rep["model_health"]:
        for m in rep["model_health"]:
            print(f"  {m['model']:<28} calls={m['calls']:<4} avg_conf={m['avg_confidence']} "
                  f"bull/bear/neutral={m['bullish']}/{m['bearish']}/{m['neutral']} "
                  f"[{m['first']} .. {m['last']}]")
    else:
        print("  (no ai_analyses rows yet)")

    sc = rep["scalper"]
    print("\n## Scalper (LLM-free loop)")
    if sc.get("n"):
        print(f"  closed: {sc['n']} | win%: {sc['win_rate']} | pnl: {sc['pnl']}")
        for reason, g in sc["by_reason"].items():
            print(f"    {reason:<18} n={g['n']:<4} pnl={g['pnl']}")
    else:
        print("  (no scalp_exit rows yet)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--json", type=Path, help="also write the full report as JSON")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"No database at {args.db} — run the app first so it can create data.", file=sys.stderr)
        return 1
    try:
        rep = build_report(args.db)
    except sqlite3.OperationalError as exc:
        print(f"Database not ready ({exc}) — has the backend initialised it?", file=sys.stderr)
        return 1

    print_report(rep)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
