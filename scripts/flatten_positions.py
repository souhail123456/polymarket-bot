#!/usr/bin/env python3
"""Manually flatten (close out) all OPEN paper positions across every bot's
JSONL datastore. Paper-only, reversible, non-destructive.

An "open" paper trade is one where `resolved` is falsy. This script moves such
trades to a CLOSED state WITHOUT deleting any history and WITHOUT faking a
market outcome:

    resolved      -> True          (so bots stop counting it as open exposure)
    manual_flatten-> True          (marker: this was a manual flatten, not a real resolution)
    close_reason  -> "manual_flatten"
    closed_at     -> ISO timestamp
    realized_pnl  -> 0.0           (no live current mark available -> close at entry, 0 P&L)
    won           -> None          (NOT a real win/loss; excluded from win counts, keeps stats honest)

The script is idempotent: a trade already carrying `manual_flatten=True` (or any
already-resolved trade) is left untouched.

To REVERSE: delete the five injected fields and set `resolved` back to False for
every trade with `manual_flatten=True`. See scripts/unflatten_positions.py.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# Every datastore that can hold open paper positions.
DATASTORES = [
    LOG_DIR / "trades.jsonl",          # EV bot
    LOG_DIR / "weather_trades.jsonl",  # Weather bot
    LOG_DIR / "econ_trades.jsonl",     # Econ bot
    LOG_DIR / "crypto_trades.jsonl",   # Crypto bot (may not exist)
]

CLOSE_REASON = "manual_flatten"


def flatten_file(path: Path) -> tuple[int, int]:
    """Return (closed_now, still_open_after)."""
    if not path.exists():
        return (0, 0)

    lines = [l for l in path.read_text().splitlines() if l.strip()]
    trades = [json.loads(l) for l in lines]

    now = datetime.now(timezone.utc).isoformat()
    closed_now = 0
    for t in trades:
        if t.get("resolved"):
            continue  # already closed/resolved
        if t.get("manual_flatten"):
            continue  # already flattened (idempotent guard)
        t["resolved"] = True
        t["manual_flatten"] = True
        t["close_reason"] = CLOSE_REASON
        t["closed_at"] = now
        t["realized_pnl"] = 0.0
        t["won"] = None
        closed_now += 1

    if closed_now:
        # Rewrite atomically-ish: write full file back.
        path.write_text("\n".join(json.dumps(t) for t in trades) + "\n")

    still_open = sum(1 for t in trades if not t.get("resolved"))
    return (closed_now, still_open)


def main() -> int:
    grand_closed = 0
    grand_open = 0
    for path in DATASTORES:
        closed, still_open = flatten_file(path)
        exists = "exists" if path.exists() else "absent"
        print(f"{path.name:32} closed_now={closed:3d}  open_after={still_open:3d}  ({exists})")
        grand_closed += closed
        grand_open += still_open
    print(f"\nTOTAL closed this run: {grand_closed}")
    print(f"TOTAL still open after: {grand_open}")
    if grand_open != 0:
        print("WARNING: some positions remain open", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
