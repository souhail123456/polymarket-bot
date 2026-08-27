#!/usr/bin/env python3
"""Reverse scripts/flatten_positions.py: reopen every manually-flattened paper
position, restoring it exactly to its pre-flatten open state.

For each trade with `manual_flatten=True`, this sets `resolved` back to False and
removes the injected fields (manual_flatten, close_reason, closed_at, realized_pnl,
won). Trades that were genuinely resolved by the market are never touched.
Idempotent.
"""
import json
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

DATASTORES = [
    LOG_DIR / "trades.jsonl",
    LOG_DIR / "weather_trades.jsonl",
    LOG_DIR / "econ_trades.jsonl",
    LOG_DIR / "crypto_trades.jsonl",
]

INJECTED = ("manual_flatten", "close_reason", "closed_at", "realized_pnl", "won")


def unflatten_file(path: Path) -> int:
    if not path.exists():
        return 0
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    trades = [json.loads(l) for l in lines]
    reopened = 0
    for t in trades:
        if not t.get("manual_flatten"):
            continue
        for k in INJECTED:
            t.pop(k, None)
        t["resolved"] = False
        reopened += 1
    if reopened:
        path.write_text("\n".join(json.dumps(t) for t in trades) + "\n")
    return reopened


def main() -> int:
    total = 0
    for path in DATASTORES:
        n = unflatten_file(path)
        print(f"{path.name:32} reopened={n}")
        total += n
    print(f"\nTOTAL reopened: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
