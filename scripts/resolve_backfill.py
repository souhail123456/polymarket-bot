#!/usr/bin/env python3
"""
Resolution backfill script for EV bot trades.

Reads logs/trades.jsonl, finds unresolved positions, queries the Polymarket
gamma API to check if they have resolved, and updates the trade records with
won/lost status and realized P&L.

Usage:
    python scripts/resolve_backfill.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

GAMMA_API = "https://gamma-api.polymarket.com"
TRADES_FILE = Path("logs/trades.jsonl")
STARTING_BANKROLL = 100.0


def load_trades() -> list:
    if not TRADES_FILE.exists():
        print(f"ERROR: {TRADES_FILE} not found")
        sys.exit(1)
    trades = []
    with open(TRADES_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return trades


def save_trades(trades: list):
    with open(TRADES_FILE, "w") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")


def query_market(market_id: str) -> dict | None:
    """Query Polymarket gamma API for market resolution status."""
    url = f"{GAMMA_API}/markets/{market_id}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            print(f"  WARNING: Market {market_id} not found (404)")
        else:
            print(f"  ERROR: HTTP {e.response.status_code if e.response else '?'} for {market_id}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: Request failed for {market_id}: {e}")
        return None


def resolve_trade(trade: dict, market_data: dict) -> bool:
    """
    Check if market has resolved and update trade record.
    Returns True if trade was updated.
    """
    if not market_data.get("closed"):
        return False

    prices = market_data.get("outcomePrices")
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except json.JSONDecodeError:
            return False
    if not prices or len(prices) != 2:
        return False

    yes_resolved = float(prices[0])
    if yes_resolved not in (0.0, 1.0):
        return False  # not cleanly resolved

    won_yes = yes_resolved == 1.0
    side = trade.get("side", "YES")
    won = won_yes if side == "YES" else not won_yes

    entry_price = trade.get("entry_price", 0)
    size_usd = trade.get("size_usd", 0)

    if won:
        shares = size_usd / entry_price if entry_price > 0 else 0
        pnl = (shares * 1.0) - size_usd
    else:
        pnl = -size_usd

    trade["resolved"] = True
    trade["won"] = won
    trade["realized_pnl"] = round(pnl, 4)
    trade["resolved_at"] = datetime.now(timezone.utc).isoformat()
    trade["backfilled"] = True

    return True


def main():
    trades = load_trades()
    total = len(trades)
    unresolved = [i for i, t in enumerate(trades) if not t.get("resolved")]

    print(f"Loaded {total} trades from {TRADES_FILE}")
    print(f"Unresolved: {len(unresolved)}")
    print()

    if not unresolved:
        print("All trades already resolved. Nothing to do.")
        return

    updated = 0
    still_open = 0
    errors = 0

    for idx in unresolved:
        t = trades[idx]
        market_id = t.get("market_id", "???")
        question = t.get("question", "")[:60]
        print(f"[{idx+1}/{total}] Checking {market_id[:12]}... {question}")

        market_data = query_market(market_id)
        if market_data is None:
            errors += 1
            continue

        if resolve_trade(t, market_data):
            status = "WIN" if t["won"] else "LOSS"
            print(f"  RESOLVED: {t['side']} -> {status} | P&L: ${t['realized_pnl']:+.2f}")
            updated += 1
        else:
            print(f"  Still open (not resolved yet)")
            still_open += 1

        # Rate limit: be nice to the API
        time.sleep(0.5)

    # Recalculate running balance for all resolved trades
    all_resolved = [t for t in trades if t.get("resolved")]
    total_pnl = sum(float(t.get("realized_pnl", 0)) for t in all_resolved)
    balance = STARTING_BANKROLL + total_pnl

    # Update balance on the last resolved trade
    for t in reversed(trades):
        if t.get("resolved"):
            t["balance"] = round(balance, 2)
            break

    if updated > 0:
        save_trades(trades)
        print()
        print(f"Updated {updated} trades. Written back to {TRADES_FILE}")
    else:
        print()
        print("No new resolutions found.")

    print(f"\nSummary:")
    print(f"  Resolved this run: {updated}")
    print(f"  Still open:        {still_open}")
    print(f"  Errors:            {errors}")
    print(f"  Total P&L:         ${total_pnl:+.2f}")
    print(f"  Balance:           ${balance:.2f}")


if __name__ == "__main__":
    main()
