"""
Polymarket Strategy Backtest
----------------------------
Pulls historical resolved markets, evaluates them with Claude AS IF trading
24 hours before resolution, and computes what the strategy's P&L would have been.

Purpose: validate (or kill) the strategy hypothesis BEFORE spending a week
paper-trading live.

Strategy being tested:
  - Target: obscure low-liquidity markets ($500-$5K)
  - Time to resolve at trade: 6-72h (we evaluate 24h before resolution)
  - Price range: 0.15-0.85
  - Trade if Claude's probability estimate gives >= 10% EV edge
  - Confidence threshold: 55%
  - Size: quarter-Kelly, capped at $5 per trade
  - Fees: 2% round-trip modeled

Critical caveat: Claude's training cutoff is Jan 2026. To avoid lookahead bias,
we only test on markets that resolved Feb 2026 or later. Even so, some topics
may leak through training data. Treat results with appropriate skepticism.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python backtest.py
  python backtest.py --max-markets 100       # limit API spend
  python backtest.py --from 2026-03-01       # start date filter
  python backtest.py --dry-run               # fetch + filter only, no Claude calls
"""

import os
import json
import time
import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from anthropic import Anthropic

GAMMA_API = "https://gamma-api.polymarket.com"
OUT_DIR = Path("./backtest_out")
OUT_DIR.mkdir(exist_ok=True)

CONFIG = {
    # Strategy thresholds
    "min_edge": 0.10,
    "min_confidence": 0.55,
    "max_position_usd": 5.0,
    "kelly_fraction": 0.25,
    "bankroll": 100.0,
    "fee_round_trip": 0.02,  # 2% total (taker fee + spread cost)

    # Market filters
    "min_liquidity_proxy": 500.0,
    "max_liquidity_proxy": 5000.0,
    "min_hours_to_resolve": 6,
    "max_hours_to_resolve": 72,
    "price_floor": 0.15,
    "price_ceiling": 0.85,

    # Evaluation timing
    "evaluate_hours_before_resolution": 24,

    # Lookahead protection
    "min_resolution_date": "2026-02-01",  # markets must resolve after this

    # Cost control
    "max_markets_to_evaluate": 200,
    "sleep_between_calls_sec": 1.0,

    # Model
    "claude_model": "claude-sonnet-4-6",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "backtest.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ---------- Fetch resolved markets ----------
def fetch_resolved_markets(from_date, limit=500):
    """Pull resolved/closed markets from Gamma API."""
    all_markets = []
    offset = 0
    page_size = 100

    while len(all_markets) < limit:
        try:
            r = requests.get(
                f"{GAMMA_API}/markets",
                params={
                    "closed": "true",
                    "limit": page_size,
                    "offset": offset,
                    "order": "endDate",
                    "ascending": "false",
                },
                timeout=15,
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            all_markets.extend(batch)
            offset += page_size
            time.sleep(0.3)
            # Stop paginating once we're before our date floor
            last_end = batch[-1].get("endDate") or batch[-1].get("end_date_iso")
            if last_end and last_end < from_date:
                break
        except Exception as e:
            log.error(f"Fetch failed at offset {offset}: {e}")
            break

    return all_markets


# ---------- Historical snapshot helpers ----------
def parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def get_price_at_time(market, target_time):
    """
    Approximate the market's YES price at target_time.

    Polymarket's Gamma API doesn't return historical prices directly on this
    endpoint. The CLOB API has price history per token. For simplicity here,
    we use the *current* final outcomePrices stored on the resolved market
    metadata as a fallback, combined with `lastTradePrice` if available.

    HONEST NOTE: this is a known weakness of the backtest. The right fix is to
    call the CLOB price-history endpoint per token and pick the price at
    target_time. That requires the token_id (clobTokenIds on market) and an
    extra API call per market. Implemented below as best-effort; falls back
    to market metadata if CLOB lookup fails.
    """
    clob_token_ids = market.get("clobTokenIds")
    if isinstance(clob_token_ids, str):
        try:
            clob_token_ids = json.loads(clob_token_ids)
        except Exception:
            clob_token_ids = None

    if clob_token_ids and len(clob_token_ids) >= 1:
        yes_token = clob_token_ids[0]
        try:
            # CLOB prices-history endpoint
            ts = int(target_time.timestamp())
            r = requests.get(
                "https://clob.polymarket.com/prices-history",
                params={
                    "market": yes_token,
                    "startTs": ts - 3600,  # 1h window around target
                    "endTs": ts + 3600,
                    "fidelity": 60,
                },
                timeout=10,
            )
            if r.ok:
                data = r.json()
                hist = data.get("history", [])
                if hist:
                    # Find closest timestamp
                    closest = min(hist, key=lambda p: abs(p["t"] - ts))
                    return float(closest["p"])
        except Exception as e:
            log.debug(f"CLOB history lookup failed: {e}")

    # Fallback: skip — we don't trust resolved-market final prices as entry prices
    return None


def liquidity_proxy(market):
    """
    Use volume as a proxy for liquidity since we can't reconstruct historical
    order book depth easily. Not perfect, but correlated.
    """
    vol = market.get("volume") or market.get("volumeNum") or 0
    try:
        return float(vol)
    except Exception:
        return 0.0


# ---------- Market filtering ----------
SKIP_KEYWORDS = ["bitcoin price", "ethereum price", "btc price", "eth price", "solana price"]


def should_skip(market):
    q = (market.get("question") or "").lower()
    return any(kw in q for kw in SKIP_KEYWORDS)


def filter_eligible(markets, cfg):
    """Return markets suitable for backtest evaluation."""
    eligible = []
    min_resolve = parse_iso(cfg["min_resolution_date"] + "T00:00:00+00:00")

    for m in markets:
        try:
            end = parse_iso(m.get("endDate") or m.get("end_date_iso"))
            if not end or end < min_resolve:
                continue

            # Needs clean resolution
            outcome_prices = m.get("outcomePrices")
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)
            if not outcome_prices or len(outcome_prices) != 2:
                continue
            final_yes = float(outcome_prices[0])
            if final_yes not in (0.0, 1.0):
                continue  # ambiguous or 50/50 resolution

            # Liquidity/volume window
            liq = liquidity_proxy(m)
            if not (cfg["min_liquidity_proxy"] <= liq <= cfg["max_liquidity_proxy"]):
                continue

            # Skip obvious sharp-dominated markets
            if should_skip(m):
                continue

            m["_final_yes"] = final_yes
            m["_end_time"] = end
            m["_eval_time"] = end - timedelta(hours=cfg["evaluate_hours_before_resolution"])
            eligible.append(m)
        except Exception as e:
            log.debug(f"Skip filter: {e}")

    return eligible


# ---------- Claude evaluation ----------
PROMPT = """You are a calibrated prediction market analyst. Estimate the TRUE probability this market resolves YES.

MARKET: {question}

DESCRIPTION: {description}

CURRENT MARKET PRICE (crowd's implied probability of YES): {yes_price:.1%}
HOURS UNTIL RESOLUTION: {hours_left:.1f}
RESOLUTION SOURCE: {resolution}

Reason through:
1. What must happen for YES?
2. Base rate for this type of event?
3. Specific evidence relevant right now?
4. Is the crowd biased (recency, narrative, longshot)?

Be calibrated. If you have no real informational edge, your estimate should be close to market price. Most markets are approximately efficient.

Output ONLY this JSON:
{{"true_probability": <0-1>, "confidence": <0-1>, "reasoning": "<one sentence>"}}"""


def estimate(client, market, yes_price_at_eval):
    prompt = PROMPT.format(
        question=(market.get("question") or "")[:500],
        description=(market.get("description") or "")[:1200],
        yes_price=yes_price_at_eval,
        hours_left=CONFIG["evaluate_hours_before_resolution"],
        resolution=(market.get("resolutionSource") or "See description")[:300],
    )
    try:
        resp = client.messages.create(
            model=CONFIG["claude_model"],
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        return float(data["true_probability"]), float(data["confidence"]), data.get("reasoning", "")
    except Exception as e:
        log.warning(f"estimate failed: {e}")
        return None, None, None


# ---------- EV + sizing (duplicate of bot logic so backtest is self-contained) ----------
def expected_value(true_p, market_p, side):
    if side == "YES":
        return true_p * (1 - market_p) - (1 - true_p) * market_p
    return (1 - true_p) * market_p - true_p * (1 - market_p)


def kelly_size(true_p, market_p, side, bankroll, fraction, cap):
    if side == "YES":
        b = (1 - market_p) / market_p if market_p > 0 else 0
        p = true_p
    else:
        b = market_p / (1 - market_p) if market_p < 1 else 0
        p = 1 - true_p
    if b <= 0:
        return 0.0
    q = 1 - p
    kelly_full = (b * p - q) / b
    bet = bankroll * fraction * max(0, kelly_full)
    return min(bet, cap)


def decide(true_p, market_p, confidence, bankroll, cfg):
    if confidence < cfg["min_confidence"]:
        return None, 0.0, 0.0
    ev_yes = expected_value(true_p, market_p, "YES")
    ev_no = expected_value(true_p, market_p, "NO")
    side, edge = ("YES", ev_yes) if ev_yes >= ev_no else ("NO", ev_no)
    if edge < cfg["min_edge"]:
        return None, 0.0, edge
    size = kelly_size(true_p, market_p, side, bankroll, cfg["kelly_fraction"], cfg["max_position_usd"])
    if size < 1.0:
        return None, 0.0, edge
    return side, size, edge


# ---------- P&L computation with fees ----------
def compute_pnl(trade, fee_round_trip):
    """Given a completed trade record, compute realized P&L net of fees."""
    side = trade["side"]
    size = trade["size_usd"]
    entry = trade["entry_price"]
    final_yes = trade["final_yes"]

    won = (side == "YES" and final_yes == 1.0) or (side == "NO" and final_yes == 0.0)
    if won:
        shares = size / entry
        gross_pnl = shares * 1.0 - size  # receive $1 per share, cost was size
    else:
        gross_pnl = -size

    fee_cost = size * fee_round_trip
    return round(gross_pnl - fee_cost, 4), won


# ---------- Main backtest loop ----------
def run_backtest(args):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        log.error("Set ANTHROPIC_API_KEY. Or use --dry-run to skip Claude calls.")
        return

    client = Anthropic(api_key=api_key) if api_key else None
    cfg = CONFIG.copy()
    if args.from_date:
        cfg["min_resolution_date"] = args.from_date
    if args.max_markets:
        cfg["max_markets_to_evaluate"] = args.max_markets

    log.info("Fetching resolved markets...")
    raw = fetch_resolved_markets(cfg["min_resolution_date"], limit=1500)
    log.info(f"Got {len(raw)} resolved markets")

    eligible = filter_eligible(raw, cfg)
    log.info(f"{len(eligible)} pass filters")

    # Cap to max_markets
    eligible = eligible[: cfg["max_markets_to_evaluate"]]
    log.info(f"Evaluating {len(eligible)} markets")

    if args.dry_run:
        log.info("Dry run: skipping Claude evaluation")
        for m in eligible[:10]:
            log.info(f"  {m['question'][:80]!r} final={m['_final_yes']} liq={liquidity_proxy(m):.0f}")
        return

    trades = []
    skipped_no_price = 0
    skipped_no_edge = 0
    evaluated = 0

    for i, m in enumerate(eligible, 1):
        log.info(f"[{i}/{len(eligible)}] {m['question'][:80]!r}")

        entry_price = get_price_at_time(m, m["_eval_time"])
        if entry_price is None:
            skipped_no_price += 1
            log.info("  no historical price available, skipping")
            continue
        if not (cfg["price_floor"] <= entry_price <= cfg["price_ceiling"]):
            log.info(f"  price {entry_price:.2f} outside range, skipping")
            continue

        true_p, conf, reason = estimate(client, m, entry_price)
        if true_p is None:
            continue
        evaluated += 1

        side, size, edge = decide(true_p, entry_price, conf, cfg["bankroll"], cfg)
        log.info(
            f"  mkt={entry_price:.2f} est={true_p:.2f} conf={conf:.2f} "
            f"edge={edge:+.1%} -> {side or 'skip'} ${size:.2f}"
        )

        if side is None:
            skipped_no_edge += 1
            time.sleep(cfg["sleep_between_calls_sec"])
            continue

        trade = {
            "market_id": str(m.get("id")),
            "question": m["question"][:200],
            "eval_time": m["_eval_time"].isoformat(),
            "resolve_time": m["_end_time"].isoformat(),
            "market_price": entry_price,
            "estimated_prob": true_p,
            "confidence": conf,
            "reasoning": reason,
            "side": side,
            "entry_price": entry_price if side == "YES" else 1 - entry_price,
            "size_usd": round(size, 2),
            "edge": round(edge, 4),
            "final_yes": m["_final_yes"],
        }
        pnl, won = compute_pnl(trade, cfg["fee_round_trip"])
        trade["realized_pnl"] = pnl
        trade["won"] = won
        trades.append(trade)
        log.info(f"  -> {'WIN' if won else 'LOSS'} pnl=${pnl:+.2f}")

        time.sleep(cfg["sleep_between_calls_sec"])

    # Save raw trades
    with open(OUT_DIR / "backtest_trades.jsonl", "w") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")

    report(trades, evaluated, skipped_no_edge, skipped_no_price)


# ---------- Reporting ----------
def report(trades, evaluated, skipped_no_edge, skipped_no_price):
    print("\n" + "=" * 60)
    print("BACKTEST REPORT")
    print("=" * 60)
    print(f"Markets evaluated by Claude: {evaluated}")
    print(f"Skipped (no price history): {skipped_no_price}")
    print(f"Skipped (no edge): {skipped_no_edge}")
    print(f"Trades placed: {len(trades)}")

    if not trades:
        print("\nNo trades. Strategy too restrictive, or no opportunities found.")
        return

    wins = sum(1 for t in trades if t["won"])
    total_pnl = sum(t["realized_pnl"] for t in trades)
    total_staked = sum(t["size_usd"] for t in trades)
    avg_edge_pred = sum(t["edge"] for t in trades) / len(trades)

    print(f"\nWin rate: {wins}/{len(trades)} ({100*wins/len(trades):.1f}%)")
    print(f"Total staked: ${total_staked:.2f}")
    print(f"Total P&L (net of 2% fees): ${total_pnl:+.2f}")
    print(f"ROI on staked capital: {100*total_pnl/total_staked:+.1f}%")
    print(f"Average predicted edge per trade: {avg_edge_pred:+.1%}")

    # Calibration
    buckets = {"50-60": [], "60-70": [], "70-80": [], "80-90": [], "90-100": []}
    for t in trades:
        p = t["estimated_prob"] if t["side"] == "YES" else 1 - t["estimated_prob"]
        if p < 0.5:
            continue
        if p < 0.6: buckets["50-60"].append(t["won"])
        elif p < 0.7: buckets["60-70"].append(t["won"])
        elif p < 0.8: buckets["70-80"].append(t["won"])
        elif p < 0.9: buckets["80-90"].append(t["won"])
        else: buckets["90-100"].append(t["won"])

    print(f"\nCalibration (predicted vs actual win rate):")
    for label, results in buckets.items():
        if not results:
            continue
        actual = sum(results) / len(results)
        print(f"  Predicted {label}%: n={len(results)}, actual={100*actual:.1f}%")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if total_pnl > 0 and len(trades) >= 30:
        print("Positive P&L over >=30 trades. Worth paper-trading live to validate.")
    elif total_pnl > 0 and len(trades) < 30:
        print("Positive P&L but <30 trades. Sample too small. Rerun with more markets.")
    elif total_pnl <= 0:
        print("Negative P&L. Strategy doesn't work on historical data. Do NOT go live.")
    print(f"\nRaw trades saved to {OUT_DIR / 'backtest_trades.jsonl'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-date", dest="from_date", default=None,
                    help="Min resolution date (YYYY-MM-DD), default 2026-02-01")
    ap.add_argument("--max-markets", type=int, default=None,
                    help="Cap number of Claude evaluations")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch + filter only, no Claude calls")
    args = ap.parse_args()
    run_backtest(args)


if __name__ == "__main__":
    main()
