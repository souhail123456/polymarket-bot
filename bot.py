"""
Polymarket EV Trading Bot
-------------------------
Scans active Polymarket markets, uses Llama 3.3 70B (via Groq) to estimate
true probability, calculates expected value, places trades when edge exceeds threshold.

Strategy: pure expected-value — trade any market where estimated probability
differs from market price by more than MIN_EDGE, sized by fractional Kelly.

Modes:
  paper    Logs hypothetical trades, does NOT submit orders (default, safe)
  live     ACTUALLY submits orders (requires wallet setup, see README)

Usage:
  python bot.py --mode paper
  python bot.py --mode paper --target-trades 15   # aim for 15 trades per scan cycle
  python bot.py --mode live                       # real money — do not use until calibrated
"""

import os
import json
import time
import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from openai import OpenAI

# ---------- Config ----------
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
LOG_DIR = Path(os.environ.get("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG = {
    # Trade decision thresholds
    "min_edge": 0.10,            # min EV per $1 staked (matches backtest)
    "min_confidence": 0.55,      # Claude's self-reported confidence
    # Sizing
    "max_position_usd": 5.0,     # cap per trade — keep small for paper
    "daily_loss_cap_usd": 20.0,
    "kelly_fraction": 0.25,      # quarter-Kelly
    # Market filters — tuned for FAST RESOLUTION so we hit 30 trades quickly
    "min_liquidity_usd": 500.0,
    "max_liquidity_usd": 500000.0, # skip mega-markets (>$500K liq)
    "max_hours_to_resolve": 168, # up to 7 days out
    "min_hours_to_resolve": 2,
    "price_floor": 0.15,         # matches backtest
    "price_ceiling": 0.85,       # matches backtest
    "max_markets_per_scan": 25,
    # Starting paper bankroll
    "starting_bankroll": 100.0,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "bot.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ---------- Market fetching ----------
def fetch_active_markets(limit=150):
    """Pull active markets from Polymarket Gamma API, sorted by 24h volume."""
    try:
        r = requests.get(
            f"{GAMMA_API}/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": limit,
                "order": "volume24hr",
                "ascending": "false",
            },
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"fetch_active_markets failed: {e}")
        return []


def fetch_market_by_id(market_id):
    """Fetch one market (used for resolution tracking)."""
    try:
        r = requests.get(f"{GAMMA_API}/markets/{market_id}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug(f"fetch_market_by_id({market_id}) failed: {e}")
        return None


def filter_markets(markets, cfg, already_traded_ids):
    """Keep only markets worth evaluating this scan."""
    kept = []
    now = datetime.now(timezone.utc)
    for m in markets:
        try:
            mid = str(m.get("id"))
            if mid in already_traded_ids:
                continue

            end_str = m.get("endDate") or m.get("end_date_iso")
            if not end_str:
                continue
            end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            hours_left = (end - now).total_seconds() / 3600
            if not (cfg["min_hours_to_resolve"] <= hours_left <= cfg["max_hours_to_resolve"]):
                continue

            liq = float(m.get("liquidity", 0) or 0)
            if liq < cfg["min_liquidity_usd"]:
                continue
            if liq > cfg["max_liquidity_usd"]:
                continue

            outcome_prices = m.get("outcomePrices")
            if not outcome_prices:
                continue
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)
            if len(outcome_prices) != 2:
                continue

            yes_price = float(outcome_prices[0])
            if not (cfg["price_floor"] <= yes_price <= cfg["price_ceiling"]):
                continue

            m["_yes_price"] = yes_price
            m["_hours_left"] = hours_left
            m["_market_id"] = mid
            kept.append(m)
        except Exception as e:
            log.debug(f"Skip market {m.get('id')}: {e}")

    # Sort by proximity to 0.5 — those are where disagreement is most valuable
    kept.sort(key=lambda m: abs(m["_yes_price"] - 0.5))
    return kept[: cfg["max_markets_per_scan"]]


# ---------- Claude probability estimation ----------
PROMPT = """You are a calibrated prediction market analyst. Your job: estimate the TRUE probability this market resolves YES.

MARKET: {question}

DESCRIPTION: {description}

CURRENT MARKET PRICE (crowd's implied probability of YES): {yes_price:.1%}
HOURS UNTIL RESOLUTION: {hours_left:.1f}
RESOLUTION SOURCE: {resolution}

Think through:
1. What exactly must happen for YES?
2. Base rate for this type of event?
3. Any concrete evidence you have right now?
4. Is the crowd likely biased here (recency, narrative, longshot bias)?

CRITICAL: If you do not have real informational edge, your estimate should be CLOSE to the market price. Being "confident the market is wrong" without specific evidence is how money gets lost. Most markets are approximately efficient.

Output ONLY this JSON, no other text:
{{"true_probability": <0-1>, "confidence": <0-1>, "reasoning": "<one sentence>"}}"""


def estimate_probability(client, market):
    prompt = PROMPT.format(
        question=(market.get("question") or "")[:500],
        description=(market.get("description") or "")[:1200],
        yes_price=market["_yes_price"],
        hours_left=market["_hours_left"],
        resolution=(market.get("resolutionSource") or "See description")[:300],
    )
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        return float(data["true_probability"]), float(data["confidence"]), data.get("reasoning", "")
    except Exception as e:
        log.warning(f"estimate_probability failed for {market.get('_market_id')}: {e}")
        return None, None, None


# ---------- EV math ----------
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


def decide_trade(true_p, market_p, confidence, bankroll, cfg):
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


# ---------- Trade log ----------
TRADES_FILE = LOG_DIR / "trades.jsonl"


def log_trade(record):
    with open(TRADES_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_trades():
    if not TRADES_FILE.exists():
        return []
    out = []
    with open(TRADES_FILE) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def save_trades(trades):
    with open(TRADES_FILE, "w") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")


def already_traded_market_ids():
    return {t["market_id"] for t in load_trades()}


def todays_realized_pnl():
    today = datetime.now(timezone.utc).date().isoformat()
    return sum(
        float(t.get("realized_pnl") or 0)
        for t in load_trades()
        if t.get("date") == today and t.get("resolved")
    )


# ---------- Resolution tracking ----------
def update_resolutions():
    """For each unresolved trade, check if market resolved; update P&L."""
    trades = load_trades()
    changed = False
    for t in trades:
        if t.get("resolved"):
            continue
        m = fetch_market_by_id(t["market_id"])
        if not m:
            continue
        # Market is resolved when closed=true and has a clear outcome
        if not m.get("closed"):
            continue
        # outcomePrices after resolution should be [1,0] or [0,1]
        prices = m.get("outcomePrices")
        if isinstance(prices, str):
            prices = json.loads(prices)
        if not prices or len(prices) != 2:
            continue
        yes_resolved = float(prices[0])
        if yes_resolved not in (0.0, 1.0):
            continue  # resolution ambiguous

        won_yes = yes_resolved == 1.0
        if t["side"] == "YES":
            won = won_yes
        else:
            won = not won_yes

        if won:
            # Bought at entry_price, receives $1 per share
            shares = t["size_usd"] / t["entry_price"]
            payout = shares * 1.0
            pnl = payout - t["size_usd"]
        else:
            pnl = -t["size_usd"]

        t["resolved"] = True
        t["won"] = won
        t["realized_pnl"] = round(pnl, 4)
        t["resolved_at"] = datetime.now(timezone.utc).isoformat()
        changed = True
        log.info(f"Resolved {t['market_id'][:8]}: {t['side']} -> {'WIN' if won else 'LOSS'} pnl=${pnl:.2f}")

    if changed:
        save_trades(trades)


# ---------- Calibration report ----------
def calibration_report():
    trades = [t for t in load_trades() if t.get("resolved")]
    if not trades:
        print("No resolved trades yet.")
        return
    total_pnl = sum(float(t["realized_pnl"]) for t in trades)
    n = len(trades)
    wins = sum(1 for t in trades if t["won"])
    avg_edge = sum(float(t["edge"]) for t in trades) / n

    # Calibration buckets
    buckets = {"50-60": [], "60-70": [], "70-80": [], "80-90": [], "90-100": []}
    for t in trades:
        p = t["estimated_prob"] if t["side"] == "YES" else 1 - t["estimated_prob"]
        if p < 0.5:
            continue
        elif p < 0.6: buckets["50-60"].append(t["won"])
        elif p < 0.7: buckets["60-70"].append(t["won"])
        elif p < 0.8: buckets["70-80"].append(t["won"])
        elif p < 0.9: buckets["80-90"].append(t["won"])
        else:         buckets["90-100"].append(t["won"])

    print(f"\n=== CALIBRATION REPORT ({n} resolved trades) ===")
    print(f"Win rate: {wins}/{n} ({100*wins/n:.1f}%)")
    print(f"Total P&L: ${total_pnl:+.2f}")
    print(f"Avg predicted edge: {avg_edge:.1%}")
    print(f"\nCalibration by confidence bucket (want actual ≈ predicted):")
    for label, results in buckets.items():
        if not results:
            continue
        actual = sum(results) / len(results)
        print(f"  Predicted {label}%: n={len(results)}, actual win rate={100*actual:.1f}%")
    print(f"\nReady for live? Need: n>=30, positive P&L, calibration within ~15%.")


# ---------- Live trading (disabled stub) ----------
def submit_live_order(market, side, size_usd, price):
    raise NotImplementedError(
        "Live mode not wired. Requires py-clob-client + wallet setup. "
        "Run paper mode until calibration report shows n>=30 and positive edge."
    )


# ---------- Main scan ----------
def run_scan(client, cfg, mode, bankroll):
    log.info(f"=== Scan | mode={mode} | bankroll=${bankroll:.2f} ===")

    # Update resolutions first
    update_resolutions()

    pnl_today = todays_realized_pnl()
    if pnl_today <= -cfg["daily_loss_cap_usd"]:
        log.warning(f"Daily loss cap hit (${pnl_today:.2f}). Skipping scan.")
        return

    markets = fetch_active_markets()
    log.info(f"Fetched {len(markets)} active markets")
    traded_ids = already_traded_market_ids()
    candidates = filter_markets(markets, cfg, traded_ids)
    log.info(f"{len(candidates)} candidates after filters")

    trades_placed = 0
    for m in candidates:
        true_p, conf, reason = estimate_probability(client, m)
        if true_p is None:
            continue
        market_p = m["_yes_price"]
        side, size, edge = decide_trade(true_p, market_p, conf, bankroll, cfg)

        log.info(
            f"[{m['_market_id'][:8]}] {m['question'][:60]!r} "
            f"mkt={market_p:.2f} est={true_p:.2f} conf={conf:.2f} edge={edge:+.1%} "
            f"=> {side or 'skip'} ${size:.2f}"
        )

        if side is None:
            continue

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "date": datetime.now(timezone.utc).date().isoformat(),
            "mode": mode,
            "market_id": m["_market_id"],
            "question": m["question"][:200],
            "hours_to_resolve": round(m["_hours_left"], 2),
            "market_price": market_p,
            "estimated_prob": true_p,
            "confidence": conf,
            "reasoning": reason,
            "side": side,
            "entry_price": market_p if side == "YES" else 1 - market_p,
            "size_usd": round(size, 2),
            "edge": round(edge, 4),
            "resolved": False,
        }

        if mode == "live":
            try:
                entry_px = market_p if side == "YES" else 1 - market_p
                submit_live_order(m, side, size, entry_px)
                record["live_submitted"] = True
            except NotImplementedError as e:
                log.error(str(e))
                return

        log_trade(record)
        trades_placed += 1

        # Small delay to avoid hammering APIs
        time.sleep(2)

    log.info(f"Scan complete. Placed {trades_placed} trades.")


# ---------- Entry point ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--loop", action="store_true", help="Run continuously every 30 min")
    ap.add_argument("--report", action="store_true", help="Print calibration report and exit")
    ap.add_argument("--max-position", type=float, default=None)
    args = ap.parse_args()

    if args.report:
        update_resolutions()
        calibration_report()
        return

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        log.error("Set GROQ_API_KEY environment variable.")
        return

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    cfg = DEFAULT_CONFIG.copy()
    if args.max_position:
        cfg["max_position_usd"] = args.max_position

    # Bankroll = starting + realized P&L from all resolved trades
    all_trades = [t for t in load_trades() if t.get("resolved")]
    realized = sum(float(t.get("realized_pnl") or 0) for t in all_trades)
    bankroll = cfg["starting_bankroll"] + realized

    if args.loop:
        while True:
            try:
                run_scan(client, cfg, args.mode, bankroll)
            except Exception as e:
                log.exception(f"Scan error: {e}")
            log.info("Sleeping 30 min...")
            time.sleep(30 * 60)
    else:
        run_scan(client, cfg, args.mode, bankroll)


if __name__ == "__main__":
    main()
