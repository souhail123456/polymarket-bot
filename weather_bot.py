"""
Polymarket Weather Trading Bot
-------------------------------
Uses GFS ensemble forecasts (31 members) from Open-Meteo to estimate
temperature probabilities, then trades Polymarket weather markets when
the model disagrees with the crowd.

No API key needed — Open-Meteo is free. No LLM calls.

Strategy: count ensemble members in each temperature bucket,
compare to market price, trade when edge exceeds threshold.

Usage:
  python weather_bot.py --mode paper
"""

import os
import json
import re
import time
import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

# ---------- Config ----------
SHANGHAI_TZ = timezone(timedelta(hours=8))
GAMMA_API = "https://gamma-api.polymarket.com"
ENSEMBLE_API = "https://ensemble-api.open-meteo.com/v1/ensemble"
LOG_DIR = Path(os.environ.get("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

CONFIG = {
    "min_edge": 0.08,             # 8% min edge for weather
    "max_position_usd": 15.0,    # max cap (for 25%+ edge)
    "kelly_fraction": 0.15,       # 15% Kelly
    "taker_fee_rate": 0.05,
    "starting_bankroll": 100.0,
    "daily_loss_cap_usd": 20.0,
}

# Scaled position sizing by edge — conservative until proven
def max_size_for_edge(edge):
    if edge >= 0.25:
        return 8.0
    elif edge >= 0.15:
        return 6.0
    else:
        return 5.0

# Cities Polymarket tracks, with coordinates and units
CITIES = {
    "nyc":     {"lat": 40.78, "lon": -73.88, "unit": "fahrenheit", "name": "NYC"},
    "chicago": {"lat": 41.97, "lon": -87.91, "unit": "fahrenheit", "name": "Chicago"},
    "miami":   {"lat": 25.79, "lon": -80.29, "unit": "fahrenheit", "name": "Miami"},
    "la":      {"lat": 33.94, "lon": -118.41, "unit": "fahrenheit", "name": "Los Angeles"},
    "denver":  {"lat": 39.72, "lon": -104.75, "unit": "fahrenheit", "name": "Denver"},
    "paris":   {"lat": 48.97, "lon": 2.44,   "unit": "celsius",    "name": "Paris"},
    "seoul":   {"lat": 37.46, "lon": 126.44, "unit": "celsius",    "name": "Seoul"},
    "shanghai":{"lat": 31.14, "lon": 121.81, "unit": "celsius",    "name": "Shanghai"},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "weather_bot.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ---------- Telegram ----------
def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")


# ---------- Trade log ----------
TRADES_FILE = LOG_DIR / "weather_trades.jsonl"


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


# ---------- Fetch weather markets from Polymarket ----------
def fetch_weather_markets():
    """Search Polymarket for active weather/temperature markets."""
    markets = []
    for city_key, city in CITIES.items():
        try:
            r = requests.get(
                f"{GAMMA_API}/public-search",
                params={"q": f"{city['name']} temperature", "events_status": "active"},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            events = data.get("events", [])
            for event in events:
                for m in event.get("markets", []):
                    m["_city_key"] = city_key
                    m["_city_name"] = city["name"]
                    m["_event_title"] = event.get("title", "")
                    markets.append(m)
        except Exception as e:
            log.warning(f"Failed to fetch weather markets for {city['name']}: {e}")
    log.info(f"Found {len(markets)} weather sub-markets")
    return markets


def parse_bucket(question, unit):
    """Parse a temperature bucket from market question.
    Returns (low, high) in the market's unit. None bounds mean open-ended.
    Examples:
      '56-57F' -> (56, 57)
      '74F or higher' -> (74, None)
      '55F or below' -> (None, 55)
      '14C' -> (14, 14)
    """
    q = question.strip()
    unit_char = "F" if unit == "fahrenheit" else "C"

    # "74F or higher" / "74°F or higher"
    m = re.search(rf'(\d+)\s*°?\s*{unit_char}\s+or\s+higher', q, re.IGNORECASE)
    if m:
        return (int(m.group(1)), None)

    # "55F or below" / "55°F or below"
    m = re.search(rf'(\d+)\s*°?\s*{unit_char}\s+or\s+below', q, re.IGNORECASE)
    if m:
        return (None, int(m.group(1)))

    # "56-57F" / "56 - 57°F"
    m = re.search(rf'(\d+)\s*-\s*(\d+)\s*°?\s*{unit_char}', q, re.IGNORECASE)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    # Single value "14C" / "14°C"
    m = re.search(rf'(\d+)\s*°?\s*{unit_char}\b', q, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        return (val, val)

    return None


def parse_event_date(event_title):
    """Extract the date from event title like 'Highest temperature in NYC on April 28'."""
    m = re.search(r'on\s+(\w+\s+\d+)', event_title, re.IGNORECASE)
    if not m:
        return None
    try:
        year = datetime.now(timezone.utc).year
        return datetime.strptime(f"{m.group(1)} {year}", "%B %d %Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


# ---------- Fetch ensemble forecast ----------
ENSEMBLE_MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless"]
# GFS: 31 members, ECMWF: 51 members, ICON: 40 members = ~122 total


def fetch_ensemble_forecast(city_key, date_str):
    """Get multi-model ensemble forecast from Open-Meteo.
    Combines GFS (31), ECMWF (51), and ICON (40) for ~122 members."""
    city = CITIES[city_key]
    all_maxes = []
    for model in ENSEMBLE_MODELS:
        try:
            r = requests.get(
                ENSEMBLE_API,
                params={
                    "latitude": city["lat"],
                    "longitude": city["lon"],
                    "hourly": "temperature_2m",
                    "models": model,
                    "start_date": date_str,
                    "end_date": date_str,
                    "temperature_unit": city["unit"],
                },
                timeout=15,
            )
            r.raise_for_status()
            maxes = compute_daily_maxes(r.json())
            all_maxes.extend(maxes)
            log.debug(f"{model}: {len(maxes)} members for {city_key} {date_str}")
        except Exception as e:
            log.warning(f"Ensemble fetch failed for {model} {city_key} {date_str}: {e}")
    return all_maxes


def compute_daily_maxes(ensemble_data):
    """Extract daily max temperature for each ensemble member."""
    hourly = ensemble_data.get("hourly", {})
    maxes = []

    for key, temps in hourly.items():
        if not key.startswith("temperature_2m"):
            continue
        if not temps:
            continue
        valid = [t for t in temps if t is not None]
        if valid:
            maxes.append(max(valid))

    return maxes


def ensemble_probability(maxes, low, high):
    """Count fraction of ensemble members whose daily max falls in [low, high]."""
    if not maxes:
        return None
    count = 0
    for t in maxes:
        t_rounded = round(t)
        if low is None and high is not None:
            if t_rounded <= high:
                count += 1
        elif high is None and low is not None:
            if t_rounded >= low:
                count += 1
        elif low is not None and high is not None:
            if low <= t_rounded <= high:
                count += 1
    return count / len(maxes)


# ---------- EV math ----------
def calculate_fee(price, size_usd, fee_rate):
    if price <= 0 or price >= 1:
        return 0.0
    shares = size_usd / price
    return fee_rate * price * (1 - price) * shares


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


def decide_trade(true_p, market_p, bankroll, cfg):
    ev_yes = expected_value(true_p, market_p, "YES")
    ev_no = expected_value(true_p, market_p, "NO")
    side, gross_edge = ("YES", ev_yes) if ev_yes >= ev_no else ("NO", ev_no)
    entry_price = market_p if side == "YES" else 1 - market_p
    fee_per_dollar = calculate_fee(entry_price, 1.0, cfg["taker_fee_rate"])
    edge = gross_edge - fee_per_dollar
    if edge < cfg["min_edge"]:
        return None, 0.0, edge, 0.0
    cap = max_size_for_edge(edge)
    size = kelly_size(true_p, market_p, side, bankroll, cfg["kelly_fraction"], cap)
    if size < 1.0:
        return None, 0.0, edge, 0.0
    fee_usd = calculate_fee(entry_price, size, cfg["taker_fee_rate"])
    return side, size, edge, fee_usd


# ---------- Resolution tracking ----------
def update_resolutions():
    trades = load_trades()
    changed = False
    for t in trades:
        if t.get("resolved"):
            continue
        try:
            r = requests.get(f"{GAMMA_API}/markets/{t['market_id']}", timeout=10)
            r.raise_for_status()
            m = r.json()
        except Exception:
            continue
        if not m.get("closed"):
            continue
        prices = m.get("outcomePrices")
        if isinstance(prices, str):
            prices = json.loads(prices)
        if not prices or len(prices) != 2:
            continue
        yes_resolved = float(prices[0])
        if yes_resolved not in (0.0, 1.0):
            continue

        won_yes = yes_resolved == 1.0
        won = won_yes if t["side"] == "YES" else not won_yes

        if won:
            shares = t["size_usd"] / t["entry_price"]
            pnl = (shares * 1.0) - t["size_usd"]
        else:
            pnl = -t["size_usd"]

        t["resolved"] = True
        t["won"] = won
        t["realized_pnl"] = round(pnl, 4)
        t["resolved_at"] = datetime.now(SHANGHAI_TZ).isoformat()
        changed = True

        # Calculate current balance
        all_resolved = [tr for tr in trades if tr.get("resolved")]
        total_pnl = sum(float(tr.get("realized_pnl", 0)) for tr in all_resolved)
        balance = CONFIG["starting_bankroll"] + total_pnl
        t["balance"] = round(balance, 2)

        log.info(f"Resolved {t['market_id'][:8]}: {t['side']} -> {'WIN' if won else 'LOSS'} pnl=${pnl:.2f} balance=${balance:.2f}")
        send_telegram(
            f"{'✅' if won else '❌'} *Weather Trade Resolved*\n"
            f"Market: {t['question'][:80]}\n"
            f"Side: {t['side']} -> *{'WIN' if won else 'LOSS'}*\n"
            f"P&L: ${pnl:+.2f}\n"
            f"Balance: ${balance:.2f}"
        )
    if changed:
        save_trades(trades)


# ---------- Main scan ----------
def run_scan(cfg, mode, bankroll):
    log.info(f"=== Weather Scan | mode={mode} | bankroll=${bankroll:.2f} ===")
    update_resolutions()

    markets = fetch_weather_markets()
    traded_ids = already_traded_market_ids()

    # Group markets by city + date for efficient ensemble fetching
    forecast_cache = {}
    trades_placed = 0

    for m in markets:
        mid = str(m.get("id", ""))
        if mid in traded_ids:
            continue

        city_key = m.get("_city_key")
        city = CITIES.get(city_key)
        if not city:
            continue

        event_title = m.get("_event_title", "")
        date_str = parse_event_date(event_title)
        if not date_str:
            continue

        question = m.get("question", "")
        # Only trade "highest temperature" markets — we fetch daily max, not min
        if "lowest" in question.lower() or "highest" not in event_title.lower():
            continue
        bucket = parse_bucket(question, city["unit"])
        if bucket is None:
            log.debug(f"Could not parse bucket from: {question}")
            continue

        outcome_prices = m.get("outcomePrices")
        if not outcome_prices:
            continue
        if isinstance(outcome_prices, str):
            outcome_prices = json.loads(outcome_prices)
        if len(outcome_prices) != 2:
            continue
        yes_price = float(outcome_prices[0])
        if yes_price <= 0.02 or yes_price >= 0.98:
            continue

        # Fetch ensemble (cached per city+date)
        cache_key = f"{city_key}_{date_str}"
        if cache_key not in forecast_cache:
            forecast_cache[cache_key] = fetch_ensemble_forecast(city_key, date_str)

        maxes = forecast_cache.get(cache_key)
        if not maxes:
            continue

        low, high = bucket
        model_prob = ensemble_probability(maxes, low, high)
        if model_prob is None:
            continue

        side, size, edge, fee_usd = decide_trade(model_prob, yes_price, bankroll, cfg)

        log.info(
            f"[{city['name']}] {question[:50]} "
            f"mkt={yes_price:.2f} model={model_prob:.2f} edge={edge:+.1%} "
            f"fee=${fee_usd:.2f} => {side or 'skip'} ${size:.2f}"
        )

        if side is None:
            continue

        # Only take NO bets — ensemble models are better at ruling out temps
        if side == "YES":
            continue

        bankroll -= size  # deduct from bankroll

        record = {
            "timestamp": datetime.now(SHANGHAI_TZ).isoformat(),
            "date": datetime.now(SHANGHAI_TZ).date().isoformat(),
            "mode": mode,
            "strategy": "weather",
            "market_id": mid,
            "question": question[:200],
            "city": city["name"],
            "event_date": date_str,
            "bucket": f"{low}-{high}",
            "market_price": yes_price,
            "model_prob": round(model_prob, 4),
            "ensemble_members": len(maxes),
            "side": side,
            "entry_price": yes_price if side == "YES" else 1 - yes_price,
            "size_usd": round(size, 2),
            "edge": round(edge, 4),
            "fee_usd": round(fee_usd, 4),
            "balance": round(bankroll, 2),
            "resolved": False,
        }

        log_trade(record)
        trades_placed += 1

        send_telegram(
            f"🌡️ *Weather Trade*\n"
            f"City: {city['name']} ({date_str})\n"
            f"Bucket: {question[:60]}\n"
            f"Side: *{side}* @ {record['entry_price']:.2f}\n"
            f"Model: {model_prob:.0%} vs Market: {yes_price:.0%}\n"
            f"Size: ${size:.2f} | Edge: {edge:+.1%}\n"
            f"Balance: ${bankroll:.2f}"
        )

        time.sleep(1)

    log.info(f"Weather scan complete. Placed {trades_placed} trades.")


# ---------- Daily summary ----------
def send_daily_summary():
    today = datetime.now(SHANGHAI_TZ).date().isoformat()
    trades = load_trades()
    todays_trades = [t for t in trades if t.get("date") == today]
    all_resolved = [t for t in trades if t.get("resolved")]
    total_pnl = sum(float(t.get("realized_pnl") or 0) for t in all_resolved)
    open_trades = [t for t in trades if not t.get("resolved")]
    wins = sum(1 for t in all_resolved if t.get("won"))

    if todays_trades:
        msg = (
            f"🌤️ *Weather Bot Summary — {today}*\n\n"
            f"Trades today: {len(todays_trades)}\n"
            f"Open trades: {len(open_trades)}\n"
            f"Total resolved: {len(all_resolved)}\n"
        )
        if all_resolved:
            msg += f"Win rate: {100*wins/len(all_resolved):.0f}%\n"
        msg += f"Total P&L: ${total_pnl:+.2f}"
    else:
        msg = (
            f"🌤️ *Weather Bot Summary — {today}*\n\n"
            f"No weather trades today.\n"
            f"Open trades: {len(open_trades)}\n"
            f"Total resolved: {len(all_resolved)}\n"
            f"Total P&L: ${total_pnl:+.2f}"
        )

    send_telegram(msg)
    log.info(f"Weather summary sent: {len(todays_trades)} trades today")


# ---------- Entry point ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--daily-summary", action="store_true")
    args = ap.parse_args()

    if args.daily_summary:
        update_resolutions()
        send_daily_summary()
        return

    cfg = CONFIG.copy()

    all_resolved = [t for t in load_trades() if t.get("resolved")]
    realized = sum(float(t.get("realized_pnl") or 0) for t in all_resolved)
    bankroll = cfg["starting_bankroll"] + realized

    run_scan(cfg, args.mode, bankroll)


if __name__ == "__main__":
    main()
