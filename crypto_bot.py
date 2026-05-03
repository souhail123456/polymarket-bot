"""
Polymarket Crypto Price Threshold Trading Bot
----------------------------------------------
Fetches active BTC/ETH price threshold markets from Polymarket Gamma API,
gets current prices + 30-day volatility from CoinGecko (no API key needed),
then calculates probability that price stays above/below the threshold using
a log-normal model with historical vol + time remaining.

No LLM calls. No API key needed.

Strategy: when BTC is at $95k and market asks "above $80k by May 5?", the
model says ~99% but market may price it at 85% — that's the edge.

Usage:
  python crypto_bot.py --mode paper
  python crypto_bot.py --daily-summary
"""

import os
import json
import re
import math
import time
import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

# ---------- Config ----------
UTC = timezone.utc
SHANGHAI_TZ = timezone(timedelta(hours=8))
GAMMA_API = "https://gamma-api.polymarket.com"
COINGECKO_PRICE_API = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_HISTORY_API = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
LOG_DIR = Path(os.environ.get("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

CONFIG = {
    "min_edge": 0.12,               # 12% min edge before fees
    "kelly_fraction": 0.15,         # fractional Kelly
    "max_position_usd": 15.0,       # hard cap per trade
    "taker_fee_rate": 0.05,         # Polymarket taker fee
    "min_model_confidence": 0.90,   # only trade near-certain outcomes
    "max_days_ahead": 3,            # skip markets more than 3 days out
    "starting_bankroll": 100.0,
    "daily_loss_cap_usd": 20.0,
}

# Assets to trade — only the most liquid
CRYPTO_ASSETS = {
    "bitcoin": {
        "cg_id": "bitcoin",
        "symbols": ["BTC", "Bitcoin", "bitcoin"],
        "search_terms": ["Bitcoin price", "BTC price", "Will Bitcoin", "Will BTC"],
    },
    "ethereum": {
        "cg_id": "ethereum",
        "symbols": ["ETH", "Ethereum", "ethereum"],
        "search_terms": ["Ethereum price", "ETH price", "Will Ethereum", "Will ETH"],
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "crypto_bot.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ---------- Shared context ----------
SHARED_CONTEXT_FILE = LOG_DIR / "shared_context.json"


def load_shared_context():
    """Load cross-bot shared context."""
    try:
        if SHARED_CONTEXT_FILE.exists():
            with open(SHARED_CONTEXT_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_shared_context(bot_name, data):
    """Update this bot's section in shared context."""
    ctx = load_shared_context()
    data["updated_at"] = datetime.now(SHANGHAI_TZ).isoformat()
    ctx[bot_name] = data
    with open(SHARED_CONTEXT_FILE, "w") as f:
        json.dump(ctx, f, indent=2)


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
TRADES_FILE = LOG_DIR / "crypto_trades.jsonl"


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


# ---------- Normal CDF via math.erf ----------
def norm_cdf(x):
    """Standard normal CDF using math.erf. No scipy needed."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------- Fetch crypto markets from Polymarket ----------
def fetch_crypto_markets():
    """Search Polymarket Gamma API for active BTC/ETH price threshold markets."""
    markets = []
    seen_ids = set()

    for asset_key, asset in CRYPTO_ASSETS.items():
        for term in asset["search_terms"]:
            try:
                r = requests.get(
                    f"{GAMMA_API}/markets",
                    params={
                        "q": term,
                        "active": "true",
                        "closed": "false",
                        "limit": 50,
                    },
                    timeout=15,
                )
                r.raise_for_status()
                data = r.json()

                # Gamma /markets can return a list or a dict with "markets" key
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = data.get("markets", data.get("results", [data]))
                else:
                    items = []

                for m in items:
                    mid = str(m.get("id", ""))
                    if mid in seen_ids:
                        continue
                    seen_ids.add(mid)
                    m["_asset_key"] = asset_key
                    markets.append(m)

            except Exception as e:
                log.warning(f"Failed to fetch markets for '{term}': {e}")
            time.sleep(0.3)  # gentle rate limiting

    log.info(f"Fetched {len(markets)} raw crypto markets from Polymarket")
    return markets


# ---------- Parse market question ----------
# Patterns we handle:
#   "Will BTC be above $80,000 on May 5?"
#   "Will Bitcoin be above $80k by May 5, 2025?"
#   "Will ETH be below $2,500 on April 30?"
#   "BTC above $95000 on May 10?"

_THRESHOLD_RE = re.compile(
    r'\$\s*([\d,]+(?:\.\d+)?)\s*(k|K)?',
    re.IGNORECASE,
)
_DIRECTION_RE = re.compile(r'\b(above|over|below|under)\b', re.IGNORECASE)
_DATE_RE = re.compile(
    r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    r'\s+(\d{1,2})(?:,?\s*(\d{4}))?',
    re.IGNORECASE,
)


def parse_threshold(question):
    """Return (direction, threshold_usd) or None.
    direction is 'above' or 'below'.
    """
    dm = _DIRECTION_RE.search(question)
    if not dm:
        return None
    direction = dm.group(1).lower()
    if direction in ("over",):
        direction = "above"
    elif direction in ("under",):
        direction = "below"

    tm = _THRESHOLD_RE.search(question)
    if not tm:
        return None
    raw = tm.group(1).replace(",", "")
    multiplier = 1000.0 if tm.group(2) else 1.0
    threshold = float(raw) * multiplier

    return direction, threshold


def parse_resolution_date(question, event_end_date=None):
    """Extract resolution date from question or fall back to endDate field.
    Returns a date string 'YYYY-MM-DD' or None.
    """
    m = _DATE_RE.search(question)
    if m:
        month_str = m.group(1)
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else datetime.now(UTC).year
        try:
            dt = datetime.strptime(f"{month_str[:3]} {day} {year}", "%b %d %Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Fall back to endDate from the market object
    if event_end_date:
        try:
            # endDate can be ISO format or just a date string
            dt = datetime.fromisoformat(event_end_date.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    return None


def identify_asset(question, asset_key):
    """Confirm the question is really about the expected asset."""
    asset = CRYPTO_ASSETS[asset_key]
    q_lower = question.lower()
    return any(s.lower() in q_lower for s in asset["symbols"])


# ---------- CoinGecko price + volatility ----------
_price_cache = {}
_vol_cache = {}


def fetch_current_prices():
    """Get current BTC and ETH prices in USD from CoinGecko."""
    global _price_cache
    if _price_cache:
        return _price_cache
    try:
        r = requests.get(
            COINGECKO_PRICE_API,
            params={"ids": "bitcoin,ethereum", "vs_currencies": "usd"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        _price_cache = {
            "bitcoin": data["bitcoin"]["usd"],
            "ethereum": data["ethereum"]["usd"],
        }
        log.info(f"Prices — BTC: ${_price_cache['bitcoin']:,.0f}  ETH: ${_price_cache['ethereum']:,.0f}")
        return _price_cache
    except Exception as e:
        log.error(f"Failed to fetch CoinGecko prices: {e}")
        return {}


def fetch_30d_volatility(coin_id):
    """Fetch 30-day daily close prices from CoinGecko and compute annualised log-vol."""
    global _vol_cache
    if coin_id in _vol_cache:
        return _vol_cache[coin_id]
    try:
        r = requests.get(
            COINGECKO_HISTORY_API.format(coin_id=coin_id),
            params={"vs_currency": "usd", "days": "30", "interval": "daily"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        closes = [p[1] for p in data.get("prices", [])]
        if len(closes) < 5:
            raise ValueError("Not enough price data")
        log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        # Annualise: daily vol * sqrt(365)
        mean_r = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
        daily_vol = math.sqrt(variance)
        annual_vol = daily_vol * math.sqrt(365)
        _vol_cache[coin_id] = annual_vol
        log.info(f"{coin_id} 30d annualised vol: {annual_vol:.1%}")
        return annual_vol
    except Exception as e:
        log.warning(f"Failed to fetch vol for {coin_id}: {e}. Using default 0.80.")
        # Conservative default: 80% annualised vol for crypto
        fallback = 0.80
        _vol_cache[coin_id] = fallback
        return fallback


# ---------- Probability model ----------
def price_threshold_probability(current_price, threshold, direction, days_to_expiry, annual_vol):
    """
    Probability that price is above (or below) threshold at expiry.

    Uses a log-normal GBM model:
        ln(S_T / S_0) ~ N(mu * T, sigma^2 * T)

    For a conservative (no-drift) model we set mu = 0. This is
    intentional — we do NOT want to bet on directional price moves,
    only on near-certainty outcomes where current price is far from
    the threshold relative to vol * sqrt(T).

    Returns float in [0, 1].
    """
    if days_to_expiry <= 0:
        # Expiry passed — evaluate current price vs threshold
        if direction == "above":
            return 1.0 if current_price > threshold else 0.0
        else:
            return 1.0 if current_price < threshold else 0.0

    T = days_to_expiry / 365.0
    sigma_sqrt_T = annual_vol * math.sqrt(T)

    if sigma_sqrt_T < 1e-9:
        # Essentially zero time — use current price
        if direction == "above":
            return 1.0 if current_price > threshold else 0.0
        else:
            return 1.0 if current_price < threshold else 0.0

    # d = [ln(S0/K)] / (sigma * sqrt(T))  (zero-drift log-normal)
    # P(S_T > K) = N(d)
    # P(S_T < K) = N(-d)
    try:
        d = math.log(current_price / threshold) / sigma_sqrt_T
    except (ValueError, ZeroDivisionError):
        return 0.5  # degenerate case

    if direction == "above":
        prob = norm_cdf(d)
    else:
        prob = norm_cdf(-d)

    return prob


# ---------- EV math (identical pattern to weather_bot) ----------
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
    size = kelly_size(true_p, market_p, side, bankroll, cfg["kelly_fraction"], cfg["max_position_usd"])
    if size < 1.0:
        return None, 0.0, edge, 0.0
    fee_usd = calculate_fee(entry_price, size, cfg["taker_fee_rate"])
    return side, size, edge, fee_usd


# ---------- Bot status ----------
BOT_STATUS_FILE = LOG_DIR / "bot_status.json"


def update_bot_status(bot_name, trades_placed, trades_skipped, next_run):
    try:
        status = {}
        if BOT_STATUS_FILE.exists():
            with open(BOT_STATUS_FILE) as f:
                status = json.load(f)
        status[bot_name] = {
            "last_run": datetime.now(SHANGHAI_TZ).isoformat(),
            "trades_placed": trades_placed,
            "trades_skipped": trades_skipped,
            "next_run": next_run,
            "status": "ok",
        }
        with open(BOT_STATUS_FILE, "w") as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        log.warning(f"update_bot_status failed: {e}")


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
        t["resolved_at"] = datetime.now(UTC).isoformat()
        changed = True

        all_resolved = [tr for tr in trades if tr.get("resolved")]
        total_pnl = sum(float(tr.get("realized_pnl", 0)) for tr in all_resolved)
        balance = CONFIG["starting_bankroll"] + total_pnl
        t["balance"] = round(balance, 2)

        log.info(
            f"Resolved {t['market_id'][:8]}: {t['side']} -> "
            f"{'WIN' if won else 'LOSS'} pnl=${pnl:.2f} balance=${balance:.2f}"
        )
        send_telegram(
            f"{'✅' if won else '❌'} *Crypto Trade Resolved*\n"
            f"Market: {t['question'][:80]}\n"
            f"Side: {t['side']} -> *{'WIN' if won else 'LOSS'}*\n"
            f"P&L: ${pnl:+.2f}\n"
            f"Balance: ${balance:.2f}\n"
            f"🕐 {datetime.now(SHANGHAI_TZ).strftime('%H:%M Shanghai')}"
        )

    if changed:
        save_trades(trades)


# ---------- Main scan ----------
def run_scan(cfg, mode, bankroll):
    log.info(f"=== Crypto Scan | mode={mode} | bankroll=${bankroll:.2f} ===")

    # Log shared context from other bots
    shared = load_shared_context()
    for bot_name, ctx in shared.items():
        if bot_name != "crypto":
            log.info(f"[shared/{bot_name}] {ctx.get('summary', 'no summary')} (as of {ctx.get('updated_at', '?')})")

    update_resolutions()

    # Fetch prices and vols upfront
    prices = fetch_current_prices()
    if not prices:
        log.error("Cannot fetch current prices — aborting scan.")
        return

    vols = {}
    for asset_key, asset in CRYPTO_ASSETS.items():
        vols[asset_key] = fetch_30d_volatility(asset["cg_id"])
        time.sleep(1)  # CoinGecko free tier rate limit

    # Save crypto prices/vols to shared context now (before market loop)
    btc_price = prices.get("bitcoin", 0)
    eth_price = prices.get("ethereum", 0)
    btc_vol = round(vols.get("bitcoin", 0.0), 4)
    eth_vol = round(vols.get("ethereum", 0.0), 4)
    save_shared_context("crypto", {
        "btc_price": btc_price,
        "eth_price": eth_price,
        "btc_30d_vol": btc_vol,
        "eth_30d_vol": eth_vol,
        "summary": f"BTC ${btc_price:,.0f}, ETH ${eth_price:,.0f}, BTC vol {btc_vol:.0%}",
    })

    markets = fetch_crypto_markets()
    traded_ids = already_traded_market_ids()
    now_utc = datetime.now(UTC)

    candidates = []

    for m in markets:
        mid = str(m.get("id", ""))
        if not mid or mid in traded_ids:
            continue

        asset_key = m.get("_asset_key")
        if asset_key not in CRYPTO_ASSETS:
            continue

        question = m.get("question", "")
        if not question:
            continue

        # Confirm question is about the right asset
        if not identify_asset(question, asset_key):
            continue

        # Parse direction + threshold
        parsed = parse_threshold(question)
        if parsed is None:
            log.debug(f"Could not parse threshold from: {question[:80]}")
            continue
        direction, threshold = parsed

        # Parse resolution date
        end_date_str = m.get("endDate") or m.get("end_date_iso") or ""
        date_str = parse_resolution_date(question, end_date_str)
        if not date_str:
            log.debug(f"Could not parse date from: {question[:80]}")
            continue

        # Filter: only trade within max_days_ahead
        try:
            resolution_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
            days_ahead = (resolution_dt - now_utc).total_seconds() / 86400
            if days_ahead < 0:
                continue  # already past
            if days_ahead > cfg["max_days_ahead"]:
                log.debug(f"Skip {question[:50]} — {days_ahead:.1f}d ahead > {cfg['max_days_ahead']}d")
                continue
        except Exception:
            continue

        # Market price
        outcome_prices = m.get("outcomePrices")
        if not outcome_prices:
            continue
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except Exception:
                continue
        if len(outcome_prices) != 2:
            continue
        try:
            yes_price = float(outcome_prices[0])
        except (ValueError, TypeError):
            continue
        if yes_price <= 0.02 or yes_price >= 0.98:
            continue  # already near-resolved by market

        # Model probability
        current_price = prices.get(asset_key)
        if not current_price:
            continue
        annual_vol = vols.get(asset_key, 0.80)

        model_prob = price_threshold_probability(
            current_price, threshold, direction, days_ahead, annual_vol
        )

        # Conservative filter: only trade when model is very confident
        if model_prob < cfg["min_model_confidence"] and model_prob > (1 - cfg["min_model_confidence"]):
            log.debug(
                f"Skip {question[:50]} — model={model_prob:.2%} not confident enough"
            )
            continue

        side, size, edge, fee_usd = decide_trade(model_prob, yes_price, bankroll, cfg)

        log.info(
            f"[{asset_key.upper()[:3]}] {question[:60]} | "
            f"price=${current_price:,.0f} threshold=${threshold:,.0f} "
            f"({direction}) days={days_ahead:.1f} "
            f"mkt={yes_price:.2f} model={model_prob:.2f} edge={edge:+.1%} "
            f"=> {side or 'skip'} ${size:.2f}"
        )

        if side is None:
            continue

        entry_price = yes_price if side == "YES" else 1 - yes_price

        candidates.append({
            "m": m,
            "mid": mid,
            "asset_key": asset_key,
            "question": question,
            "direction": direction,
            "threshold": threshold,
            "date_str": date_str,
            "days_ahead": days_ahead,
            "current_price": current_price,
            "annual_vol": annual_vol,
            "yes_price": yes_price,
            "model_prob": model_prob,
            "side": side,
            "size": size,
            "edge": edge,
            "fee_usd": fee_usd,
            "entry_price": entry_price,
        })

    # Sort by edge descending — best trades first
    candidates.sort(key=lambda c: c["edge"], reverse=True)

    trades_placed = 0
    asset_bets = {}  # limit correlated risk: max 2 bets per asset per run

    for c in candidates:
        asset_key = c["asset_key"]
        if asset_bets.get(asset_key, 0) >= 2:
            log.info(f"[{asset_key}] Skip — already 2 bets for this asset this run")
            continue

        bankroll -= c["size"]
        asset_bets[asset_key] = asset_bets.get(asset_key, 0) + 1

        record = {
            "timestamp": datetime.now(SHANGHAI_TZ).isoformat(),
            "date": datetime.now(SHANGHAI_TZ).date().isoformat(),
            "mode": mode,
            "strategy": "crypto_threshold",
            "market_id": c["mid"],
            "question": c["question"][:200],
            "asset": asset_key,
            "current_price_usd": c["current_price"],
            "threshold_usd": c["threshold"],
            "direction": c["direction"],
            "resolution_date": c["date_str"],
            "days_to_expiry": round(c["days_ahead"], 2),
            "annual_vol": round(c["annual_vol"], 4),
            "market_price_yes": c["yes_price"],
            "model_prob": round(c["model_prob"], 4),
            "side": c["side"],
            "entry_price": c["entry_price"],
            "size_usd": round(c["size"], 2),
            "edge": round(c["edge"], 4),
            "fee_usd": round(c["fee_usd"], 4),
            "balance": round(bankroll, 2),
            "resolved": False,
        }

        log_trade(record)
        trades_placed += 1

        coin_symbol = "BTC" if asset_key == "bitcoin" else "ETH"
        send_telegram(
            f"*Crypto Threshold Trade*\n"
            f"Asset: {coin_symbol} @ ${c['current_price']:,.0f}\n"
            f"Market: {c['question'][:70]}\n"
            f"Threshold: ${c['threshold']:,.0f} ({c['direction']})\n"
            f"Side: *{c['side']}* @ {c['entry_price']:.2f}\n"
            f"Model: {c['model_prob']:.0%} vs Market: {c['yes_price']:.0%}\n"
            f"Days to expiry: {c['days_ahead']:.1f} | Vol: {c['annual_vol']:.0%}\n"
            f"Size: ${c['size']:.2f} | Edge: {c['edge']:+.1%}\n"
            f"Balance: ${bankroll:.2f}\n"
            f"🕐 {datetime.now(SHANGHAI_TZ).strftime('%H:%M Shanghai')}"
        )

        time.sleep(1)

    trades_skipped = len(candidates) - trades_placed

    # Update shared context with open trades and win rate
    all_trades = load_trades()
    all_resolved = [t for t in all_trades if t.get("resolved")]
    open_trades_list = [t for t in all_trades if not t.get("resolved")]
    wins = sum(1 for t in all_resolved if t.get("won"))
    win_rate = round(wins / len(all_resolved), 2) if all_resolved else 0.0
    save_shared_context("crypto", {
        "btc_price": btc_price,
        "eth_price": eth_price,
        "btc_30d_vol": btc_vol,
        "eth_30d_vol": eth_vol,
        "trades_open": len(open_trades_list),
        "win_rate": win_rate,
        "summary": f"BTC ${btc_price:,.0f}, ETH ${eth_price:,.0f}, {len(open_trades_list)} open, {win_rate:.0%} wr",
    })

    log.info(f"Crypto scan complete. Placed {trades_placed} trades.")
    update_bot_status("crypto", trades_placed, trades_skipped, "every 30min")
    send_telegram(
        f"🔄 *Crypto Bot Scan Complete*\n"
        f"Placed: {trades_placed} | Skipped: {trades_skipped}\n"
        f"🕐 {datetime.now(SHANGHAI_TZ).strftime('%H:%M')} Shanghai | Next: 30min"
    )


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
            f"*Crypto Bot Summary — {today}*\n\n"
            f"Trades today: {len(todays_trades)}\n"
            f"Open trades: {len(open_trades)}\n"
            f"Total resolved: {len(all_resolved)}\n"
        )
        if all_resolved:
            msg += f"Win rate: {100 * wins / len(all_resolved):.0f}%\n"
        msg += f"Total P&L: ${total_pnl:+.2f}"
    else:
        msg = (
            f"*Crypto Bot Summary — {today}*\n\n"
            f"No crypto trades today.\n"
            f"Open trades: {len(open_trades)}\n"
            f"Total resolved: {len(all_resolved)}\n"
            f"Total P&L: ${total_pnl:+.2f}"
        )

    send_telegram(msg)
    log.info(f"Crypto summary sent: {len(todays_trades)} trades today")


# ---------- Entry point ----------
def main():
    ap = argparse.ArgumentParser(description="Polymarket crypto price threshold bot")
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--daily-summary", action="store_true", help="Send daily summary and exit")
    args = ap.parse_args()

    if args.daily_summary:
        update_resolutions()
        send_daily_summary()
        return

    if args.mode == "live":
        log.error("Live mode not wired. Run paper mode until calibration shows n>=30 and positive edge.")
        return

    cfg = CONFIG.copy()

    all_resolved = [t for t in load_trades() if t.get("resolved")]
    realized = sum(float(t.get("realized_pnl") or 0) for t in all_resolved)
    bankroll = cfg["starting_bankroll"] + realized

    run_scan(cfg, args.mode, bankroll)


if __name__ == "__main__":
    main()
