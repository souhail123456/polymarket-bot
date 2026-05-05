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
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from llm_router import call_llm

# ---------- Config ----------
SHANGHAI_TZ = timezone(timedelta(hours=8))
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
LOG_DIR = Path(os.environ.get("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

DEFAULT_CONFIG = {
    # Trade decision thresholds
    "min_edge": 0.10,            # min EV per $1 staked
    "min_confidence": 0.55,      # LLM self-reported confidence
    # Sizing
    "max_position_usd": 10.0,    # was $5 — size up, LLM edge pays when it hits
    "daily_loss_cap_usd": 25.0,
    "kelly_fraction": 0.25,      # quarter-Kelly
    "taker_fee_rate": 0.05,
    # Market filters — opened up to catch more opportunities
    "min_liquidity_usd": 500.0,
    "max_liquidity_usd": 500000.0,
    "max_hours_to_resolve": 720, # was 168 (7d) — now 30 days. 92/150 markets were filtered out
    "min_hours_to_resolve": 2,
    "price_floor": 0.05,         # was 0.15 — allow cheap YES bets (LLM can spot longshot value)
    "price_ceiling": 0.95,       # was 0.85 — allow high-confidence NO bets
    "max_markets_per_scan": 15,  # cap per scan to avoid draining bankroll
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


# ---------- Telegram notifications ----------
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


# ---------- Market category & live context ----------
def classify_market_category(question: str, description: str = "") -> str:
    """Classify market into category for context enrichment."""
    text = (question + " " + description).lower()

    if any(w in text for w in ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "dogecoin"]):
        return "crypto"
    if any(w in text for w in ["oil", "wti", "brent", "gold", "silver", "commodity", "natural gas"]):
        return "commodity"
    if any(w in text for w in ["nba", "nfl", "mlb", "nhl", "premier league", "la liga", "serie a",
                                "bundesliga", "champions league", "world cup", "tennis", "boxing",
                                "ufc", "mma", "handicap", "vs.", "match", "game ", " win ",
                                "cavaliers", "lakers", "celtics", "yankees", "dodgers",
                                "arsenal", "manchester", "barcelona", "real madrid"]):
        return "sports"
    if any(w in text for w in ["stock", "s&p", "nasdaq", "dow jones", "spy", "qqq", "share price"]):
        return "stocks"
    return "general"  # politics, geopolitics, celebrity, etc.


def fetch_live_context(category: str, question: str) -> str:
    """Fetch live data relevant to the market category."""
    if category == "crypto":
        return _fetch_crypto_prices()
    if category == "commodity":
        return _fetch_commodity_context(question)
    if category == "sports":
        return "NOTE: You have NO live sports data. Be honest about uncertainty. If you cannot determine a clear edge from public knowledge alone, set confidence below 0.5."
    return ""


def _fetch_crypto_prices() -> str:
    """Fetch BTC/ETH prices from CoinGecko free API."""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd&include_24hr_change=true"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "polymarket-bot/1.0")
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        lines = ["LIVE CRYPTO PRICES (use these for your estimate):"]
        for coin, info in data.items():
            price = info.get("usd", "?")
            change = info.get("usd_24h_change", 0)
            lines.append(f"  {coin.upper()}: ${price:,.2f} (24h: {change:+.1f}%)")
        return "\n".join(lines)
    except Exception as e:
        return f"WARNING: Could not fetch live crypto prices ({e}). Be very cautious with probability estimates."


def _fetch_commodity_context(question: str) -> str:
    """Return a warning about commodity prices — no free reliable API for WTI etc."""
    return "NOTE: You have NO live commodity price data. For price-threshold markets (e.g., 'Will oil hit $X'), be very conservative. If the threshold seems extreme (e.g., oil >$120, gold >$3000), the base rate for such spikes is very low."


# ---------- Pre-filter: math-obvious markets ----------
# Cached live prices so we only fetch once per scan
_LIVE_PRICES_CACHE: dict = {}

def _get_live_prices() -> dict:
    """Return {coin: price_usd} dict — cached within a scan run."""
    global _LIVE_PRICES_CACHE
    if _LIVE_PRICES_CACHE:
        return _LIVE_PRICES_CACHE
    try:
        url = (
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum,solana,dogecoin&vs_currencies=usd"
        )
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "polymarket-bot/1.0")
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        # Normalise to flat {ticker: price} map
        mapping = {
            "bitcoin": data.get("bitcoin", {}).get("usd"),
            "btc":     data.get("bitcoin", {}).get("usd"),
            "ethereum": data.get("ethereum", {}).get("usd"),
            "eth":     data.get("ethereum", {}).get("usd"),
            "solana":  data.get("solana",   {}).get("usd"),
            "sol":     data.get("solana",   {}).get("usd"),
            "dogecoin": data.get("dogecoin", {}).get("usd"),
            "doge":    data.get("dogecoin",  {}).get("usd"),
        }
        _LIVE_PRICES_CACHE = {k: v for k, v in mapping.items() if v is not None}
        return _LIVE_PRICES_CACHE
    except Exception as e:
        log.debug(f"_get_live_prices failed: {e}")
        return {}


def _reset_live_prices_cache():
    """Call once per scan so prices are re-fetched fresh next scan."""
    global _LIVE_PRICES_CACHE
    _LIVE_PRICES_CACHE = {}


import re

# Patterns: "Will BTC be above $50,000" / "below $50k" / "exceed $50,000" / "reach $50k"
_PRICE_THRESHOLD_RE = re.compile(
    r"\b(bitcoin|btc|ethereum|eth|solana|sol|dogecoin|doge)\b"
    r".{0,80}"
    r"\b(above|over|exceed|reach|hit|below|under|drop below|fall below)\b"
    r".{0,30}"
    r"\$?([\d,]+(?:\.\d+)?)\s*([km]?)",
    re.IGNORECASE,
)


def _parse_price_threshold(question: str):
    """
    Try to extract (coin, direction, threshold_usd) from question.
    direction is 'above' or 'below'.
    Returns None if no clear threshold found.
    """
    m = _PRICE_THRESHOLD_RE.search(question)
    if not m:
        return None
    coin = m.group(1).lower()
    verb = m.group(2).lower()
    raw_num = m.group(3).replace(",", "")
    suffix = m.group(4).lower()

    try:
        threshold = float(raw_num)
    except ValueError:
        return None

    if suffix == "k":
        threshold *= 1_000
    elif suffix == "m":
        threshold *= 1_000_000

    direction = "above" if any(w in verb for w in ("above", "over", "exceed", "reach", "hit")) else "below"
    return coin, direction, threshold


def pre_filter_market(market) -> "tuple[float, float, str] | None":
    """
    Fast math-based pre-filter. No LLM needed.

    Returns (true_probability, confidence, reasoning) when the outcome is
    mathematically obvious, or None when LLM evaluation is required.

    Currently handles:
    - Crypto price threshold markets (BTC/ETH/SOL/DOGE above/below $X)
    - Markets whose resolution date has already passed
    """
    question = market.get("question", "")
    hours_left = market.get("_hours_left", 999)

    # --- Guard: resolution already passed ---
    # (filter_markets already enforces min_hours_to_resolve > 0, but belt-and-suspenders)
    if hours_left <= 0:
        return 0.5, 0.5, "Resolution date already passed — outcome ambiguous"

    # --- Crypto price threshold ---
    parsed = _parse_price_threshold(question)
    if parsed is None:
        return None  # not a price-threshold market — let LLM handle it

    coin, direction, threshold = parsed
    prices = _get_live_prices()
    current_price = prices.get(coin)
    if current_price is None:
        return None  # can't determine live price — fall back to LLM

    ratio = current_price / threshold  # >1 means current > threshold

    if direction == "above":
        pct_above = (current_price - threshold) / threshold  # positive = current above threshold
        if pct_above >= 0.20:
            # Current price is ≥20% above threshold — very likely YES even with time left
            conf = min(0.97, 0.80 + pct_above * 0.5)
            return (
                min(0.99, 0.92 + pct_above * 0.3),
                round(conf, 2),
                f"{coin.upper()} is ${current_price:,.0f}, {pct_above:.0%} above threshold ${threshold:,.0f} — very likely YES",
            )
        if pct_above <= -0.20:
            # Current price is ≥20% below threshold — very likely NO
            pct_below = abs(pct_above)
            conf = min(0.97, 0.80 + pct_below * 0.5)
            return (
                max(0.01, 0.08 - pct_below * 0.3),
                round(conf, 2),
                f"{coin.upper()} is ${current_price:,.0f}, {pct_below:.0%} below threshold ${threshold:,.0f} — very likely NO",
            )
    else:  # direction == "below"
        pct_below = (threshold - current_price) / threshold  # positive = current below threshold
        if pct_below >= 0.20:
            # Current price is ≥20% below threshold — very likely YES (below target)
            conf = min(0.97, 0.80 + pct_below * 0.5)
            return (
                min(0.99, 0.92 + pct_below * 0.3),
                round(conf, 2),
                f"{coin.upper()} is ${current_price:,.0f}, {pct_below:.0%} below threshold ${threshold:,.0f} — very likely YES (below target)",
            )
        if pct_below <= -0.20:
            # Current price is ≥20% above threshold — very likely NO (not below target)
            pct_above = abs(pct_below)
            conf = min(0.97, 0.80 + pct_above * 0.5)
            return (
                max(0.01, 0.08 - pct_above * 0.3),
                round(conf, 2),
                f"{coin.upper()} is ${current_price:,.0f}, {pct_above:.0%} above threshold ${threshold:,.0f} — very likely NO (not below target)",
            )

    # Price is within 20% of threshold — not obvious, send to LLM
    return None


# ---------- Time-decay sniper ----------
def _apply_time_decay_sniper(
    true_p: float,
    conf: float,
    reason: str,
    market_p: float,
    hours_left: float,
) -> "tuple[float, float, str]":
    """
    Near-expiry booster: when a market has <6 hours left and the crowd price is
    extreme (>0.85 YES or <0.15 YES), unlikely reversals become even less likely.

    Rule: nudge true_p 10% closer to market_p and floor confidence at 0.70.
    This captures the time-decay edge — status-quo persistence accelerates as
    resolution approaches.

    Example: market_p=0.90, LLM true_p=0.80 → boosted true_p=0.81 (10% of gap closed)
    """
    if hours_left >= 6:
        return true_p, conf, reason
    if not (market_p > 0.85 or market_p < 0.15):
        return true_p, conf, reason

    original_true_p = true_p
    # Close 10% of the gap between LLM estimate and market price
    true_p = true_p + 0.10 * (market_p - true_p)
    true_p = round(max(0.01, min(0.99, true_p)), 4)
    conf = round(max(0.70, conf), 2)

    direction = "YES" if market_p > 0.85 else "NO"
    boost_note = (
        f" [time-decay sniper: {hours_left:.1f}h left, mkt={market_p:.2f} strongly implies"
        f" {direction}; true_p nudged {original_true_p:.2f}→{true_p:.2f}, conf floored at {conf:.2f}]"
    )
    reason = (reason or "").rstrip(".") + boost_note

    log.debug(
        f"Time-decay sniper applied: hours_left={hours_left:.1f} market_p={market_p:.2f} "
        f"true_p {original_true_p:.2f}→{true_p:.2f} conf→{conf:.2f}"
    )
    return true_p, conf, reason


# ---------- Claude probability estimation ----------
PROMPT = """You are an aggressive prediction market trader. Your job: estimate the TRUE probability this market resolves YES, then explain why the crowd is wrong.

MARKET: {question}

DESCRIPTION: {description}

CURRENT MARKET PRICE (crowd's implied probability of YES): {yes_price:.1%}
HOURS UNTIL RESOLUTION: {hours_left:.1f}
RESOLUTION SOURCE: {resolution}
{live_context}

Think through:
1. What exactly must happen for YES?
2. Base rate for this type of event? Historical frequency?
3. What concrete evidence do you have NOW that the crowd is missing?
4. Common crowd biases: recency bias, narrative bias, longshot overpricing, favorite-longshot bias, anchoring to round numbers.
5. Time decay: if resolution is soon and nothing has changed, current state likely persists.

YOUR EDGE: You have broad knowledge of base rates, historical patterns, and cognitive biases. The crowd often overprices dramatic outcomes and underprices boring ones. If you see a clear mispricing, be bold — that's where money is made.

MARKET REGIME: {regime}
{regime_guidance}

CRITICAL RULES:
- If this market involves a PRICE (crypto, oil, stocks), you MUST check the live prices above. If no live price is available, set confidence below 0.5.
- If this is a SPORTS match, you have NO live stats. Only bet if you have strong public knowledge of a massive skill gap. Otherwise set confidence below 0.5.
- Your reasoning MUST cite specific evidence (numbers, dates, facts). "Recency bias" alone is NOT valid reasoning.
- Confidence MUST vary per trade. Using 0.80 for everything means you're not calibrating.

Output ONLY this JSON, no other text:
{{"true_probability": <0-1>, "confidence": <0-1 — MUST vary: use 0.5-0.6 for low info, 0.7-0.8 for moderate, 0.9+ only with strong data>, "reasoning": "<one sentence with SPECIFIC evidence, not generic bias names>"}}"""


def _load_regime():
    """Load market regime from trading-admin shared context."""
    try:
        with open("/tmp/shared_global_state.json") as f:
            state = json.load(f)
        regime = state.get("regime", "UNKNOWN")
        guidance = {
            "CRISIS": "CAUTION: Market is in CRISIS mode (high VIX). Require higher confidence threshold — only trade if edge > 15%. Prefer YES on safe-haven/hedging markets.",
            "VOLATILE": "Market is VOLATILE. Widen your confidence threshold — only trade if edge > 10%.",
            "RANGING": "",
            "TRENDING": "",
        }.get(regime, "")
        return regime, guidance
    except Exception:
        return "UNKNOWN", ""


def estimate_probability(market):
    """Estimate probability for a single market (fallback if batch fails)."""
    regime, regime_guidance = _load_regime()
    category = classify_market_category(market.get("question", ""), market.get("description", ""))
    live_context = fetch_live_context(category, market.get("question", ""))
    prompt = PROMPT.format(
        question=(market.get("question") or "")[:500],
        description=(market.get("description") or "")[:1200],
        yes_price=market["_yes_price"],
        hours_left=market["_hours_left"],
        resolution=(market.get("resolutionSource") or "See description")[:300],
        regime=regime,
        regime_guidance=regime_guidance,
        live_context=live_context,
    )
    try:
        text, provider, model = call_llm(prompt, max_tokens=400, skip="gemini")
        log.info(f"LLM response from {provider}/{model} for {market.get('_market_id', '')[:8]}")
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        true_p = float(data["true_probability"])
        conf = float(data["confidence"])
        reason = data.get("reasoning", "")
        true_p, conf, reason = _apply_time_decay_sniper(
            true_p, conf, reason,
            market_p=market["_yes_price"],
            hours_left=market["_hours_left"],
        )
        return true_p, conf, reason
    except Exception as e:
        log.warning(f"estimate_probability failed for {market.get('_market_id')}: {e}")
        return None, None, None


def estimate_probability_batch(markets):
    """Evaluate multiple markets in one LLM call. Falls back to individual calls on failure."""
    if not markets:
        return {}

    regime, regime_guidance = _load_regime()

    # Build one prompt with all markets
    market_lines = []
    for i, m in enumerate(markets):
        category = classify_market_category(m.get("question", ""), m.get("description", ""))
        live_ctx = fetch_live_context(category, m.get("question", ""))
        market_lines.append(
            f"MARKET {i+1} (id={m['_market_id'][:12]}):\n"
            f"  Question: {(m.get('question') or '')[:300]}\n"
            f"  Description: {(m.get('description') or '')[:400]}\n"
            f"  Market price (YES): {m['_yes_price']:.1%}\n"
            f"  Hours to resolve: {m['_hours_left']:.1f}\n"
            f"  Resolution: {(m.get('resolutionSource') or 'See description')[:200]}\n"
            f"  {live_ctx}"
        )

    batch_prompt = f"""You are an aggressive prediction market trader. Evaluate ALL markets below and estimate TRUE probability each resolves YES.

REGIME: {regime}
{regime_guidance}

{"---".join(market_lines)}

CRITICAL RULES:
- If a market involves a PRICE (crypto, oil, stocks), you MUST use live prices if provided. If none, confidence below 0.5.
- If a SPORTS match with no stats, confidence below 0.5.
- Reasoning MUST cite specific evidence. "Recency bias" alone is NOT valid.
- Confidence MUST vary per market.

Output ONLY a JSON array, one object per market, in order. No other text:
[{{"market_id": "<id>", "true_probability": <0-1>, "confidence": <0-1>, "reasoning": "<one sentence>"}}]"""

    try:
        text, provider, model = call_llm(batch_prompt, max_tokens=200 * len(markets), temperature=0.3, prefer="gemini")
        log.info(f"Batch LLM response from {provider}/{model} for {len(markets)} markets")
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        # Extract JSON array if surrounded by other text
        if "[" in text and "]" in text:
            text = text[text.index("["):text.rindex("]") + 1]
        results = json.loads(text)
        if not isinstance(results, list) or len(results) != len(markets):
            raise ValueError(f"Expected {len(markets)} results, got {len(results) if isinstance(results, list) else 'non-list'}")

        # Map results back to market IDs, then apply time-decay sniper
        output = {}
        for i, m in enumerate(markets):
            r = results[i]
            true_p = float(r["true_probability"])
            conf = float(r["confidence"])
            reason = r.get("reasoning", "")

            true_p, conf, reason = _apply_time_decay_sniper(
                true_p, conf, reason,
                market_p=m["_yes_price"],
                hours_left=m["_hours_left"],
            )

            output[m["_market_id"]] = (true_p, conf, reason)
        log.info(f"Batch evaluation succeeded: {len(output)} markets in 1 LLM call")
        return output
    except Exception as e:
        log.warning(f"Batch evaluation failed ({e}), falling back to individual calls")
        output = {}
        for m in markets:
            result = estimate_probability(m)
            output[m["_market_id"]] = result
        return output


# ---------- EV math ----------
def calculate_fee(price, size_usd, fee_rate):
    """Polymarket taker fee: feeRate * price * (1 - price) * shares.
    Shares = size_usd / price, so fee = feeRate * (1 - price) * size_usd."""
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


def decide_trade(true_p, market_p, confidence, bankroll, cfg):
    if confidence < cfg["min_confidence"]:
        return None, 0.0, 0.0, 0.0
    ev_yes = expected_value(true_p, market_p, "YES")
    ev_no = expected_value(true_p, market_p, "NO")
    side, gross_edge = ("YES", ev_yes) if ev_yes >= ev_no else ("NO", ev_no)
    # Estimate fee as fraction of $1 staked
    entry_price = market_p if side == "YES" else 1 - market_p
    fee_per_dollar = calculate_fee(entry_price, 1.0, cfg["taker_fee_rate"])
    edge = gross_edge - fee_per_dollar
    if edge < cfg["min_edge"]:
        return None, 0.0, edge, fee_per_dollar
    size = kelly_size(true_p, market_p, side, bankroll, cfg["kelly_fraction"], cfg["max_position_usd"])
    if size < 1.0:
        return None, 0.0, edge, fee_per_dollar
    fee_usd = calculate_fee(entry_price, size, cfg["taker_fee_rate"])
    return side, size, edge, fee_usd


# ---------- LLM cache — skip markets whose price hasn't moved ----------
EVAL_CACHE_FILE = LOG_DIR / "eval_cache.json"
CACHE_PRICE_THRESHOLD = 0.03  # re-evaluate only if price moved > 3%


def load_eval_cache():
    if not EVAL_CACHE_FILE.exists():
        return {}
    try:
        with open(EVAL_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_eval_cache(cache):
    with open(EVAL_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def should_reevaluate(market_id, current_price, cache):
    """Return True if we should call LLM, False if cached result is still valid."""
    if market_id not in cache:
        return True
    cached = cache[market_id]
    old_price = cached.get("price", 0)
    if abs(current_price - old_price) >= CACHE_PRICE_THRESHOLD:
        return True
    # Cache is fresh (price barely moved) — skip LLM call
    return False


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
    today = datetime.now(SHANGHAI_TZ).date().isoformat()
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
        t["resolved_at"] = datetime.now(SHANGHAI_TZ).isoformat()
        changed = True

        all_resolved = [tr for tr in trades if tr.get("resolved")]
        total_pnl = sum(float(tr.get("realized_pnl", 0)) for tr in all_resolved)
        balance = DEFAULT_CONFIG["starting_bankroll"] + total_pnl
        t["balance"] = round(balance, 2)

        log.info(f"Resolved {t['market_id'][:8]}: {t['side']} -> {'WIN' if won else 'LOSS'} pnl=${pnl:.2f} balance=${balance:.2f}")
        send_telegram(
            f"{'✅' if won else '❌'} *Trade Resolved*\n"
            f"Market: {t['question'][:80]}\n"
            f"Side: {t['side']} → *{'WIN' if won else 'LOSS'}*\n"
            f"P&L: ${pnl:+.2f}\n"
            f"Balance: ${balance:.2f}"
        )

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


# ---------- Daily summary ----------
def send_daily_summary():
    today = datetime.now(SHANGHAI_TZ).date().isoformat()
    trades = load_trades()
    todays_trades = [t for t in trades if t.get("date") == today]
    all_resolved = [t for t in trades if t.get("resolved")]
    total_pnl = sum(float(t.get("realized_pnl") or 0) for t in all_resolved)
    todays_pnl = sum(float(t.get("realized_pnl") or 0) for t in todays_trades if t.get("resolved"))
    open_trades = [t for t in trades if not t.get("resolved")]
    wins = sum(1 for t in all_resolved if t.get("won"))

    msg = (
        f"📋 *Daily Summary — {today}*\n\n"
        f"Trades today: {len(todays_trades)}\n"
        f"Today's P&L: ${todays_pnl:+.2f}\n\n"
        f"Open trades: {len(open_trades)}\n"
        f"Total resolved: {len(all_resolved)}\n"
        f"Win rate: {100*wins/len(all_resolved):.0f}%\n" if all_resolved else ""
        f"Total P&L: ${total_pnl:+.2f}\n"
    )

    if not todays_trades:
        msg = (
            f"📋 *Daily Summary — {today}*\n\n"
            f"No trades today.\n\n"
            f"Open trades: {len(open_trades)}\n"
            f"Total resolved: {len(all_resolved)}\n"
            f"Total P&L: ${total_pnl:+.2f}"
        )

    send_telegram(msg)
    log.info(f"Daily summary sent: {len(todays_trades)} trades today, total P&L ${total_pnl:+.2f}")


# ---------- Live trading (disabled stub) ----------
def submit_live_order(market, side, size_usd, price):
    raise NotImplementedError(
        "Live mode not wired. Requires py-clob-client + wallet setup. "
        "Run paper mode until calibration report shows n>=30 and positive edge."
    )


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


# ---------- Main scan ----------
def run_scan(cfg, mode, bankroll):
    log.info(f"=== Scan | mode={mode} | bankroll=${bankroll:.2f} ===")

    # Log shared context from other bots
    shared = load_shared_context()
    for bot_name, ctx in shared.items():
        if bot_name != "ev":
            log.info(f"[shared/{bot_name}] {ctx.get('summary', 'no summary')} (as of {ctx.get('updated_at', '?')})")

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

    eval_cache = load_eval_cache()
    cache_hits = 0
    trades_placed = 0

    # Reset live-price cache so we get fresh prices for this scan
    _reset_live_prices_cache()

    # Pre-filter: resolve mathematically obvious markets without LLM
    pre_filtered_results: dict = {}
    pre_filter_hits = 0
    remaining_candidates = []
    for m in candidates:
        pf = pre_filter_market(m)
        if pf is not None:
            true_p, conf, reason = pf
            pre_filtered_results[m["_market_id"]] = (true_p, conf, reason)
            pre_filter_hits += 1
            log.info(
                f"[{m['_market_id'][:8]}] Pre-filter hit: {m['question'][:60]!r} "
                f"=> true_p={true_p:.2f} conf={conf:.2f} | {reason}"
            )
        else:
            remaining_candidates.append(m)
    if pre_filter_hits:
        log.info(f"{pre_filter_hits} markets resolved by pre-filter (no LLM needed)")

    # Split remaining candidates into cached (skip) and needs_eval (batch LLM)
    needs_eval = []
    for m in remaining_candidates:
        market_id = m["_market_id"]
        market_p = m["_yes_price"]
        if not should_reevaluate(market_id, market_p, eval_cache):
            log.info(f"[{market_id[:8]}] Cache hit — price {market_p:.2f} unchanged, skipping LLM")
            cache_hits += 1
        else:
            needs_eval.append(m)

    # Batch evaluate all non-cached markets in one LLM call
    batch_results = estimate_probability_batch(needs_eval) if needs_eval else {}

    # Merge pre-filter and batch results
    all_results = {**pre_filtered_results, **batch_results}

    for m in candidates:
        if bankroll < 2.0:
            log.info(f"Bankroll ${bankroll:.2f} too low — stopping scan")
            break
        market_id = m["_market_id"]
        market_p = m["_yes_price"]

        # Get result from pre-filter / batch, or skip if cached
        if market_id in all_results:
            true_p, conf, reason = all_results[market_id]
            if true_p is None:
                continue
            # Cache this evaluation (pre-filter results also cached to avoid re-checking)
            eval_cache[market_id] = {"price": market_p, "true_p": true_p, "conf": conf}
        else:
            continue  # cached — already counted above

        side, size, edge, fee_usd = decide_trade(true_p, market_p, conf, bankroll, cfg)

        log.info(
            f"[{m['_market_id'][:8]}] {m['question'][:60]!r} "
            f"mkt={market_p:.2f} est={true_p:.2f} conf={conf:.2f} edge={edge:+.1%} "
            f"fee=${fee_usd:.2f} => {side or 'skip'} ${size:.2f}"
        )

        if side is None:
            continue

        bankroll -= size

        record = {
            "timestamp": datetime.now(SHANGHAI_TZ).isoformat(),
            "date": datetime.now(SHANGHAI_TZ).date().isoformat(),
            "mode": mode,
            "market_id": m["_market_id"],
            "question": m["question"][:200],
            "category": classify_market_category(m.get("question", ""), m.get("description", "")),
            "hours_to_resolve": round(m["_hours_left"], 2),
            "market_price": market_p,
            "estimated_prob": true_p,
            "confidence": conf,
            "reasoning": reason,
            "side": side,
            "entry_price": market_p if side == "YES" else 1 - market_p,
            "size_usd": round(size, 2),
            "edge": round(edge, 4),
            "fee_usd": round(fee_usd, 4),
            "balance": round(bankroll, 2),
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

        send_telegram(
            f"📊 *New Paper Trade*\n"
            f"Market: {m['question'][:80]}\n"
            f"Side: *{side}* @ {record['entry_price']:.2f}\n"
            f"Size: ${size:.2f} | Edge: {edge:+.1%}\n"
            f"Reasoning: {reason[:150]}\n"
            f"Balance: ${bankroll:.2f}"
        )

        # Small delay to avoid hammering APIs
        time.sleep(2)

    trades_skipped = len(candidates) - trades_placed

    # Save shared context
    all_trades = load_trades()
    all_resolved = [t for t in all_trades if t.get("resolved")]
    open_trades = [t for t in all_trades if not t.get("resolved")]
    wins = sum(1 for t in all_resolved if t.get("won"))
    win_rate = round(wins / len(all_resolved), 2) if all_resolved else 0.0
    save_shared_context("ev", {
        "trades_open": len(open_trades),
        "trades_resolved": len(all_resolved),
        "win_rate": win_rate,
        "summary": f"{len(open_trades)} open trades, {win_rate:.0%} win rate",
    })

    save_eval_cache(eval_cache)
    log.info(
        f"Scan complete. Placed {trades_placed} trades. "
        f"Pre-filter: {pre_filter_hits} | Cache hits: {cache_hits} | LLM evals: {len(needs_eval)}"
    )
    update_bot_status("ev", trades_placed, trades_skipped, "every 30min")
    send_telegram(
        f"🔄 *EV Bot Scan Complete*\n"
        f"Placed: {trades_placed} | Skipped: {trades_skipped}\n"
        f"Pre-filter: {pre_filter_hits} | Cached: {cache_hits} | LLM: {len(needs_eval)}\n"
        f"🕐 {datetime.now(SHANGHAI_TZ).strftime('%H:%M')} Shanghai | Next: 30min"
    )


# ---------- Entry point ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    ap.add_argument("--loop", action="store_true", help="Run continuously every 30 min")
    ap.add_argument("--report", action="store_true", help="Print calibration report and exit")
    ap.add_argument("--daily-summary", action="store_true", help="Send daily summary to Telegram and exit")
    ap.add_argument("--max-position", type=float, default=None)
    args = ap.parse_args()

    if args.report:
        update_resolutions()
        calibration_report()
        return

    if args.daily_summary:
        update_resolutions()
        send_daily_summary()
        return

    # LLM calls go through llm_router.py (Groq -> Gemini -> Cerebras)
    # At least one of GROQ_API_KEY, GEMINI_API_KEY, CEREBRAS_API_KEY must be set
    if not any(os.environ.get(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY")):
        log.error("Set at least one LLM API key: GROQ_API_KEY, GEMINI_API_KEY, or CEREBRAS_API_KEY")
        return

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
                run_scan(cfg, args.mode, bankroll)
            except Exception as e:
                log.exception(f"Scan error: {e}")
            log.info("Sleeping 30 min...")
            time.sleep(30 * 60)
    else:
        run_scan(cfg, args.mode, bankroll)


if __name__ == "__main__":
    main()
