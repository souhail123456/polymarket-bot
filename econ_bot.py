"""
Polymarket Economic Data Trading Bot
-------------------------------------
Uses public nowcasting models from the Federal Reserve and CME FedWatch
to estimate economic indicator probabilities, then trades Polymarket
markets when the data strongly disagrees with the crowd.

No LLM calls — pure data-driven.

Data sources (all free, no key required except FRED):
  1. Cleveland Fed CPI Nowcast  — monthly CPI prediction
  2. Atlanta Fed GDPNow         — real-time GDP estimate
  3. CME FedWatch               — Fed rate cut/hike probabilities
  4. FRED API                   — recent actuals / trend (key from fred.stlouisfed.org)
  5. BLS public data            — unemployment / nonfarm payrolls

Strategy: fetch the relevant nowcast for each market type, compare to
market-implied probability, trade when edge exceeds threshold.

Usage:
  python econ_bot.py --mode paper
  python econ_bot.py --daily-summary
"""

import os
import re
import json
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

# ---------- Config ----------
UTC = timezone.utc
SHANGHAI_TZ = timezone(timedelta(hours=8))
GAMMA_API = "https://gamma-api.polymarket.com"
LOG_DIR = Path(os.environ.get("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# FRED is free — get a key at fred.stlouisfed.org (optional; falls back to scraping)
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

CONFIG = {
    "min_edge": 0.12,
    "kelly_fraction": 0.15,
    "max_position_usd": 15.0,
    "taker_fee_rate": 0.02,       # economic markets tend to have lower fees
    "starting_bankroll": 100.0,
    "daily_loss_cap_usd": 20.0,
}

# Search terms we use to find relevant Polymarket markets
ECON_SEARCH_TERMS = [
    "CPI",
    "inflation",
    "unemployment",
    "jobs",
    "Fed",
    "GDP",
    "interest rate",
    "nonfarm payrolls",
    "federal funds",
    "core CPI",
    "PCE",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "econ_bot.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ---------- Telegram ----------
def send_telegram(msg: str):
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
TRADES_FILE = LOG_DIR / "econ_trades.jsonl"


def log_trade(record: dict):
    with open(TRADES_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_trades() -> list:
    if not TRADES_FILE.exists():
        return []
    out = []
    with open(TRADES_FILE) as f:
        for line in f:
            try:
                out.append(json.loads(line.strip()))
            except Exception:
                pass
    return out


def save_trades(trades: list):
    with open(TRADES_FILE, "w") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")


def already_traded_market_ids() -> set:
    return {t["market_id"] for t in load_trades()}


# ==========================================================================
# SECTION 1: Fetch active economic markets from Polymarket
# ==========================================================================

def fetch_econ_markets() -> list:
    """
    Search Polymarket Gamma API for active economic/financial markets.
    Returns a flat list of market dicts annotated with _search_term.
    """
    seen_ids = set()
    markets = []

    for term in ECON_SEARCH_TERMS:
        try:
            r = requests.get(
                f"{GAMMA_API}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": 50,
                    "q": term,
                },
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            # Gamma API returns either a list directly or {"markets": [...]}
            if isinstance(data, list):
                batch = data
            elif isinstance(data, dict):
                batch = data.get("markets", data.get("data", []))
            else:
                batch = []

            for m in batch:
                mid = str(m.get("id", ""))
                if not mid or mid in seen_ids:
                    continue
                seen_ids.add(mid)
                m["_search_term"] = term
                markets.append(m)

        except Exception as e:
            log.warning(f"fetch_econ_markets failed for '{term}': {e}")

    # Also try the public-search endpoint which sometimes returns more
    for term in ["CPI", "Fed rate", "unemployment", "nonfarm"]:
        try:
            r = requests.get(
                f"{GAMMA_API}/public-search",
                params={"q": term, "events_status": "active"},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            for event in data.get("events", []):
                for m in event.get("markets", []):
                    mid = str(m.get("id", ""))
                    if not mid or mid in seen_ids:
                        continue
                    seen_ids.add(mid)
                    m["_search_term"] = term
                    m["_event_title"] = event.get("title", "")
                    markets.append(m)
        except Exception as e:
            log.warning(f"public-search failed for '{term}': {e}")

    log.info(f"Found {len(markets)} unique econ markets")
    return markets


# ==========================================================================
# SECTION 2: Nowcast data fetchers
# ==========================================================================

# --- 2a. Cleveland Fed CPI Nowcast ---

def fetch_cleveland_fed_cpi() -> dict | None:
    """
    Cleveland Fed publishes CPI and Core CPI nowcasts at:
    https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting

    They have a JSON endpoint for their data tool. We try the JSON first,
    then fall back to scraping the HTML table.

    Returns: {"cpi_yoy": float, "core_cpi_yoy": float, "month": "YYYY-MM"} or None
    """
    # Attempt 1: Their published data JSON (format may change — wrapped in try)
    json_url = (
        "https://www.clevelandfed.org/en/indicators-and-data/inflation-nowcasting"
        "/-/media/project/cleveland-federalreserve/cleveland-fed-site/indicators-and-data"
        "/inflation-nowcasting/inflation-nowcasting-data-latest-available.xlsx"
    )
    # JSON API attempt — Cleveland Fed has changed their data format before;
    # we use a known working endpoint that returns JSON for the chart data.
    try:
        api_url = (
            "https://www.clevelandfed.org/en/indicators-and-data"
            "/inflation-nowcasting.aspx"
        )
        r = requests.get(api_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        # Try to extract JSON embedded in the page
        text = r.text
        # Look for patterns like "cpiNowcast":2.87 or similar
        m = re.search(r'"cpiNowcast"\s*:\s*([\d.]+)', text)
        core_m = re.search(r'"coreCpiNowcast"\s*:\s*([\d.]+)', text)
        if m:
            result = {
                "cpi_yoy": float(m.group(1)),
                "core_cpi_yoy": float(core_m.group(1)) if core_m else None,
                "month": datetime.now(UTC).strftime("%Y-%m"),
                "source": "cleveland_fed_scrape",
            }
            log.info(f"Cleveland Fed CPI nowcast: {result}")
            return result
    except Exception as e:
        log.debug(f"Cleveland Fed scrape attempt 1 failed: {e}")

    # Attempt 2: FRED series for Cleveland Fed nowcast (CPINOWCAST)
    # The Cleveland Fed publishes to FRED under specific series IDs
    if FRED_API_KEY:
        try:
            # EXPINF1YR = 1-year inflation expectations from Cleveland Fed
            r = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": "EXPINF1YR",
                    "api_key": FRED_API_KEY,
                    "file_type": "json",
                    "limit": 3,
                    "sort_order": "desc",
                },
                timeout=10,
            )
            r.raise_for_status()
            obs = r.json().get("observations", [])
            if obs:
                val = float(obs[0]["value"])
                log.info(f"Cleveland Fed 1Y inflation expectation (FRED): {val}")
                return {
                    "cpi_yoy": val,
                    "core_cpi_yoy": None,
                    "month": obs[0]["date"][:7],
                    "source": "fred_expinf1yr",
                }
        except Exception as e:
            log.debug(f"FRED EXPINF1YR fetch failed: {e}")

    # Attempt 3: Use most recent actual CPI from FRED as a floor estimate
    # (not a nowcast, but gives us the trend direction)
    if FRED_API_KEY:
        try:
            r = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": "CPIAUCSL",
                    "api_key": FRED_API_KEY,
                    "file_type": "json",
                    "limit": 13,
                    "sort_order": "desc",
                    "units": "pc1",           # percent change year-over-year
                },
                timeout=10,
            )
            r.raise_for_status()
            obs = r.json().get("observations", [])
            if len(obs) >= 2:
                latest = float(obs[0]["value"])
                prev = float(obs[1]["value"])
                # Simple trend extrapolation — not a real nowcast, but
                # directionally useful for large discrepancies
                trend_estimate = latest + (latest - prev) * 0.5
                log.info(
                    f"CPI YoY actual (FRED): {latest:.2f}%, trend est: {trend_estimate:.2f}%"
                )
                return {
                    "cpi_yoy": latest,
                    "cpi_yoy_trend": round(trend_estimate, 2),
                    "month": obs[0]["date"][:7],
                    "source": "fred_actual_trend",
                    "is_actual_not_nowcast": True,
                }
        except Exception as e:
            log.debug(f"FRED CPI actual fetch failed: {e}")

    log.warning("Cleveland Fed CPI: all fetch methods failed — skipping CPI markets")
    return None


# --- 2b. Atlanta Fed GDPNow ---

def fetch_gdpnow() -> dict | None:
    """
    Atlanta Fed GDPNow: https://www.atlantafed.org/cqer/research/gdpnow
    They publish a JSON/CSV of the current estimate.

    Returns: {"gdp_growth_pct": float, "quarter": "QXYYY", "as_of": "YYYY-MM-DD"} or None
    """
    # Attempt 1: Atlanta Fed publishes GDPNow as a downloadable file
    try:
        # Known direct data URL for GDPNow tracking
        data_url = (
            "https://www.atlantafed.org/cqer/research/gdpnow"
        )
        r = requests.get(data_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        text = r.text

        # Look for the current estimate in page text/JS data
        # Pattern: "GDPNow model estimate for real GDP growth ... X.X percent"
        m = re.search(
            r'GDPNow\s+model\s+estimate[^:]*:\s*([-\d.]+)\s*percent',
            text,
            re.IGNORECASE,
        )
        if not m:
            # Try JSON-like patterns embedded in page
            m = re.search(r'"gdpnow"\s*:\s*([-\d.]+)', text, re.IGNORECASE)
        if not m:
            # Try: "estimate of X.X%" type patterns
            m = re.search(
                r'estimate\s+(?:of|for)[^:]*?(?:is\s+)?([-\d.]+)\s*(?:percent|%)',
                text,
                re.IGNORECASE,
            )

        if m:
            val = float(m.group(1))
            log.info(f"GDPNow estimate: {val}%")
            return {
                "gdp_growth_pct": val,
                "source": "atlanta_fed_scrape",
                "as_of": datetime.now(UTC).strftime("%Y-%m-%d"),
            }
    except Exception as e:
        log.debug(f"GDPNow scrape failed: {e}")

    # Attempt 2: FRED for advance GDP estimate (GDPNow is not on FRED,
    # but we can use the most recent actual + nowcast from PCE data)
    if FRED_API_KEY:
        try:
            r = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": "A191RL1Q225SBEA",  # Real GDP growth QoQ annualised
                    "api_key": FRED_API_KEY,
                    "file_type": "json",
                    "limit": 4,
                    "sort_order": "desc",
                },
                timeout=10,
            )
            r.raise_for_status()
            obs = r.json().get("observations", [])
            if obs:
                latest = float(obs[0]["value"])
                log.info(f"Real GDP growth (FRED actual): {latest}%")
                return {
                    "gdp_growth_pct": latest,
                    "source": "fred_gdp_actual",
                    "as_of": obs[0]["date"],
                    "is_actual_not_nowcast": True,
                }
        except Exception as e:
            log.debug(f"FRED GDP fetch failed: {e}")

    log.warning("GDPNow: all fetch methods failed — skipping GDP markets")
    return None


# --- 2c. CME FedWatch ---

def fetch_fedwatch() -> dict | None:
    """
    CME FedWatch publishes Fed rate cut/hike probabilities.
    https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html

    We scrape or use their published JSON API for the next FOMC meeting.

    Returns: {
        "next_meeting_date": "YYYY-MM-DD",
        "cut_prob": 0.XX,       # probability of at least 25bp cut
        "hold_prob": 0.XX,
        "hike_prob": 0.XX,
        "current_rate_bps": 525,
    } or None
    """
    # CME publishes FedWatch data as JSON via their CDN
    # This endpoint returns the probability table for upcoming FOMC meetings
    cme_endpoints = [
        "https://www.cmegroup.com/CmeWS/mvc/aggregatedView/FedFundsFuturesViewer.json",
        "https://www.cmegroup.com/CmeWS/mvc/FedWatch/fedwatch.json",
    ]

    for url in cme_endpoints:
        try:
            r = requests.get(
                url,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://www.cmegroup.com/",
                },
            )
            r.raise_for_status()
            data = r.json()
            # Parse their data structure — it varies by endpoint
            result = _parse_cme_fedwatch_json(data)
            if result:
                log.info(f"CME FedWatch: {result}")
                return result
        except Exception as e:
            log.debug(f"CME FedWatch endpoint {url} failed: {e}")

    # Fallback: scrape the FedWatch HTML page
    try:
        r = requests.get(
            "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        text = r.text

        # Look for JSON blobs embedded in the page
        # Pattern: probabilities in the page script data
        m = re.search(r'"probabilities"\s*:\s*(\[.*?\])', text, re.DOTALL)
        if m:
            probs = json.loads(m.group(1))
            if probs:
                result = _parse_cme_probs_array(probs)
                if result:
                    log.info(f"CME FedWatch (HTML scrape): {result}")
                    return result
    except Exception as e:
        log.debug(f"CME FedWatch HTML scrape failed: {e}")

    # Last resort: use Fed Funds futures from FRED (FEDFUNDS)
    if FRED_API_KEY:
        try:
            r = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": "FEDFUNDS",
                    "api_key": FRED_API_KEY,
                    "file_type": "json",
                    "limit": 2,
                    "sort_order": "desc",
                },
                timeout=10,
            )
            r.raise_for_status()
            obs = r.json().get("observations", [])
            if obs:
                current_rate = float(obs[0]["value"])
                log.info(f"Fed Funds actual rate (FRED): {current_rate}%")
                return {
                    "current_rate_bps": round(current_rate * 100),
                    "current_rate_pct": current_rate,
                    "source": "fred_fedfunds_actual",
                    "cut_prob": None,   # Can't derive cut prob from actual rate
                    "hold_prob": None,
                    "hike_prob": None,
                    "is_actual_not_market": True,
                }
        except Exception as e:
            log.debug(f"FRED FEDFUNDS fetch failed: {e}")

    log.warning("CME FedWatch: all fetch methods failed — skipping Fed rate markets")
    return None


def _parse_cme_fedwatch_json(data: dict) -> dict | None:
    """Parse various CME FedWatch JSON formats. Returns None if unrecognised."""
    try:
        # Format A: {"meetings": [{"date": ..., "probabilities": {...}}]}
        meetings = data.get("meetings") or data.get("fedFundsMeetings") or []
        if meetings:
            next_m = meetings[0]
            probs = next_m.get("probabilities") or next_m.get("prob") or {}
            cut_prob = float(probs.get("cut", probs.get("decrease", 0)) or 0) / 100
            hold_prob = float(probs.get("hold", probs.get("nochange", 0)) or 0) / 100
            hike_prob = float(probs.get("hike", probs.get("increase", 0)) or 0) / 100
            return {
                "next_meeting_date": str(next_m.get("date", "")),
                "cut_prob": cut_prob,
                "hold_prob": hold_prob,
                "hike_prob": hike_prob,
                "source": "cme_json",
            }
        # Format B: {"cut": XX.X, "hold": XX.X, "hike": XX.X} at top level
        if "cut" in data or "hold" in data:
            return {
                "cut_prob": float(data.get("cut", 0)) / 100,
                "hold_prob": float(data.get("hold", 0)) / 100,
                "hike_prob": float(data.get("hike", 0)) / 100,
                "source": "cme_json_flat",
            }
    except Exception as e:
        log.debug(f"_parse_cme_fedwatch_json failed: {e}")
    return None


def _parse_cme_probs_array(probs: list) -> dict | None:
    """Parse a CME probabilities array from HTML scrape."""
    try:
        # Expect list of {outcome, probability} or similar
        cut_p = hold_p = hike_p = 0.0
        for item in probs:
            label = str(item.get("label", item.get("outcome", ""))).lower()
            prob = float(item.get("probability", item.get("prob", 0)) or 0) / 100
            if "cut" in label or "decrease" in label:
                cut_p += prob
            elif "hike" in label or "increase" in label:
                hike_p += prob
            else:
                hold_p += prob
        if cut_p + hold_p + hike_p > 0:
            return {
                "cut_prob": cut_p,
                "hold_prob": hold_p,
                "hike_prob": hike_p,
                "source": "cme_html_scrape",
            }
    except Exception as e:
        log.debug(f"_parse_cme_probs_array failed: {e}")
    return None


# --- 2d. BLS unemployment / nonfarm payrolls via FRED ---

def fetch_unemployment() -> dict | None:
    """
    Fetch most recent unemployment rate from FRED (series UNRATE).
    Also fetch the trend for simple direction extrapolation.

    Returns: {"rate_pct": 3.9, "month": "2024-03", "prev_rate_pct": 3.8} or None
    """
    if not FRED_API_KEY:
        log.info("No FRED_API_KEY — unemployment data unavailable")
        return None

    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "UNRATE",
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "limit": 3,
                "sort_order": "desc",
            },
            timeout=10,
        )
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if len(obs) >= 2:
            latest = float(obs[0]["value"])
            prev = float(obs[1]["value"])
            log.info(f"Unemployment (FRED): {latest}% (prev: {prev}%)")
            return {
                "rate_pct": latest,
                "prev_rate_pct": prev,
                "month": obs[0]["date"][:7],
                "source": "fred_unrate",
            }
    except Exception as e:
        log.debug(f"FRED UNRATE fetch failed: {e}")

    return None


def fetch_nonfarm_payrolls() -> dict | None:
    """
    Fetch most recent nonfarm payroll change from FRED (series PAYEMS).
    Returns the month-over-month change in thousands.

    Returns: {"change_k": 275, "month": "2024-03"} or None
    """
    if not FRED_API_KEY:
        return None

    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "PAYEMS",
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "limit": 3,
                "sort_order": "desc",
                "units": "chg",   # month-over-month change (thousands)
            },
            timeout=10,
        )
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if obs:
            latest = float(obs[0]["value"])
            prev = float(obs[1]["value"]) if len(obs) > 1 else None
            log.info(f"Nonfarm payrolls change (FRED): +{latest}K")
            return {
                "change_k": latest,
                "prev_change_k": prev,
                "month": obs[0]["date"][:7],
                "source": "fred_payems",
            }
    except Exception as e:
        log.debug(f"FRED PAYEMS fetch failed: {e}")

    return None


def fetch_core_pce() -> dict | None:
    """
    Core PCE YoY from FRED (PCEPI or PCEPILFE).
    Fed targets 2% PCE — useful for rate cut markets.

    Returns: {"pce_yoy": 2.7, "month": "2024-03"} or None
    """
    if not FRED_API_KEY:
        return None

    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "PCEPILFE",   # Core PCE price index (ex food & energy)
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "limit": 14,
                "sort_order": "desc",
                "units": "pc1",            # percent change year-over-year
            },
            timeout=10,
        )
        r.raise_for_status()
        obs = r.json().get("observations", [])
        if obs:
            latest = float(obs[0]["value"])
            log.info(f"Core PCE YoY (FRED): {latest}%")
            return {
                "pce_yoy": latest,
                "month": obs[0]["date"][:7],
                "source": "fred_pcepilfe",
            }
    except Exception as e:
        log.debug(f"FRED Core PCE fetch failed: {e}")
    return None


# ==========================================================================
# SECTION 3: Market type classifier and question parser
# ==========================================================================

def classify_market(market: dict) -> str | None:
    """
    Classify a market question into one of our supported types.

    Returns one of: "cpi", "fed_rate", "unemployment", "gdp", "payrolls" or None
    """
    q = (market.get("question", "") + " " + market.get("description", "")).lower()

    # CPI / inflation
    cpi_terms = ["cpi", "consumer price index", "inflation rate", "core cpi", "cpi yoy"]
    if any(t in q for t in cpi_terms):
        return "cpi"

    # PCE (Fed's preferred inflation measure)
    if "pce" in q or "personal consumption expenditures" in q:
        return "pce"

    # Fed rate decision
    fed_terms = [
        "federal funds rate", "fed rate", "fomc", "rate cut", "rate hike",
        "basis points", "interest rate decision", "fed will", "fed cut", "fed raise",
    ]
    if any(t in q for t in fed_terms):
        return "fed_rate"

    # Unemployment
    if "unemployment" in q or "jobless rate" in q:
        return "unemployment"

    # GDP
    if "gdp" in q or "gross domestic product" in q:
        return "gdp"

    # Nonfarm payrolls
    if "nonfarm payroll" in q or "non-farm payroll" in q or "payrolls" in q:
        return "payrolls"

    return None


def parse_threshold(question: str, market_type: str) -> dict | None:
    """
    Parse a numeric threshold from a market question.

    Examples:
      "Will CPI be above 3.5%?" -> {"direction": "above", "value": 3.5, "unit": "%"}
      "Will unemployment be below 4%?" -> {"direction": "below", "value": 4.0}
      "Will the Fed cut rates in June?" -> {"direction": "cut", "value": None}
      "Will nonfarm payrolls exceed 200K?" -> {"direction": "above", "value": 200.0, "unit": "K"}

    Returns None if no threshold could be extracted.
    """
    q = question.lower()

    # Fed rate: cut / hike / hold
    if market_type == "fed_rate":
        if any(w in q for w in ["cut", "lower", "reduce", "decrease"]):
            # How many basis points?
            bp_m = re.search(r'(\d+)\s*(?:basis points?|bp)', q)
            bps = int(bp_m.group(1)) if bp_m else 25
            return {"direction": "cut", "value": bps, "unit": "bp"}
        if any(w in q for w in ["hike", "raise", "increase", "higher"]):
            bp_m = re.search(r'(\d+)\s*(?:basis points?|bp)', q)
            bps = int(bp_m.group(1)) if bp_m else 25
            return {"direction": "hike", "value": bps, "unit": "bp"}
        if "hold" in q or "pause" in q or "unchanged" in q:
            return {"direction": "hold", "value": None}
        # Generic "Fed rate" market — try to detect cut vs hold
        return {"direction": "unknown_fed", "value": None}

    # Payrolls: "exceed 200K" / "above 200,000"
    if market_type == "payrolls":
        # Match patterns like "200K", "200,000", "200 thousand"
        m = re.search(
            r'(above|below|exceed|over|under|more than|less than|at least|at most)\s*'
            r'([\d,]+)\s*(?:k\b|thousand)?',
            q,
            re.IGNORECASE,
        )
        if m:
            raw = m.group(2).replace(",", "")
            val = float(raw)
            # Normalise to thousands
            if val > 10000:
                val /= 1000
            direction = "above" if m.group(1).lower() in (
                "above", "exceed", "over", "more than", "at least"
            ) else "below"
            return {"direction": direction, "value": val, "unit": "K"}

    # CPI / PCE / GDP / unemployment: look for "above X%" / "below X%" patterns
    # Covers: "above 3.5%", "below 4%", "over 3.0 percent", "exceed 2.5%"
    m = re.search(
        r'(above|below|exceed|over|under|more than|less than|at least|at most|higher than|lower than)'
        r'\s*([\d.]+)\s*(?:%|percent)?',
        q,
        re.IGNORECASE,
    )
    if m:
        direction = "above" if m.group(1).lower() in (
            "above", "exceed", "over", "more than", "at least", "higher than"
        ) else "below"
        return {"direction": direction, "value": float(m.group(2)), "unit": "%"}

    # Range: "between 3.0% and 3.5%"
    m = re.search(r'between\s*([\d.]+)\s*%?\s*and\s*([\d.]+)\s*%', q, re.IGNORECASE)
    if m:
        return {
            "direction": "between",
            "low": float(m.group(1)),
            "high": float(m.group(2)),
            "unit": "%",
        }

    # Exact: "equal to X%" or "at X%"
    m = re.search(r'(?:equal to|at exactly|be)\s*([\d.]+)\s*%', q, re.IGNORECASE)
    if m:
        return {"direction": "equal", "value": float(m.group(1)), "unit": "%"}

    return None


# ==========================================================================
# SECTION 4: Compute model probability for each market type
# ==========================================================================

def model_prob_cpi(threshold: dict, cpi_data: dict) -> float | None:
    """
    Given Cleveland Fed CPI nowcast data and a threshold, return the
    model's probability of YES.

    For a point estimate (not a distribution), we use a confidence interval
    based on recent forecast errors (~0.15% std dev for 1-month CPI nowcast).
    """
    if cpi_data is None:
        return None

    nowcast = cpi_data.get("cpi_yoy")
    if nowcast is None:
        return None

    # Standard deviation of Cleveland Fed 1-month CPI nowcast errors
    # Based on published accuracy: ~0.15% YoY RMSE at 1-month horizon
    # Use a wider sigma if we're using actual data not a nowcast
    sigma = 0.25 if cpi_data.get("is_actual_not_nowcast") else 0.15

    direction = threshold.get("direction")
    target = threshold.get("value")
    if target is None:
        return None

    # Z-score: how many sigmas is the nowcast from the threshold?
    z = (nowcast - target) / sigma

    # Convert to probability using cumulative normal approximation
    prob = _normal_cdf(z)

    if direction == "above":
        return prob          # P(actual > target) ≈ Φ(z) when nowcast > target
    elif direction == "below":
        return 1 - prob
    elif direction == "between":
        low = threshold.get("low", target)
        high = threshold.get("high", target)
        z_low = (nowcast - low) / sigma
        z_high = (nowcast - high) / sigma
        return _normal_cdf(z_high) - _normal_cdf(z_low)
    return None


def model_prob_pce(threshold: dict, pce_data: dict) -> float | None:
    """Same logic as CPI but for PCE data."""
    if pce_data is None:
        return None

    nowcast = pce_data.get("pce_yoy")
    if nowcast is None:
        return None

    sigma = 0.20   # PCE slightly smoother than CPI
    direction = threshold.get("direction")
    target = threshold.get("value")
    if target is None:
        return None

    z = (nowcast - target) / sigma
    prob = _normal_cdf(z)

    if direction == "above":
        return prob
    elif direction == "below":
        return 1 - prob
    return None


def model_prob_fed_rate(threshold: dict, fedwatch_data: dict) -> float | None:
    """
    Use CME FedWatch probabilities directly.
    """
    if fedwatch_data is None:
        return None

    # If we don't have market-derived probabilities, can't reliably estimate
    if fedwatch_data.get("is_actual_not_market"):
        log.debug("FedWatch: only have actual rate, not forward probabilities — skip")
        return None

    direction = threshold.get("direction")
    if direction == "cut":
        return fedwatch_data.get("cut_prob")
    elif direction == "hike":
        return fedwatch_data.get("hike_prob")
    elif direction == "hold":
        return fedwatch_data.get("hold_prob")
    return None


def model_prob_unemployment(threshold: dict, unemp_data: dict) -> float | None:
    """
    Use most recent unemployment rate + trend to estimate probability of
    next print being above/below a threshold.
    """
    if unemp_data is None:
        return None

    current = unemp_data.get("rate_pct")
    prev = unemp_data.get("prev_rate_pct", current)
    if current is None:
        return None

    # Unemployment month-to-month std dev historically ~0.1-0.15%
    sigma = 0.15
    # Simple trend adjustment: if rate has been rising/falling, shift center
    trend = current + (current - prev) * 0.3 if prev else current

    direction = threshold.get("direction")
    target = threshold.get("value")
    if target is None:
        return None

    z = (trend - target) / sigma
    prob = _normal_cdf(z)

    if direction == "above":
        return prob
    elif direction == "below":
        return 1 - prob
    return None


def model_prob_gdp(threshold: dict, gdp_data: dict) -> float | None:
    """
    Use GDPNow / Atlanta Fed estimate against market threshold.
    GDP growth estimates have higher uncertainty (~0.5-1.0pp).
    """
    if gdp_data is None:
        return None

    nowcast = gdp_data.get("gdp_growth_pct")
    if nowcast is None:
        return None

    # GDPNow RMSE is ~1.0pp on final GDP at start of quarter,
    # narrows to ~0.5pp near end of quarter
    sigma = 0.7 if gdp_data.get("is_actual_not_nowcast") else 1.0

    direction = threshold.get("direction")
    target = threshold.get("value")
    if target is None:
        return None

    z = (nowcast - target) / sigma
    prob = _normal_cdf(z)

    if direction == "above":
        return prob
    elif direction == "below":
        return 1 - prob
    return None


def model_prob_payrolls(threshold: dict, payroll_data: dict) -> float | None:
    """
    Use recent payroll trend against threshold.
    Payroll revisions are large — use wide sigma (~80K).
    """
    if payroll_data is None:
        return None

    latest = payroll_data.get("change_k")
    prev = payroll_data.get("prev_change_k", latest)
    if latest is None:
        return None

    sigma = 80.0  # historical payroll surprise std dev ~80K
    trend = latest + (latest - prev) * 0.2 if prev else latest

    direction = threshold.get("direction")
    target = threshold.get("value")
    if target is None:
        return None

    z = (trend - target) / sigma
    prob = _normal_cdf(z)

    if direction == "above":
        return prob
    elif direction == "below":
        return 1 - prob
    return None


def _normal_cdf(z: float) -> float:
    """
    Approximation of Φ(z) — the cumulative standard normal distribution.
    Accurate to ~0.003 (Horner's method approximation).
    """
    import math
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ==========================================================================
# SECTION 5: EV math (same as weather_bot.py)
# ==========================================================================

def calculate_fee(price: float, size_usd: float, fee_rate: float) -> float:
    if price <= 0 or price >= 1:
        return 0.0
    shares = size_usd / price
    return fee_rate * price * (1 - price) * shares


def expected_value(true_p: float, market_p: float, side: str) -> float:
    if side == "YES":
        return true_p * (1 - market_p) - (1 - true_p) * market_p
    return (1 - true_p) * market_p - true_p * (1 - market_p)


def kelly_size(true_p: float, market_p: float, side: str,
               bankroll: float, fraction: float, cap: float) -> float:
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


def decide_trade(true_p: float, market_p: float,
                 bankroll: float, cfg: dict) -> tuple:
    """Returns (side, size_usd, edge, fee_usd). side=None means skip."""
    ev_yes = expected_value(true_p, market_p, "YES")
    ev_no = expected_value(true_p, market_p, "NO")
    side, gross_edge = ("YES", ev_yes) if ev_yes >= ev_no else ("NO", ev_no)
    entry_price = market_p if side == "YES" else 1 - market_p
    fee_per_dollar = calculate_fee(entry_price, 1.0, cfg["taker_fee_rate"])
    edge = gross_edge - fee_per_dollar
    if edge < cfg["min_edge"]:
        return None, 0.0, edge, 0.0
    size = kelly_size(
        true_p, market_p, side, bankroll,
        cfg["kelly_fraction"], cfg["max_position_usd"],
    )
    if size < 1.0:
        return None, 0.0, edge, 0.0
    fee_usd = calculate_fee(entry_price, size, cfg["taker_fee_rate"])
    return side, size, edge, fee_usd


# ==========================================================================
# SECTION 6: Resolution tracking
# ==========================================================================

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
            f"{'✅' if won else '❌'} *Econ Trade Resolved*\n"
            f"Market: {t['question'][:80]}\n"
            f"Side: {t['side']} -> *{'WIN' if won else 'LOSS'}*\n"
            f"P&L: ${pnl:+.2f}\n"
            f"Balance: ${balance:.2f}\n"
            f"🕐 {datetime.now(SHANGHAI_TZ).strftime('%H:%M Shanghai')}"
        )

    if changed:
        save_trades(trades)


# ==========================================================================
# SECTION 7: Main scan
# ==========================================================================

def run_scan(cfg: dict, mode: str, bankroll: float):
    log.info(f"=== Econ Scan | mode={mode} | bankroll=${bankroll:.2f} ===")
    update_resolutions()

    # -- Fetch all data sources once --
    log.info("Fetching nowcast data sources...")
    cpi_data = fetch_cleveland_fed_cpi()
    gdp_data = fetch_gdpnow()
    fedwatch_data = fetch_fedwatch()
    unemp_data = fetch_unemployment()
    payroll_data = fetch_nonfarm_payrolls()
    pce_data = fetch_core_pce()

    log.info(
        f"Data sources: CPI={'ok' if cpi_data else 'SKIP'} | "
        f"GDP={'ok' if gdp_data else 'SKIP'} | "
        f"FedWatch={'ok' if fedwatch_data else 'SKIP'} | "
        f"Unemployment={'ok' if unemp_data else 'SKIP'} | "
        f"Payrolls={'ok' if payroll_data else 'SKIP'} | "
        f"PCE={'ok' if pce_data else 'SKIP'}"
    )

    # -- Fetch markets --
    markets = fetch_econ_markets()
    traded_ids = already_traded_market_ids()
    trades_placed = 0

    # Collect candidates then sort by edge
    candidates = []

    for m in markets:
        mid = str(m.get("id", ""))
        if not mid or mid in traded_ids:
            continue

        # Parse price
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
        yes_price = float(outcome_prices[0])

        # Skip very stale prices
        if yes_price <= 0.03 or yes_price >= 0.97:
            continue

        # Classify market type
        mtype = classify_market(m)
        if mtype is None:
            log.debug(f"[{mid[:8]}] Unclassified: {m.get('question', '')[:60]}")
            continue

        # Parse threshold
        question = m.get("question", "")
        threshold = parse_threshold(question, mtype)
        if threshold is None:
            log.debug(f"[{mid[:8]}] No threshold parsed: {question[:60]}")
            continue

        # Get model probability for this market type
        model_p = None
        data_src = None

        if mtype == "cpi":
            if cpi_data is None:
                continue
            model_p = model_prob_cpi(threshold, cpi_data)
            data_src = cpi_data.get("source", "cpi")

        elif mtype == "pce":
            if pce_data is None:
                continue
            model_p = model_prob_pce(threshold, pce_data)
            data_src = pce_data.get("source", "pce")

        elif mtype == "fed_rate":
            if fedwatch_data is None:
                continue
            model_p = model_prob_fed_rate(threshold, fedwatch_data)
            data_src = fedwatch_data.get("source", "fedwatch")

        elif mtype == "unemployment":
            if unemp_data is None:
                continue
            model_p = model_prob_unemployment(threshold, unemp_data)
            data_src = unemp_data.get("source", "unemployment")

        elif mtype == "gdp":
            if gdp_data is None:
                continue
            model_p = model_prob_gdp(threshold, gdp_data)
            data_src = gdp_data.get("source", "gdp")

        elif mtype == "payrolls":
            if payroll_data is None:
                continue
            model_p = model_prob_payrolls(threshold, payroll_data)
            data_src = payroll_data.get("source", "payrolls")

        if model_p is None:
            log.debug(f"[{mid[:8]}] model_p=None for type={mtype} threshold={threshold}")
            continue

        # Clamp model probability to avoid overconfidence
        model_p = max(0.01, min(0.99, model_p))

        side, size, edge, fee_usd = decide_trade(model_p, yes_price, bankroll, cfg)

        log.info(
            f"[{mid[:8]}] type={mtype} | {question[:55]} | "
            f"mkt={yes_price:.2f} model={model_p:.2f} edge={edge:+.1%} "
            f"=> {side or 'skip'} ${size:.2f}"
        )

        if side is None:
            continue

        entry_price = yes_price if side == "YES" else 1 - yes_price

        candidates.append({
            "mid": mid,
            "market": m,
            "question": question,
            "mtype": mtype,
            "threshold": threshold,
            "yes_price": yes_price,
            "model_p": model_p,
            "side": side,
            "size": size,
            "edge": edge,
            "fee_usd": fee_usd,
            "entry_price": entry_price,
            "data_src": data_src,
        })

    # Sort by edge descending — pick best opportunities first
    candidates.sort(key=lambda c: c["edge"], reverse=True)

    for c in candidates:
        bankroll -= c["size"]

        record = {
            "timestamp": datetime.now(SHANGHAI_TZ).isoformat(),
            "date": datetime.now(SHANGHAI_TZ).date().isoformat(),
            "mode": mode,
            "strategy": "econ",
            "market_id": c["mid"],
            "question": c["question"][:200],
            "market_type": c["mtype"],
            "threshold": c["threshold"],
            "market_price": c["yes_price"],
            "model_prob": round(c["model_p"], 4),
            "data_source": c["data_src"],
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

        send_telegram(
            f"*Econ Trade*\n"
            f"Type: {c['mtype'].upper()} | {c['question'][:70]}\n"
            f"Threshold: {c['threshold']}\n"
            f"Model: {c['model_p']:.0%} vs Market: {c['yes_price']:.0%}\n"
            f"Side: *{c['side']}* @ {c['entry_price']:.2f}\n"
            f"Size: ${c['size']:.2f} | Edge: {c['edge']:+.1%}\n"
            f"Source: {c['data_src']}\n"
            f"Balance: ${bankroll:.2f}\n"
            f"🕐 {datetime.now(SHANGHAI_TZ).strftime('%H:%M Shanghai')}"
        )

        time.sleep(1)

    log.info(f"Econ scan complete. Placed {trades_placed} trades.")


# ==========================================================================
# SECTION 8: Daily summary
# ==========================================================================

def send_daily_summary():
    today = datetime.now(SHANGHAI_TZ).date().isoformat()
    trades = load_trades()
    todays_trades = [t for t in trades if t.get("date") == today]
    all_resolved = [t for t in trades if t.get("resolved")]
    total_pnl = sum(float(t.get("realized_pnl") or 0) for t in all_resolved)
    open_trades = [t for t in trades if not t.get("resolved")]
    wins = sum(1 for t in all_resolved if t.get("won"))

    # Break down by market type
    type_counts: dict = {}
    for t in todays_trades:
        mt = t.get("market_type", "unknown")
        type_counts[mt] = type_counts.get(mt, 0) + 1

    type_str = ", ".join(f"{k}:{v}" for k, v in type_counts.items())

    if todays_trades:
        msg = (
            f"*Econ Bot Summary — {today}*\n\n"
            f"Trades today: {len(todays_trades)} ({type_str})\n"
            f"Open trades: {len(open_trades)}\n"
            f"Total resolved: {len(all_resolved)}\n"
        )
        if all_resolved:
            msg += f"Win rate: {100*wins/len(all_resolved):.0f}%\n"
        msg += f"Total P&L: ${total_pnl:+.2f}"
    else:
        msg = (
            f"*Econ Bot Summary — {today}*\n\n"
            f"No econ trades today.\n"
            f"Open trades: {len(open_trades)}\n"
            f"Total resolved: {len(all_resolved)}\n"
            f"Total P&L: ${total_pnl:+.2f}"
        )

    send_telegram(msg)
    log.info(f"Econ summary sent: {len(todays_trades)} trades today, P&L ${total_pnl:+.2f}")


# ==========================================================================
# SECTION 9: Entry point
# ==========================================================================

def main():
    ap = argparse.ArgumentParser(description="Polymarket Economic Data Trading Bot")
    ap.add_argument("--mode", choices=["paper", "live"], default="paper",
                    help="paper = log only, live = submit orders (not yet wired)")
    ap.add_argument("--daily-summary", action="store_true",
                    help="Send daily summary to Telegram and exit")
    args = ap.parse_args()

    if args.daily_summary:
        update_resolutions()
        send_daily_summary()
        return

    cfg = CONFIG.copy()

    # Bankroll = starting + realized P&L from all resolved trades
    all_resolved = [t for t in load_trades() if t.get("resolved")]
    realized = sum(float(t.get("realized_pnl") or 0) for t in all_resolved)
    bankroll = cfg["starting_bankroll"] + realized

    if args.mode == "live":
        log.error(
            "Live mode is not yet wired. "
            "Run paper mode until calibration shows n>=30 and positive edge."
        )
        return

    run_scan(cfg, args.mode, bankroll)


if __name__ == "__main__":
    main()
