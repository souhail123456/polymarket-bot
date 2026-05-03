"""
Polymarket Bot — API Health Check
----------------------------------
Tests all external APIs and writes results to logs/health_status.json.
Uses only stdlib — no pip dependencies.
"""

import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

SHANGHAI_TZ = timezone(timedelta(hours=8))
OUT_FILE = "logs/health_status.json"


def _request(method: str, url: str, headers: dict = None, body: bytes = None, timeout: int = 10) -> tuple[int, str]:
    """Make an HTTP request and return (status_code, body_text)."""
    req = urllib.request.Request(url, data=body, method=method)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def check_service(name: str, method: str, url: str, headers: dict = None, body: bytes = None) -> dict:
    t0 = time.monotonic()
    try:
        status_code, body_text = _request(method, url, headers=headers, body=body)
        latency_ms = round((time.monotonic() - t0) * 1000)
        if 200 <= status_code < 300:
            return {"status": "ok", "latency_ms": latency_ms}
        else:
            return {"status": "error", "latency_ms": latency_ms, "error": f"{status_code}"}
    except Exception as e:
        latency_ms = round((time.monotonic() - t0) * 1000)
        return {"status": "error", "latency_ms": latency_ms, "error": str(e)[:120]}


def run_checks() -> dict:
    services = {}

    # Polymarket Gamma API — no auth
    services["polymarket_gamma"] = check_service(
        "polymarket_gamma", "GET",
        "https://gamma-api.polymarket.com/markets?limit=1"
    )

    # CoinGecko — no auth
    services["coingecko"] = check_service(
        "coingecko", "GET",
        "https://api.coingecko.com/api/v3/ping"
    )

    # FRED API
    fred_key = os.environ.get("FRED_API_KEY", "")
    if fred_key:
        services["fred"] = check_service(
            "fred", "GET",
            f"https://api.stlouisfed.org/fred/series?series_id=CPIAUCSL&api_key={fred_key}&file_type=json"
        )
    else:
        services["fred"] = {"status": "error", "latency_ms": 0, "error": "FRED_API_KEY not set"}

    # Groq API
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        groq_body = json.dumps({
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1
        }).encode()
        services["groq"] = check_service(
            "groq", "POST",
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json",
            },
            body=groq_body
        )
    else:
        services["groq"] = {"status": "error", "latency_ms": 0, "error": "GROQ_API_KEY not set"}

    # Gemini API
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        services["gemini"] = check_service(
            "gemini", "GET",
            f"https://generativelanguage.googleapis.com/v1/models?key={gemini_key}"
        )
    else:
        services["gemini"] = {"status": "error", "latency_ms": 0, "error": "GEMINI_API_KEY not set"}

    # Cerebras API
    cerebras_key = os.environ.get("CEREBRAS_API_KEY", "")
    if cerebras_key:
        services["cerebras"] = check_service(
            "cerebras", "GET",
            "https://api.cerebras.ai/v1/models",
            headers={"Authorization": f"Bearer {cerebras_key}"}
        )
    else:
        services["cerebras"] = {"status": "error", "latency_ms": 0, "error": "CEREBRAS_API_KEY not set"}

    # Telegram
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if tg_token:
        services["telegram"] = check_service(
            "telegram", "GET",
            f"https://api.telegram.org/bot{tg_token}/getMe"
        )
    else:
        services["telegram"] = {"status": "error", "latency_ms": 0, "error": "TELEGRAM_BOT_TOKEN not set"}

    return services


def main():
    os.makedirs("logs", exist_ok=True)
    print("Running Polymarket Bot health checks...")

    services = run_checks()
    checked_at = datetime.now(SHANGHAI_TZ).isoformat()

    result = {
        "checked_at": checked_at,
        "repo": "polymarket-bot",
        "services": services,
    }

    with open(OUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    ok_count = sum(1 for s in services.values() if s["status"] == "ok")
    total = len(services)
    print(f"Health check complete: {ok_count}/{total} services healthy")
    for name, info in services.items():
        icon = "✓" if info["status"] == "ok" else "✗"
        err = f" — {info.get('error', '')}" if info["status"] == "error" else ""
        print(f"  {icon} {name}: {info['latency_ms']}ms{err}")
    print(f"Results written to {OUT_FILE}")


if __name__ == "__main__":
    main()
