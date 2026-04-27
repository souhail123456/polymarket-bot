"""Generate a static HTML dashboard from trade logs."""

import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

SHANGHAI_TZ = timezone(timedelta(hours=8))
LOG_DIR = Path("./logs")
OUT = Path("./dashboard/index.html")


def load_trades(filepath):
    if not filepath.exists():
        return []
    trades = []
    with open(filepath) as f:
        for line in f:
            try:
                trades.append(json.loads(line))
            except Exception:
                pass
    return trades


def compute_stats(trades):
    resolved = [t for t in trades if t.get("resolved")]
    open_trades = [t for t in trades if not t.get("resolved")]
    wins = [t for t in resolved if t.get("won")]
    total_pnl = sum(float(t.get("realized_pnl", 0)) for t in resolved)
    total_risked = sum(float(t.get("size_usd", 0)) for t in resolved)
    return {
        "total": len(trades),
        "open": len(open_trades),
        "resolved": len(resolved),
        "wins": len(wins),
        "win_rate": f"{100 * len(wins) / len(resolved):.0f}%" if resolved else "—",
        "total_pnl": total_pnl,
        "total_risked": total_risked,
    }


def trade_row(t):
    status = "open"
    pnl = "—"
    if t.get("resolved"):
        status = "win" if t.get("won") else "loss"
        pnl = f"${float(t.get('realized_pnl', 0)):+.2f}"

    strategy = t.get("strategy", "ev")
    city = t.get("city", "—")
    question = t.get("question", "")[:60]
    side = t.get("side", "")
    entry = f"${float(t.get('entry_price', 0)):.2f}"
    size = f"${float(t.get('size_usd', 0)):.2f}"
    edge = f"{float(t.get('edge', 0)):+.1%}"
    ts = t.get("timestamp", "")[:16].replace("T", " ")

    return f"""<tr class="{status}">
        <td>{ts}</td>
        <td><span class="badge {strategy}">{strategy}</span></td>
        <td>{city}</td>
        <td title="{t.get('question', '')}">{question}</td>
        <td><span class="side {side.lower()}">{side}</span></td>
        <td>{entry}</td>
        <td>{size}</td>
        <td>{edge}</td>
        <td class="pnl">{pnl}</td>
        <td><span class="status-dot {status}"></span> {status}</td>
    </tr>"""


def generate():
    ev_trades = load_trades(LOG_DIR / "trades.jsonl")
    weather_trades = load_trades(LOG_DIR / "weather_trades.jsonl")
    all_trades = sorted(ev_trades + weather_trades, key=lambda t: t.get("timestamp", ""), reverse=True)

    ev_stats = compute_stats(ev_trades)
    w_stats = compute_stats(weather_trades)
    total_stats = compute_stats(all_trades)

    now = datetime.now(SHANGHAI_TZ).strftime("%Y-%m-%d %H:%M Shanghai")
    rows = "\n".join(trade_row(t) for t in all_trades)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Polymarket Bot Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0f; color: #e0e0e0; padding: 20px; }}
h1 {{ font-size: 1.5rem; margin-bottom: 5px; color: #fff; }}
.updated {{ color: #666; font-size: 0.85rem; margin-bottom: 20px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; margin-bottom: 25px; }}
.card {{ background: #14141f; border: 1px solid #2a2a3a; border-radius: 10px; padding: 18px; }}
.card h3 {{ font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }}
.card .big {{ font-size: 2rem; font-weight: 700; }}
.card .row {{ display: flex; justify-content: space-between; margin-top: 8px; font-size: 0.9rem; }}
.card .label {{ color: #666; }}
.green {{ color: #4ade80; }}
.red {{ color: #f87171; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
th {{ text-align: left; padding: 10px 8px; color: #666; border-bottom: 1px solid #2a2a3a; font-weight: 500; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 1px; }}
td {{ padding: 10px 8px; border-bottom: 1px solid #1a1a2a; }}
tr:hover {{ background: #1a1a2f; }}
tr.win td.pnl {{ color: #4ade80; }}
tr.loss td.pnl {{ color: #f87171; }}
.badge {{ padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
.badge.weather {{ background: #1e3a5f; color: #60a5fa; }}
.badge.ev {{ background: #3a1e5f; color: #a78bfa; }}
.side {{ font-weight: 600; }}
.side.yes {{ color: #4ade80; }}
.side.no {{ color: #f87171; }}
.status-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }}
.status-dot.open {{ background: #facc15; }}
.status-dot.win {{ background: #4ade80; }}
.status-dot.loss {{ background: #f87171; }}
.table-wrap {{ overflow-x: auto; background: #14141f; border: 1px solid #2a2a3a; border-radius: 10px; }}
</style>
</head>
<body>
<h1>Polymarket Bot Dashboard</h1>
<p class="updated">Last updated: {now}</p>

<div class="cards">
    <div class="card">
        <h3>Total P&L</h3>
        <div class="big {'green' if total_stats['total_pnl'] >= 0 else 'red'}">${total_stats['total_pnl']:+.2f}</div>
        <div class="row"><span class="label">Trades</span> {total_stats['total']}</div>
        <div class="row"><span class="label">Open</span> {total_stats['open']}</div>
        <div class="row"><span class="label">Resolved</span> {total_stats['resolved']}</div>
        <div class="row"><span class="label">Win Rate</span> {total_stats['win_rate']}</div>
    </div>
    <div class="card">
        <h3>EV Bot (Groq/Llama)</h3>
        <div class="big {'green' if ev_stats['total_pnl'] >= 0 else 'red'}">${ev_stats['total_pnl']:+.2f}</div>
        <div class="row"><span class="label">Trades</span> {ev_stats['total']}</div>
        <div class="row"><span class="label">Open</span> {ev_stats['open']}</div>
        <div class="row"><span class="label">Win Rate</span> {ev_stats['win_rate']}</div>
    </div>
    <div class="card">
        <h3>Weather Bot (Ensemble)</h3>
        <div class="big {'green' if w_stats['total_pnl'] >= 0 else 'red'}">${w_stats['total_pnl']:+.2f}</div>
        <div class="row"><span class="label">Trades</span> {w_stats['total']}</div>
        <div class="row"><span class="label">Open</span> {w_stats['open']}</div>
        <div class="row"><span class="label">Win Rate</span> {w_stats['win_rate']}</div>
    </div>
</div>

<div class="table-wrap">
<table>
<thead>
<tr>
    <th>Time</th>
    <th>Strategy</th>
    <th>City</th>
    <th>Market</th>
    <th>Side</th>
    <th>Entry</th>
    <th>Size</th>
    <th>Edge</th>
    <th>P&L</th>
    <th>Status</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>
</div>
</body>
</html>"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"Dashboard generated: {OUT} ({len(all_trades)} trades)")


if __name__ == "__main__":
    generate()
