# Polymarket EV Trading Bot

Paper-first Polymarket trading bot that uses Claude to estimate true probabilities, calculates expected value vs market price, and trades when edge is large enough.

## Strategy (plain English)

1. Pull active short-dated Polymarket markets (resolve within 2-72 hours)
2. For each, ask Claude: "what's the true probability of YES?"
3. Compare Claude's estimate to market price → compute expected value
4. If EV edge ≥ 8% and Claude is confident ≥ 55%, place a trade
5. Size with quarter-Kelly, capped at $5/trade
6. Log everything, track resolutions automatically
7. After 30+ resolved trades, check calibration before going live

## Setup

```bash
cd polymarket_bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## Run

```bash
# Single scan in paper mode
python bot.py --mode paper

# Continuous loop (every 30 min) — recommended for accumulating trades
python bot.py --mode paper --loop

# Check progress
python bot.py --report
```

## Files it creates

- `logs/bot.log` — everything it does
- `logs/trades.jsonl` — every trade, one JSON per line

## Decision gate before live mode

Run `python bot.py --report` after ~1 week of paper trading. You need:

- **n ≥ 30 resolved trades** (statistical minimum)
- **Positive total P&L** (otherwise strategy is broken)
- **Calibration within ~15%**: when bot says "70%" things should resolve YES roughly 70% of the time
- No single trade dominating the results (variance check)

If any of those fail, DO NOT go live. Tune the prompt, adjust `min_edge` / `min_confidence`, or accept that the strategy doesn't work.

## Going live (only after paper gate passes)

Not included by default. Requires:
1. `pip install py-clob-client`
2. A funded Polygon wallet (USDC + small MATIC for gas)
3. API credentials from Polymarket (keys + signing)
4. Implementing `submit_live_order()` in `bot.py`

Start live with `max_position_usd=2` and `daily_loss_cap_usd=10`. Scale up only if paper results continue to hold.

## Honest expectations

Most people who build bots like this lose money. Claude has no inherent edge on markets where information is already public and crowd-priced. The only ways this makes money:

- Obscure markets with few sharp traders (possible)
- Crowd is systematically biased (possible)
- Claude calibration genuinely beats average Polymarket participant (unknown)

The bot is built so you can find out cheaply whether it works. Treat the API cost (~$5-15 for a week of paper trading) as the price of the experiment.

## Known limitations

- Gamma API may rate-limit with heavy scanning
- Market filter assumes binary YES/NO markets; skips multi-outcome
- Resolution detection depends on Polymarket marking markets closed
- No slippage modeling in paper mode — real fills will be slightly worse
- Claude API latency (2-8s per market) means you won't catch fast-moving events
