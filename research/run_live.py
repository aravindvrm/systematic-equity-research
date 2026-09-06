#!/usr/bin/env python
"""Daily runner: compute target weights, diff against the account, plan orders.

DRY RUN BY DEFAULT. Nothing is submitted unless you pass --submit.

    ./.venv/bin/python run_live.py                  # plan only, no connection needed
    ./.venv/bin/python run_live.py --connect        # plan against real paper account
    ./.venv/bin/python run_live.py --connect --submit   # actually trade (paper)
    ./.venv/bin/python run_live.py --connect --submit --live   # REAL MONEY

Env vars:
    ALPACA_PAPER_KEY / ALPACA_PAPER_SECRET
    ALPACA_LIVE_KEY  / ALPACA_LIVE_SECRET     (only for --live)
"""
import argparse
import logging
import sys

from algo import broker, data, portfolio

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

p = argparse.ArgumentParser()
p.add_argument("--connect", action="store_true", help="connect to Alpaca for real account state")
p.add_argument("--submit", action="store_true", help="actually place the planned orders")
p.add_argument("--live", action="store_true", help="use the LIVE account instead of paper")
p.add_argument("--band", type=float, default=0.10, help="rebalance band (default 0.10)")
p.add_argument("--equity", type=float, default=1000.0, help="assumed equity when not connecting")
p.add_argument("--max-order-pct", type=float, default=0.25, help="max order as fraction of equity")
args = p.parse_args()

if args.live and args.submit:
    print("\n*** LIVE MONEY MODE ***")
    if input("Type 'yes I am sure' to continue: ").strip() != "yes I am sure":
        sys.exit("aborted")

print(f"\nloading prices for {len(data.ETF_UNIVERSE)} ETFs ...")
px = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01").dropna(how="any")
target = portfolio.build(px).iloc[-1]
print(f"signal date: {px.index[-1].date()}\n")
print("target weights:")
for sym, wt in target[target > 0.001].sort_values(ascending=False).items():
    print(f"  {sym:6s} {wt*100:5.2f}%")
print(f"  {'CASH':6s} {(1-target.sum())*100:5.2f}%\n")

client, current, equity = None, None, args.equity
if args.connect:
    client = broker.connect(paper=not args.live)
    equity, current = broker.current_state(client)
    print(f"connected to {'LIVE' if args.live else 'PAPER'} account\n")

limits = broker.RiskLimits(max_order_pct=args.max_order_pct)
plan = broker.plan_orders(target, equity, current, band=args.band, limits=limits)
print(plan)

if args.submit:
    if not args.connect:
        sys.exit("\n--submit requires --connect")
    if not plan.ok:
        sys.exit("\nplan is blocked; nothing submitted")
    sent = broker.submit_orders(client, plan, live=True)
    print(f"\nsubmitted {len(sent)} order(s)")
else:
    print("\n(dry run -- pass --connect --submit to place these)")
