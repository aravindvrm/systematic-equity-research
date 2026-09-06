"""Can we rebalance daily in a cash account?

PDT is irrelevant here -- it applies only to margin accounts. The two things
that actually bind are:

  1. ORDER BUDGET. Every order costs at least $0.35. That is a fixed dollar
     cost, so your account size sets a hard cap on how many orders per year you
     can afford before commissions dominate.

  2. T+1 SETTLEMENT. Selling and re-buying with unsettled proceeds risks a Good
     Faith Violation. Three in 12 months = 90-day settled-cash-only restriction.

This tests whether "check daily, but only trade when it matters" recovers the
responsiveness of daily rebalancing without the order count.
"""
import numpy as np, pandas as pd
from algo import data, portfolio, orderlevel, metrics

print("ORDER BUDGET: how many $0.35 orders can you afford per year?\n")
print(f"{'capital':>10} " + "".join(f"{c:>16}" for c in ["at 0.5% cost","at 1% cost","at 2% cost"]))
for cap in [1_000, 5_000, 10_000, 25_000, 100_000]:
    cells = "".join(f"{int(cap*p/0.35):>16,}" for p in [0.005, 0.01, 0.02])
    print(f"{'$'+format(cap,','):>10} {cells}")
print("\n  A 10-position portfolio uses ~10 orders per full rebalance.")
print("  At $1,000 and a 1% cost budget you get ~28 orders/yr = ~3 rebalances.")

px = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")
w = portfolio.build(px)
years = len(px) / 252


def band_rebalance(weights, band):
    """Check every day; only move a position when it has drifted past `band`."""
    out = weights.copy()
    cur = np.zeros(weights.shape[1])
    rows = []
    for _, tgt in weights.iterrows():
        t = tgt.to_numpy()
        move = np.abs(t - cur) > band
        cur = np.where(move, t, cur)
        rows.append(cur.copy())
    return pd.DataFrame(rows, index=weights.index, columns=weights.columns)


print("\n\nBAND REBALANCING AT $1,000  (check daily, trade only past the band)\n")
print(f"{'band':>8} {'CAGR %':>9} {'Sharpe':>8} {'MaxDD %':>9} {'orders':>8} {'orders/yr':>10} {'cost/yr %':>10}")
for band in [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]:
    bw = band_rebalance(w, band) if band > 0 else w
    r = orderlevel.run(px, bw, capital=1000, rebalance=None, min_trade=25.0)
    print(f"{band*100:>7.0f}% {r['cagr']*100:>9.2f} {r['sharpe']:>8.2f} {r['max_drawdown']*100:>9.2f}"
          f" {r['n_orders']:>8,} {r['n_orders']/years:>10.0f} {r['cost_pct_of_capital']/years*100:>10.2f}")

print("\nSAME BANDS AT $25,000\n")
print(f"{'band':>8} {'CAGR %':>9} {'Sharpe':>8} {'MaxDD %':>9} {'orders/yr':>10} {'cost/yr %':>10}")
for band in [0.0, 0.01, 0.02, 0.05, 0.10]:
    bw = band_rebalance(w, band) if band > 0 else w
    r = orderlevel.run(px, bw, capital=25000, rebalance=None, min_trade=25.0)
    print(f"{band*100:>7.0f}% {r['cagr']*100:>9.2f} {r['sharpe']:>8.2f} {r['max_drawdown']*100:>9.2f}"
          f" {r['n_orders']/years:>10.0f} {r['cost_pct_of_capital']/years*100:>10.2f}")
