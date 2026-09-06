"""How much did the v1 turnover bug distort things?"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from algo import backtest, backtest2, data, metrics, strategies
from algo.costs import IBKR_US_EQUITY, ZERO_COST

tk = json.load(open("data/trading_universe.json"))
px = data.load_panel(tk, start="2010-01-01", end="2026-09-01", refresh=False)
px = px.dropna(axis=1, thresh=int(0.9 * len(px))).ffill(limit=5)
print(f"{px.shape[1]} names, {len(px)} bars\n")

print("TURNOVER AND COST: v1 (target-change) vs v2 (true drift)\n")
print(f"{'strategy':<28} {'engine':<5} {'ann turnover':>13} {'ann cost bp':>12} "
      f"{'Sharpe':>8} {'CAGR':>8}")

cases = {
    "equal weight (constant tgt)": strategies.equal_weight(px),
    "inverse vol (slow tgt)":      strategies.inverse_volatility(px, 60),
    "xsec momentum (fast tgt)":    strategies.cross_sectional_momentum(px, 126),
}
for name, w in cases.items():
    for tag, eng in (("v1", backtest), ("v2", backtest2)):
        res = eng.run(px, w, cost_model=IBKR_US_EQUITY)
        s = res.summary()
        ann_to = res.turnover.mean() * 252
        ann_cost = res.costs.mean() * 252 * 1e4
        print(f"{name:<28} {tag:<5} {ann_to:>13.2f} {ann_cost:>12.1f} "
              f"{s['sharpe']:>8.2f} {s['cagr']*100:>7.2f}%")
    print()

print("=" * 78)
print("THE BIAS: cost charged to the BENCHMARK vs to a SIGNAL strategy")
print("=" * 78)
bench_w = strategies.equal_weight(px)
sig_w = strategies.cross_sectional_momentum(px, 126)
for tag, eng in (("v1", backtest), ("v2", backtest2)):
    b = eng.run(px, bench_w, cost_model=IBKR_US_EQUITY)
    s = eng.run(px, sig_w, cost_model=IBKR_US_EQUITY)
    bc = b.costs.mean() * 252 * 1e4
    sc = s.costs.mean() * 252 * 1e4
    print(f"  {tag}: benchmark {bc:6.1f}bp/yr   signal {sc:6.1f}bp/yr   "
          f"handicap to signal {sc-bc:6.1f}bp/yr   "
          f"Sharpe gap {s.summary()['sharpe']-b.summary()['sharpe']:+.3f}")
