"""Cross-validate the homegrown metrics against reference implementations.

Everything in this project has run on 536 lines of my own backtest/metrics/
diagnostics code with no standard library underneath it. Before re-evaluating
anything, the arithmetic itself needs checking against empyrical (the library
under pyfolio) and quantstats.

A disagreement here would invalidate every number produced this week.
"""
import warnings

import numpy as np
import pandas as pd
import empyrical as ep

warnings.filterwarnings("ignore")
from algo import backtest, data, metrics, strategies
from algo.costs import IBKR_US_EQUITY

rng = np.random.default_rng(0)
idx = pd.bdate_range("2015-01-01", periods=2000)
r = pd.Series(rng.normal(0.0004, 0.011, len(idx)), index=idx)
eq = (1 + r).cumprod()

print("METRIC CROSS-VALIDATION  (synthetic series, 2000 bars)\n")
print(f"{'metric':<22} {'mine':>12} {'empyrical':>12} {'diff':>12}  {'ok':>4}")
checks = [
    ("sharpe",       metrics.sharpe(r),              ep.sharpe_ratio(r, risk_free=0)),
    ("sortino",      metrics.sortino(r),             ep.sortino_ratio(r)),
    ("max_drawdown", metrics.max_drawdown(eq),       ep.max_drawdown(r)),
    ("cagr",         metrics.cagr(eq),               ep.annual_return(r)),
    ("volatility",   metrics.volatility(r),          ep.annual_volatility(r)),
    ("calmar",       metrics.calmar(eq),             ep.calmar_ratio(r)),
]
worst = 0.0
for name, mine, ref in checks:
    d = abs(mine - ref)
    rel = d / max(abs(ref), 1e-9)
    worst = max(worst, rel)
    print(f"{name:<22} {mine:>12.6f} {ref:>12.6f} {d:>12.2e}  "
          f"{'OK' if rel < 0.02 else 'MISMATCH':>4}")

print(f"\nworst relative disagreement: {worst:.2%}")

# Same check on a REAL backtest, where the engine's own return construction is
# also in play -- not just the metric formulas.
print("\n\nSAME CHECK ON A REAL BACKTEST (equal-weight, 30-name book)\n")
import json
tk = json.load(open("data/trading_universe.json"))
px = data.load_panel(tk, start="2010-01-01", end="2026-09-01", refresh=False)
px = px.dropna(axis=1, thresh=int(0.9 * len(px))).ffill(limit=5)
w = strategies.equal_weight(px)
res = backtest.run(px, w, cost_model=IBKR_US_EQUITY)
rr = res.returns

print(f"{'metric':<22} {'mine':>12} {'empyrical':>12} {'diff':>12}")
for name, mine, ref in [
    ("sharpe",       metrics.sharpe(rr),            ep.sharpe_ratio(rr, risk_free=0)),
    ("max_drawdown", metrics.max_drawdown(res.equity), ep.max_drawdown(rr)),
    ("cagr",         metrics.cagr(res.equity),      ep.annual_return(rr)),
    ("volatility",   metrics.volatility(rr),        ep.annual_volatility(rr)),
]:
    print(f"{name:<22} {mine:>12.6f} {ref:>12.6f} {abs(mine-ref):>12.2e}")

# The engine's own accounting: does gross - costs actually equal net?
print("\n\nENGINE ACCOUNTING IDENTITY")
implied = res.gross_returns - res.costs / res.equity.shift(1).fillna(res.equity.iloc[0])
err = (implied - res.returns).abs().max()
print(f"  max |(gross - cost/equity) - net| = {err:.2e}  "
      f"{'OK' if err < 1e-8 else 'ACCOUNTING BUG'}")
print(f"  total cost drag: {res.costs.sum():,.2f} on final equity {res.equity.iloc[-1]:,.2f}")
