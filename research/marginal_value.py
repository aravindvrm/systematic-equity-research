"""You already own the beta. So what is the marginal value of adding this sleeve?

Benchmarking a strategy against SPY assumes SPY is the ALTERNATIVE. If you
already hold VOO/QQQM in a main portfolio, SPY is not an alternative -- it is
what you already have. A sleeve that merely replicates it adds nothing, no
matter how good its standalone Sharpe.

The right question: does adding X% of this sleeve improve the COMBINED
portfolio? That depends far more on correlation than on standalone Sharpe.
"""
import numpy as np, pandas as pd
from algo import backtest, data, metrics, portfolio
from algo.costs import IBKR_US_EQUITY

px = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")
w = portfolio.build(px)
sleeve = backtest.run(px, w, IBKR_US_EQUITY)
core = backtest.buy_and_hold(px, "SPY", cost_model=IBKR_US_EQUITY)

s_r, c_r = sleeve.returns, core.returns
print("STANDALONE")
print(f"  core (VOO/SPY proxy)  Sharpe {metrics.sharpe(c_r):.2f}  CAGR {metrics.cagr(core.equity)*100:6.2f}%"
      f"  MaxDD {metrics.max_drawdown(core.equity)*100:6.1f}%")
print(f"  trend sleeve          Sharpe {metrics.sharpe(s_r):.2f}  CAGR {metrics.cagr(sleeve.equity)*100:6.2f}%"
      f"  MaxDD {metrics.max_drawdown(sleeve.equity)*100:6.1f}%")

corr = s_r.corr(c_r)
print(f"\n  CORRELATION to core: {corr:.2f}")
print(f"  {'-> highly correlated: adds little diversification' if corr > 0.7 else '-> meaningfully diversifying' if corr < 0.5 else '-> moderately diversifying'}")

# down-market behaviour: the thing you actually buy a sleeve for
down = c_r < c_r.quantile(0.05)
print(f"\n  On the core's worst 5% of days:")
print(f"    core   average {c_r[down].mean()*100:6.2f}%")
print(f"    sleeve average {s_r[down].mean()*100:6.2f}%")

print("\n\nBLENDED PORTFOLIO  (rebalanced daily between core and sleeve)")
print(f"{'sleeve %':>9} {'Sharpe':>8} {'CAGR %':>8} {'Vol %':>7} {'MaxDD %':>9} {'Calmar':>8}")
best = (-9, None)
for alloc in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    blend = (1 - alloc) * c_r + alloc * s_r
    eq = (1 + blend).cumprod()
    sh, cg = metrics.sharpe(blend), metrics.cagr(eq)
    dd, cal = metrics.max_drawdown(eq), metrics.calmar(eq)
    if sh > best[0]: best = (sh, alloc)
    mark = ""
    print(f"{alloc*100:>8.0f}% {sh:>8.2f} {cg*100:>8.2f} {metrics.volatility(blend)*100:>7.2f}"
          f" {dd*100:>9.2f} {cal:>8.2f}{mark}")
print(f"\n  Sharpe-optimal blend: {best[1]*100:.0f}% sleeve  (Sharpe {best[0]:.2f} vs {metrics.sharpe(c_r):.2f} for core alone)")

print("\n\nWHICH ASSETS IS THE SLEEVE ACTUALLY HOLDING?  (avg weight)")
avg = sleeve.weights.mean().sort_values(ascending=False)
for tick, v in avg.items():
    bar = "#" * int(v * 100)
    print(f"  {tick:5s} {v*100:5.1f}%  {bar}")
eq_names = ["SPY","QQQ","IWM","EFA","EEM","XLE","XLF"]
print(f"\n  equity-like: {avg[eq_names].sum()*100:.1f}%   "
      f"bonds/gold: {avg[['TLT','IEF','GLD']].sum()*100:.1f}%")
