"""Stress the crypto result before believing it."""
import numpy as np, pandas as pd
from scipy import stats
from algo import backtest, data, metrics, strategies
from algo.costs import IBKR_US_EQUITY, IBKR_CRYPTO

CRYPTO = ["BTC-USD","ETH-USD","LTC-USD","BCH-USD","SOL-USD",
          "ADA-USD","DOGE-USD","AVAX-USD","LINK-USD","XRP-USD"]
eq = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")
cr = data.load_panel(CRYPTO, start="2015-01-01", end="2026-09-01").dropna(how="any")

print("1) DISTRIBUTION SHAPE  (Sharpe assumes roughly Gaussian; fat tails break it)")
for nm, px, ppy in [("equity", eq, 252), ("crypto", cr, 365)]:
    r = px.pct_change().dropna().mean(axis=1)
    print(f"   {nm:7s} skew {stats.skew(r):6.2f}   excess kurtosis {stats.kurtosis(r):7.2f}"
          f"   worst day {r.min()*100:6.1f}%   best day {r.max()*100:5.1f}%")

print("\n2) MULTIPLE TESTING  (the sweep was 6 lookbacks x 2 venues = 12 trials)")
for nm, px, cm, ppy in [("equity", eq, IBKR_US_EQUITY, 252), ("crypto", cr, IBKR_CRYPTO, 365)]:
    w = strategies.cross_sectional_momentum(px, lookback=21, skip=1, top_n=3)
    r = backtest.run(px, w, cm, periods_per_year=ppy)
    s = metrics.sharpe(r.returns, periods_per_year=ppy)
    print(f"   {nm:7s} raw {s:5.2f}  deflated(12 trials) {metrics.deflated_sharpe(s,12,len(px)):5.2f}"
          f"  deflated(100) {metrics.deflated_sharpe(s,100,len(px)):5.2f}")

print("\n3) REGIME DEPENDENCE  (crypto history is short; split it in half)")
w = strategies.cross_sectional_momentum(cr, lookback=21, skip=1, top_n=3)
r = backtest.run(cr, w, IBKR_CRYPTO, periods_per_year=365)
mid = len(r.returns)//2
h1, h2 = r.returns.iloc[:mid], r.returns.iloc[mid:]
print(f"   1st half {h1.index[0].date()}..{h1.index[-1].date()}  Sharpe {metrics.sharpe(h1,periods_per_year=365):5.2f}")
print(f"   2nd half {h2.index[0].date()}..{h2.index[-1].date()}  Sharpe {metrics.sharpe(h2,periods_per_year=365):5.2f}")
w2 = strategies.cross_sectional_momentum(eq, lookback=21, skip=1, top_n=3)
r2 = backtest.run(eq, w2, IBKR_US_EQUITY, periods_per_year=252)
m2 = len(r2.returns)//2
print(f"   equity 1st half Sharpe {metrics.sharpe(r2.returns.iloc[:m2]):5.2f}"
      f"   2nd half {metrics.sharpe(r2.returns.iloc[m2:]):5.2f}")

print("\n4) BENCHMARK  (is the crypto strategy beating just holding BTC?)")
bh = backtest.buy_and_hold(cr, "BTC-USD", cost_model=IBKR_CRYPTO, periods_per_year=365)
print(f"   BTC buy & hold      Sharpe {metrics.sharpe(bh.returns,periods_per_year=365):5.2f}"
      f"   CAGR {metrics.cagr(bh.equity,365)*100:7.1f}%   MaxDD {metrics.max_drawdown(bh.equity)*100:6.1f}%")
print(f"   XS momentum (1m)    Sharpe {metrics.sharpe(r.returns,periods_per_year=365):5.2f}"
      f"   CAGR {metrics.cagr(r.equity,365)*100:7.1f}%   MaxDD {metrics.max_drawdown(r.equity)*100:6.1f}%")
