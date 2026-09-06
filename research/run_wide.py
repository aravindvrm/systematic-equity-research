"""Does a wider universe buy breadth -- and does cross-sectional beat time-series?"""
import numpy as np, pandas as pd
from algo import backtest, data, features, metrics, research, strategies
from algo.costs import ZERO_COST

pd.set_option("display.width", 200)
narrow = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")
wide = data.load_panel(data.WIDE_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")
print(f"narrow: {narrow.shape[1]} ETFs, {len(narrow)} bars")
print(f"wide:   {wide.shape[1]} ETFs, {len(wide)} bars, {wide.index[0].date()}..{wide.index[-1].date()}\n")

print("BREADTH")
for nm, p in [("narrow (10)", narrow), (f"wide ({wide.shape[1]})", wide)]:
    eb = research.effective_bets(p.pct_change().dropna())
    print(f"  {nm:<14} effective bets {eb:5.2f}   BR@daily {eb*252:7.0f}   "
          f"IR@IC=0.03 {0.03*np.sqrt(eb*252):.2f}")

print("\n\nIC: CROSS-SECTIONAL vs TIME-SERIES, narrow vs wide (h=1)\n")
print(f"{'feature':<14} {'xsec narrow':>12} {'xsec wide':>11} {'ts narrow':>11} {'ts wide':>9}")
for name in ["mom_12_1", "mom_252", "mom_126", "mom_63", "low_vol_60", "reversal_5"]:
    row = []
    for p in (narrow, wide):
        f = features.REGISTRY[name](p); fwd = research.forward_returns(p, 1)
        row.append(research.cross_sectional_ic(f, fwd).mean())
    for p in (narrow, wide):
        f = features.REGISTRY[name](p); fwd = research.forward_returns(p, 1)
        row.append(research.time_series_ic(f, fwd).mean())
    print(f"{name:<14} {row[0]:>12.4f} {row[1]:>11.4f} {row[2]:>11.4f} {row[3]:>9.4f}")

print("\n\nSTRATEGY: time-series vs cross-sectional momentum, gross of costs")
print("(daily rebalance, inverse-vol weighted, vol-targeted)\n")
def stack(p, raw):
    iv = strategies.inverse_volatility(p, 60)
    c = raw * iv; tot = c.sum(axis=1)
    c = c.div(tot.where(tot > 0), axis=0).fillna(0.0).mul(raw.sum(axis=1), axis=0)
    return strategies.volatility_target(c, p, target_vol=0.10, max_leverage=1.0)

print(f"{'universe':<12} {'signal':<26} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8}")
for nm, p in [("narrow", narrow), ("wide", wide)]:
    ts = strategies.ensemble(*[strategies.time_series_momentum(p, lb) for lb in (63,126,189,252,315)])
    top = max(3, p.shape[1] // 4)
    xs = strategies.ensemble(*[strategies.cross_sectional_momentum(p, lb, 21, top) for lb in (63,126,189,252,315)])
    for label, w in [("time-series (current)", ts), (f"cross-sectional (top {top})", xs)]:
        r = backtest.run(p, stack(p, w), ZERO_COST)
        print(f"{nm:<12} {label:<26} {metrics.sharpe(r.returns):>8.2f} "
              f"{metrics.cagr(r.equity)*100:>7.2f}% {metrics.max_drawdown(r.equity)*100:>7.1f}%")
bh = backtest.buy_and_hold(narrow, "SPY", cost_model=ZERO_COST)
print(f"{'--':<12} {'SPY buy & hold':<26} {metrics.sharpe(bh.returns):>8.2f} "
      f"{metrics.cagr(bh.equity)*100:>7.2f}% {metrics.max_drawdown(bh.equity)*100:>7.1f}%")
