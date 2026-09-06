"""Two demonstrations of where retail edge actually comes from."""
import numpy as np, pandas as pd
from algo import backtest, data, metrics, strategies
from algo.costs import IBKR_US_EQUITY

px = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")
spy = px[["SPY"]]

print("=" * 72)
print("1. VOLATILITY TARGETING -- improves Sharpe while forecasting nothing")
print("=" * 72)
bh_w = pd.DataFrame(1.0, index=spy.index, columns=spy.columns)
bh = backtest.run(spy, bh_w, IBKR_US_EQUITY)
print(f"  {'SPY buy & hold':<28} Sharpe {metrics.sharpe(bh.returns):5.2f}"
      f"  CAGR {metrics.cagr(bh.equity)*100:6.2f}%  vol {metrics.volatility(bh.returns)*100:5.1f}%"
      f"  MaxDD {metrics.max_drawdown(bh.equity)*100:6.1f}%")
for tv in [0.08, 0.10, 0.12, 0.15]:
    w = strategies.volatility_target(bh_w, spy, target_vol=tv, max_leverage=1.0)
    r = backtest.run(spy, w, IBKR_US_EQUITY)
    print(f"  {'vol-targeted @ '+str(int(tv*100))+'%':<28} Sharpe {metrics.sharpe(r.returns):5.2f}"
          f"  CAGR {metrics.cagr(r.equity)*100:6.2f}%  vol {metrics.volatility(r.returns)*100:5.1f}%"
          f"  MaxDD {metrics.max_drawdown(r.equity)*100:6.1f}%  turnover {r.turnover.mean():.3f}")
print("\n  Note: CAGR mostly FALLS (exposure is only ever cut, never levered).")
print("  Sharpe and drawdown improve. It is a risk trade, not a return trade.")

print("\n" + "=" * 72)
print("2. PARAMETER ENSEMBLING -- beats picking the best backtested parameter")
print("=" * 72)
LOOKBACKS = [63, 126, 189, 252, 315]
split = len(px) // 2
first, second = px.iloc[:split], px.iloc[split:]

print(f"  in-sample  {first.index[0].date()} .. {first.index[-1].date()}")
print(f"  out-sample {second.index[0].date()} .. {second.index[-1].date()}\n")
print(f"  {'spec':<22} {'in-sample':>10} {'out-sample':>11} {'decay':>8}")
res = {}
for lb in LOOKBACKS:
    w = strategies.cross_sectional_momentum(px, lookback=lb, skip=21, top_n=3)
    i = metrics.sharpe(backtest.run(first, w.loc[first.index], IBKR_US_EQUITY).returns)
    o = metrics.sharpe(backtest.run(second, w.loc[second.index], IBKR_US_EQUITY).returns)
    res[lb] = (i, o)
    print(f"  {'lookback '+str(lb):<22} {i:10.2f} {o:11.2f} {o-i:8.2f}")

best_lb = max(res, key=lambda k: res[k][0])
ens_w = strategies.ensemble(*[strategies.cross_sectional_momentum(px, lookback=lb, skip=21, top_n=3)
                              for lb in LOOKBACKS])
ei = metrics.sharpe(backtest.run(first, ens_w.loc[first.index], IBKR_US_EQUITY).returns)
eo = metrics.sharpe(backtest.run(second, ens_w.loc[second.index], IBKR_US_EQUITY).returns)
print(f"  {'ENSEMBLE (all 5)':<22} {ei:10.2f} {eo:11.2f} {eo-ei:8.2f}")
print(f"\n  Picking the in-sample winner (lookback {best_lb}) gave OOS Sharpe {res[best_lb][1]:.2f}")
print(f"  The ensemble gave OOS Sharpe {eo:.2f}.")
