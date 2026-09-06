"""IBKR Lite vs Pro for a small systematic account.

Lite: $0 commission on US exchange-listed stocks/ETFs, but orders are routed to
      wholesalers for payment-for-order-flow -> worse fills.
Pro:  $0.35/order minimum, but SmartRouting with no PFOF and documented price
      improvement.

Lite trades a FIXED cost ($0.35/order) for a PROPORTIONAL one (worse fill).
Fixed costs crush small orders; proportional ones don't. So the answer depends
entirely on order size.
"""
import numpy as np, pandas as pd
from algo import data, portfolio, orderlevel

px = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")
w = portfolio.build(px)
years = len(px) / 252


def band(weights, b):
    cur = np.zeros(weights.shape[1]); rows = []
    for _, tgt in weights.iterrows():
        t = tgt.to_numpy(); mv = np.abs(t - cur) > b
        cur = np.where(mv, t, cur); rows.append(cur.copy())
    return pd.DataFrame(rows, index=weights.index, columns=weights.columns)


bw = band(w, 0.10)

print("CROSSOVER: at what order size does Pro's $0.35 beat Lite's worse fill?\n")
print(f"{'order size':>12} {'Pro commission':>16} {'Lite penalty (1.5bp)':>22} {'winner':>8}")
for n in [100, 250, 500, 1000, 2333, 5000, 10000, 25000]:
    pro = 0.35 / n * 1e4
    lite = 1.5
    print(f"{'$'+format(n,','):>12} {pro:>13.1f}bp {lite:>19.1f}bp {'Lite' if pro>lite else 'Pro':>8}")
print("\n  Crossover near $2,300/order. Below it, the fixed fee dominates.")

print("\n\nFULL BACKTEST, 10% band rebalancing\n")
print(f"{'capital':>10}  {'PRO ($0.35 min, 1.5bp)':>24}  {'LITE ($0 comm, 3.0bp)':>23}")
print(f"{'':>10}  {'CAGR':>10} {'Sharpe':>6} {'cost/yr':>6}  {'CAGR':>9} {'Sharpe':>6} {'cost/yr':>6}")
for cap in [500, 1_000, 5_000, 25_000, 100_000]:
    pro = orderlevel.run(px, bw, capital=cap, rebalance=None, min_trade=25.0,
                         spread_slip_bps=1.5)
    # Lite: no commission floor, but wider effective spread from PFOF routing.
    orderlevel.MIN_ORDER, orderlevel.PER_SHARE = 0.0, 0.0
    lite = orderlevel.run(px, bw, capital=cap, rebalance=None, min_trade=25.0,
                          spread_slip_bps=3.0)
    orderlevel.MIN_ORDER, orderlevel.PER_SHARE = 0.35, 0.0035
    print(f"{'$'+format(cap,','):>10}  {pro['cagr']*100:>9.2f}% {pro['sharpe']:>6.2f} "
          f"{pro['cost_pct_of_capital']/years*100:>5.2f}%  {lite['cagr']*100:>8.2f}% "
          f"{lite['sharpe']:>6.2f} {lite['cost_pct_of_capital']/years*100:>5.2f}%")

print("\n\nAND WITHOUT THE BAND (daily rebalance) -- does Lite rescue it?\n")
print(f"{'capital':>10}  {'PRO CAGR':>10}  {'LITE CAGR':>10}")
for cap in [1_000, 25_000]:
    pro = orderlevel.run(px, w, capital=cap, rebalance=None, min_trade=25.0, spread_slip_bps=1.5)
    orderlevel.MIN_ORDER, orderlevel.PER_SHARE = 0.0, 0.0
    lite = orderlevel.run(px, w, capital=cap, rebalance=None, min_trade=25.0, spread_slip_bps=3.0)
    orderlevel.MIN_ORDER, orderlevel.PER_SHARE = 0.35, 0.0035
    print(f"{'$'+format(cap,','):>10}  {pro['cagr']*100:>9.2f}%  {lite['cagr']*100:>9.2f}%")
