"""Three ways to actually run this at $1,000."""
import numpy as np, pandas as pd
from algo import data, portfolio, orderlevel

px = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")
w = portfolio.build(px)
years = len(px) / 252

def band(weights, b):
    cur = np.zeros(weights.shape[1]); rows = []
    for _, t in weights.iterrows():
        t = t.to_numpy(); mv = np.abs(t - cur) > b
        cur = np.where(mv, t, cur); rows.append(cur.copy())
    return pd.DataFrame(rows, index=weights.index, columns=weights.columns)

def run(weights, cap, commission_min, per_share, slip):
    orderlevel.MIN_ORDER, orderlevel.PER_SHARE = commission_min, per_share
    r = orderlevel.run(px, weights, capital=cap, rebalance=None, min_trade=25.0, spread_slip_bps=slip)
    orderlevel.MIN_ORDER, orderlevel.PER_SHARE = 0.35, 0.0035
    return r

bw10 = band(w, 0.10)

print("AT $1,000, 10% BAND REBALANCING (~30 orders/yr, ~2-3 trades/month)\n")
print(f"{'path':<38} {'CAGR':>8} {'Sharpe':>8} {'cost/yr':>9} {'trades/yr':>10}")
cfgs = [
    ("1. IBKR Pro Tiered + API",        bw10, 0.35, 0.0035, 1.5),
    ("2. IBKR Lite + MANUAL execution", bw10, 0.00, 0.0,    3.0),
    ("3. Alpaca + API (commission-free)",bw10, 0.00, 0.0,   3.0),
]
for name, ww, cm, ps, sl in cfgs:
    r = run(ww, 1000, cm, ps, sl)
    print(f"{name:<38} {r['cagr']*100:>7.2f}% {r['sharpe']:>8.2f} "
          f"{r['cost_pct_of_capital']/years*100:>8.2f}% {r['n_orders']/years:>10.0f}")

print("\n\nSAME PATHS, DAILY REBALANCING (no band) -- the frequency Pro can't afford\n")
print(f"{'path':<38} {'CAGR':>8} {'Sharpe':>8} {'cost/yr':>9} {'trades/yr':>10}")
for name, _, cm, ps, sl in cfgs:
    r = run(w, 1000, cm, ps, sl)
    print(f"{name:<38} {r['cagr']*100:>7.2f}% {r['sharpe']:>8.2f} "
          f"{r['cost_pct_of_capital']/years*100:>8.2f}% {r['n_orders']/years:>10.0f}")

print("\n\nMANUAL EXECUTION WORKLOAD BY BAND (at $1,000)\n")
print(f"{'band':>7} {'trades/yr':>11} {'trades/month':>14} {'CAGR (0-comm)':>15}")
for b in [0.02, 0.05, 0.10, 0.15]:
    bb = band(w, b)
    r = run(bb, 1000, 0.0, 0.0, 3.0)
    print(f"{b*100:>6.0f}% {r['n_orders']/years:>11.0f} {r['n_orders']/years/12:>14.1f} {r['cagr']*100:>14.2f}%")
