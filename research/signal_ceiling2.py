"""Does signal quality pay off at the rebalance frequency a small account can afford?"""
import numpy as np, pandas as pd
from algo import backtest, data, metrics, strategies, orderlevel
from algo.costs import IBKR_US_EQUITY

px = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")


def synth(prices, ic, seed=0):
    r = np.random.default_rng(seed)
    fut = prices.pct_change().shift(-1)
    z = (fut - fut.mean()) / fut.std()
    noise = pd.DataFrame(r.normal(size=z.shape), index=z.index, columns=z.columns)
    fc = ic * z + np.sqrt(1 - ic**2) * noise
    sig = (fc > 0).astype(float)
    n = sig.sum(axis=1).replace(0, np.nan)
    return sig.div(n, axis=0).fillna(0.0)


def hold_for(w, days):
    """Only act on the signal every `days` bars; hold in between."""
    mask = pd.Series(False, index=w.index)
    mask.iloc[::days] = True
    return w.where(mask).ffill().fillna(0.0)


print("SHARPE vs SIGNAL QUALITY, AT DIFFERENT REBALANCE FREQUENCIES")
print("(gross of the per-order minimum; vectorized costs only)\n")
print(f"{'IC':>6} " + "".join(f"{lab:>12}" for lab in ["daily","weekly(5d)","monthly(21d)","quarterly"]))
for ic in [0.00, 0.03, 0.05, 0.10, 0.20]:
    w = synth(px, ic, seed=int(ic * 1000) + 7)
    cells = ""
    for d in [1, 5, 21, 63]:
        r = backtest.run(px, hold_for(w, d), IBKR_US_EQUITY)
        cells += f"{metrics.sharpe(r.returns):>12.2f}"
    print(f"{ic:>6.2f} {cells}")

print("\n  Breadth = effective_assets x rebalances/yr. Slowing down throws breadth")
print("  away, so the payoff to signal quality collapses with it.\n")

print("\nNOW WITH REAL PER-ORDER COSTS AT $1,000 (order-level sim)\n")
print(f"{'IC':>6} {'daily CAGR':>12} {'monthly CAGR':>14}")
for ic in [0.03, 0.05, 0.10, 0.20]:
    w = synth(px, ic, seed=int(ic * 1000) + 7)
    d = orderlevel.run(px, w, capital=1000, rebalance=None, min_trade=50.0)
    m = orderlevel.run(px, hold_for(w, 21), capital=1000, rebalance="ME", min_trade=50.0)
    print(f"{ic:>6.2f} {d['cagr']*100:>11.2f}% {m['cagr']*100:>13.2f}%")
