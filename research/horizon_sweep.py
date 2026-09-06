"""Is there a viable middle ground -- weeks to months?

Two forces pull opposite ways as holding period lengthens:
  + IC tends to RISE (slower signals are less noisy)
  + cost per year FALLS (fewer round trips)
  - BREADTH falls (fewer independent bets per year)

IR = IC x sqrt(BR), so there may be an interior optimum. This finds it, using
the DYNAMIC (static-tilt-removed) IC rather than the inflated raw number.
"""
import numpy as np, pandas as pd
from algo import data, features, features2 as f2, research

BLUE = ["MSFT","JPM","JNJ","XOM","PG","HD","CAT","NEE","LIN","VZ","AMT",
        "UNH","WMT","CVX","KO","MCD","HON","TXN","DE","GS"]
DIV = ["TLT","IEF","GLD","DBC"]
px = data.load_panel(BLUE + DIV, start="2012-01-01", end="2026-09-01").dropna(how="any")
eff = research.effective_bets(px.pct_change().dropna())
print(f"universe: {px.shape[1]} names, {len(px)} bars, effective bets {eff:.2f}\n")

HORIZONS = [5, 10, 21, 42, 63, 126]
CANDS = {
    "mom_12_1":     features.momentum_12_1(px),
    "mom_252":      features.momentum(px, 252),
    "mom_126":      features.momentum(px, 126),
    "resid_mom_252": f2.residual_momentum(px, 252),
    "rel_str_252":  f2.relative_strength(px, 252),
    "reversal_5":   features.reversal(px, 5),
    "reversal_21":  features.reversal(px, 21),
}

print("DYNAMIC IC BY HORIZON (static tilt removed)\n")
print(f"{'feature':<15} " + "".join(f"{'h='+str(h):>9}" for h in HORIZONS))
dyn = {}
for name, f in CANDS.items():
    row = []
    for h in HORIZONS:
        r = research.static_vs_dynamic(px, f, horizon=h)
        row.append(r["ic_dynamic"])
    dyn[name] = row
    print(f"{name:<15} " + "".join(f"{v:>9.4f}" for v in row))

print("\n\nIMPLIED NET SHARPE  (IR = IC x sqrt(eff_bets x 252/h), minus costs)")
print("assumes 100% turnover per rebalance -- an upper bound on cost")
print("Alpaca: $0 commission, ~3bp round-trip spread+slippage\n")
COST_RT = 3e-4
VOL = 0.10
print(f"{'feature':<15} " + "".join(f"{'h='+str(h):>9}" for h in HORIZONS))
best = (-9, None, None)
for name, ics in dyn.items():
    row = []
    for h, ic in zip(HORIZONS, ics):
        br = eff * (252 / h)
        ir = ic * np.sqrt(br)
        cost_drag = (252 / h) * COST_RT / VOL      # in Sharpe units
        net = ir - cost_drag
        row.append(net)
        if net > best[0]:
            best = (net, name, h)
    print(f"{name:<15} " + "".join(f"{v:>9.2f}" for v in row))

print(f"\n  best cell: {best[1]} at h={best[2]}  ->  net Sharpe {best[0]:.2f}")
print(f"\n  For reference: SPY buy & hold = 0.86, our current portfolio = 0.81")
print("\n  CAVEAT: these are IMPLIED from IC, not backtested. They assume the")
print("  IC is real and stable. Several of these ICs failed significance.")
