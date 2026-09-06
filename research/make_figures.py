"""Compute the data behind the writeup's figures. No invented numbers.

Three datasets:
  1. IC convergence  -- combined IC as k signals are added at rho = 0.234
  2. Phantom alpha   -- equity curves for a vol-targeted ZERO-INFORMATION book
                        vs the same book untargeted, through 2008
  3. Null floor      -- the distribution of FF6 alpha t for random signals,
                        with the real strategies placed against it
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from algo import backtest, data, strategies
from algo.costs import IBKR_US_EQUITY

out = {}

# ---- 1. IC convergence -------------------------------------------------------
c, rho = 0.0108, 0.234          # measured: mean individual IC, mean pairwise IC corr
ks = list(range(1, 101))
out["ic_curve"] = [{"k": k, "ic": c * np.sqrt(k) / np.sqrt(1 + (k - 1) * rho)} for k in ks]
out["ic_asymptote"] = c / np.sqrt(rho)
out["ic_required"] = 0.036
print(f"1. IC convergence: k=1 {out['ic_curve'][0]['ic']:.4f} -> "
      f"k=100 {out['ic_curve'][-1]['ic']:.4f}, asymptote {out['ic_asymptote']:.4f}")

# ---- 2. Phantom alpha --------------------------------------------------------
uni = pd.read_parquet("data/collection_universe.parquet")["ticker"].tolist()
px = data.load_panel(uni, start="2005-01-01", end="2026-09-01", refresh=False)
px = px.dropna(axis=1, thresh=int(0.9 * len(px))).ffill(limit=5)
iv = strategies.inverse_volatility(px, 60)

def book(vol_target):
    rng = np.random.default_rng(7)           # a ZERO-INFORMATION signal
    noise = pd.DataFrame(rng.normal(size=px.shape), index=px.index, columns=px.columns)
    m = np.zeros(len(noise), dtype=bool); m[::21] = True
    s = noise.where(pd.Series(m, index=noise.index), np.nan).ffill()
    raw = (s.rank(axis=1, pct=True, ascending=False) <= 0.3).astype(float) * iv
    w = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if vol_target:
        w = strategies.volatility_target(w, px, target_vol=0.10, max_leverage=1.0)
    w = w.where(pd.Series(m, index=w.index), np.nan).ffill()
    return backtest.run(px, w, cost_model=IBKR_US_EQUITY).equity

eq_t, eq_u = book(True), book(False)
mo = eq_t.resample("ME").last().index
out["phantom"] = [
    {"d": d.strftime("%Y-%m"),
     "targeted": float(eq_t.resample("ME").last()[d] / eq_t.iloc[0]),
     "untargeted": float(eq_u.resample("ME").last()[d] / eq_u.iloc[0])}
    for d in mo if pd.notna(eq_t.resample("ME").last()[d])
]
def dd(e): return float((e / e.cummax() - 1).min())
out["phantom_dd"] = {"targeted": dd(eq_t), "untargeted": dd(eq_u)}
print(f"2. phantom alpha: max drawdown targeted {dd(eq_t)*100:.1f}% vs "
      f"untargeted {dd(eq_u)*100:.1f}%  ({len(out['phantom'])} monthly points)")

# ---- 3. Null floor -----------------------------------------------------------
D = pd.read_csv("results/reassess_all.csv")
out["null"] = {"mean": 2.75, "sd": 0.36, "p95": 3.27, "max": 3.45}
out["strategies"] = [
    {"name": r.strategy, "t": float(r.t_u), "pct": float(r.pct_u)}
    for r in D.itertuples()
    if r.strategy in ("0 NO-SIGNAL large-cap EW", "9 rel:rel_strength_126",
                      "11 filings:jaccard", "1 px:COMPOSITE", "12 pead:sue",
                      "4 insider:ins_net_value")
]
print(f"3. null floor: {len(out['strategies'])} strategies placed against "
      f"mean {out['null']['mean']}, p95 {out['null']['p95']}")

pathlib_out = "results/figure_data.json"
with open(pathlib_out, "w") as f:
    json.dump(out, f, indent=1)
print(f"\nwritten: {pathlib_out}")
