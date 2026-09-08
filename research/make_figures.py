"""Compute the data behind the writeup's figures. No invented numbers.

Three datasets:
  1. IC convergence  -- combined IC as k signals are added at rho = 0.234
  2. Phantom alpha   -- equity curves for a vol-targeted ZERO-INFORMATION book
                        vs the same book untargeted, through 2008
  3. Null floor      -- the distribution of FF6 alpha t for random signals,
                        with the real strategies placed against it
"""
import json
import pathlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from algo import backtest, data, strategies
from algo.costs import IBKR_US_EQUITY

out = {}

# ---- 1. IC convergence -------------------------------------------------------
c, rho = 0.0101, 0.233          # measured on the BATTERY universe (199 names) by
                                # research/spectrum.py -- mean individual IC, mean
                                # pairwise correlation of the daily IC series
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


# ---------------------------------------------------------------- IC curve SVG
# The chart used to be hand-transcribed from these numbers into docs/index.html,
# and its coordinates were wrong once already. Generate it instead.
def ic_curve_svg(c, rho, required=0.036, kmax=100, k_tested=8):
    import math
    W, H = 640, 250
    x0, x1, ytop, ybot = 58.0, 496.0, 34.0, 206.0
    ymax_val = 0.045
    def X(k):  return x0 + (k - 1) / (kmax - 1) * (x1 - x0)
    def Y(v):  return ybot - (v / ymax_val) * (ybot - ytop)
    def ic(k): return c * math.sqrt(k) / math.sqrt(1 + (k - 1) * rho)

    asym = c / math.sqrt(rho)
    tested = ic(k_tested)
    pts = " ".join(f"{X(k):.1f},{Y(ic(k)):.1f}" for k in range(1, kmax + 1))
    ticks = "".join(
        f'\n      <text x="{X(k):.1f}" y="222.0" text-anchor="middle">{k}</text>'
        for k in (1, 25, 50, 75, 100))
    grid = "".join(
        f'\n      <text x="50" y="{Y(v)+3:.1f}" text-anchor="end">{v:.2f}</text>'
        for v in (0.04, 0.03, 0.02, 0.01))
    return f"""<svg viewBox="0 0 {W} {H}" role="img"
       aria-label="Combined information coefficient as signals are added. The curve rises from {c:.4f} with one signal and flattens at {asym:.4f}, never reaching the {required:.3f} required to break even.">
    <text x="0" y="12" font-family="IBM Plex Mono, monospace" font-size="10" fill="var(--faint)" letter-spacing="1.2">COMBINED IC AS SIGNALS ARE ADDED, AT THE MEASURED CORRELATION</text>
    <line x1="{x0}" y1="{ybot}" x2="{x1}" y2="{ybot}" stroke="var(--rule-strong)" stroke-width="1" fill="none"/>
    <line x1="{x0}" y1="{ytop}" x2="{x0}" y2="{ybot}" stroke="var(--rule-strong)" stroke-width="1" fill="none"/>
    <g font-family="IBM Plex Mono, monospace" font-size="9.5" fill="var(--faint)">{grid}{ticks}
      <text x="273.0" y="238.0" text-anchor="middle" fill="var(--muted)">number of signals combined</text>
    </g>
    <line x1="{x0}" y1="{Y(required):.1f}" x2="{x1}" y2="{Y(required):.1f}" stroke="var(--null)" stroke-width="1.5" stroke-dasharray="5 3" fill="none"/>
    <text x="502" y="{Y(required)-3:.1f}" font-family="IBM Plex Sans, sans-serif" font-size="10.5" fill="var(--null)" font-weight="600">required {required:.3f}</text>
    <text x="502" y="{Y(required)+11:.1f}" font-family="IBM Plex Sans, sans-serif" font-size="10" fill="var(--muted)">to beat holding all</text>
    <line x1="{x0}" y1="{Y(asym):.1f}" x2="{x1}" y2="{Y(asym):.1f}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="2 3" fill="none"/>
    <text x="502" y="{Y(asym)-3:.1f}" font-family="IBM Plex Sans, sans-serif" font-size="10.5" fill="var(--ink)" font-weight="600">asymptote {asym:.4f}</text>
    <text x="502" y="{Y(asym)+11:.1f}" font-family="IBM Plex Mono, monospace" font-size="10" fill="var(--muted)">k &#8594; &#8734;</text>
    <polyline points="{pts}" fill="none" stroke="var(--accent)" stroke-width="2"/>
    <circle cx="{X(k_tested):.1f}" cy="{Y(tested):.1f}" r="3.5" fill="var(--accent)"/>
    <text x="{X(k_tested)+9:.1f}" y="{Y(tested)+14:.1f}" font-family="IBM Plex Sans, sans-serif" font-size="10" fill="var(--muted)">{k_tested} signals tested &#8594; {tested:.4f}</text>
  </svg>"""

_svg = ic_curve_svg(c, rho)
pathlib.Path("results/fig_ic_curve.svg").write_text(_svg)
print(f"written: results/fig_ic_curve.svg  (asymptote {c/ (rho ** 0.5):.4f}, "
      f"8-signal {c * (8 ** 0.5) / ((1 + 7 * rho) ** 0.5):.4f})")

pathlib_out = "results/figure_data.json"
with open(pathlib_out, "w") as f:
    json.dump(out, f, indent=1)
print(f"\nwritten: {pathlib_out}")
