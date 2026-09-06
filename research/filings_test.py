"""Lazy Prices: does year-over-year filing similarity predict returns?

PRE-SPECIFIED, count fixed before running:
    3 measures (sim_cosine, sim_cosine_tfidf, sim_jaccard)
  x 3 horizons (h = 21, 63, 126 trading days)
  = 9 tests.

Horizons are monthly-to-semiannual because that is the frequency the effect is
documented at -- filings are quarterly and staggered, so the panel updates
continuously but any individual name's signal only refreshes four times a year.
Testing h=1 would be testing something the mechanism does not claim.

SIGN: the paper says firms that CHANGE their language underperform. The feature
is SIMILARITY, so high = unchanged = expected outperformance, and a positive IC
confirms the paper.

POINT-IN-TIME: everything anchors to filing_date + 1 day. Never report_date --
the period ends weeks before anyone can read the document.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sst

from algo import data, filings, research

pd.set_option("display.width", 220)

uni = pd.read_parquet("data/collection_universe.parquet").dropna(subset=["cik"])
uni["cik"] = uni["cik"].astype(int).astype(str)

# ---- assemble measures -------------------------------------------------------
mp = Path("data/edgar/measures.parquet")
if mp.exists():
    M = pd.read_parquet(mp)
else:
    rows = []
    for i, r in enumerate(uni.itertuples(), 1):
        p = Path(f"data/edgar/index/{r.cik}.parquet")
        if not p.exists():
            continue
        idx = pd.read_parquet(p)
        idx = idx[idx.filing_date >= "2005-01-01"]
        m = filings.similarity_measures(r.cik, idx)
        if len(m):
            m["ticker"] = r.ticker
            rows.append(m)
        if i % 25 == 0:
            print(f"  measured {i}/{len(uni)}", flush=True)
    M = pd.concat(rows, ignore_index=True)
    M.to_parquet(mp)

print(f"{len(M):,} filing-pairs, {M.ticker.nunique()} names, "
      f"{M.filing_date.min().date()}..{M.filing_date.max().date()}\n")

px = data.load_panel(sorted(M.ticker.unique()), start="2005-01-01", end="2026-09-01",
                     field="close", refresh=False)
px = px.dropna(axis=1, thresh=int(0.9 * len(px))).ffill(limit=5)
M = M[M.ticker.isin(px.columns)]
tick = dict(zip(M.cik.astype(str), M.ticker))
print(f"prices: {px.shape[1]} names, {len(px)} bars")

MEASURES = ["sim_cosine", "sim_cosine_tfidf", "sim_jaccard"]
HZ = (21, 63, 126)
K = len(MEASURES) * len(HZ)
bar = sst.norm.ppf(1 - 0.05 / (2 * K))
print(f"\n{'='*80}\n{K} pre-specified tests -> Bonferroni bar |t| > {bar:.2f}\n{'='*80}\n")
print(f"{'measure':<20} {'h':>4} {'IC':>9} {'t':>8} {'n days':>8} {'coverage':>9}")

rows = []
panels = {}
for meas in MEASURES:
    f = filings.to_daily(M, px.index, tick, col=meas, lag_days=1, hold_days=126)
    panels[meas] = f
    cov = f.notna().mean(axis=1).mean()
    for hz in HZ:
        ic = research.cross_sectional_ic(f, research.forward_returns(px, hz)).dropna()
        if len(ic) < 100:
            continue
        m = float(ic.mean())
        t = (m / (ic.std(ddof=1) / np.sqrt(len(ic)))) / np.sqrt(hz)
        rows.append((meas, hz, m, t))
        print(f"{meas:<20} {hz:>4} {m:>9.4f} {t:>8.2f} {len(ic):>8} {cov*100:>8.0f}%")

df = pd.DataFrame(rows, columns=["measure", "h", "ic", "t"])
surv = df[df.t.abs() > bar]
print(f"\nsurvivors: {len(surv)}")
for _, r in surv.iterrows():
    print(f"   {r.measure:<20} h={int(r.h):<4} IC {r.ic:+.4f}  t {r.t:+.2f}")
print(f"\nlargest |t|: {df.t.abs().max():.2f}   "
      f"expected max under the null: ~{sst.norm.ppf(1-1/(2*K)):.2f}")
print(f"largest |IC|: {df.ic.abs().max():.4f}   required to break even: ~0.036")

# ---- the paper's own construction: quintile spread ---------------------------
print(f"\n{'-'*80}\nQUINTILE SPREAD (the paper's construction), h=63\n{'-'*80}")
fwd = research.forward_returns(px, 63)
print(f"{'measure':<20} {'Q1 (changed)':>14} {'Q5 (unchanged)':>15} {'Q5-Q1':>9} {'t':>7}")
for meas in MEASURES:
    f = panels[meas]
    q = f.rank(axis=1, pct=True)
    lo = fwd.where(q <= 0.2).mean(axis=1)
    hi = fwd.where(q >= 0.8).mean(axis=1)
    d = (hi - lo).dropna()
    if len(d) < 100:
        continue
    t = (d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) / np.sqrt(63)
    print(f"{meas:<20} {lo.mean()*100:>13.2f}% {hi.mean()*100:>14.2f}% "
          f"{d.mean()*100:>8.2f}% {t:>7.2f}")
df.to_csv("filings_test.csv", index=False)
