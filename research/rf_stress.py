"""Stress-test the random forest result. Designed NOT to manufacture a winner.

THE TRAP THIS AVOIDS
--------------------
A grid search that reports its best configuration is a machine for producing
false positives: with 18 configs, the best one is the max of 18 draws, and the
max of 18 noise draws looks impressive. Reporting it as "the model" is exactly
the overfitting the whole project has been trying to avoid.

So this reports FOUR things, in order of how much they matter:

  1. THE FULL DISTRIBUTION over configs. If the effect is real, MOST configs
     work and the spread is tight. If only one works, it is noise.
  2. SEED STABILITY. Same config, different random seeds. A real effect barely
     moves; noise swings.
  3. BEST-OF-GRID vs a NULL OF BEST-OF-GRID. The comparison must be
     like-for-like: max of 18 fitted configs against max of 18 configs fitted to
     RANDOM targets. Comparing best-of-18 to a single-draw null is cheating.
  4. FEATURE IMPORTANCE STABILITY across years. A real signal uses the same
     features every year; an overfit one reshuffles.

OVERLAP: observations are monthly with 21-day forward returns, so labels do not
overlap and purged CV is unnecessary here. It WOULD be required with daily
sampling -- noted so the omission is deliberate rather than forgotten.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from scipy import stats as sst

from algo import (anomalies, data, features, features2 as f2, filings, pead,
                  research)

pd.set_option("display.width", 220)

M = pd.read_parquet("data/edgar/measures.parquet")
cl = data.load_panel(sorted(M.ticker.unique()), start="2009-01-01", end="2026-09-01",
                     field="close", refresh=False)
cl = cl.dropna(axis=1, thresh=int(0.9 * len(cl))).ffill(limit=5)
M = M[M.ticker.isin(cl.columns)]
P = {f: data.load_panel(sorted(M.ticker.unique()), start="2009-01-01",
                        end="2026-09-01", field=f, refresh=False)
     .reindex(index=cl.index, columns=cl.columns).ffill(limit=5)
     for f in ("open", "high", "low", "volume")}
o, hi, lo, vo = P["open"], P["high"], P["low"], P["volume"]

FEATS = {
    "mom_12_1": features.momentum_12_1(cl), "mom_252": features.momentum(cl, 252),
    "mom_63": features.momentum(cl, 63), "mom_21": features.momentum(cl, 21),
    "reversal_5": features.reversal(cl, 5), "reversal_21": features.reversal(cl, 21),
    "low_vol_60": features.low_vol(cl, 60), "vol_change": features.vol_change(cl),
    "near_high_252": features.near_high(cl, 252), "drawdown_252": features.drawdown(cl, 252),
    "skew_126": features.skewness(cl, 126), "ma_dist_200": features.ma_distance(cl, 200),
    "trend_r2": features.trend_strength(cl, 126),
    "vol_shock": f2.volume_shock(vo), "vol_trend": f2.volume_trend(vo),
    "dollar_volume": f2.dollar_volume(cl, vo),
    "px_vol_diverge": f2.price_volume_divergence(cl, vo),
    "garman_klass": f2.garman_klass_vol(o, hi, lo, cl),
    "close_in_range": f2.close_position_in_range(hi, lo, cl),
    "overnight_vs_day": f2.intraday_vs_overnight(o, cl),
    "true_range": f2.true_range_pct(hi, lo, cl),
    "resid_mom_252": f2.residual_momentum(cl, 252),
    "beta_to_univ": f2.beta_to_universe(cl), "corr_to_univ": f2.correlation_to_universe(cl),
    "idio_vol_share": f2.idio_vol_share(cl), "lead_lag": f2.lead_lag(cl),
}
F = pd.read_parquet("data/fundamentals/all_facts.parquet")
uni = pd.read_parquet("data/collection_universe.parquet").dropna(subset=["cik"])
uni["cik"] = uni["cik"].astype(int).astype(str)
c2t = {k: v for k, v in zip(uni["cik"], uni["ticker"]) if v in cl.columns}
V = anomalies.compute(F)
for f in anomalies.FEATURES:
    FEATS[f] = anomalies.to_panel(V, cl.index, c2t, f)
FEATS.update(anomalies.market_ratios(V, cl, c2t))
FEATS["filing_sim"] = filings.to_daily(M, cl.index, dict(zip(M.cik.astype(str), M.ticker)),
                                       col="sim_jaccard", lag_days=1, hold_days=126)
R = pd.read_parquet("data/edgar/pead_reactions.parquet")
FEATS["pead_sue"] = pead.to_daily(R, cl.index, cl.columns, col="sue", hold_days=63)

H = 21
fwd = research.forward_returns(cl, H)
fwd_x = fwd.sub(fwd.mean(axis=1), axis=0)
dates = cl.index[::H]
Xr = {k: v.rank(axis=1, pct=True).loc[dates] for k, v in FEATS.items()}
rows = []
for d in dates:
    r = pd.DataFrame({k: Xr[k].loc[d] for k in FEATS})
    r["y"] = fwd_x.loc[d]; r["date"] = d
    rows.append(r.reset_index().rename(columns={"index": "ticker"}))
L = pd.concat(rows, ignore_index=True).dropna(subset=["y"])
cols = list(FEATS)
L = L[L[cols].notna().mean(axis=1) > 0.7]
L[cols] = L[cols].fillna(0.5)
print(f"{len(L):,} rows, {len(cols)} features, {L.date.nunique()} months\n", flush=True)

test_years = [y for y in sorted(L.date.dt.year.unique()) if y >= 2015]


def walk_forward(depth, leaf, mfeat, seed, shuffle=False):
    """Annual walk-forward. shuffle=True permutes y WITHIN each date, which
    destroys cross-sectional signal while preserving every other structure --
    the correct null for this design."""
    ics, imps = [], []
    for y in test_years:
        tr = L[L.date.dt.year < y]
        te = L[L.date.dt.year == y]
        if len(tr) < 2000 or te.empty:
            continue
        ytr = tr["y"].to_numpy()
        if shuffle:
            rng = np.random.default_rng(seed * 1000 + y)
            ytr = (tr.assign(y=ytr).groupby("date")["y"]
                   .transform(lambda s: rng.permutation(s.to_numpy())).to_numpy())
        m = RandomForestRegressor(n_estimators=100, max_depth=depth,
                                  min_samples_leaf=leaf, max_features=mfeat,
                                  n_jobs=-1, random_state=seed)
        m.fit(tr[cols].to_numpy(), ytr)
        p = te[["date"]].copy()
        p["pred"] = m.predict(te[cols].to_numpy()); p["y"] = te["y"].to_numpy()
        ics.append(p)
        imps.append(pd.Series(m.feature_importances_, index=cols, name=y))
    D = pd.concat(ics, ignore_index=True)
    ic = D.groupby("date").apply(
        lambda g: sst.spearmanr(g["pred"], g["y"]).statistic if len(g) > 5 else np.nan
    ).dropna()
    return float(ic.mean()), ic, pd.DataFrame(imps)


GRID = [(d, l, f) for d in (3, 6, 10) for l in (20, 40, 100) for f in (0.3, 0.6)]
print(f"1. GRID: {len(GRID)} configs, walk-forward each\n", flush=True)
res = []
for d, l, f in GRID:
    m, ic, imp = walk_forward(d, l, f, seed=0)
    res.append(dict(depth=d, leaf=l, mfeat=f, ic=m,
                    t=m / (ic.std(ddof=1) / np.sqrt(len(ic)))))
    print(f"  depth={d:<3} leaf={l:<4} mfeat={f}  IC {m:+.4f}", flush=True)
G = pd.DataFrame(res)
print(f"\n  IC across {len(G)} configs: mean {G.ic.mean():+.4f}  sd {G.ic.std():.4f}")
print(f"  min {G.ic.min():+.4f}  median {G.ic.median():+.4f}  max {G.ic.max():+.4f}")
print(f"  configs with IC > 0: {(G.ic > 0).sum()}/{len(G)}")
best = G.loc[G.ic.idxmax()]
print(f"  BEST: depth={int(best.depth)} leaf={int(best.leaf)} "
      f"mfeat={best.mfeat} IC {best.ic:+.4f}")

print("\n2. SEED STABILITY (best config, 6 seeds)\n", flush=True)
seeds = []
for s in range(6):
    m, _, _ = walk_forward(int(best.depth), int(best.leaf), best.mfeat, seed=s)
    seeds.append(m)
    print(f"  seed {s}: IC {m:+.4f}", flush=True)
seeds = np.array(seeds)
print(f"\n  mean {seeds.mean():+.4f}  sd {seeds.std():.4f}  "
      f"range [{seeds.min():+.4f}, {seeds.max():+.4f}]")
print(f"  seed sd as % of mean: {seeds.std()/abs(seeds.mean())*100:.0f}%")

print(f"\n3. NULL: same grid, y SHUFFLED within each date ({len(GRID)} configs)\n",
      flush=True)
null = []
for d, l, f in GRID:
    m, _, _ = walk_forward(d, l, f, seed=1, shuffle=True)
    null.append(m)
null = np.array(null)
print(f"  null IC across configs: mean {null.mean():+.4f}  sd {null.std():.4f}  "
      f"max {null.max():+.4f}")
print(f"\n  LIKE-FOR-LIKE: best-of-{len(GRID)} real = {G.ic.max():+.4f}   "
      f"best-of-{len(GRID)} null = {null.max():+.4f}")
print(f"  real best exceeds null best by {G.ic.max()-null.max():+.4f}")

print("\n4. FEATURE IMPORTANCE STABILITY (best config)\n", flush=True)
_, _, imp = walk_forward(int(best.depth), int(best.leaf), best.mfeat, seed=0)
top = imp.mean().nlargest(8)
print("  top features by mean importance, and their rank each year:")
ranks = imp.rank(axis=1, ascending=False)
for f in top.index:
    rr = ranks[f].astype(int).tolist()
    print(f"    {f:<18} mean imp {top[f]:.4f}  yearly ranks {rr}")
print(f"\n  mean rank correlation between consecutive years: "
      f"{np.mean([ranks.iloc[i].corr(ranks.iloc[i+1], method='spearman') for i in range(len(ranks)-1)]):.3f}")
print("  (near 1 = the model uses the same features every year; near 0 = it reshuffles)")
G.to_csv("rf_stress.csv", index=False)
