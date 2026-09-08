"""Effective breadth from the eigenvalue spectrum, not from the mean correlation.

The published ceiling of 0.0223 uses the EQUICORRELATION approximation: every
pair of signals assumed to share one correlation rho, every signal assumed to
carry the same IC, and the combination assumed equal-weighted. This replaces a
k x k matrix with a single number. That is a simplification in three places,
and all three point the same way -- it understates what a good combination of
the same signals can reach.

This computes the real thing:

  * the full correlation matrix of the daily IC series
  * its eigenvalue spectrum, which is what actually governs how many
    independent bets are present
  * the participation ratio, a spectrum-based effective rank
  * the optimal-combination IC, sqrt(c' R^-1 c), against the equal-weight
    equicorrelation figure the paper reports

Reuses the feature set and IC construction from orthogonality.py verbatim.
"""
import numpy as np
import pandas as pd

from algo import data, features, features2 as f2, research

pd.set_option("display.width", 220)

CACHE = "results/signal_ic_series.parquet"   # keyed to the battery universe
import os
if os.path.exists(CACHE):
    ICs = pd.read_parquet(CACHE)
    print(f"IC series loaded from {CACHE}")
else:
    ICs = None

# Same universe as the strategy battery (reassess_all.py): measures.parquet
# filtered to 90% coverage. The earlier version of this analysis used
# collection_universe.parquet -- 211 names against the battery's 199 -- so the
# ceiling and the results were computed on different panels.
M = pd.read_parquet("data/edgar/measures.parquet")
tick = sorted(M.ticker.unique())
c = data.load_panel(tick, start="2005-01-01", end="2026-09-01",
                    field="close", refresh=False)
c = c.dropna(axis=1, thresh=int(0.9 * len(c))).ffill(limit=5)
P = {f: data.load_panel(tick, start="2005-01-01", end="2026-09-01",
                        field=f, refresh=False)
     .reindex(index=c.index, columns=c.columns).ffill(limit=5)
     for f in ("open", "high", "low", "volume")}
o, h, l, v = P["open"], P["high"], P["low"], P["volume"]
print(f"{c.shape[1]} names, {len(c)} bars", flush=True)

FEATS = {
    "reversal_5":       features.reversal(c, 5),
    "mom_12_1":         features.momentum_12_1(c),
    "vol_shock":        f2.volume_shock(v),
    "close_in_range":   f2.close_position_in_range(h, l, c),
    "resid_mom_252":    f2.residual_momentum(c, 252),
    "mom_252":          features.momentum(c, 252),
    "rel_strength_252": f2.relative_strength(c, 252),
    "overnight_vs_day": f2.intraday_vs_overnight(o, c),
}
if ICs is None:
    fwd = research.forward_returns(c, 1)
    IC = pd.DataFrame({n: research.cross_sectional_ic(f, fwd)
                       for n, f in FEATS.items()}).dropna()
    ICs = IC.mul(np.sign(IC.mean()), axis=1)      # orient every signal positive
    ICs.to_parquet(CACHE)
R = ICs.corr()
k = R.shape[0]
cvec = ICs.mean().to_numpy()

print(f"\ndaily IC series aligned on {len(ICs)} days\n")

# ---- per-signal IC and whether it survives being split in half.
# The paper's "best stable feature" figure came from a range quoted in an
# untracked note. Define stable and measure it: same sign in both halves, and
# the weaker half still positive.
half = len(ICs) // 2
A, B = ICs.iloc[:half], ICs.iloc[half:]
print("INDIVIDUAL IC, AND STABILITY ACROSS HALVES")
print(f"{'signal':<20}{'full':>9}{'first half':>12}{'second half':>13}{'t (full)':>10}   stable")
stable = []
for n in ICs.columns:
    f_, a_, b_ = ICs[n].mean(), A[n].mean(), B[n].mean()
    t = f_ / (ICs[n].std(ddof=1) / np.sqrt(len(ICs)))
    ok = (a_ > 0) and (b_ > 0)
    if ok:
        stable.append((n, f_, min(a_, b_)))
    print(f"{n:<20}{f_:>+9.4f}{a_:>+12.4f}{b_:>+13.4f}{t:>+10.2f}   {'yes' if ok else 'NO'}")
if stable:
    best_full = max(stable, key=lambda x: x[1])
    best_weak = max(stable, key=lambda x: x[2])
    print(f"\n  best STABLE signal, full-sample IC     {best_full[0]} = {best_full[1]:.4f}")
    print(f"  best STABLE signal, weaker-half IC     {best_weak[0]} = {best_weak[2]:.4f}")
    print(f"  -> a defensible 'best stable feature' range: "
          f"{min(best_full[1], best_weak[2]):.4f} to {best_full[1]:.4f}")

print("CORRELATION BETWEEN DAILY IC SERIES")
print(R.round(2).to_string())

off = R.to_numpy()[np.triu_indices(k, 1)]
rho = float(off.mean())
print(f"\nmean off-diagonal rho = {rho:.3f}   min {off.min():+.3f}   max {off.max():+.3f}")
print("\nmost and least correlated pairs")
pairs = [(R.index[i], R.columns[j], R.iloc[i, j])
         for i in range(k) for j in range(i + 1, k)]
for a, b, r in sorted(pairs, key=lambda x: -abs(x[2]))[:4]:
    print(f"   {a:<18}{b:<18}{r:+.3f}")
print("   ...")
for a, b, r in sorted(pairs, key=lambda x: abs(x[2]))[:3]:
    print(f"   {a:<18}{b:<18}{r:+.3f}")

# ---- spectrum
w = np.linalg.eigvalsh(R.to_numpy())[::-1]
print(f"\n\nEIGENVALUE SPECTRUM  (sum = k = {k}; equal spread would be 1.0 each)")
cum = np.cumsum(w) / k
for i, (ev, cu) in enumerate(zip(w, cum), 1):
    bar = "#" * int(round(ev * 18))
    print(f"  PC{i}  {ev:6.3f}   {cu*100:5.1f}% of variance   {bar}")

part_ratio = float(w.sum() ** 2 / (w ** 2).sum())
k_eff_equi = k / (1 + (k - 1) * rho)
print(f"\n  effective rank, participation ratio (sum L)^2/sum L^2 = {part_ratio:.2f}")
print(f"  effective count under equicorrelation k/(1+(k-1)rho)   = {k_eff_equi:.2f}")

# ---- a pair at rho = 1.000 is one signal counted twice, and it makes R singular
dupes = [(R.index[i], R.columns[j]) for i in range(k) for j in range(i + 1, k)
         if abs(R.iloc[i, j]) > 0.9995]
if dupes:
    drop = sorted({b for _, b in dupes})
    print(f"\n  EXACT DUPLICATES at |rho| > 0.9995: {dupes}")
    print(f"  -> dropping {drop}; the matrix is singular with them in, and the")
    print(f"     'eight signals' are really {k - len(drop)}.")
    keep = [n for n in R.index if n not in drop]
    Rd = ICs[keep].corr()
    cvd = ICs[keep].mean().to_numpy()
else:
    keep, Rd, cvd = list(R.index), R, cvec

kd = len(keep)
offd = Rd.to_numpy()[np.triu_indices(kd, 1)]
rhod = float(offd.mean())
wd = np.linalg.eigvalsh(Rd.to_numpy())[::-1]
pr_d = float(wd.sum() ** 2 / (wd ** 2).sum())
print(f"  de-duplicated: k = {kd}, rho = {rhod:.3f}, participation ratio = {pr_d:.2f}")

# ---- what a combination actually reaches
cbar = float(cvec.mean())
equal_equi = cbar * np.sqrt(k) / np.sqrt(1 + (k - 1) * rho)
asym_equi = cbar / np.sqrt(rho)

one = np.ones(kd)
cbar_d = float(cvd.mean())
equal_real = float(cbar_d * kd / np.sqrt(one @ Rd.to_numpy() @ one))  # equal weight, true R
Rinv = np.linalg.inv(Rd.to_numpy())
optimal = float(np.sqrt(cvd @ Rinv @ cvd))                            # optimal weights, true R

print(f"\n\nCOMBINED IC, same eight signals, four ways")
print(f"  mean individual IC  c                                  = {cbar:.4f}")
print(f"  best individual IC                                     = {cvec.max():.4f}")
print(f"  equal weight, EQUICORRELATION (what the paper reports)  = {equal_equi:.4f}")
print(f"  equal weight, TRUE correlation matrix                   = {equal_real:.4f}")
print(f"  OPTIMAL weights, true matrix  sqrt(c' R^-1 c)           = {optimal:.4f}")
print(f"  equicorrelation asymptote as k->inf  c/sqrt(rho)        = {asym_equi:.4f}")
print(f"  break-even required                                     = 0.0360")

R.to_csv("results/signal_ic_correlation.csv")
pd.DataFrame({"eigenvalue": w, "cum_var_share": cum}).to_csv(
    "results/signal_ic_spectrum.csv", index=False)
print("\nwritten: results/signal_ic_correlation.csv, results/signal_ic_spectrum.csv")
