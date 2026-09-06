"""How many INDEPENDENT signals do we actually have, and what do they combine to?

The composite in oos_backtest.py underperformed, and I never diagnosed why. The
hypothesis: the seven surviving features are not seven signals. mom_252,
rel_strength_252 and resid_mom_252 are near-duplicates; reversal_5 and mom_21 are
sign mirrors. If the effective count is 2-3 rather than 7, the combination was
never going to reach the target and no weighting scheme would have saved it.

THE ARITHMETIC
--------------
For k signals with individual IC c and average pairwise IC-correlation rho, the
combined IC of an equal-weighted composite is approximately

    IC_combined = c * k / sqrt(k + k*(k-1)*rho)
                = c * sqrt(k) / sqrt(1 + (k-1)*rho)

At rho=0 this is c*sqrt(k) -- the full diversification benefit. At rho=1 it is
just c: combining identical signals buys nothing. The effective independent
count is k_eff = k / (1 + (k-1)*rho).

Correlations are measured between the daily IC SERIES, not between the raw
feature values. Two features can be highly correlated in level while their
predictive errors are not, and it is the errors that determine whether combining
them diversifies.
"""
import numpy as np
import pandas as pd

from algo import data, features, features2 as f2, research

pd.set_option("display.width", 220)

uni = pd.read_parquet("data/collection_universe.parquet")
sym = [c for c in uni.columns if c.lower() in ("symbol", "ticker")][0]
c = data.load_panel(uni[sym].tolist(), start="2005-01-01", end="2026-09-01",
                    field="close", refresh=False)
c = c.dropna(axis=1, thresh=int(0.9 * len(c))).ffill(limit=5)
P = {f: data.load_panel(uni[sym].tolist(), start="2005-01-01", end="2026-09-01",
                        field=f, refresh=False)
     .reindex(index=c.index, columns=c.columns).ffill(limit=5)
     for f in ("open", "high", "low", "volume")}
o, h, l, v = P["open"], P["high"], P["low"], P["volume"]
print(f"{c.shape[1]} names, {len(c)} bars\n")

# The eight cells that cleared Bonferroni on the long sample, at h=1.
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

fwd = research.forward_returns(c, 1)
ics = {}
for name, f in FEATS.items():
    ics[name] = research.cross_sectional_ic(f, fwd)
IC = pd.DataFrame(ics).dropna()
print(f"daily IC series aligned on {len(IC)} days\n")

means = IC.mean()
# Orient every signal so its IC is positive -- a signal that predicts with a
# negative sign is still a signal, and leaving it inverted would show up as
# spurious negative correlation with the others.
signs = np.sign(means)
ICs = IC.mul(signs, axis=1)

print("INDIVIDUAL IC (sign-oriented)")
for n in ICs.columns:
    t = ICs[n].mean() / (ICs[n].std(ddof=1) / np.sqrt(len(ICs)))
    print(f"  {n:<18} IC {ICs[n].mean():+.4f}   t {t:+.2f}")

R = ICs.corr()
print("\n\nCORRELATION BETWEEN DAILY IC SERIES")
print("(this, not feature-level correlation, determines diversification)\n")
print(R.round(2).to_string())

k = len(ICs.columns)
off = R.to_numpy()[np.triu_indices(k, 1)]
rho = float(off.mean())
k_eff = k / (1 + (k - 1) * rho)
cbar = float(ICs.mean().mean())
combined = cbar * np.sqrt(k) / np.sqrt(1 + (k - 1) * rho)

print(f"\n\n{'='*72}")
print(f"  signals tried              k       = {k}")
print(f"  mean pairwise IC corr      rho     = {rho:.3f}")
print(f"  EFFECTIVE independent      k_eff   = {k_eff:.2f}")
print(f"  mean individual IC         c       = {cbar:.4f}")
print(f"  predicted combined IC              = {combined:.4f}")
print(f"  target (break-even, monthly)       = 0.0360")
print(f"{'='*72}")

# Empirical check: does an actual equal-weighted composite match the prediction?
def z(f):
    zz = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1).replace(0, np.nan), axis=0)
    return zz.clip(-3, 3)

comp = sum(z(FEATS[n]) * signs[n] for n in FEATS) / k
ic_c = research.cross_sectional_ic(comp, fwd).dropna()
t_c = ic_c.mean() / (ic_c.std(ddof=1) / np.sqrt(len(ic_c)))
print(f"\n  ACTUAL equal-weight composite IC   = {ic_c.mean():.4f}  (t {t_c:+.2f})")
print(f"  predicted was                      = {combined:.4f}")

print("\n\nHOW MANY MORE SIGNALS WOULD IT TAKE?")
print("(same individual IC and same average correlation, adding independent-ish signals)\n")
print(f"{'k':>5} {'k_eff':>8} {'combined IC':>13}")
for kk in (8, 12, 20, 30, 50, 100):
    ke = kk / (1 + (kk - 1) * rho)
    print(f"{kk:>5} {ke:>8.2f} {cbar*np.sqrt(kk)/np.sqrt(1+(kk-1)*rho):>13.4f}")
print(f"\n  asymptote as k -> infinity: {cbar/np.sqrt(rho):.4f}" if rho > 0 else "")
