"""Two tests the market-wide work could not reach.

A. PAIRWISE RELATIONAL FEATURES -- a name measured against its closest peer,
   rather than against the 30-name basket. Never tested; features2.py only ever
   compared a name to the aggregate.

B. PER-NAME CONDITIONING -- does a feature's IC differ between names that are in
   their own high state and names in their own low state, measured on the SAME
   DAY? This is the version of the regime question that has breadth: the unit of
   observation is a day (~3,500 of them), not a market-wide episode (7-12).

Multiple testing is counted across BOTH parts and reported at the bottom. The
difference series in part B is paired by construction (both states measured on
the same day), so its t-stat needs no cross-sectional correlation adjustment --
the common market move is differenced out.
"""
import json
import numpy as np
import pandas as pd

from algo import data, features, pername, relational, research

pd.set_option("display.width", 220)

TICKERS = json.load(open("data/trading_universe.json"))
START, END = "2010-01-01", "2026-09-01"

px = data.load_panel(TICKERS, start=START, end=END, refresh=False)
px = px.dropna(axis=1, thresh=int(0.9 * len(px))).ffill(limit=5)
print(f"universe: {px.shape[1]} names, {len(px)} bars, "
      f"{px.index[0].date()} .. {px.index[-1].date()}\n")

FEATS = {
    "mom_21":      lambda p: features.momentum(p, 21),
    "mom_63":      lambda p: features.momentum(p, 63),
    "mom_252":     lambda p: features.momentum(p, 252),
    "reversal_5":  lambda p: features.reversal(p, 5),
    "reversal_21": lambda p: features.reversal(p, 21),
    "low_vol_60":  lambda p: features.low_vol(p, 60),
}

results = []   # (part, label, t, extra) -- collected for one honest count at the end


def tstat(s: pd.Series, horizon: int = 1) -> tuple[float, float, int]:
    """Mean, overlap-deflated t, n for a series of daily IC observations."""
    s = s.dropna()
    if len(s) < 60:
        return np.nan, np.nan, len(s)
    t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
    return float(s.mean()), float(t / np.sqrt(horizon)), len(s)


# ---------------------------------------------------------------- part A -----
print("=" * 96)
print("A. PAIRWISE RELATIONAL FEATURES  (name vs its closest trailing peer)")
print("=" * 96)

peers = relational.peer_map(px, window=252, refit_every=63)
cov = peers.notna().mean(axis=1)
print(f"peer map: assigned for {cov.mean()*100:.0f}% of name-days from "
      f"{peers.notna().any(axis=1).idxmax().date()}\n")

# What do the pairs actually look like? If these are nonsense the rest is moot.
last = peers.dropna(how="all").iloc[-1].dropna()
print("current peers (most-correlated over trailing 252d):")
print("  " + ",  ".join(f"{k}->{v}" for k, v in list(last.items())[:12]) + ",  ...")
rets = px.pct_change()
pair_corr = [rets[k].tail(252).corr(rets[v].tail(252)) for k, v in last.items()]
basket_corr = [rets[k].tail(252).corr(rets.mean(axis=1).tail(252)) for k in last.index]
print(f"\n  median corr to PEER   {np.nanmedian(pair_corr):.3f}")
print(f"  median corr to BASKET {np.nanmedian(basket_corr):.3f}   "
      "<- if these are close, pairs add nothing over what features2 already tested\n")

print(f"{'feature':<16} {'h':>3} {'IC':>9} {'t':>8} {'n':>7}")
for name, fn in relational.REGISTRY.items():
    f = fn(px, peers)
    for h in (1, 5, 21):
        fwd = research.forward_returns(px, h)
        ic = research.cross_sectional_ic(f, fwd)
        m, t, n = tstat(ic, h)
        print(f"{name:<16} {h:>3} {m:>9.4f} {t:>8.2f} {n:>7}")
        results.append(("A", f"{name}@h{h}", t))

# ---------------------------------------------------------------- part B -----
print("\n" + "=" * 96)
print("B. PER-NAME CONDITIONING  (IC among names in own-high state vs own-low, same day)")
print("=" * 96)

vol_panel = data.load_panel(TICKERS, start=START, end=END, field="volume",
                            refresh=False).reindex(columns=px.columns)
states = {k: fn(px) for k, fn in pername.REGISTRY.items()}
states["own_volume"] = pername.own_volume_state(vol_panel)

for k, s in states.items():
    share = s.mean(axis=1).dropna()
    print(f"  {k:<14} names in HIGH state per day: mean {share.mean()*100:4.1f}%  "
          f"min {share.min()*100:4.1f}%  max {share.max()*100:4.1f}%")

H = 5
fwd = research.forward_returns(px, H)
print(f"\nhorizon h={H}, IC computed separately within each state each day "
      f"(min 5 names per side)\n")
print(f"{'conditioner':<14} {'feature':<13} {'IC|high':>9} {'IC|low':>9} "
      f"{'diff':>9} {'t(diff)':>9} {'n days':>8}")

for cname, st in states.items():
    for fname, fn in FEATS.items():
        f = fn(px)
        hi = research.cross_sectional_ic(f.where(st == 1.0), fwd)
        lo = research.cross_sectional_ic(f.where(st == 0.0), fwd)
        both = pd.concat([hi.rename("hi"), lo.rename("lo")], axis=1).dropna()
        if len(both) < 60:
            continue
        d = both["hi"] - both["lo"]
        m, t, n = tstat(d, H)
        print(f"{cname:<14} {fname:<13} {both['hi'].mean():>9.4f} "
              f"{both['lo'].mean():>9.4f} {m:>9.4f} {t:>9.2f} {n:>8}")
        results.append(("B", f"{cname}/{fname}", t))

# ------------------------------------------------------------------ tally ----
print("\n" + "=" * 96)
ts = np.array([r[2] for r in results if np.isfinite(r[2])])
k = len(ts)
from scipy import stats as sst
bar = sst.norm.ppf(1 - 0.05 / (2 * k))
surv = [r for r in results if np.isfinite(r[2]) and abs(r[2]) > bar]
print(f"{k} tests run  ->  Bonferroni bar |t| > {bar:.2f} at family-wise 5%")
print(f"largest |t| observed: {np.abs(ts).max():.2f} "
      f"({[r[1] for r in results if abs(r[2]) == np.abs(ts).max()][0]})")
print(f"survivors: {len(surv)}")
for r in surv:
    print(f"   {r[0]}  {r[1]:<28} t={r[2]:+.2f}")
if not surv:
    # Under the null the max of k roughly-independent t's lands near this.
    exp_max = sst.norm.ppf(1 - 1 / (2 * k))
    print(f"\nexpected max |t| under the null with {k} tests: ~{exp_max:.2f}")
    print("observed max is " + ("BELOW" if np.abs(ts).max() < exp_max else "above")
          + " that -- i.e. " + ("less" if np.abs(ts).max() < exp_max else "more")
          + " signal than pure noise would have produced.")
