"""Two questions that decide whether sim_jaccard matters.

1. DECAY. Lazy Prices was published in 2020 on a 1995-2014 sample. Our data runs
   2006-2026. Split at 2015: the first half overlaps the paper's own sample, the
   second is out-of-sample AND mostly post-publication. If the effect is only in
   the first half, this is McLean-Pontiff decay and the signal is dead. If it is
   stable, it is real but small.

2. ORTHOGONALITY. Per orthogonality.py, what a new signal is worth depends almost
   entirely on its correlation to what we already hold. A weak signal uncorrelated
   with momentum beats a strong one that is momentum in disguise. Both must be
   measured AT THE SAME HORIZON -- comparing an h=1 composite to an h=126 signal
   would be meaningless.
"""
import numpy as np
import pandas as pd

from algo import data, features, features2 as f2, filings, research

pd.set_option("display.width", 200)

M = pd.read_parquet("data/edgar/measures.parquet")
uni = pd.read_parquet("data/collection_universe.parquet").dropna(subset=["cik"])
uni["cik"] = uni["cik"].astype(int).astype(str)

c = data.load_panel(sorted(M.ticker.unique()), start="2005-01-01", end="2026-09-01",
                    field="close", refresh=False)
c = c.dropna(axis=1, thresh=int(0.9 * len(c))).ffill(limit=5)
M = M[M.ticker.isin(c.columns)]
tick = dict(zip(M.cik.astype(str), M.ticker))
P = {f: data.load_panel(sorted(M.ticker.unique()), start="2005-01-01",
                        end="2026-09-01", field=f, refresh=False)
     .reindex(index=c.index, columns=c.columns).ffill(limit=5)
     for f in ("open", "high", "low", "volume")}
o, h_, l, v = P["open"], P["high"], P["low"], P["volume"]
print(f"{c.shape[1]} names, {len(c)} bars\n")

jac = filings.to_daily(M, c.index, tick, col="sim_jaccard", lag_days=1, hold_days=126)

print("=" * 78)
print("1. DECAY -- paper's sample era vs post-publication")
print("=" * 78)
print(f"\n{'period':<18} {'h':>4} {'IC':>9} {'t':>8} {'n':>7}")
for lbl, a, b in [("2006-2014 (in-samp)", "2006-01-01", "2014-12-31"),
                  ("2015-2026 (out)", "2015-01-01", "2026-09-01")]:
    m = (c.index >= a) & (c.index <= b)
    for hz in (21, 63, 126):
        ic = research.cross_sectional_ic(jac[m], research.forward_returns(c, hz)[m]).dropna()
        if len(ic) < 100:
            continue
        mu = float(ic.mean())
        t = (mu / (ic.std(ddof=1) / np.sqrt(len(ic)))) / np.sqrt(hz)
        print(f"{lbl:<18} {hz:>4} {mu:>9.4f} {t:>8.2f} {len(ic):>7}")

print("\n" + "=" * 78)
print("2. ORTHOGONALITY to the price composite (same horizon, as required)")
print("=" * 78)

PRICE = {
    "reversal_5":       features.reversal(c, 5),
    "mom_12_1":         features.momentum_12_1(c),
    "vol_shock":        f2.volume_shock(v),
    "close_in_range":   f2.close_position_in_range(h_, l, c),
    "resid_mom_252":    f2.residual_momentum(c, 252),
    "mom_252":          features.momentum(c, 252),
    "overnight_vs_day": f2.intraday_vs_overnight(o, c),
}
def z(f):
    zz = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1).replace(0, np.nan), axis=0)
    return zz.clip(-3, 3)

print(f"\n{'h':>4} {'IC price':>10} {'IC jaccard':>12} {'corr(IC)':>10} "
      f"{'combined':>10} {'target':>8}")
for hz in (21, 63, 126):
    fwd = research.forward_returns(c, hz)
    icp_each = {n: research.cross_sectional_ic(f, fwd) for n, f in PRICE.items()}
    signs = {n: np.sign(s.mean()) for n, s in icp_each.items()}
    comp = sum(z(PRICE[n]) * signs[n] for n in PRICE) / len(PRICE)
    icp = research.cross_sectional_ic(comp, fwd)
    icj = research.cross_sectional_ic(jac, fwd)
    both = pd.concat([icp.rename("p"), icj.rename("j")], axis=1).dropna()
    r = float(both.p.corr(both.j))
    a, b = float(both.p.mean()), float(both.j.mean())
    # Orthogonal signals combine in quadrature under optimal weighting; the
    # general two-signal case discounts by their correlation.
    comb = np.sqrt(max(a**2 + b**2 - 2*r*a*b, 0.0) / max(1 - r**2, 1e-9))
    print(f"{hz:>4} {a:>10.4f} {b:>12.4f} {r:>10.3f} {comb:>10.4f} {0.036:>8.3f}")

print("\ncorr(IC) near zero means the filing signal is genuinely additive;")
print("near 1 would mean it is momentum wearing a different name.")
