"""Common-ownership features: the first NON-price-derived test in this project.

Three features x three horizons = 9 pre-specified tests, count fixed before
running. Everything is anchored to period_end + 45 days (the 13F reporting
deadline), never to the period end itself.

Sanity checks come FIRST. If the co-ownership peers turn out to be the same
names correlation already gives us, the whole exercise measures nothing new and
the IC table is not worth reading.
"""
import numpy as np
import pandas as pd
from scipy import stats as sst

from algo import coownership as co, data, relational, research

pd.set_option("display.width", 220)

cmap = pd.read_parquet("data/thirteenf/cusip_map.parquet")
hold = co.load_holdings()
print(f"holdings: {len(hold):,} filer-positions across "
      f"{hold['period'].nunique()} quarters "
      f"({sorted(hold['period'].unique())[0]} .. {sorted(hold['period'].unique())[-1]})")

uni = pd.read_parquet("data/collection_universe.parquet")
syms = [t for t in uni["ticker"] if t in set(cmap["ticker"])]
px = data.load_panel(syms, start="2013-01-01", end="2026-09-01", field="close",
                     refresh=False)
px = px.dropna(axis=1, thresh=int(0.9 * len(px))).ffill(limit=5)
sectors = uni.set_index("ticker")["sector"].reindex(px.columns)
print(f"prices: {px.shape[1]} names, {len(px)} bars, "
      f"{px.index[0].date()}..{px.index[-1].date()}\n")

print("building connectedness panels (active managers only, 5-100 positions)...",
      flush=True)
peers, central = co.build_panels(px, cmap, hold)
cov = peers.notna().mean().mean()
print(f"  co-ownership peer assigned for {cov*100:.0f}% of name-days\n")

# ---- SANITY: is this actually different from correlation? -------------------
print("SANITY CHECK -- does co-ownership find DIFFERENT peers than correlation?\n")
corr_peers = pd.DataFrame(np.nan, index=px.index, columns=px.columns, dtype=object)
for sec, grp in sectors.groupby(sectors):
    cols = [c for c in grp.index if c in px.columns]
    if len(cols) >= 3:
        corr_peers[cols] = relational.peer_map(px[cols], 252, 63)

both = peers.notna() & corr_peers.notna()
agree = (peers == corr_peers)[both].to_numpy()
agree = agree[~pd.isna(agree)]
print(f"  co-ownership peer == correlation peer on {agree.mean()*100:.1f}% of name-days")
print("  (high agreement would mean this is correlation wearing a hat)")

last = peers.dropna(how="all").iloc[-1].dropna()
lastc = corr_peers.dropna(how="all").iloc[-1].dropna()
shared = [t for t in last.index if t in lastc.index][:10]
print("\n  ticker   co-own peer   correlation peer   same-sector?")
for t in shared:
    same = "yes" if sectors.get(t) == sectors.get(last[t]) else "NO"
    print(f"  {t:<8} {str(last[t]):<13} {str(lastc[t]):<18} {same}")

rets = px.pct_change()
cop = [rets[k].tail(252).corr(rets[v].tail(252))
       for k, v in last.items() if v in rets.columns]
print(f"\n  median return-correlation to the CO-OWNERSHIP peer: {np.nanmedian(cop):.3f}")
print("  (if this is as high as the correlation-peer figure ~0.60, the two "
      "measures are picking the same links)")

# ---- the features -----------------------------------------------------------
def peer_align(vals, peers):
    cols = list(vals.columns)
    idx = {c: j for j, c in enumerate(cols)}
    v, pv = vals.to_numpy(), peers.to_numpy()
    out = np.full(v.shape, np.nan)
    for t in range(v.shape[0]):
        for i in range(v.shape[1]):
            p = pv[t, i]
            if isinstance(p, str) and p in idx:
                out[t, i] = v[t, idx[p]]
    return pd.DataFrame(out, index=vals.index, columns=cols)

lp = np.log(px)
spread = lp - peer_align(lp, peers)
mu = spread.rolling(126, min_periods=63).mean()
sd = spread.rolling(126, min_periods=63).std()

FEATS = {
    # Contagion then reversal: a co-owned peer's recent move is non-fundamental
    # pressure on this name, so NEGATE it (peer up -> this name expected down).
    "coown_peer_ret_21": -peer_align(px.pct_change(21), peers),
    "coown_peer_ret_63": -peer_align(px.pct_change(63), peers),
    # Highly connected names carry more exposure to other people's redemptions.
    "coown_centrality":  -central,
    "coown_spread_z":    -((spread - mu) / sd.replace(0, np.nan)),
}
HZ = (1, 5, 21)
K = len(FEATS) * len(HZ)
bar = sst.norm.ppf(1 - 0.05 / (2 * K))
print(f"\n{'='*76}\n{K} pre-specified tests -> Bonferroni bar |t| > {bar:.2f}\n{'='*76}\n")
print(f"{'feature':<20} {'h':>3} {'IC':>9} {'t':>8} {'n':>7}")
rows = []
for name, f in FEATS.items():
    for hz in HZ:
        ic = research.cross_sectional_ic(f, research.forward_returns(px, hz)).dropna()
        if len(ic) < 100:
            continue
        m = float(ic.mean())
        t = (m / (ic.std(ddof=1) / np.sqrt(len(ic)))) / np.sqrt(hz)
        rows.append((name, hz, m, t))
        print(f"{name:<20} {hz:>3} {m:>9.4f} {t:>8.2f} {len(ic):>7}")

df = pd.DataFrame(rows, columns=["feature", "h", "ic", "t"])
surv = df[df.t.abs() > bar]
print(f"\nsurvivors: {len(surv)}")
for _, r in surv.iterrows():
    print(f"   {r.feature:<20} h={int(r.h):<3} IC {r.ic:+.4f}  t {r.t:+.2f}")
print(f"\nlargest |t|: {df.t.abs().max():.2f}   largest |IC|: {df.ic.abs().max():.4f}")
print(f"IC required to break even (large cap, monthly rebal): ~0.036")
df.to_csv("coown_test.csv", index=False)


# ---------------------------------------------------------------------------
# VARIANT: top-k connected peers instead of the single argmax.
#
# Motivation is estimator variance, not a search for significance. For a mega-cap
# the active-manager slice is thin, so argmax is decided by whichever single
# concentrated fund happens to hold both names -- a very noisy read on a network
# edge. Averaging the k strongest edges, weighted by connectedness, is the
# standard construction and is strictly better-conditioned.
#
# This adds 6 tests. Total for the family is therefore 18, and the bar below is
# recomputed on 18, not on 6.
# ---------------------------------------------------------------------------
print(f"\n\n{'='*76}\nVARIANT: top-k connected peers (k=10), connectedness-weighted\n{'='*76}\n")

cm = cmap.dropna(subset=["cusip"]).drop_duplicates("ticker")
cm = cm[cm["ticker"].isin(px.columns)]
c2t = dict(zip(cm["cusip"], cm["ticker"]))
cusips = list(cm["cusip"])
periods = sorted(hold["period"].unique())

K_PEERS = 10
r21, r63 = px.pct_change(21), px.pct_change(63)
agg21 = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
agg63 = pd.DataFrame(np.nan, index=px.index, columns=px.columns)

for k, p in enumerate(periods):
    start = co.period_end(p) + co.REPORTING_LAG
    end = (co.period_end(periods[k + 1]) + co.REPORTING_LAG
           if k + 1 < len(periods) else px.index[-1] + pd.Timedelta(days=1))
    mask = (px.index >= start) & (px.index < end)
    if not mask.any():
        continue
    c = co.connectedness(hold[hold["period"] == p], cusips)
    if c.to_numpy().sum() == 0:
        continue
    c = c.rename(index=c2t, columns=c2t)
    c = c.loc[[i for i in c.index if i in px.columns],
              [j for j in c.columns if j in px.columns]]
    # keep only the k strongest edges per row, renormalise to weights
    arr = c.to_numpy(copy=True)
    if arr.shape[1] > K_PEERS:
        cut = np.partition(arr, -K_PEERS, axis=1)[:, -K_PEERS][:, None]
        arr[arr < cut] = 0.0
    rs = arr.sum(axis=1, keepdims=True)
    w = np.divide(arr, rs, out=np.zeros_like(arr), where=rs > 0)
    W = pd.DataFrame(w, index=c.index, columns=c.columns)
    for src, dst in ((r21, agg21), (r63, agg63)):
        blk = src.loc[mask, W.columns].fillna(0.0)
        dst.loc[mask, W.index] = blk.to_numpy() @ W.to_numpy().T

FEATS2 = {
    "coown_topk_ret_21": -agg21,
    "coown_topk_ret_63": -agg63,
}
rows2 = []
print(f"{'feature':<20} {'h':>3} {'IC':>9} {'t':>8} {'n':>7}")
for name, f in FEATS2.items():
    for hz in HZ:
        ic = research.cross_sectional_ic(f, research.forward_returns(px, hz)).dropna()
        if len(ic) < 100:
            continue
        m = float(ic.mean())
        t = (m / (ic.std(ddof=1) / np.sqrt(len(ic)))) / np.sqrt(hz)
        rows2.append((name, hz, m, t))
        print(f"{name:<20} {hz:>3} {m:>9.4f} {t:>8.2f} {len(ic):>7}")

allr = pd.concat([df, pd.DataFrame(rows2, columns=["feature", "h", "ic", "t"])])
KT = len(allr)
bart = sst.norm.ppf(1 - 0.05 / (2 * KT))
print(f"\n{'-'*76}")
print(f"FAMILY TOTAL: {KT} tests -> Bonferroni bar |t| > {bart:.2f}")
print(f"survivors: {(allr.t.abs() > bart).sum()}")
print(f"largest |t|: {allr.t.abs().max():.2f}    "
      f"expected max under the null: ~{sst.norm.ppf(1 - 1/(2*KT)):.2f}")
print(f"largest |IC|: {allr.ic.abs().max():.4f}   required to break even: ~0.036")
