"""Pairwise relational features on a universe that actually HAS pairs.

The 30-name trading universe was selected by MINIMIZING correlation -- it was
optimized for effective bets. Asking it for pairs is a contradiction in terms,
and the first run confirmed it: median correlation to "closest peer" 0.338 vs
0.318 to the basket. There were no pairs to find.

Here peers are drawn from the 262-name collection universe and CONSTRAINED TO
THE SAME GICS SECTOR, which is where genuine relative-value structure lives.
Peer selection remains point-in-time (trailing 252d, refit quarterly).
"""
import numpy as np
import pandas as pd

from algo import data, relational, research
from scipy import stats as sst

pd.set_option("display.width", 220)

uni = pd.read_parquet("data/collection_universe.parquet")
sec_col = [c for c in uni.columns if "sector" in c.lower()][0]
sym_col = [c for c in uni.columns if c.lower() in ("symbol", "ticker")][0]
print(f"collection universe: {len(uni)} names across {uni[sec_col].nunique()} sectors")

px = data.load_panel(uni[sym_col].tolist(), start="2005-01-01", end="2026-09-01",
                     refresh=False)
px = px.dropna(axis=1, thresh=int(0.9 * len(px))).ffill(limit=5)
sectors = uni.set_index(sym_col)[sec_col].reindex(px.columns)
print(f"with usable history: {px.shape[1]} names, {len(px)} bars, "
      f"{px.index[0].date()} .. {px.index[-1].date()}\n")

# --- peer map, one sector at a time so pairs are economically related --------
peers = pd.DataFrame(np.nan, index=px.index, columns=px.columns, dtype=object)
for sec, grp in sectors.groupby(sectors):
    cols = [c for c in grp.index if c in px.columns]
    if len(cols) < 3:
        continue
    pm = relational.peer_map(px[cols], window=252, refit_every=63)
    peers[cols] = pm

rets = px.pct_change()
last = peers.dropna(how="all").iloc[-1].dropna()
pair_corr = np.array([rets[k].tail(252).corr(rets[v].tail(252)) for k, v in last.items()])
basket = rets.mean(axis=1)
basket_corr = np.array([rets[k].tail(252).corr(basket.tail(252)) for k in last.index])
print(f"median corr to SECTOR PEER  {np.nanmedian(pair_corr):.3f}")
print(f"median corr to BASKET       {np.nanmedian(basket_corr):.3f}")
print(f"  -> peers are {np.nanmedian(pair_corr) - np.nanmedian(basket_corr):+.3f} "
      "tighter than the basket. Above ~+0.15 there is real structure to trade.\n")
top = pd.Series(pair_corr, index=last.index).nlargest(10)
print("tightest pairs:  " + ",  ".join(f"{k}-{last[k]}({v:.2f})" for k, v in top.items()))

# --- the same three features, on structure that exists ----------------------
print(f"\n{'feature':<16} {'h':>3} {'IC':>9} {'t':>8} {'n':>7}")
ts = []
for name, fn in relational.REGISTRY.items():
    f = fn(px, peers)
    for h in (1, 5, 21):
        fwd = research.forward_returns(px, h)
        ic = research.cross_sectional_ic(f, fwd).dropna()
        m = ic.mean()
        t = (m / (ic.std(ddof=1) / np.sqrt(len(ic)))) / np.sqrt(h)
        print(f"{name:<16} {h:>3} {m:>9.4f} {t:>8.2f} {len(ic):>7}")
        ts.append((f"{name}@h{h}", t))

# --- restrict to the pairs that are actually tight --------------------------
# A feature that works only where the pair is real is a coherent claim; one that
# needs every pair to work is not. Trailing correlation gate, so still PIT.
print("\nrestricted to name-days whose trailing-252d pair correlation > 0.6")
roll_pc = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for i, sym in enumerate(px.columns):
    pc = peers[sym]
    for p, block in pc.groupby(pc):
        if not isinstance(p, str):
            continue
        c = rets[sym].rolling(252, min_periods=126).corr(rets[p])
        roll_pc.loc[block.index, sym] = c.reindex(block.index)
gate = roll_pc > 0.6
print(f"  gate keeps {gate.to_numpy().mean()*100:.1f}% of name-days\n")
print(f"{'feature':<16} {'h':>3} {'IC':>9} {'t':>8} {'n':>7}")
for name, fn in relational.REGISTRY.items():
    f = fn(px, peers).where(gate)
    for h in (1, 5, 21):
        fwd = research.forward_returns(px, h)
        ic = research.cross_sectional_ic(f, fwd).dropna()
        if len(ic) < 60:
            continue
        m = ic.mean()
        t = (m / (ic.std(ddof=1) / np.sqrt(len(ic)))) / np.sqrt(h)
        print(f"{name:<16} {h:>3} {m:>9.4f} {t:>8.2f} {len(ic):>7}")
        ts.append((f"gated {name}@h{h}", t))

k = len(ts)
bar = sst.norm.ppf(1 - 0.05 / (2 * k))
print(f"\n{k} tests -> Bonferroni bar |t| > {bar:.2f}")
surv = [x for x in ts if abs(x[1]) > bar]
print(f"survivors: {len(surv)}" + ("" if not surv else ""))
for s in surv:
    print(f"   {s[0]:<28} t={s[1]:+.2f}")
