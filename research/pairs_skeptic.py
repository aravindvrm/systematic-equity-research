"""Four ways pair_spread_z could be fake. Run them before believing anything.

1. Is 0.6 a cherry-picked gate? -> sweep it. A real mechanism gives a MONOTONE
   response; a lucky threshold gives a spike.
2. Is it stable? -> split the sample in half.
3. Is it just reversal in disguise? -> orthogonalize against own 21d reversal
   cross-sectionally, and re-measure the residual signal.
4. Does it survive being sector-neutral? -> the pair IS the sector bet if the
   whole sector moved; demean the feature within sector and re-measure.
"""
import numpy as np
import pandas as pd
from scipy import stats as sst

from algo import data, relational, research

pd.set_option("display.width", 220)

uni = pd.read_parquet("data/collection_universe.parquet")
sec_col = [c for c in uni.columns if "sector" in c.lower()][0]
sym_col = [c for c in uni.columns if c.lower() in ("symbol", "ticker")][0]
px = data.load_panel(uni[sym_col].tolist(), start="2005-01-01", end="2026-09-01",
                     refresh=False)
px = px.dropna(axis=1, thresh=int(0.9 * len(px))).ffill(limit=5)
sectors = uni.set_index(sym_col)[sec_col].reindex(px.columns)
rets = px.pct_change()

peers = pd.DataFrame(np.nan, index=px.index, columns=px.columns, dtype=object)
for sec, grp in sectors.groupby(sectors):
    cols = [c for c in grp.index if c in px.columns]
    if len(cols) >= 3:
        peers[cols] = relational.peer_map(px[cols], 252, 63)

f_raw = relational.pair_spread_z(px, peers)
roll_pc = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for sym in px.columns:
    pc = peers[sym]
    for p, block in pc.groupby(pc):
        if isinstance(p, str):
            roll_pc.loc[block.index, sym] = (
                rets[sym].rolling(252, min_periods=126).corr(rets[p]).reindex(block.index))

H = 21
fwd = research.forward_returns(px, H)


def t_of(f, fwd=fwd, h=H):
    ic = research.cross_sectional_ic(f, fwd).dropna()
    if len(ic) < 60:
        return np.nan, np.nan, 0
    m = ic.mean()
    return m, (m / (ic.std(ddof=1) / np.sqrt(len(ic)))) / np.sqrt(h), len(ic)


print("1. GATE SWEEP  (is 0.6 special, or is the response monotone?)\n")
print(f"{'min pair corr':>14} {'% name-days':>12} {'IC':>9} {'t':>8}")
for g in [-1.0, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    gate = roll_pc > g
    m, t, n = t_of(f_raw.where(gate))
    print(f"{g:>14.1f} {gate.to_numpy().mean()*100:>11.1f}% {m:>9.4f} {t:>8.2f}")

print("\n\n2. SPLIT-HALF STABILITY  (gate 0.6)\n")
gate = roll_pc > 0.6
f = f_raw.where(gate)
ic = research.cross_sectional_ic(f, fwd).dropna()
mid = len(ic) // 2
for lbl, s in [("full", ic), ("1st half", ic.iloc[:mid]), ("2nd half", ic.iloc[mid:])]:
    t = (s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) / np.sqrt(H)
    print(f"  {lbl:<10} {s.index[0].date()}..{s.index[-1].date()}  "
          f"IC {s.mean():+.4f}  t {t:+.2f}  n {len(s)}")

print("\n\n3. IS IT JUST REVERSAL?  (orthogonalize vs own 21d return, per day)\n")
own_rev = -px.pct_change(21)
print(f"  mean cross-sectional rank corr(pair_spread_z, reversal_21) = "
      f"{f.rank(axis=1).corrwith(own_rev.rank(axis=1), axis=1).mean():.3f}")


def resid_vs(a, b):
    """Cross-sectionally regress ranks of a on ranks of b, keep the residual."""
    ar = a.rank(axis=1); br = b.rank(axis=1)
    valid = ar.notna() & br.notna()
    ar = ar.where(valid); br = br.where(valid)
    ac = ar.sub(ar.mean(axis=1), axis=0); bc = br.sub(br.mean(axis=1), axis=0)
    beta = (ac * bc).sum(axis=1) / (bc**2).sum(axis=1).replace(0, np.nan)
    return ac - bc.mul(beta, axis=0)


m, t, n = t_of(own_rev.where(gate))
print(f"  reversal_21 alone, same gated cells     IC {m:+.4f}  t {t:+.2f}")
m, t, n = t_of(resid_vs(f, own_rev))
print(f"  pair_spread_z ORTHOGONAL to reversal_21 IC {m:+.4f}  t {t:+.2f}")
m, t, n = t_of(resid_vs(own_rev.where(gate), f))
print(f"  reversal_21 ORTHOGONAL to pair_spread_z IC {m:+.4f}  t {t:+.2f}")

print("\n\n4. SECTOR-NEUTRAL  (demean the feature within sector each day)\n")
sn = f.copy()
for sec, grp in sectors.groupby(sectors):
    cols = [c for c in grp.index if c in sn.columns]
    if len(cols) >= 3:
        blk = sn[cols]
        sn[cols] = blk.sub(blk.mean(axis=1), axis=0)
m, t, n = t_of(sn)
print(f"  sector-neutral pair_spread_z            IC {m:+.4f}  t {t:+.2f}  n {n}")

print("\n\n5. HORIZON PROFILE (gated, sector-neutral) -- where does it live?\n")
print(f"{'h':>4} {'IC':>9} {'t':>8}")
for h in (1, 3, 5, 10, 21, 42, 63):
    m, t, _ = t_of(sn, research.forward_returns(px, h), h)
    print(f"{h:>4} {m:>9.4f} {t:>8.2f}")
