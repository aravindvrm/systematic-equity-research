"""Stress the two survivors before believing them."""
import numpy as np, pandas as pd
from algo import form4, insider_state, universe as U, data, research

f4 = form4.load(); tab = U.collection_table()
tk = [t for t in tab.ticker if t not in U.DIVERSIFIERS]
px = data.load_panel(tk, start="2018-01-01", end="2026-09-01")
px = px[[c for c in px.columns if px[c].notna().mean() > 0.95]].ffill().dropna()
F = insider_state.build(f4, px, window_days=126)
A, B = F["ins_buy_share"], F["ins_buyer_ratio"]

print("1. ARE THE TWO SURVIVORS ONE SIGNAL OR TWO?")
print(f"   cross-sectional rank correlation: "
      f"{A.rank(axis=1).corrwith(B.rank(axis=1), axis=1).mean():.3f}")
print("   -> effectively the same feature; count as ONE discovery, not two.\n")

def dyn_ic(F, h):
    d = F.sub(F.mean(axis=0), axis=1)
    return research.cross_sectional_ic(d, research.forward_returns(px, h))

print("2. SHUFFLED CONTROL (cross-section permuted each day -> signal destroyed)")
rng = np.random.default_rng(0)
for lab, Fx in [("ins_buy_share", A)]:
    real = dyn_ic(Fx, 21)
    sh = Fx.apply(lambda r: pd.Series(rng.permutation(r.values), index=r.index), axis=1)
    shuf = dyn_ic(sh, 21)
    print(f"   {lab}: real IC {real.mean():+.4f} (t {research.ic_stats(real,21)['ic_t']:+.2f})"
          f"   shuffled {shuf.mean():+.4f} (t {research.ic_stats(shuf,21)['ic_t']:+.2f})")

print("\n3. SPLIT STABILITY (does it work in BOTH halves?)")
mid = len(px)//2
print(f"   {'feature':<18} {'1st half IC':>12} {'2nd half IC':>12} {'retention':>10}")
for lab, Fx in [("ins_buy_share", A), ("ins_buyer_ratio", B)]:
    ic = dyn_ic(Fx, 21)
    i1, i2 = ic[:ic.index[mid] if mid < len(ic) else ic.index[-1]], ic[ic.index[mid] if mid < len(ic) else ic.index[-1]:]
    ret = i2.mean()/i1.mean()*100 if i1.mean() else np.nan
    print(f"   {lab:<18} {i1.mean():>12.4f} {i2.mean():>12.4f} {ret:>9.0f}%")

print("\n4. IC BY YEAR (is it one regime?)")
ic = dyn_ic(A, 21)
by = ic.groupby(ic.index.year).mean()
for y, v in by.items():
    bar = "#"*max(0,int(abs(v)*600)); sign = "" if v>=0 else "  (negative)"
    print(f"   {y}  {v:+.4f}  {bar}{sign}")
print(f"\n   positive years: {(by>0).sum()}/{len(by)}")

print("\n5. WHAT IS THE STATIC COMPONENT IT WAS MASKED BY?")
avg = A.mean(axis=0).sort_values()
print(f"   names with persistently LOW buy-share: {', '.join(avg.index[:5])}")
print(f"   names with persistently HIGH buy-share: {', '.join(avg.index[-5:])}")
fwd_tot = (px.iloc[-1]/px.iloc[0]-1)
print(f"   corr(avg buy-share, total return over sample): "
      f"{avg.corr(fwd_tot.reindex(avg.index)):+.3f}")
print("   -> a NEGATIVE correlation here means the persistent insider-buying names")
print("      underperformed, which is what masked the time-varying signal.")
