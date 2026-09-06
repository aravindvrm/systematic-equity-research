"""How often must the universe be re-selected?

Two costs pull against each other:
  - re-select too rarely -> correlation structure drifts, breadth decays
  - re-select too often  -> turnover costs, and you start fitting noise

Measure the first directly: select on a window, then track the selection's
effective bets forward without re-selecting. Where it decays to near-random,
that is the shelf life.
"""
import numpy as np, pandas as pd
from algo import universe as U, data

tab = U.collection_table(); m = tab.set_index("ticker")
px = data.load_panel(tab.ticker.tolist(), start="2018-01-01", end="2026-09-01")
keep = [c for c in px.columns if px[c].notna().mean() > 0.98]
pxk = px[keep].dropna()
R = pxk.pct_change().dropna()
CORE = ["TLT", "GLD", "DBC", "HYG", "EFA", "TIP"]
N, TRAIN = 30, 504          # 2y training window

def eff(r):
    lam = np.linalg.eigvalsh(r.corr().to_numpy()); lam = lam[lam > 0]
    return float(lam.sum() ** 2 / (lam ** 2).sum())

rng = np.random.default_rng(4)
HOLDS = [63, 126, 252, 504, 756]     # 3m, 6m, 1y, 2y, 3y
starts = range(TRAIN, len(R) - max(HOLDS), 126)
rows = []
for s in starts:
    tr = R.iloc[s - TRAIN:s]
    sel = U.trading_universe(pxk.iloc[s - TRAIN:s], n=N, must_include=CORE,
                             method="cluster", sector_floor=1, sectors=m.sector)
    base = eff(tr[sel])
    for h in HOLDS:
        fw = R.iloc[s:s + h]
        if len(fw) < h * 0.9:
            continue
        rnd = np.median([eff(fw[list(rng.choice(R.columns, N, replace=False))])
                         for _ in range(15)])
        rows.append(dict(start=R.index[s].date(), hold=h, train=base,
                         held=eff(fw[sel]), rand=rnd))

df = pd.DataFrame(rows)
print(f"{len(starts)} selection dates, 2-year training window\n")
print(f"{'hold':<10} {'train bets':>11} {'held bets':>10} {'random':>8} "
      f"{'edge':>7} {'retained':>9}")
for h in HOLDS:
    d = df[df.hold == h]
    if d.empty: continue
    edge0 = (df[df.hold == 63].held - df[df.hold == 63].rand).mean()
    edge = (d.held - d.rand).mean()
    lab = {63:"3 months",126:"6 months",252:"1 year",504:"2 years",756:"3 years"}[h]
    print(f"{lab:<10} {d.train.mean():>11.2f} {d.held.mean():>10.2f} "
          f"{d.rand.mean():>8.2f} {edge:>7.2f} {edge/edge0*100:>8.0f}%")

print("\n\nSELECTION TURNOVER — how many of 30 names change?\n")
sels = {}
for s in starts:
    sels[R.index[s].date()] = set(U.trading_universe(
        pxk.iloc[s-TRAIN:s], n=N, must_include=CORE, method="cluster",
        sector_floor=1, sectors=m.sector))
ks = sorted(sels)
print(f"{'gap':<12} {'names changed':>14} {'% turnover':>11}")
for gap, lab in [(1,"6 months"), (2,"1 year"), (4,"2 years")]:
    ch = [len(sels[ks[i+gap]] - sels[ks[i]]) for i in range(len(ks)-gap)]
    if ch:
        print(f"{lab:<12} {np.mean(ch):>14.1f} {np.mean(ch)/N*100:>10.0f}%")
