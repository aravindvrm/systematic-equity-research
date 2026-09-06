"""Does hierarchical clustering really beat greedy? Multi-split validation.

A single split said clustering delivered 14.81 test effective bets against
greedy's 13.30, and IMPROVED out of sample (+3.19) where greedy decayed (-1.34).
That is exactly the shape of result that fooled me twice already this session, so
it gets the same treatment: multiple disjoint windows, plus a random baseline.
"""
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from algo import universe as U, data

tab = U.collection_table()
px = data.load_panel(tab.ticker.tolist(), start='2018-01-01', end='2026-09-01')
keep = [c for c in px.columns if px[c].notna().mean() > 0.98]
R = px[keep].dropna().pct_change().dropna()
N = 30

def eff(r):
    lam = np.linalg.eigvalsh(r.corr().to_numpy()); lam = lam[lam > 0]
    return float(lam.sum()**2/(lam**2).sum())

def greedy(tr, n=N):
    chosen, rem = [], list(tr.columns)
    while len(chosen) < n and rem:
        best, bv = rem[0], -np.inf
        for t in rem:
            v = eff(tr[chosen+[t]]) if chosen else tr[t].std()
            if v > bv: best, bv = t, v
        chosen.append(best); rem.remove(best)
    return sorted(chosen)

def cluster(tr, n=N):
    c = tr.corr().to_numpy()
    d = np.sqrt(np.clip(0.5*(1-c), 0, None)); np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method="average")
    lab = fcluster(Z, t=n, criterion="maxclust")
    return sorted(tr.columns[lab==k][tr[tr.columns[lab==k]].corr().mean().argmin()]
                  for k in np.unique(lab))

rng = np.random.default_rng(9)
rows = []
n = len(R)
for k, frac in enumerate([0.40, 0.50, 0.60, 0.70, 0.80], 1):
    cut, end = int(n*frac), min(int(n*(frac+0.20)), n)
    TR, TE = R.iloc[:cut], R.iloc[cut:end]
    if len(TE) < 200: continue
    g, c = greedy(TR), cluster(TR)
    rnd = [list(rng.choice(R.columns, N, replace=False)) for _ in range(30)]
    r_te = np.array([eff(TE[s]) for s in rnd])
    rows.append(dict(split=k, test=f"{TE.index[0].date()}..{TE.index[-1].date()}",
        greedy_tr=eff(TR[g]), greedy_te=eff(TE[g]),
        clust_tr=eff(TR[c]), clust_te=eff(TE[c]),
        rand_te=float(np.median(r_te)), overlap=len(set(g)&set(c))))

df = pd.DataFrame(rows)
print(f"{'split':<6} {'test window':<24} {'greedy TE':>10} {'clust TE':>9} "
      f"{'random':>8} {'winner':>9} {'overlap':>8}")
for _, r in df.iterrows():
    w = 'cluster' if r.clust_te > r.greedy_te else 'greedy'
    print(f"{r.split:<6} {r.test:<24} {r.greedy_te:>10.2f} {r.clust_te:>9.2f} "
          f"{r.rand_te:>8.2f} {w:>9} {int(r.overlap):>8}")
print()
print(f"  clustering beat greedy in {(df.clust_te>df.greedy_te).sum()}/{len(df)} splits")
print(f"  mean test bets  -- greedy {df.greedy_te.mean():.2f}  "
      f"clustering {df.clust_te.mean():.2f}  random {df.rand_te.mean():.2f}")
print(f"  mean decay      -- greedy {(df.greedy_te-df.greedy_tr).mean():+.2f}  "
      f"clustering {(df.clust_te-df.clust_tr).mean():+.2f}")
