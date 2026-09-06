"""Were we exhaustive about structural selection objectives? No -- we tested two.

Tested previously:
  1. maximize effective bets  -> held up out of sample (8/8 splits, 94th pctile)
  2. maximize Sharpe          -> no better than random

Untested until now. All are STRUCTURAL (never look at returns), so all are
legitimate candidates under the spec:
  3. minimum variance                -- classic, but concentrates in low-vol names
  4. maximum diversification ratio   -- Choueifaty: wtd-avg vol / portfolio vol
  5. minimum average pairwise corr   -- simpler than eigenvalues; is it equivalent?
  6. hierarchical clustering         -- different METHOD: cluster, take one each
  7. maximum eigenvalue entropy      -- spreads variance across ALL components

Each is fitted on TRAIN and scored on TEST, because a single in-sample number
misled us twice already.
"""
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from algo import universe as U, data, research

tab = U.collection_table()
px = data.load_panel(tab.ticker.tolist(), start='2018-01-01', end='2026-09-01')
keep = [c for c in px.columns if px[c].notna().mean() > 0.98]
pxk = px[keep].dropna()
R = pxk.pct_change().dropna()
mid = len(R) // 2
TR, TE = R.iloc[:mid], R.iloc[mid:]
N = 30
print(f"{R.shape[1]} names, train {TR.index[0].date()}..{TR.index[-1].date()}, "
      f"test {TE.index[0].date()}..{TE.index[-1].date()}\n")


def eff_bets(r):
    lam = np.linalg.eigvalsh(r.corr().to_numpy()); lam = lam[lam > 0]
    return float(lam.sum()**2 / (lam**2).sum())

def div_ratio(r):
    w = np.ones(r.shape[1]) / r.shape[1]
    return float((w @ r.std()) / np.sqrt(w @ r.cov().to_numpy() @ w))

def eig_entropy(r):
    lam = np.linalg.eigvalsh(r.corr().to_numpy()); lam = lam[lam > 1e-10]
    p = lam / lam.sum()
    return float(-(p * np.log(p)).sum())

def port_var(r):
    w = np.ones(r.shape[1]) / r.shape[1]
    return float(w @ r.cov().to_numpy() @ w)

def avg_corr(r):
    c = r.corr().to_numpy()
    return float(c[np.triu_indices(len(c), 1)].mean())


def greedy(train, score, n=N, maximize=True):
    chosen, rem = [], list(train.columns)
    while len(chosen) < n and rem:
        best, bv = rem[0], (-np.inf if maximize else np.inf)
        for t in rem:
            if len(chosen) < 1:
                v = -train[t].std() if not maximize else train[t].std()
            else:
                v = score(train[chosen + [t]])
            if (v > bv) if maximize else (v < bv):
                best, bv = t, v
        chosen.append(best); rem.remove(best)
    return sorted(chosen)


def cluster_select(train, n=N):
    """Cluster into n groups by correlation distance, take the least-correlated
    member of each. Structurally different from greedy: global, not sequential."""
    c = train.corr().to_numpy()
    d = np.sqrt(np.clip(0.5 * (1 - c), 0, None))
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method="average")
    lab = fcluster(Z, t=n, criterion="maxclust")
    out = []
    for k in np.unique(lab):
        members = train.columns[lab == k]
        sub = train[members].corr().mean()
        out.append(sub.idxmin())
    return sorted(out)


METHODS = {
    "max effective bets":     lambda tr: greedy(tr, eff_bets),
    "max diversification rat":lambda tr: greedy(tr, div_ratio),
    "max eigenvalue entropy": lambda tr: greedy(tr, eig_entropy),
    "min avg pairwise corr":  lambda tr: greedy(tr, avg_corr, maximize=False),
    "min variance":           lambda tr: greedy(tr, port_var, maximize=False),
    "hierarchical clustering":cluster_select,
}

rng = np.random.default_rng(5)
rand = [list(rng.choice(R.columns, N, replace=False)) for _ in range(40)]
r_eb = np.array([eff_bets(TE[s]) for s in rand])

print(f"{'objective':<26} {'TRAIN bets':>11} {'TEST bets':>10} {'decay':>7} "
      f"{'TEST vol%':>10} {'pctile':>7}")
res = {}
for name, fn in METHODS.items():
    sel = fn(TR)
    tr_b, te_b = eff_bets(TR[sel]), eff_bets(TE[sel])
    vol = (TE[sel].std() * np.sqrt(252) * 100).mean()
    pct = (r_eb < te_b).mean() * 100
    res[name] = sel
    print(f"{name:<26} {tr_b:>11.2f} {te_b:>10.2f} {te_b-tr_b:>7.2f} "
          f"{vol:>10.1f} {pct:>6.0f}%")
print(f"{'random (median of 40)':<26} {'--':>11} {np.median(r_eb):>10.2f} "
      f"{'--':>7} {'--':>10}")

print("\n\nOVERLAP BETWEEN METHODS (of 30)\n")
names = list(res)
print(f"{'':<26}" + "".join(f"{n[:9]:>11}" for n in names))
for a in names:
    print(f"{a:<26}" + "".join(f"{len(set(res[a])&set(res[b])):>11}" for b in names))
