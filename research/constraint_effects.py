"""What do sector / beta / liquidity floors cost, and what do they buy?"""
import numpy as np, pandas as pd, json
from algo import universe as U, data, research

tab = U.collection_table(); m = tab.set_index('ticker')
px = data.load_panel(sorted(set(tab.ticker.tolist()+["SPY"])), start='2018-01-01', end='2026-09-01')
keep = [c for c in px.columns if px[c].notna().mean() > 0.98]
pxk = px[keep].dropna(); R = pxk.pct_change().dropna()
SPY = R['SPY']; POOL = R.drop(columns=['SPY'])
BETA = POOL.apply(lambda c: c.cov(SPY)/SPY.var())
DV = m.dollar_volume.reindex(POOL.columns)
CORE6 = ['TLT','GLD','DBC','HYG','EFA','TIP']
N = 30

def eff(r):
    lam = np.linalg.eigvalsh(r.corr().to_numpy()); lam = lam[lam>0]
    return float(lam.sum()**2/(lam**2).sum())

def greedy(pool, n=N, must=(), sector_floor=0, beta_floor=None, dv_floor=None):
    cols = list(pool.columns)
    if beta_floor is not None:
        cols = [c for c in cols if c in U.DIVERSIFIERS or BETA.get(c, 0) >= beta_floor]
    if dv_floor is not None:
        cols = [c for c in cols if c in U.DIVERSIFIERS or (DV.get(c) or 0) >= dv_floor]
    chosen = [t for t in must if t in cols]
    rem = [c for c in cols if c not in chosen]

    # sector floor: reserve slots so every sector gets minimum representation
    if sector_floor:
        for sec in sorted(set(m.sector) - {'Diversifier'}):
            cands = [c for c in rem if c in m.index and m.loc[c,'sector']==sec]
            for _ in range(sector_floor):
                if not cands or len(chosen) >= n: break
                best, bv = cands[0], -np.inf
                for t in cands:
                    v = eff(pool[chosen+[t]]) if chosen else pool[t].std()
                    if v > bv: best, bv = t, v
                chosen.append(best); rem.remove(best); cands.remove(best)
    while len(chosen) < n and rem:
        best, bv = rem[0], -np.inf
        for t in rem:
            v = eff(pool[chosen+[t]]) if chosen else pool[t].std()
            if v > bv: best, bv = t, v
        chosen.append(best); rem.remove(best)
    return sorted(chosen)

def report(lab, sel):
    r = pxk[sel].pct_change().dropna()
    eqs = [t for t in sel if t not in U.DIVERSIFIERS]
    eb = eff(r); br = eb*52
    ic = (0.5 + 52*0.5*3e-4/0.10)/np.sqrt(br)
    nsec = len({m.loc[t,'sector'] for t in eqs if t in m.index})
    print(f"{lab:<28} {eb:>8.2f} {ic:>7.3f} {(r.std()*np.sqrt(252)*100).mean():>7.1f} "
          f"{BETA.reindex(eqs).mean():>6.2f} {DV.reindex(eqs).median()/1e6:>8.0f} {nsec:>6}")
    return eb

print(f"{'constraint':<28} {'eff.bets':>8} {'IC@wk':>7} {'vol%':>7} {'beta':>6} "
      f"{'$M/day':>8} {'sectors':>6}")
base = report("core6 (current)", greedy(POOL, must=CORE6))
report("+ sector floor 1/sector", greedy(POOL, must=CORE6, sector_floor=1))
report("+ sector floor 2/sector", greedy(POOL, must=CORE6, sector_floor=2))
report("+ beta floor 0.7", greedy(POOL, must=CORE6, beta_floor=0.7))
report("+ beta floor 0.9", greedy(POOL, must=CORE6, beta_floor=0.9))
report("+ liquidity floor $500M", greedy(POOL, must=CORE6, dv_floor=5e8))
report("+ liquidity floor $1B", greedy(POOL, must=CORE6, dv_floor=1e9))
report("all three (1/sec,0.7,$500M)", greedy(POOL, must=CORE6, sector_floor=1,
                                             beta_floor=0.7, dv_floor=5e8))
print(f"\n  baseline effective bets: {base:.2f}")
print("  IC@wk = IC required for IR 0.5 at weekly rebalance")
