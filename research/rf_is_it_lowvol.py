"""Is the random forest finding something, or rediscovering low-volatility?

The stress tests passed, but the feature importances are dominated by
low_vol_60, true_range, garman_klass and beta_to_univ -- three measures of the
same thing plus beta. That is the low-volatility anomaly / betting-against-beta
(Frazzini & Pedersen 2014), which is NOT in Fama-French 5 + momentum. So alpha
against FF6 could be pure BAB exposure that the factor model cannot see.

Three tests:
  1. Portfolio-level vs the null floor, using the BEST grid config (defensible
     because all 18 configs worked -- this is not cherry-picking a fluke).
  2. Attribution against FF6 PLUS a home-built low-volatility factor from our
     own universe. If alpha dies, the RF is BAB.
  3. Direct comparison: how much of the RF prediction is explained by the single
     low_vol feature, and does a naive low-vol strategy do as well?
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from scipy import stats as sst

from algo import (anomalies, backtest, data, evaluation, factors, features,
                  features2 as f2, filings, metrics, pead, research, strategies)
from algo.costs import IBKR_US_EQUITY

pd.set_option("display.width", 220)
exec(open("research/rf_stress.py").read().split("test_years =")[0].split('print(f"{len(L):,}')[0])

test_years = [y for y in sorted(L.date.dt.year.unique()) if y >= 2015]
BEST = dict(max_depth=10, min_samples_leaf=100, max_features=0.3)
print(f"best config: {BEST}\n", flush=True)

preds = []
for y in test_years:
    tr, te = L[L.date.dt.year < y], L[L.date.dt.year == y]
    if len(tr) < 2000 or te.empty:
        continue
    m = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0, **BEST)
    m.fit(tr[cols].to_numpy(), tr["y"].to_numpy())
    p = te[["date", "ticker"]].copy()
    p["pred"] = m.predict(te[cols].to_numpy())
    preds.append(p)
D = pd.concat(preds, ignore_index=True)
rf_panel = D.pivot(index="date", columns="ticker", values="pred").reindex(
    cl.index).ffill().reindex(columns=cl.columns)

def stack(score, top_frac=0.3, rebal=21):
    s = score.copy()
    mm = np.zeros(len(s), dtype=bool); mm[::rebal] = True
    s = s.where(pd.Series(mm, index=s.index), np.nan).ffill()
    raw = (s.rank(axis=1, pct=True, ascending=False) <= top_frac).astype(float)
    w = raw * strategies.inverse_volatility(cl, 60)
    tot = w.sum(axis=1)
    w = w.div(tot.where(tot > 0), axis=0).fillna(0.0)
    return strategies.volatility_target(w, cl, target_vol=0.10, max_leverage=1.0)

def rets(score, top_frac=0.3):
    return backtest.run(cl, stack(score, top_frac), cost_model=IBKR_US_EQUITY).returns

first = pd.Timestamp(f"{test_years[0]}-01-01")
oos = cl.index >= first
ff = factors.load()

# --- home-built low-volatility factor, from OUR universe ----------------------
lv = FEATS["low_vol_60"]
lv_rank = lv.rank(axis=1, pct=True)
long_lv = (lv_rank >= 0.7).astype(float)
short_lv = (lv_rank <= 0.3).astype(float)
r1 = cl.pct_change()
LOWVOL = ((r1 * long_lv.shift(1)).sum(axis=1) / long_lv.shift(1).sum(axis=1).replace(0, np.nan)
          - (r1 * short_lv.shift(1)).sum(axis=1) / short_lv.shift(1).sum(axis=1).replace(0, np.nan))
LOWVOL = LOWVOL.rename("LOWVOL")
print(f"home-built LOWVOL factor: ann return {LOWVOL.mean()*252*100:+.2f}%, "
      f"Sharpe {metrics.sharpe(LOWVOL):.2f}\n")

print(f"running 40 null draws...", flush=True)
nu, nc = [], []
for i in range(40):
    rng = np.random.default_rng(9000 + i)
    noise = pd.DataFrame(rng.normal(size=cl.shape), index=cl.index, columns=cl.columns)
    r = rets(noise)[oos]
    nu.append(factors.attribution(r, ff)["alpha_t"])
    nc.append(factors.conditional_attribution(r, ff)["alpha_t"])
nu = np.array([x for x in nu if np.isfinite(x)]); nc = np.array([x for x in nc if np.isfinite(x)])
print(f"null floor: uncond p95 {np.percentile(nu,95):+.2f}  cond p95 {np.percentile(nc,95):+.2f}\n")

ff2 = ff.join(LOWVOL, how="inner")
F7 = factors.FACTORS + ["LOWVOL"]

print("=" * 104)
print("1+2. PORTFOLIO, vs FF6 and vs FF6+LOWVOL")
print("=" * 104 + "\n")
print(f"{'strategy':<26} {'Sharpe':>7} {'FF6 a%':>8} {'t':>6} {'pct':>5} | "
      f"{'+LOWVOL a%':>11} {'t':>6} {'b_LOWVOL':>9}")
for lbl, sc in [("RF (best config)", rf_panel),
                ("naive low_vol_60 only", lv),
                ("naive inverse beta", -FEATS["beta_to_univ"])]:
    r = rets(sc)[oos]
    a6 = factors.attribution(r, ff)
    a7 = factors.attribution(r, ff2, factors=F7)
    print(f"{lbl:<26} {metrics.sharpe(r):>7.2f} {a6['alpha_ann']*100:>7.2f}% "
          f"{a6['alpha_t']:>6.2f} {evaluation.percentile_vs_null(a6['alpha_t'], nu):>4.0f}% | "
          f"{a7['alpha_ann']*100:>10.2f}% {a7['alpha_t']:>6.2f} {a7['b_LOWVOL']:>9.2f}")

print("\n" + "=" * 104)
print("3. HOW MUCH OF THE RF IS JUST low_vol_60?")
print("=" * 104)
lvr = lv.rank(axis=1, pct=True).reindex(rf_panel.index)
rfr = rf_panel.rank(axis=1, pct=True)
cs = rfr.corrwith(lvr, axis=1).dropna()
print(f"\n  mean cross-sectional rank corr(RF prediction, low_vol_60) = {cs.mean():.3f}")
print(f"  RF OOS IC        {sst.spearmanr(D['pred'], D.merge(L[['date','ticker','y']], on=['date','ticker'])['y']).statistic:.4f}")
ic_lv = research.cross_sectional_ic(lv, fwd_x).loc[lambda s: s.index >= first]
print(f"  low_vol_60 alone IC {ic_lv.mean():.4f}")
print("\n  If the correlation is high AND the naive low-vol strategy scores")
print("  similarly above, the RF is an expensive way to buy the low-vol anomaly.")
