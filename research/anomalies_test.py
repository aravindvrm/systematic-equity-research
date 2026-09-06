"""Value, profitability and investment — the three untested categories.

PRE-SPECIFIED: 8 anomalies x 3 horizons = 24 IC tests, plus a portfolio-level
run of each against the NULL FLOOR calibrated to this stack.

Every one of these is a canonical published predictor, so this is a REPLICATION
attempt, not a discovery attempt. That matters for interpretation: if they fail
here it tells us about our universe and era, not about whether the effects were
ever real.

WHAT TO EXPECT, STATED BEFORE RUNNING
-------------------------------------
These are the most studied and most arbitraged anomalies in finance. Chordia,
Subrahmanyam & Tong (2014) measured the class as roughly HALVING after
decimalization. They are also BUYABLE -- Vanguard and iShares sell value,
quality and profitability exposure for 5-25bp -- so even a clean success here is
a rediscovery of an ETF, not an edge. The FF6 attribution already showed our
strategies carrying R2 of 0.66-0.75 on these factors passively.

The honest question is therefore not "do they work" but "does our coverage claim
hold" -- we tested two of six anomaly categories and called it exhaustive.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sst

warnings.filterwarnings("ignore")
from algo import (anomalies, backtest, data, evaluation, factors, metrics,
                  research, strategies)
from algo.costs import IBKR_US_EQUITY

pd.set_option("display.width", 220)
N_RANDOM = 40

F = pd.read_parquet("data/fundamentals/all_facts.parquet")
uni = pd.read_parquet("data/collection_universe.parquet").dropna(subset=["cik"])
uni["cik"] = uni["cik"].astype(int).astype(str)
c2t = dict(zip(uni["cik"], uni["ticker"]))
print(f"{len(F):,} filings, {F.cik.nunique()} companies, "
      f"{F.filed.min().date()}..{F.filed.max().date()}")

tickers = sorted(set(c2t.values()))
px = data.load_panel(tickers, start="2009-01-01", end="2026-09-01",
                     field="close", refresh=False)
px = px.dropna(axis=1, thresh=int(0.9 * len(px))).ffill(limit=5)
c2t = {k: v for k, v in c2t.items() if v in px.columns}
print(f"prices: {px.shape[1]} names, {len(px)} bars\n")

V = anomalies.compute(F)
print("filing-level coverage:")
for f in anomalies.FEATURES + ["book_equity", "earnings_ttm", "shares_out"]:
    print(f"  {f:<16} {V[f].notna().mean()*100:5.0f}%")

panels = {f: anomalies.to_panel(V, px.index, c2t, f) for f in anomalies.FEATURES}
panels.update(anomalies.market_ratios(V, px, c2t))
print("\ndaily cross-sectional coverage:")
for k, p in panels.items():
    print(f"  {k:<16} {p.notna().mean(axis=1).mean()*100:5.0f}% of names/day")

HZ = (21, 63, 126)
K = len(panels) * len(HZ)
bar = sst.norm.ppf(1 - 0.05 / (2 * K))
print(f"\n{'='*74}\nIC: {K} pre-specified tests -> Bonferroni |t| > {bar:.2f}\n{'='*74}\n")
print(f"{'anomaly':<18} {'h':>5} {'IC':>9} {'t':>8} {'n':>7}")
ic_rows = []
for k, p in panels.items():
    for hz in HZ:
        ic = research.cross_sectional_ic(p, research.forward_returns(px, hz)).dropna()
        if len(ic) < 200:
            continue
        m = float(ic.mean())
        t = (m / (ic.std(ddof=1) / np.sqrt(len(ic)))) / np.sqrt(hz)
        ic_rows.append((k, hz, m, t))
        print(f"{k:<18} {hz:>5} {m:>9.4f} {t:>8.2f} {len(ic):>7}")
IC = pd.DataFrame(ic_rows, columns=["anomaly", "h", "ic", "t"])
print(f"\nIC survivors: {(IC.t.abs() > bar).sum()}   "
      f"largest |t|: {IC.t.abs().max():.2f}   largest |IC|: {IC.ic.abs().max():.4f}")

# ---- portfolio level, against the null floor --------------------------------
def z(f):
    zz = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1).replace(0, np.nan), axis=0)
    return zz.clip(-3, 3)

def stack(score, top_frac=0.3, rebal=21):
    s = score.copy()
    mm = np.zeros(len(s), dtype=bool); mm[::rebal] = True
    s = s.where(pd.Series(mm, index=s.index), np.nan).ffill()
    raw = (s.rank(axis=1, pct=True, ascending=False) <= top_frac).astype(float)
    w = raw * strategies.inverse_volatility(px, 60)
    tot = w.sum(axis=1)
    w = w.div(tot.where(tot > 0), axis=0).fillna(0.0)
    return strategies.volatility_target(w, px, target_vol=0.10, max_leverage=1.0)

def rets(score, top_frac=0.3):
    return backtest.run(px, stack(score, top_frac), cost_model=IBKR_US_EQUITY).returns

ff = factors.load()
print(f"\nrunning {N_RANDOM} null draws on this universe/sample...", flush=True)
nu, nc = [], []
for i in range(N_RANDOM):
    rng = np.random.default_rng(7000 + i)
    noise = pd.DataFrame(rng.normal(size=px.shape), index=px.index, columns=px.columns)
    r = rets(noise)
    nu.append(factors.attribution(r, ff)["alpha_t"])
    nc.append(factors.conditional_attribution(r, ff)["alpha_t"])
nu = np.array([x for x in nu if np.isfinite(x)])
nc = np.array([x for x in nc if np.isfinite(x)])
print(f"\nNULL FLOOR  uncond t: mean {nu.mean():+.2f} p95 {np.percentile(nu,95):+.2f}   "
      f"cond t: mean {nc.mean():+.2f} p95 {np.percentile(nc,95):+.2f}")

print(f"\n{'='*104}\nPORTFOLIO LEVEL, FF6 + Ferson-Schadt, vs the null\n{'='*104}\n")
print(f"{'anomaly':<18} {'Sharpe':>7} {'alpha%':>8} {'uncond t':>9} {'pct':>5} "
      f"{'cond t':>7} {'pct':>5} {'R2':>6}")
rows = []
for k, p in panels.items():
    r = rets(z(p))
    u = factors.attribution(r, ff); cc = factors.conditional_attribution(r, ff)
    pu = evaluation.percentile_vs_null(u["alpha_t"], nu)
    pc = evaluation.percentile_vs_null(cc["alpha_t"], nc)
    rows.append(dict(anomaly=k, sharpe=metrics.sharpe(r), alpha=u["alpha_ann"],
                     t_u=u["alpha_t"], pct_u=pu, t_c=cc["alpha_t"], pct_c=pc,
                     r2=u["r2"]))
    print(f"{k:<18} {metrics.sharpe(r):>7.2f} {u['alpha_ann']*100:>7.2f}% "
          f"{u['alpha_t']:>9.2f} {pu:>4.0f}% {cc['alpha_t']:>7.2f} {pc:>4.0f}% "
          f"{u['r2']:>6.2f}")
D = pd.DataFrame(rows)
print(f"\nclearing null p95 unconditional: {(D.pct_u >= 95).sum()} of {len(D)}")
print(f"clearing null p95 conditional:   {(D.pct_c >= 95).sum()} of {len(D)}")
print(f"expected by chance at 5%:        {0.05*len(D):.1f}")
D.to_csv("anomalies_test.csv", index=False)
IC.to_csv("anomalies_ic.csv", index=False)
