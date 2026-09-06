"""Is the signal bad, or is our SIZING RULE bad?

Every strategy in this project used one crude rule: rank names, keep the top
30%, inverse-vol weight them, zero the rest. That rule discards MAGNITUDE (99th
and 71st percentile get identical weight), discards the bottom 70% entirely, and
cuts holdings from 199 to 60 — losing diversification.

The concentration cost is what made the RF portfolio underperform holding
everything. That is a claim about the SIZING RULE, not necessarily the signal.

Tested here on the same RF prediction:
    top-k truncation  what we always did, at k = 10/30/50%
    tilt              hold ALL names, tilt weights around equal by signal rank
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor

from algo import (anomalies, backtest, data, evaluation, factors, features,
                  features2 as f2, filings, metrics, pead, research, strategies)
from algo.costs import IBKR_US_EQUITY

pd.set_option("display.width", 220)
exec(open("research/rf_stress.py").read().split("test_years =")[0].split('print(f"{len(L):,}')[0])
test_years = [y for y in sorted(L.date.dt.year.unique()) if y >= 2015]

preds = []
for y in test_years:
    tr, te = L[L.date.dt.year < y], L[L.date.dt.year == y]
    if len(tr) < 2000 or te.empty:
        continue
    m = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=100,
                              max_features=0.3, n_jobs=-1, random_state=0)
    m.fit(tr[cols].to_numpy(), tr["y"].to_numpy())
    p = te[["date", "ticker"]].copy()
    p["pred"] = m.predict(te[cols].to_numpy())
    preds.append(p)
rf = (pd.concat(preds, ignore_index=True)
      .pivot(index="date", columns="ticker", values="pred")
      .reindex(cl.index).ffill().reindex(columns=cl.columns))

first = pd.Timestamp(f"{test_years[0]}-01-01")
oos = cl.index >= first
ff = factors.load()
iv = strategies.inverse_volatility(cl, 60)


def run(raw):
    """Normalise raw weights, monthly-hold, vol-target, backtest."""
    tot = raw.sum(axis=1)
    w = raw.div(tot.where(tot > 0), axis=0).fillna(0.0)
    w = strategies.volatility_target(w, cl, target_vol=0.10, max_leverage=1.0)
    m = np.zeros(len(w), dtype=bool); m[::21] = True
    w = w.where(pd.Series(m, index=w.index), np.nan).ffill()
    return backtest.run(cl, w, cost_model=IBKR_US_EQUITY)


def topk(score, frac):
    return (score.rank(axis=1, pct=True, ascending=False) <= frac).astype(float) * iv


def tilt(score, strength):
    """All names held; weight tilted around equal by signal rank.

    strength=0 is equal weight; higher tilts harder. Diversification preserved.
    """
    r = score.rank(axis=1, pct=True)
    return (1.0 + strength * 2.0 * (r - 0.5)).clip(lower=0.0) * iv


RULES = {
    "top 10%":                lambda s: topk(s, 0.10),
    "top 30% (what we used)": lambda s: topk(s, 0.30),
    "top 50%":                lambda s: topk(s, 0.50),
    "tilt 0.5 (all names)":   lambda s: tilt(s, 0.5),
    "tilt 1.0 (all names)":   lambda s: tilt(s, 1.0),
    "tilt 1.8 (all names)":   lambda s: tilt(s, 1.8),
}

print(f"OOS {first.date()}..{cl.index[-1].date()}, {cl.shape[1]} names")
print("running null draws per sizing rule (40 each)...\n", flush=True)

print(f"{'sizing rule':<26} {'held':>6} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} "
      f"{'alpha%':>8} {'t':>6} {'pct':>5} {'null p95':>9}")
for lbl, fn in RULES.items():
    res = run(fn(rf))
    r = res.returns[oos]
    nu = []
    for i in range(40):
        rng = np.random.default_rng(9000 + i)
        n = pd.DataFrame(rng.normal(size=cl.shape), index=cl.index, columns=cl.columns)
        nu.append(factors.attribution(run(fn(n)).returns[oos], ff)["alpha_t"])
    nu = np.array([x for x in nu if np.isfinite(x)])
    a = factors.attribution(r, ff)
    held = (res.weights[oos] > 1e-6).sum(axis=1).mean()
    eq = (1 + r).cumprod()
    print(f"{lbl:<26} {held:>6.0f} {metrics.sharpe(r):>7.2f} "
          f"{metrics.cagr(eq)*100:>7.2f}% {metrics.max_drawdown(eq)*100:>7.1f}% "
          f"{a['alpha_ann']*100:>7.2f}% {a['alpha_t']:>6.2f} "
          f"{evaluation.percentile_vs_null(a['alpha_t'], nu):>4.0f}% "
          f"{np.percentile(nu,95):>9.2f}", flush=True)

r0 = run(pd.DataFrame(1.0, index=cl.index, columns=cl.columns) * iv).returns[oos]
eq0 = (1 + r0).cumprod()
print(f"\n{'NO SIGNAL (all names)':<26} {cl.shape[1]:>6} {metrics.sharpe(r0):>7.2f} "
      f"{metrics.cagr(eq0)*100:>7.2f}% {metrics.max_drawdown(eq0)*100:>7.1f}%")
