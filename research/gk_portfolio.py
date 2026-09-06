"""Does switching the volatility estimator to Garman-Klass improve the portfolios?

Garman-Klass forecasts next-month realised vol better than close-to-close
(rank corr 0.368 vs 0.283, better on 16 of 18 names). Since the vol-targeting
overlay is what generates essentially all of this project's results, a better
vol estimate should feed straight through.

Two places the estimator is used, and both are changed here:
    inverse-volatility WEIGHTS  (how much of each name)
    volatility TARGETING        (how much of the book is invested)
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from algo import backtest, data, evaluation, factors, metrics, strategies
from algo.costs import IBKR_US_EQUITY

pd.set_option("display.width", 200)

uni = pd.read_parquet("data/collection_universe.parquet")["ticker"].tolist()
P = {f: data.load_panel(uni, start="2005-01-01", end="2026-09-01", field=f,
                        refresh=False) for f in ("open", "high", "low", "close")}
cl = P["close"].dropna(axis=1, thresh=int(0.9 * len(P["close"]))).ffill(limit=5)
for k in P:
    P[k] = P[k].reindex(columns=cl.columns).ffill(limit=5)
o, hi, lo = P["open"], P["high"], P["low"]
print(f"{cl.shape[1]} names, {len(cl)} bars\n")

ANN = np.sqrt(252)

def vol_cc(window=60):
    return cl.pct_change().rolling(window).std() * ANN

def vol_gk(window=60):
    """Garman-Klass: uses the daily RANGE, so it sees intraday variation that a
    close-to-close estimate misses entirely."""
    term = (0.5 * np.log(hi / lo) ** 2
            - (2 * np.log(2) - 1) * np.log(cl / o) ** 2)
    return np.sqrt(term.rolling(window).mean().clip(lower=0)) * ANN

def build(volfn, target_vol=0.10, rebal=21):
    v = volfn()
    w = (1.0 / v.replace(0, np.nan))
    w = w.div(w.sum(axis=1), axis=0).fillna(0.0)
    # vol-target using the SAME estimator, applied to the portfolio series
    pr = (w.shift(1) * cl.pct_change()).sum(axis=1)
    realised = pr.rolling(60).std() * ANN
    scale = (target_vol / realised.replace(0, np.nan)).clip(upper=1.0)
    w = w.mul(scale, axis=0).fillna(0.0)
    m = np.zeros(len(w), dtype=bool); m[::rebal] = True
    return backtest.run(cl, w.where(pd.Series(m, index=w.index), np.nan).ffill(),
                        cost_model=IBKR_US_EQUITY)

core_px = data.load_panel(["SPY", "AGG"], start="2005-01-01", end="2026-09-01",
                          refresh=False).ffill().dropna()
cw = pd.DataFrame({"SPY": 0.6, "AGG": 0.4}, index=core_px.index)
m = np.zeros(len(cw), dtype=bool); m[::21] = True
core = backtest.run(core_px, cw.where(pd.Series(m, index=cw.index), np.nan).ffill(),
                    cost_model=IBKR_US_EQUITY).returns
ff = factors.load()

print("=" * 104)
print("NO-SIGNAL BOOK: close-to-close vol vs Garman-Klass vol")
print("=" * 104 + "\n")
print(f"{'estimator':<26} {'Sharpe':>7} {'CAGR':>8} {'vol':>7} {'MaxDD':>8} "
      f"{'Calmar':>7} {'FF6 a%':>8} {'t':>6} {'delta@20%':>10}")
out = {}
for lbl, fn in (("close-to-close (ours)", vol_cc), ("Garman-Klass", vol_gk)):
    res = build(fn)
    r = res.returns
    out[lbl] = r
    a = factors.attribution(r, ff)
    x, c = evaluation._align(r, core)
    d = metrics.sharpe(0.8 * c + 0.2 * x) - metrics.sharpe(c)
    print(f"{lbl:<26} {metrics.sharpe(r):>7.2f} {metrics.cagr(res.equity)*100:>7.2f}% "
          f"{metrics.volatility(r)*100:>6.2f}% {metrics.max_drawdown(res.equity)*100:>7.1f}% "
          f"{metrics.calmar(res.equity):>7.2f} {a['alpha_ann']*100:>7.2f}% "
          f"{a['alpha_t']:>6.2f} {d:>+10.4f}")

diff = metrics.sharpe(out["Garman-Klass"]) - metrics.sharpe(out["close-to-close (ours)"])
n = len(out["Garman-Klass"]) / 252
se = np.sqrt((1 + 0.5 * metrics.sharpe(out["Garman-Klass"]) ** 2) / n)
print(f"\n  Sharpe difference: {diff:+.4f}")
print(f"  SE of a Sharpe estimate over {n:.1f}y: +/-{se:.3f}  "
      f"-> {abs(diff)/se:.2f} standard errors")

print("\n  How well did each estimator actually hit the 10% vol target?")
for lbl, r in out.items():
    rv = r.rolling(252).std() * ANN
    print(f"    {lbl:<24} realised vol: mean {r.std()*ANN*100:.2f}%  "
          f"sd of rolling-1y vol {rv.std()*100:.2f}pp")
