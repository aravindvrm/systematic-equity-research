"""Market-neutral, done properly: best signal, best sizing, real borrow costs.

WHY REVISIT
-----------
Long-short was tested ONCE, early, on the equal-weighted price composite -- the
worst construction in this project (2nd percentile of the null) -- at weekly
rebalancing. It was never run on the random forest, and never with the sizing
fix. Sharpe came out 0.16 / 0.00 / -0.05 and the idea was dropped.

WHY IT MATTERS CONCEPTUALLY
---------------------------
Long-short STRIPS THE EQUITY BETA. Every long-only Sharpe near 1.0 in this
project was mostly market exposure -- the FF6 attribution showed MKT loadings of
0.4-0.5 and R2 of 0.66-0.75. Market-neutral removes that and leaves the pure
signal. Whatever Sharpe survives IS the signal quality, undisguised.

Clarke, de Silva & Thorley (2002): the long-only constraint gives a TRANSFER
COEFFICIENT of roughly 0.5-0.6, i.e. it costs about half the achievable IR. So
market-neutral should roughly DOUBLE the IR for a given IC -- if the IC is real.

COSTS THAT LONG-ONLY DOES NOT PAY
---------------------------------
  borrow fee     ~25-50bp/yr general collateral for S&P 500 names; far more for
                 hard-to-borrow. Stressed at 150bp here.
  no rebate      institutions earn interest on short proceeds. Retail Reg T
                 accounts generally do NOT -- the broker keeps it. Modelled as
                 zero rebate, which is the conservative retail case.
  double costs   200% gross exposure means twice the trading.

ACCOUNT REALITY: this needs a margin account. IBKR Reg T minimum is $2,000, and
Reg T requires 50% initial / 25% maintenance margin.
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
r1 = cl.pct_change().fillna(0.0)
BORROW = {"none (institutional)": 0.0, "50bp/yr (GC)": 0.0050, "150bp/yr (stress)": 0.0150}


def ls_returns(score, frac=0.2, rebal=21, borrow=0.0050, cost_bps=2.2):
    """Dollar-neutral long-short. Longs and shorts each 100% of capital.

    Weights are held between rebalances (no free continuous rebalancing), and
    the short leg pays a borrow fee accrued daily on the short notional.
    """
    r = score.rank(axis=1, pct=True, ascending=False)
    lo = (r <= frac).astype(float)
    sh = (r >= 1 - frac).astype(float)
    wl = lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    ws = sh.div(sh.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w = wl - ws
    m = np.zeros(len(w), dtype=bool); m[::rebal] = True
    w = w.where(pd.Series(m, index=w.index), np.nan).ffill().fillna(0.0)

    held = w.shift(1).fillna(0.0)
    gross = (held * r1).sum(axis=1)
    turn = (w.shift(1) - w.shift(2)).abs().sum(axis=1).fillna(0.0)
    cost = turn * cost_bps * 1e-4
    short_notional = held.clip(upper=0).abs().sum(axis=1)
    fee = short_notional * borrow / 252.0
    return gross - cost - fee


print(f"OOS {first.date()}..{cl.index[-1].date()}\n")
print("=" * 96)
print("MARKET-NEUTRAL on the random forest signal")
print("=" * 96 + "\n")
print(f"{'top/bottom':<12} {'borrow':<22} {'Sharpe':>8} {'CAGR':>8} {'vol':>7} "
      f"{'MaxDD':>8} {'mkt beta':>9}")
mkt = ff["Mkt-RF"]
for frac in (0.1, 0.2, 0.3):
    for lbl, b in BORROW.items():
        r = ls_returns(rf, frac=frac, borrow=b)[oos]
        eq = (1 + r).cumprod()
        both = pd.concat([r.rename("r"), mkt.rename("m")], axis=1).dropna()
        beta = np.polyfit(both["m"], both["r"], 1)[0] if len(both) > 100 else np.nan
        print(f"{frac*100:>10.0f}% {lbl:<22} {metrics.sharpe(r):>8.2f} "
              f"{metrics.cagr(eq)*100:>7.2f}% {metrics.volatility(r)*100:>6.2f}% "
              f"{metrics.max_drawdown(eq)*100:>7.1f}% {beta:>9.2f}")
    print()

print("=" * 96)
print("NULL FLOOR for market-neutral (40 random signals, same construction)")
print("=" * 96)
nu = []
for i in range(40):
    rng = np.random.default_rng(9000 + i)
    n = pd.DataFrame(rng.normal(size=cl.shape), index=cl.index, columns=cl.columns)
    nu.append(metrics.sharpe(ls_returns(n, frac=0.2, borrow=0.0050)[oos]))
nu = np.array(nu)
real = metrics.sharpe(ls_returns(rf, frac=0.2, borrow=0.0050)[oos])
print(f"\n  random-signal market-neutral Sharpe: mean {nu.mean():+.3f} "
      f"sd {nu.std():.3f}  p95 {np.percentile(nu,95):+.3f}  max {nu.max():+.3f}")
print(f"  RF market-neutral Sharpe:           {real:+.3f}   "
      f"-> {evaluation.percentile_vs_null(real, nu):.0f}th percentile")

print("\n" + "=" * 96)
print("WHAT THE LONG-ONLY CONSTRAINT COSTS (Clarke/de Silva/Thorley transfer coefficient)")
print("=" * 96)
iv = strategies.inverse_volatility(cl, 60)
def long_only(score, frac=0.2):
    r = score.rank(axis=1, pct=True, ascending=False)
    raw = (r <= frac).astype(float) * iv
    w = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    m = np.zeros(len(w), dtype=bool); m[::21] = True
    w = w.where(pd.Series(m, index=w.index), np.nan).ffill()
    return backtest.run(cl, w, cost_model=IBKR_US_EQUITY).returns
lo_r = long_only(rf)[oos]
ls_r = ls_returns(rf, frac=0.2, borrow=0.0050)[oos]
print(f"\n  long-only Sharpe   {metrics.sharpe(lo_r):+.3f}  (mostly equity beta)")
print(f"  market-neutral     {metrics.sharpe(ls_r):+.3f}  (pure signal, beta stripped)")
print("\n  Theory says market-neutral should roughly DOUBLE the IR of a real signal.")
print("  If it does not, the long-only Sharpe was beta, not skill.")
