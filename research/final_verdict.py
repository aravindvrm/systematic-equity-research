"""Is the no-signal book worth running, or is a two-ETF allocation just as good?

This is the comparison that was never made. The book has been judged against
SPY, against equal-weight, against a 60/40 -- but never against the SIMPLEST
thing that produces the same risk profile: a static mix of SPY and AGG, matched
on volatility, rebalanced monthly, requiring two tickers and no research.

If a 30-name vol-targeted book cannot beat that, it is a complicated way to buy
something simple, and the complication has real costs: 30 positions instead of
2, 30 sets of tax lots, 30 corporate actions to handle, and a daily job that can
break.
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from algo import backtest, data, evaluation, factors, metrics, strategies
from algo.costs import IBKR_US_EQUITY

pd.set_option("display.width", 200)
ff = factors.load()

tk = json.load(open("data/trading_universe.json"))
px = data.load_panel(tk, start="2010-01-01", end="2026-09-01", refresh=False)
px = px.dropna(axis=1, thresh=int(0.9 * len(px))).ffill(limit=5)
etf = data.load_panel(["SPY", "AGG"], start="2010-01-01", end="2026-09-01",
                      refresh=False).ffill().dropna()
idx = px.index.intersection(etf.index)
px, etf = px.loc[idx], etf.loc[idx]
print(f"{px.shape[1]} names vs 2 ETFs, {idx[0].date()}..{idx[-1].date()}\n")


def monthly(w):
    m = np.zeros(len(w), dtype=bool); m[::21] = True
    return w.where(pd.Series(m, index=w.index), np.nan).ffill()


def book(prices, target_vol=0.10):
    w = strategies.inverse_volatility(prices, 60)
    tot = w.sum(axis=1)
    w = w.div(tot.where(tot > 0), axis=0).fillna(0.0)
    w = strategies.volatility_target(w, prices, target_vol=target_vol, max_leverage=1.0)
    return backtest.run(prices, monthly(w), cost_model=IBKR_US_EQUITY)


def etf_mix(spy_w):
    w = pd.DataFrame({"SPY": spy_w, "AGG": 1 - spy_w}, index=etf.index)
    return backtest.run(etf, monthly(w), cost_model=IBKR_US_EQUITY)


nb = book(px)
nbs = nb.summary()
print(f"{'strategy':<32} {'Sharpe':>7} {'CAGR':>8} {'vol':>7} {'MaxDD':>8} "
      f"{'Calmar':>7} {'alpha%':>8} {'t':>6}")

def line(lbl, res):
    s = res.summary()
    a = factors.attribution(res.returns, ff)
    print(f"{lbl:<32} {s['sharpe']:>7.2f} {s['cagr']*100:>7.2f}% "
          f"{s['vol']*100:>6.2f}% {s['max_drawdown']*100:>7.1f}% {s['calmar']:>7.2f} "
          f"{a['alpha_ann']*100:>7.2f}% {a['alpha_t']:>6.2f}")
    return s

line("30-name book (no signal)", nb)
print()
for w in (0.40, 0.50, 0.60, 0.70, 0.80, 1.00):
    line(f"SPY/AGG static {int(w*100)}/{int((1-w)*100)}", etf_mix(w))

# Which static mix matches the book's volatility? That is the fair comparison.
target = nbs["vol"]
best, bestd = None, 9e9
for w in np.arange(0.30, 1.01, 0.02):
    r = etf_mix(w)
    d = abs(r.summary()["vol"] - target)
    if d < bestd:
        best, bestd, bw = r, d, w
print(f"\n{'-'*104}")
print(f"VOLATILITY-MATCHED COMPARISON  (book vol {target*100:.2f}%)")
print(f"{'-'*104}")
bs = best.summary()
print(f"  book                     Sharpe {nbs['sharpe']:.3f}  CAGR {nbs['cagr']*100:.2f}%  "
      f"MaxDD {nbs['max_drawdown']*100:.1f}%  vol {nbs['vol']*100:.2f}%")
print(f"  SPY/AGG {bw*100:.0f}/{100-bw*100:.0f} (matched)   Sharpe {bs['sharpe']:.3f}  "
      f"CAGR {bs['cagr']*100:.2f}%  MaxDD {bs['max_drawdown']*100:.1f}%  vol {bs['vol']*100:.2f}%")
print(f"\n  book advantage: Sharpe {nbs['sharpe']-bs['sharpe']:+.3f}   "
      f"CAGR {(nbs['cagr']-bs['cagr'])*100:+.2f}%   "
      f"MaxDD {(nbs['max_drawdown']-bs['max_drawdown'])*100:+.1f}pp")

n = len(nb.returns) / 252
se = np.sqrt((1 + 0.5 * nbs["sharpe"] ** 2) / n)
print(f"\n  standard error of a Sharpe estimate over {n:.1f} years: +/-{se:.3f}")
print(f"  the observed advantage is {abs(nbs['sharpe']-bs['sharpe'])/se:.2f} standard errors")

print(f"\n{'-'*104}")
print("OPERATIONAL COST OF THE COMPLICATED VERSION")
print(f"{'-'*104}")
print(f"  positions to hold           {px.shape[1]} vs 2")
print(f"  annual turnover             {nb.turnover.mean()*252:.2f} vs "
      f"{best.turnover.mean()*252:.2f}")
print(f"  annual cost drag            {nb.costs.mean()*252*1e4:.1f}bp vs "
      f"{best.costs.mean()*252*1e4:.1f}bp")
print(f"  at $1,000 account           ${1000/px.shape[1]:.0f} per position "
      f"vs ${1000/2:.0f}")
print("  plus: 30 sets of tax lots, 30 corporate-action streams, a daily job")
print("        that can fail silently, and a universe that needs re-selection")
