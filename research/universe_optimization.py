"""Can we SELECT a universe intelligently -- or is that just overfitting?

Two candidate objectives, both fitted on the first half of the sample and tested
on the second:

  A) maximize EFFECTIVE BETS  (diversification -- a correlation property)
  B) maximize SHARPE          (performance -- a return property)

Earlier we measured that effective bets is stable across random draws while
returns are not. If that stability is real, (A) should hold up out of sample and
(B) should not.
"""
import numpy as np, pandas as pd
from algo import backtest, data, metrics, research, strategies
from algo.costs import ZERO_COST

POOL = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","JPM","V","UNH","XOM",
        "JNJ","WMT","PG","MA","HD","CVX","ABBV","MRK","KO","PEP",
        "BAC","COST","TMO","MCD","CSCO","ACN","ABT","LIN","ADBE","DHR",
        "VZ","TXN","NEE","NKE","PM","RTX","UNP","LOW","IBM","CAT",
        "GS","QCOM","HON","BA","DE","AMT","SO","DUK","MMM","GE"]
DIV = ["TLT","IEF","GLD","DBC"]
N_SELECT = 15

pp = data.load_panel(POOL, start="2012-01-01", end="2026-09-01").dropna(how="any")
dv = data.load_panel(DIV, start="2012-01-01", end="2026-09-01").dropna(how="any")
idx = pp.index.intersection(dv.index); pp, dv = pp.loc[idx], dv.loc[idx]
mid = len(idx) // 2
train, test = idx[:mid], idx[mid:]
print(f"train {train[0].date()}..{train[-1].date()}   test {test[0].date()}..{test[-1].date()}\n")


def stack(p, raw):
    iv = strategies.inverse_volatility(p, 60); c = raw * iv; t = c.sum(axis=1)
    c = c.div(t.where(t > 0), axis=0).fillna(0.0).mul(raw.sum(axis=1), axis=0)
    return strategies.volatility_target(c, p, target_vol=0.10, max_leverage=1.0)


def perf(tickers, period):
    p = pd.concat([pp[tickers], dv], axis=1).loc[period]
    raw = strategies.ensemble(*[strategies.time_series_momentum(p, lb)
                                for lb in (63, 126, 189, 252, 315)])
    r = backtest.run(p, stack(p, raw), ZERO_COST)
    return metrics.sharpe(r.returns), research.effective_bets(p.pct_change().dropna())


def greedy(period, objective):
    """Greedily add the name that most improves `objective` on `period`."""
    chosen = []
    remaining = list(pp.columns)
    while len(chosen) < N_SELECT:
        best, best_v = remaining[0], -np.inf
        for t in remaining:
            v = objective(chosen + [t], period)
            if np.isfinite(v) and v > best_v:
                best, best_v = t, v
        chosen.append(best); remaining.remove(best)
    return chosen


def obj_bets(tk, period):
    p = pd.concat([pp[tk], dv], axis=1).loc[period]
    return research.effective_bets(p.pct_change().dropna())


def obj_sharpe(tk, period):
    # A single name plus the diversifiers is a perfectly valid portfolio, so
    # there is no need to special-case the first pick.
    return perf(tk, period)[0]


print("selecting universes on the TRAIN half ...")
by_bets = greedy(train, obj_bets)
by_sharpe = greedy(train, obj_sharpe)
rng = np.random.default_rng(3)
randoms = [list(rng.choice(pp.columns, N_SELECT, replace=False)) for _ in range(40)]

print(f"\n  max-breadth picks : {', '.join(sorted(by_bets))}")
print(f"  max-Sharpe picks  : {', '.join(sorted(by_sharpe))}")
print(f"  overlap between them: {len(set(by_bets) & set(by_sharpe))} of {N_SELECT}\n")

print(f"{'universe':<26} {'TRAIN Sharpe':>13} {'TEST Sharpe':>12} {'decay':>8} {'TEST bets':>10}")
rs = np.array([perf(r, test)[0] for r in randoms])
rb = np.array([obj_bets(r, test) for r in randoms])
for name, tk in [("selected for BREADTH", by_bets), ("selected for SHARPE", by_sharpe)]:
    tr, _ = perf(tk, train); te, eb = perf(tk, test)
    pct = (rs < te).mean() * 100
    print(f"{name:<26} {tr:>13.2f} {te:>12.2f} {te-tr:>8.2f} {eb:>10.2f}   "
          f"({pct:.0f}th pctile of random)")
print(f"{'random (median of 40)':<26} {'--':>13} {np.median(rs):>12.2f} {'--':>8} {np.median(rb):>10.2f}")
