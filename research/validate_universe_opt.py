"""Validate the breadth-optimization result across multiple splits and pools.

The single split earlier could have been luck. This runs the same experiment
over several disjoint train/test boundaries and two different candidate pools,
and asks two separate questions:

  Q1 (direct): does selecting for breadth actually DELIVER breadth out of
      sample? That is the thing it optimizes, and it should hold if correlation
      structure is persistent.
  Q2 (what we care about): does it produce better RISK-ADJUSTED RETURNS than
      random selection out of sample?

Q1 passing and Q2 failing would still be informative -- it would mean structure
persists but does not pay.
"""
import numpy as np, pandas as pd
from algo import backtest, data, metrics, research, strategies
from algo.costs import ZERO_COST

POOL_A = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","JPM","V","UNH","XOM",
          "JNJ","WMT","PG","MA","HD","CVX","ABBV","MRK","KO","PEP",
          "BAC","COST","TMO","MCD","CSCO","ACN","ABT","LIN","ADBE","DHR"]
POOL_B = ["VZ","TXN","NEE","NKE","PM","RTX","UNP","LOW","IBM","CAT",
          "GS","QCOM","HON","BA","DE","AMT","SO","DUK","MMM","GE",
          "SPGI","INTC","T","CL","GILD","ADP","ISRG","NOW","MDT","BKNG"]
DIV = ["TLT","IEF","GLD","DBC"]
N_SELECT, N_RANDOM = 15, 40

dv = data.load_panel(DIV, start="2012-01-01", end="2026-09-01").dropna(how="any")


def stack(p, raw):
    iv = strategies.inverse_volatility(p, 60); c = raw * iv; t = c.sum(axis=1)
    c = c.div(t.where(t > 0), axis=0).fillna(0.0).mul(raw.sum(axis=1), axis=0)
    return strategies.volatility_target(c, p, target_vol=0.10, max_leverage=1.0)


def make_eval(pool_px):
    def perf(tk, period):
        p = pd.concat([pool_px[tk], dv], axis=1).loc[period]
        raw = strategies.ensemble(*[strategies.time_series_momentum(p, lb)
                                    for lb in (63, 126, 189, 252, 315)])
        r = backtest.run(p, stack(p, raw), ZERO_COST)
        return metrics.sharpe(r.returns)

    def bets(tk, period):
        p = pd.concat([pool_px[tk], dv], axis=1).loc[period]
        return research.effective_bets(p.pct_change().dropna())
    return perf, bets


def greedy(pool_px, period, objective, n=N_SELECT):
    chosen, remaining = [], list(pool_px.columns)
    while len(chosen) < n:
        best, best_v = remaining[0], -np.inf
        for t in remaining:
            v = objective(chosen + [t], period)
            if np.isfinite(v) and v > best_v:
                best, best_v = t, v
        chosen.append(best); remaining.remove(best)
    return chosen


results = []
for pool_name, pool in [("A", POOL_A), ("B", POOL_B)]:
    px = data.load_panel(pool, start="2012-01-01", end="2026-09-01").dropna(how="any")
    idx = px.index.intersection(dv.index)
    px = px.loc[idx]
    perf, bets = make_eval(px)
    rng = np.random.default_rng(11)

    # four disjoint test windows, each with all prior data as training
    n = len(idx)
    for k, frac in enumerate([0.45, 0.58, 0.71, 0.84], start=1):
        cut, end = int(n * frac), int(n * (frac + 0.15))
        if end > n:
            end = n
        train, test = idx[:cut], idx[cut:end]
        if len(test) < 250:
            continue

        sel_b = greedy(px, train, bets)
        sel_s = greedy(px, train, perf)
        rnd = [list(rng.choice(px.columns, N_SELECT, replace=False)) for _ in range(N_RANDOM)]

        r_sh = np.array([perf(r, test) for r in rnd])
        r_bt = np.array([bets(r, test) for r in rnd])
        results.append(dict(
            pool=pool_name, split=k,
            test_start=str(test[0].date()), test_end=str(test[-1].date()),
            breadth_sharpe=perf(sel_b, test), breadth_bets=bets(sel_b, test),
            breadth_pct=float((r_sh < perf(sel_b, test)).mean() * 100),
            breadth_bets_pct=float((r_bt < bets(sel_b, test)).mean() * 100),
            sharpe_sharpe=perf(sel_s, test),
            sharpe_pct=float((r_sh < perf(sel_s, test)).mean() * 100),
            rand_sharpe=float(np.median(r_sh)), rand_bets=float(np.median(r_bt)),
        ))
        print(f"  pool {pool_name} split {k} done ({test[0].date()}..{test[-1].date()})", flush=True)

df = pd.DataFrame(results)
print("\n" + "=" * 92)
print("Q1 (DIRECT): does breadth-selection deliver BREADTH out of sample?")
print("=" * 92)
print(f"{'pool/split':<12} {'test window':<24} {'sel bets':>9} {'rand bets':>10} {'pctile':>8}")
for _, r in df.iterrows():
    print(f"{r['pool']+'/'+str(r['split']):<12} {r['test_start']+'..'+r['test_end']:<24} "
          f"{r['breadth_bets']:>9.2f} {r['rand_bets']:>10.2f} {r['breadth_bets_pct']:>7.0f}%")
print(f"\n  beat random breadth in {(df.breadth_bets > df.rand_bets).sum()}/{len(df)} splits, "
      f"mean percentile {df.breadth_bets_pct.mean():.0f}%")

print("\n" + "=" * 92)
print("Q2 (WHAT WE CARE ABOUT): does it deliver better RETURNS out of sample?")
print("=" * 92)
print(f"{'pool/split':<12} {'breadth Sh':>11} {'pctile':>8} {'sharpe-opt Sh':>14} {'pctile':>8} {'random Sh':>10}")
for _, r in df.iterrows():
    print(f"{r['pool']+'/'+str(r['split']):<12} {r['breadth_sharpe']:>11.2f} {r['breadth_pct']:>7.0f}% "
          f"{r['sharpe_sharpe']:>14.2f} {r['sharpe_pct']:>7.0f}% {r['rand_sharpe']:>10.2f}")
print(f"\n  breadth-selected beat random in {(df.breadth_sharpe > df.rand_sharpe).sum()}/{len(df)}"
      f" splits, mean percentile {df.breadth_pct.mean():.0f}%")
print(f"  sharpe-selected  beat random in {(df.sharpe_sharpe > df.rand_sharpe).sum()}/{len(df)}"
      f" splits, mean percentile {df.sharpe_pct.mean():.0f}%")
df.to_csv("universe_validation.csv", index=False)
