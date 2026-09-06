"""How much does the CHOICE OF NAMES matter, versus the strategy?

Draw many random 11-stock subsets from a large-cap pool, run the identical
strategy on each, and look at the spread of outcomes. If that spread is wide,
then which names you picked dominates whatever the strategy is doing -- and a
single hand-picked backtest tells you almost nothing.
"""
import numpy as np, pandas as pd
from algo import backtest, data, metrics, research, strategies
from algo.costs import ZERO_COST

POOL = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","JPM","V","UNH","XOM",
        "JNJ","WMT","PG","MA","HD","CVX","ABBV","MRK","KO","PEP",
        "BAC","COST","TMO","MCD","CSCO","ACN","ABT","LIN","ADBE","DHR",
        "VZ","TXN","NEE","NKE","PM","RTX","UNP","LOW","IBM","CAT",
        "GS","QCOM","HON","BA","DE","AMT","SO","DUK","MMM","GE"]
MY_PICKS = ["MSFT","JPM","JNJ","XOM","PG","HD","CAT","NEE","LIN","VZ","AMT"]
DIVERSIFIERS = ["TLT","IEF","GLD","DBC"]

px_pool = data.load_panel(POOL, start="2012-01-01", end="2026-09-01").dropna(how="any")
px_div = data.load_panel(DIVERSIFIERS, start="2012-01-01", end="2026-09-01").dropna(how="any")
common = px_pool.index.intersection(px_div.index)
px_pool, px_div = px_pool.loc[common], px_div.loc[common]
print(f"pool: {px_pool.shape[1]} large caps, {len(common)} bars, "
      f"{common[0].date()}..{common[-1].date()}\n")


def stack(p, raw):
    iv = strategies.inverse_volatility(p, 60)
    c = raw * iv; tot = c.sum(axis=1)
    c = c.div(tot.where(tot > 0), axis=0).fillna(0.0).mul(raw.sum(axis=1), axis=0)
    return strategies.volatility_target(c, p, target_vol=0.10, max_leverage=1.0)


def evaluate(tickers):
    p = pd.concat([px_pool[tickers], px_div], axis=1)
    raw = strategies.ensemble(*[strategies.time_series_momentum(p, lb)
                                for lb in (63, 126, 189, 252, 315)])
    r = backtest.run(p, stack(p, raw), ZERO_COST)
    return (metrics.sharpe(r.returns), metrics.cagr(r.equity),
            metrics.max_drawdown(r.equity), research.effective_bets(p.pct_change().dropna()))


rng = np.random.default_rng(42)
rows = [evaluate(list(rng.choice(px_pool.columns, 11, replace=False))) for _ in range(200)]
df = pd.DataFrame(rows, columns=["sharpe", "cagr", "maxdd", "eff_bets"])
mine = evaluate(MY_PICKS)

print("200 RANDOM 11-STOCK DRAWS + the same 4 diversifiers, identical strategy\n")
print(f"{'metric':<12} {'min':>8} {'p25':>8} {'median':>8} {'p75':>8} {'max':>8} {'MY PICKS':>10} {'%ile':>6}")
for i, (name, fmt) in enumerate([("sharpe", 2), ("cagr", 2), ("maxdd", 1), ("eff_bets", 2)]):
    col = df[name]
    pct = (col < mine[i]).mean() * 100
    scale = 100 if name in ("cagr", "maxdd") else 1
    print(f"{name:<12} {col.min()*scale:>8.{fmt}f} {col.quantile(.25)*scale:>8.{fmt}f} "
          f"{col.median()*scale:>8.{fmt}f} {col.quantile(.75)*scale:>8.{fmt}f} "
          f"{col.max()*scale:>8.{fmt}f} {mine[i]*scale:>10.{fmt}f} {pct:>5.0f}%")

print(f"\n  Sharpe spread across draws: {df.sharpe.min():.2f} to {df.sharpe.max():.2f} "
      f"(range {df.sharpe.max()-df.sharpe.min():.2f})")
print(f"  Sharpe std dev from PICKS ALONE: {df.sharpe.std():.3f}")
print(f"  CAGR spread: {df.cagr.min()*100:.1f}% to {df.cagr.max()*100:.1f}%")
print("\n  Note: effective bets is STABLE across draws while returns are not.")
print("  Breadth is a property of the universe; returns are a property of luck.")
