"""Baseline run: several simple strategies on a fixed liquid-ETF universe.

Everything here is IN-SAMPLE and reported with costs. Treat the numbers as a
sanity check on the pipeline, not as evidence any strategy works.
"""
import logging
import pandas as pd

from algo import backtest, data, metrics, strategies
from algo.costs import IBKR_US_EQUITY, IBKR_CRYPTO, ZERO_COST

logging.basicConfig(level=logging.WARNING)
pd.set_option("display.width", 200)

START, END = "2010-01-01", "2026-09-01"

print(f"Loading {len(data.ETF_UNIVERSE)} ETFs {START} -> {END} ...")
px = data.load_panel(data.ETF_UNIVERSE, start=START, end=END)
px = px.dropna(how="any")
print(f"  {px.shape[0]} bars x {px.shape[1]} tickers, {px.index[0].date()} .. {px.index[-1].date()}\n")

runs = {
    "SPY buy & hold":       backtest.buy_and_hold(px, "SPY", cost_model=IBKR_US_EQUITY),
    "Equal weight":         backtest.run(px, strategies.equal_weight(px), IBKR_US_EQUITY),
    "Inverse vol":          backtest.run(px, strategies.inverse_volatility(px), IBKR_US_EQUITY),
    "TS momentum (12m)":    backtest.run(px, strategies.time_series_momentum(px), IBKR_US_EQUITY),
    "XS momentum (top 3)":  backtest.run(px, strategies.cross_sectional_momentum(px), IBKR_US_EQUITY),
    "SMA 50/200":           backtest.run(px, strategies.sma_crossover(px), IBKR_US_EQUITY),
}

rows = []
for name, res in runs.items():
    s = res.summary()
    rows.append({
        "strategy": name,
        "CAGR %": s["cagr"] * 100,
        "Vol %": s["vol"] * 100,
        "Sharpe": s["sharpe"],
        "MaxDD %": s["max_drawdown"] * 100,
        "Calmar": s["calmar"],
        "Turnover": s["avg_turnover"],
        "Gross Shrp": s["gross_sharpe"],
    })

df = pd.DataFrame(rows).set_index("strategy")
print(df.round(2).to_string())

print("\n--- cost sensitivity: XS momentum under different venues ---")
xs_w = strategies.cross_sectional_momentum(px)
for cm in (ZERO_COST, IBKR_US_EQUITY, IBKR_CRYPTO):
    r = backtest.run(px, xs_w, cost_model=cm)
    print(f"  {cm.name:24s} round-trip {cm.round_trip_bps:5.1f}bp  "
          f"Sharpe {metrics.sharpe(r.returns):6.2f}  CAGR {metrics.cagr(r.equity)*100:6.2f}%")

print("\n--- deflated Sharpe: the 6 strategies above were 6 trials ---")
best = df["Sharpe"].max()
best_name = df["Sharpe"].idxmax()
n_obs = len(px)
print(f"  best raw Sharpe   {best:.2f}  ({best_name})")
print(f"  deflated (6)      {metrics.deflated_sharpe(best, 6, n_obs):.2f}")
print(f"  deflated (100)    {metrics.deflated_sharpe(best, 100, n_obs):.2f}   <- if you tune 100 variants")
