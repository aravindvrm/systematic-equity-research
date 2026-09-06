"""How much money does this portfolio actually need?

Order-level simulation: real share counts, real $0.35 per-order minimums, real
spread. The question is not "what Sharpe does the strategy have" -- it is "at
what account size does the strategy survive contact with commissions".
"""
import pandas as pd
from algo import data, portfolio, orderlevel

px = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")
w = portfolio.build(px)
years = len(px) / 252

print(f"portfolio: ensembled trend + inv-vol + vol target, 10 ETFs")
print(f"period:    {px.index[0].date()} .. {px.index[-1].date()} ({years:.1f} years)\n")

for rebal, label in [(None, "DAILY (as currently built)"), ("ME", "MONTHLY"), ("QE", "QUARTERLY")]:
    print(f"=== rebalanced {label} ===")
    rows = []
    for cap in [500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 1_000_000]:
        r = orderlevel.run(px, w, capital=cap, rebalance=rebal, min_trade=50.0)
        rows.append({
            "capital": f"${cap:,}",
            "CAGR %": r["cagr"] * 100,
            "Sharpe": r["sharpe"],
            "MaxDD %": r["max_drawdown"] * 100,
            "orders": r["n_orders"],
            "cost/yr %": r["cost_pct_of_capital"] / years * 100,
        })
    print(pd.DataFrame(rows).set_index("capital").round(2).to_string(), "\n")

print("=" * 66)
print("REFERENCE: SPY buy & hold, order-level, same costs")
print("=" * 66)
spy_w = pd.DataFrame(0.0, index=px.index, columns=px.columns); spy_w["SPY"] = 1.0
for cap in [1_000, 10_000, 100_000]:
    r = orderlevel.run(px, spy_w, capital=cap, rebalance="YE", min_trade=50.0)
    print(f"  ${cap:>7,}  CAGR {r['cagr']*100:6.2f}%   Sharpe {r['sharpe']:.2f}   "
          f"orders {r['n_orders']:>4}   cost/yr {r['cost_pct_of_capital']/years*100:.3f}%")
