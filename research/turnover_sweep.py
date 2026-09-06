"""Where does the crypto cost structure actually hurt?

Cost drag scales with turnover, so the penalty depends entirely on holding
period. Sweep lookback (slow -> fast) and watch the two venues diverge.
"""
import pandas as pd
from algo import backtest, data, metrics, strategies
from algo.costs import IBKR_US_EQUITY, IBKR_CRYPTO, ZERO_COST

CRYPTO = ["BTC-USD","ETH-USD","LTC-USD","BCH-USD","SOL-USD",
          "ADA-USD","DOGE-USD","AVAX-USD","LINK-USD","XRP-USD"]

eq = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")
cr = data.load_panel(CRYPTO, start="2015-01-01", end="2026-09-01").dropna(how="any")

rows = []
for lb, label in [(252,"12 month"),(126,"6 month"),(63,"3 month"),(21,"1 month"),(10,"2 week"),(5,"1 week")]:
    for name, px, cm, ppy in [("equity", eq, IBKR_US_EQUITY, 252), ("crypto", cr, IBKR_CRYPTO, 365)]:
        skip = max(1, lb // 12)
        w = strategies.cross_sectional_momentum(px, lookback=lb, skip=skip, top_n=3)
        g = backtest.run(px, w, ZERO_COST, periods_per_year=ppy)
        n = backtest.run(px, w, cm, periods_per_year=ppy)
        rows.append({
            "lookback": label, "venue": name,
            "turnover": n.turnover.mean(),
            "gross": metrics.sharpe(g.returns, periods_per_year=ppy),
            "net": metrics.sharpe(n.returns, periods_per_year=ppy),
            "drag": metrics.sharpe(g.returns,periods_per_year=ppy) - metrics.sharpe(n.returns,periods_per_year=ppy),
        })

df = pd.DataFrame(rows)
piv = df.pivot(index="lookback", columns="venue", values=["turnover","gross","net","drag"])
piv = piv.reindex(["12 month","6 month","3 month","1 month","2 week","1 week"])
print(piv.round(3).to_string())
print("\nSharpe LOST to costs, by holding period:")
for lab in ["12 month","6 month","3 month","1 month","2 week","1 week"]:
    e = df[(df.lookback==lab)&(df.venue=="equity")].iloc[0]
    c = df[(df.lookback==lab)&(df.venue=="crypto")].iloc[0]
    print(f"  {lab:9s}  equity -{e.drag:.2f}   crypto -{c.drag:.2f}   "
          f"crypto penalty {c.drag/max(e.drag,1e-9):5.1f}x")
