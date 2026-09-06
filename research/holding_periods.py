"""How long do we actually hold? And is intraday a different game?"""
import numpy as np, pandas as pd
from algo import backtest, data, portfolio, strategies
from algo.costs import IBKR_US_EQUITY

px = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")
w = portfolio.build(px)

def band(weights, b):
    cur = np.zeros(weights.shape[1]); rows = []
    for _, t in weights.iterrows():
        t = t.to_numpy(); mv = np.abs(t - cur) > b
        cur = np.where(mv, t, cur); rows.append(cur.copy())
    return pd.DataFrame(rows, index=weights.index, columns=weights.columns)

def holding_stats(weights, thresh=0.005):
    """Average consecutive bars a position stays open."""
    held = weights > thresh
    spans = []
    for col in held.columns:
        s = held[col].to_numpy(); run = 0
        for v in s:
            if v: run += 1
            elif run: spans.append(run); run = 0
        if run: spans.append(run)
    spans = np.array(spans)
    return dict(n_spells=len(spans), mean_days=spans.mean(), median_days=np.median(spans),
                max_days=spans.max())

print("HOLDING PERIOD OF WHAT WE BUILT\n")
print(f"{'construction':<24} {'spells':>8} {'mean days':>11} {'median':>8} {'max':>7} {'turnover':>10}")
for label, ww in [("daily rebalance", w), ("2% band", band(w, .02)),
                  ("5% band", band(w, .05)), ("10% band", band(w, .10))]:
    h = holding_stats(ww)
    t = backtest.run(px, ww, IBKR_US_EQUITY).turnover.mean()
    print(f"{label:<24} {h['n_spells']:>8} {h['mean_days']:>11.0f} {h['median_days']:>8.0f}"
          f" {h['max_days']:>7} {t:>10.3f}")

print("\n  ~1 year average holds. This is a LOW-frequency strategy -- closer to")
print("  tactical asset allocation than to trading.\n")

print("\n" + "="*78)
print("WHY INTRADAY IS A STRUCTURALLY DIFFERENT GAME")
print("="*78)
print("\nFundamental Law says IR = IC x sqrt(BR). Frequency buys BR cheaply...")
print("...but cost scales LINEARLY with trade count while IR scales as sqrt.\n")
print(f"{'holding period':<18} {'trades/yr':>10} {'BR':>9} {'IR@IC=.01':>11} "
      f"{'cost@5bp':>10} {'cost@1bp':>10}")
EFF = 2.65
for label, per_yr in [("1 year", 1), ("1 quarter", 4), ("1 month", 12), ("1 week", 52),
                      ("1 day", 252), ("1 hour", 252*7), ("5 minutes", 252*78)]:
    br = EFF * per_yr
    ir = 0.01 * np.sqrt(br)
    c5 = per_yr * 2 * 5e-4       # round trip at 5bp
    c1 = per_yr * 2 * 1e-4
    print(f"{label:<18} {per_yr:>10,} {br:>9,.0f} {ir:>11.2f} "
          f"{c5*100:>9.1f}% {c1*100:>9.1f}%")

print("\n  At 5-minute holds you would need ~196% of capital per year in costs")
print("  at 5bp, or 39% at 1bp. An IR of 2.3 on 10% vol is a 23% gross return.")
print("  The arithmetic does not close as a LIQUIDITY TAKER at any IC.")
print("\n  This is why intraday profitability generally requires EARNING the")
print("  spread (market making) rather than paying it. That is a different")
print("  business, gated on speed and exchange relationships, not on signal.")
