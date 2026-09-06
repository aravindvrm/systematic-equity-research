"""The small-account constraint, quantified.

IBKR Pro Tiered: $0.0035/share, MINIMUM $0.35 per order, capped at 1% of trade
value. That per-order minimum is the binding constraint for a small account: it
turns into a large percentage cost when orders are small, which caps how many
positions you can hold and how often you can rebalance.
"""
import numpy as np, pandas as pd

MIN_ORDER = 0.35
PER_SHARE = 0.0035
CAP_PCT = 0.01
SPREAD_SLIP_BPS = 1.5  # half-spread + slippage on a liquid name, one side


def commission_bps(notional, price=100.0):
    """Effective one-way commission in bps of notional."""
    shares = notional / price
    comm = min(max(PER_SHARE * shares, MIN_ORDER), CAP_PCT * notional)
    return comm / notional * 1e4


print("ONE-WAY COMMISSION AS bp OF NOTIONAL  (IBKR Pro Tiered, $100 stock)")
print(f"{'order size':>12} {'commission':>12} {'bp':>8}  {'+spread/slip':>13}")
for n in [250, 500, 1000, 2500, 5000, 10000, 25000]:
    bp = commission_bps(n)
    print(f"{'$'+format(n,','):>12} {'$'+format(min(max(PER_SHARE*n/100,MIN_ORDER),CAP_PCT*n),'.2f'):>12}"
          f" {bp:7.1f}  {bp+SPREAD_SLIP_BPS:12.1f}")

print("\n\nANNUAL COST DRAG  (%/yr) -- account size x positions x rebalance frequency")
print("assumes full turnover of each position at each rebalance\n")
for rebal, per_yr in [("monthly", 12), ("quarterly", 4), ("annual", 1)]:
    print(f"--- rebalanced {rebal} ({per_yr}x/yr) ---")
    rows = []
    for acct in [5_000, 10_000, 25_000, 50_000, 100_000]:
        row = {"account": f"${acct:,}"}
        for k in [5, 10, 20, 40]:
            pos = acct / k
            one_way = commission_bps(pos) + SPREAD_SLIP_BPS
            # round trip per rebalance, times rebalances per year
            row[f"{k} pos"] = f"{2 * one_way * per_yr / 100:.2f}%"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("account").to_string(), "\n")

print("Rule of thumb: keep per-order commission under ~5bp, which means")
print(f"orders of at least ~${MIN_ORDER/0.0005:,.0f}. Below that the per-order")
print("minimum, not your strategy, is deciding your returns.")
