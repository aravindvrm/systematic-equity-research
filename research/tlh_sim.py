"""How much is tax-loss harvesting actually worth on the no-signal book?

WHAT TLH IS
-----------
Hold individual names rather than an ETF. When a position falls below its cost
basis, sell it to REALISE the loss, and immediately buy something similar (not
"substantially identical" -- see the wash-sale note) so market exposure is
unchanged. The realised loss offsets capital gains elsewhere; the tax you do not
pay this year stays invested.

WHAT IT IS NOT
--------------
It is not return alpha. It is a DEFERRAL. Harvesting lowers your cost basis, so
the gain resurfaces on eventual sale. The benefit is (a) the time value of money
on deferred tax, (b) converting short-term gains into long-term where possible,
and (c) the option value of banking losses when you have gains to offset. The
deferral becomes permanent only via step-up at death or donating appreciated
shares -- neither of which is a trading strategy.

WHY THE BENEFIT DECAYS
----------------------
As a portfolio appreciates, fewer positions sit below basis. Harvesting capacity
is largest in year one and shrinks. This simulation shows that decay explicitly
rather than quoting a flat "100bp/yr", which is the number most marketing uses
and which is only true early.

SIMPLIFICATIONS (all of which FLATTER the result)
- assumes a replacement security with identical returns is always available
- ignores the 30-day wash-sale window entirely
- ignores transaction costs on the harvest trades
- assumes you always have gains elsewhere to offset at the full rate
Treat every number here as an upper bound.
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from algo import data

LTCG_RATE = 0.15 + 0.038          # federal LTCG + NIIT; state extra, varies
HARVEST_THRESHOLD = 0.05          # only harvest losses deeper than 5%

tk = json.load(open("data/trading_universe.json"))
px = data.load_panel(tk, start="2010-01-01", end="2026-09-01", refresh=False)
px = px.dropna(axis=1, thresh=int(0.9 * len(px))).ffill(limit=5).dropna(axis=1)
print(f"{px.shape[1]} names, {px.index[0].date()}..{px.index[-1].date()}\n")


def simulate(prices, capital, check_every=21):
    """Equal-weight buy-and-hold with monthly loss harvesting.

    Tracks a cost basis per name. On each check, any position more than
    HARVEST_THRESHOLD below basis is sold and rebought: the loss is banked and
    the basis resets to the current price.
    """
    n = prices.shape[1]
    shares = (capital / n) / prices.iloc[0]
    basis = prices.iloc[0].copy()
    realised = []
    for i in range(check_every, len(prices), check_every):
        p = prices.iloc[i]
        unreal = (p / basis - 1.0)
        hit = unreal < -HARVEST_THRESHOLD
        if hit.any():
            loss = ((p[hit] - basis[hit]) * shares[hit]).sum()
            realised.append((prices.index[i], float(loss)))
            basis[hit] = p[hit]          # rebuy at today's price
    df = pd.DataFrame(realised, columns=["date", "loss"]).set_index("date")
    df["year"] = df.index.year
    return df


print("HARVESTABLE LOSSES BY YEAR  (30-name book, $100k, 5% threshold)\n")
sim = simulate(px, 100_000)
by_year = sim.groupby("year")["loss"].sum()
print(f"{'year':>6} {'realised losses':>17} {'as % of $100k':>15}")
for y, v in by_year.items():
    print(f"{y:>6} {v:>17,.0f} {abs(v)/100_000*100:>14.2f}%")
avg = abs(by_year.mean())
print(f"\n  mean per year: ${avg:,.0f} = {avg/100_000*100:.2f}% of portfolio")
print(f"  first 3 years: {abs(by_year.iloc[:3].mean())/1000:.1f}k/yr   "
      f"last 3 years: {abs(by_year.iloc[-3:].mean())/1000:.1f}k/yr   <- the decay")

print(f"\n\nTAX BENEFIT BY PORTFOLIO SIZE (at {LTCG_RATE*100:.1f}% rate)")
print("assumes you have gains elsewhere to offset -- otherwise capped at")
print("$3,000/yr against ordinary income, with the excess carried forward\n")
print(f"{'portfolio':>12} {'losses/yr':>12} {'tax deferred':>14} {'vs $3k cap':>12} {'bp/yr':>8}")
for cap in (1_000, 10_000, 50_000, 100_000, 500_000):
    s = simulate(px, cap)
    L = abs(s.groupby(s.index.year)["loss"].sum().mean())
    benefit = L * LTCG_RATE
    capped = min(L, 3000) * 0.24        # ordinary-income offset if no gains
    print(f"{cap:>12,} {L:>12,.0f} {benefit:>14,.0f} {capped:>12,.0f} "
          f"{benefit/cap*1e4:>8.0f}")

print("\n" + "=" * 78)
print("READ THE $1,000 ROW.")
print("=" * 78)
s1 = simulate(px, 1_000)
L1 = abs(s1.groupby(s1.index.year)["loss"].sum().mean())
print(f"  harvestable losses:  ${L1:,.0f}/yr")
print(f"  tax deferred:        ${L1*LTCG_RATE:,.0f}/yr")
print(f"  positions:           ${1000/px.shape[1]:,.0f} each across {px.shape[1]} names")
print("\n  Each position is tiny, so every harvest trade is a separate order with")
print("  its own commission floor. At this size the technique cannot pay for its")
print("  own transaction costs, let alone the bookkeeping.")
