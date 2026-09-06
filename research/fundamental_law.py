"""Grinold's Fundamental Law of Active Management.

    IR  ~=  IC  x  sqrt(BR)

IR = information ratio (Sharpe of your active bets)
IC = information coefficient: correlation between your forecast and what
     actually happens. Professional quant signals run IC 0.02-0.05. That is
     nearly no predictive power per bet -- and it is enough, given breadth.
BR = breadth: number of INDEPENDENT bets per year.

The law says skill and breadth are substitutes, and breadth enters as a square
root -- so quadrupling your independent bets is worth doubling your skill.
"""
import numpy as np
from algo import data

EFFECTIVE_BETS_ETF = 2.65  # measured in compare_universes.py


def required_ic(target_ir, n_assets_effective, rebals_per_year):
    return target_ir / np.sqrt(n_assets_effective * rebals_per_year)


print("WHAT IC DO YOU NEED FOR AN IR OF 1.0?")
print("(pro quant signals are IC 0.02-0.05; IC 0.20 is fantasy)\n")
print(f"{'universe':<38} {'eff.bets':>9} {'rebal/yr':>9} {'BR':>7} {'IC needed':>10}")
setups = [
    ("10 ETFs, monthly",            EFFECTIVE_BETS_ETF,  12),
    ("10 ETFs, weekly",             EFFECTIVE_BETS_ETF,  52),
    ("10 ETFs, daily",              EFFECTIVE_BETS_ETF, 252),
    ("100 stocks, monthly",         25,                  12),
    ("500 stocks, monthly",         60,                  12),
    ("2000 stocks, daily (RenTec)", 150,                252),
]
for name, eb, rb in setups:
    br = eb * rb
    print(f"{name:<38} {eb:9.1f} {rb:9d} {br:7.0f} {required_ic(1.0, eb, rb):10.3f}")

print("\n\nWHAT IR CAN YOU ACTUALLY GET, AT A REALISTIC IC OF 0.03?\n")
print(f"{'universe':<38} {'IR':>7}  {'verdict'}")
for name, eb, rb in setups:
    ir = 0.03 * np.sqrt(eb * rb)
    verdict = ("negligible" if ir < 0.3 else
               "marginal" if ir < 0.6 else
               "a real business" if ir < 1.2 else "institutional-grade")
    print(f"{name:<38} {ir:7.2f}  {verdict}")

print("\n\nTHE LEVER: breadth, not signal quality.")
print("Going from 10 ETFs to 500 stocks (same monthly rebalance, same mediocre")
print(f"signal) takes IR from {0.03*np.sqrt(EFFECTIVE_BETS_ETF*12):.2f} to {0.03*np.sqrt(60*12):.2f}"
      " -- a 4.8x improvement with no")
print("better forecasting at all. Improving IC 4.8x is not achievable; adding")
print("names is a data problem.")
