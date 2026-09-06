"""How good would a non-price signal have to be for the middle horizon to work?

Invert the Fundamental Law. Given the universe we settled on and Alpaca's costs,
solve for the IC required to reach a target NET Sharpe at each holding period.

  net_Sharpe = IC x sqrt(eff_bets x 252/h)  -  (252/h) x turnover x cost / vol
"""
import numpy as np, pandas as pd

EFF_BETS = 6.5      # measured: 20 blue chips + 4 diversifiers
COST_RT = 3e-4      # Alpaca: $0 commission, ~3bp spread + slippage round trip
VOL = 0.10          # vol target
TURNOVER = 0.5      # half the book changes per rebalance


def required_ic(target_sharpe, h, turnover=TURNOVER):
    br = EFF_BETS * (252 / h)
    cost_drag = (252 / h) * turnover * COST_RT / VOL
    return (target_sharpe + cost_drag) / np.sqrt(br)


HORIZONS = [(5, "1 week"), (10, "2 weeks"), (21, "1 month"),
            (42, "2 months"), (63, "1 quarter"), (126, "6 months")]

print("IC REQUIRED TO HIT A TARGET NET SHARPE")
print(f"(24-name universe, {EFF_BETS} effective bets, Alpaca costs, 10% vol target)\n")
print(f"{'horizon':<12} {'BR':>7} {'cost drag':>10} " +
      "".join(f"{'S='+str(s):>9}" for s in [0.3, 0.5, 0.8, 1.0]))
for h, lab in HORIZONS:
    br = EFF_BETS * (252 / h)
    cd = (252 / h) * TURNOVER * COST_RT / VOL
    cells = "".join(f"{required_ic(s, h):>9.3f}" for s in [0.3, 0.5, 0.8, 1.0])
    print(f"{lab:<12} {br:>7.0f} {cd:>10.2f} {cells}")

print("\n  CALIBRATION:  professional quant signals run IC 0.02-0.05")
print("                IC 0.10 would be exceptional")
print("                IC 0.20+ is not real\n")

print("\nSAME TABLE, MARKED FEASIBLE / NOT\n")
print(f"{'horizon':<12} " + "".join(f"{'S='+str(s):>12}" for s in [0.3, 0.5, 0.8, 1.0]))
for h, lab in HORIZONS:
    cells = ""
    for s in [0.3, 0.5, 0.8, 1.0]:
        ic = required_ic(s, h)
        tag = "easy" if ic < 0.02 else "plausible" if ic < 0.05 else \
              "hard" if ic < 0.10 else "no"
        cells += f"{tag+' ('+format(ic,'.3f')+')':>12}"
    print(f"{lab:<12} {cells}")

print("\n\nWHAT A BIGGER UNIVERSE WOULD DO  (target net Sharpe 0.5, monthly)\n")
print(f"{'universe':<34} {'eff bets':>9} {'IC needed':>11}")
for label, eb in [("10 broad ETFs", 2.65), ("24 blue chips + div (ours)", 6.5),
                  ("50 stocks long-only", 6.1), ("100 stocks long-only", 8.0),
                  ("50 stocks MARKET-NEUTRAL", 26.4)]:
    br = eb * 12
    cd = 12 * TURNOVER * COST_RT / VOL
    print(f"{label:<34} {eb:>9.1f} {(0.5+cd)/np.sqrt(br):>11.3f}")
