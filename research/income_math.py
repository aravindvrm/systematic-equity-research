"""What does 'income' actually require, in capital terms?

Income = capital x return rate. Strategy sets the rate; capital sets the scale.
No amount of skill substitutes for the multiplication.
"""
import pandas as pd

print("ANNUAL INCOME BY ACCOUNT SIZE AND RETURN RATE\n")
rates = [0.06, 0.10, 0.15, 0.25, 0.50]
labels = ["6%\n(realistic)", "10%\n(good)", "15%\n(very good)", "25%\n(exceptional)", "50%\n(fantasy)"]
rows = []
for cap in [500, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000]:
    row = {"capital": f"${cap:,}"}
    for r, lab in zip(rates, ["6%", "10%", "15%", "25%", "50%"]):
        row[lab] = f"${cap*r:,.0f}/yr"
    rows.append(row)
print(pd.DataFrame(rows).set_index("capital").to_string())

print("\n\nSAME THING AS MONTHLY CASH FLOW\n")
rows = []
for cap in [1_000, 10_000, 25_000, 50_000, 100_000, 250_000]:
    row = {"capital": f"${cap:,}"}
    for r, lab in zip(rates, ["6%", "10%", "15%", "25%", "50%"]):
        row[lab] = f"${cap*r/12:,.0f}/mo"
    rows.append(row)
print(pd.DataFrame(rows).set_index("capital").to_string())

print("\n\nCAPITAL REQUIRED TO HIT AN INCOME TARGET\n")
print(f"{'target/mo':>11} " + " ".join(f"{l:>12}" for l in ["at 6%","at 10%","at 15%","at 25%"]))
for tgt in [100, 250, 500, 1000, 2500]:
    cells = " ".join(f"{'$'+format(int(tgt*12/r),','):>12}" for r in [0.06,0.10,0.15,0.25])
    print(f"{'$'+str(tgt):>11} {cells}")

print("\n\nOPTIONS INCOME: COLLATERAL NEEDED IN A CASH ACCOUNT")
print("A cash-secured put requires strike x 100 held in cash.\n")
print(f"{'underlying':>12} {'~price':>8} {'collateral for 1 put':>22}")
for name, p in [("SPY", 640), ("QQQ", 580), ("QQQM", 240), ("IWM", 240),
                ("GLD", 310), ("XLF", 52), ("F", 12)]:
    print(f"{name:>12} {'$'+str(p):>8} {'$'+format(p*100, ','):>22}")
print("\n  Covered calls need 100 shares of the underlying -- same arithmetic.")
print("  At $1,000 the entire tradeable universe is single-digit-priced stocks,")
print("  where one bad earnings print wipes out a year of premium.")
