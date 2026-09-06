"""IBKR Pro: Fixed vs Tiered pricing.

FIXED   $0.005/share, min $1.00/order, max 1% of trade value.
        All-in -- exchange and clearing fees included. Only regulatory fees added.
        Predictable. No rebates, ever.

TIERED  $0.0035/share, min $0.35/order, max 1% of trade value.
        PLUS exchange + clearing + pass-through fees. Limit orders that ADD
        liquidity can earn rebates; market orders that REMOVE it pay more.
        Reported real-world markup for market-order-heavy trading: ~40% over
        the base rate, so ~$0.0049/share all-in -- essentially the same as Fixed.

So per-share they are near-identical. The decisive difference is the ORDER
MINIMUM: $1.00 vs $0.35.
"""
import pandas as pd

def fixed(shares, notional):
    return min(max(0.005 * shares, 1.00), 0.01 * notional)

def tiered(shares, notional, aggressive=True):
    base = 0.0035 * shares * (1.40 if aggressive else 1.0)   # ~40% markup on marketable orders
    return min(max(base, 0.35), 0.01 * notional)

print("COMMISSION BY ORDER SIZE  (assumes a $100 share price)\n")
print(f"{'order':>10} {'shares':>7} {'Fixed':>9} {'Tiered':>9} {'Fixed bp':>10} {'Tiered bp':>10} {'winner':>8}")
for notional in [200, 500, 1_000, 2_500, 5_000, 10_000, 20_000, 50_000, 100_000]:
    sh = notional / 100
    f, t = fixed(sh, notional), tiered(sh, notional)
    print(f"{'$'+format(notional,','):>10} {sh:>7.0f} {'$'+format(f,'.2f'):>9} {'$'+format(t,'.2f'):>9}"
          f" {f/notional*1e4:>9.1f} {t/notional*1e4:>9.1f} {'Tiered' if t<f else 'Fixed' if f<t else 'tie':>8}")

print("\n\nWHERE EACH MINIMUM STOPS BINDING\n")
print("  Fixed:  $1.00 min binds until 0.005 x shares > $1.00  ->  200 shares")
print("  Tiered: $0.35 min binds until 0.0049 x shares > $0.35 ->   71 shares")
print("\n  At a $100 stock that means Fixed charges its $1.00 minimum on every")
print("  order up to $20,000. Tiered clears its minimum at ~$7,100.")

print("\n\nEFFECT ON THE ACTUAL PORTFOLIO (10 positions, 10% band, ~30 orders/yr)\n")
print(f"{'capital':>10} {'position':>10} {'Fixed/yr':>10} {'Tiered/yr':>11} {'Fixed %':>9} {'Tiered %':>9}")
for cap in [1_000, 5_000, 10_000, 25_000, 100_000]:
    pos = cap / 10
    sh = pos / 100
    f, t = fixed(sh, pos) * 30, tiered(sh, pos) * 30
    print(f"{'$'+format(cap,','):>10} {'$'+format(int(pos),','):>10} {'$'+format(f,'.2f'):>10}"
          f" {'$'+format(t,'.2f'):>11} {f/cap*100:>8.2f}% {t/cap*100:>8.2f}%")

print("\n\nRULE: Tiered is cheaper or equal at every retail order size.")
print("Fixed only wins on predictability, and on very low-priced stocks where")
print("high share counts make the per-share rate bite (the 1% cap protects both).")
