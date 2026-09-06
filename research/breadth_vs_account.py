"""Does breadth require a big account?

Two ways to buy breadth, with OPPOSITE cost profiles:

    BR = (effective assets) x (rebalances per year)
          ^^^^ WIDTH               ^^^^ FREQUENCY

IR gain scales as sqrt(BR). Costs do not:

  - WIDTH costs the per-order minimum ($0.35) x N per rebalance. Shrinks to
    nothing as the account grows, because position size grows with it.
  - FREQUENCY costs spread + slippage (~1.5bp per side) on every turnover.
    This is a PERCENTAGE and never shrinks, at any account size.

So width is cheap for a large account and expensive for a small one; frequency
is expensive for everyone. That asymmetry is the whole answer.
"""
import numpy as np, pandas as pd
from algo import data

MIN_ORDER, IC, TARGET_VOL, SPREAD_SLIP = 0.35, 0.03, 0.10, 1.5e-4

MEGACAP = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","BRK-B","JPM","V",
           "UNH","XOM","JNJ","WMT","PG","MA","HD","CVX","ABBV","MRK",
           "KO","PEP","BAC","AVGO","COST","TMO","MCD","CSCO","ACN","ABT",
           "LIN","ADBE","DHR","VZ","TXN","NEE","NKE","PM","RTX","UNP",
           "LOW","INTC","IBM","CAT","GS","QCOM","SPGI","HON","BA","DE"]


def enb(corr):
    lam = np.linalg.eigvalsh(corr); lam = lam[lam > 0]
    return float(lam.sum()**2 / (lam**2).sum())


print("MEASURED effective bets -- real stocks, not an approximation")
print("(current tickers, so survivorship-biased; fine for correlation structure)\n")
px = data.load_panel(MEGACAP, start="2015-01-01", end="2026-09-01").dropna(how="any")
rets = px.pct_change().dropna()
raw = enb(rets.corr().to_numpy())

# Strip the market factor (first principal component) -- what a market-neutral
# book effectively trades. Long-only accounts CANNOT do this.
X = (rets - rets.mean()) / rets.std()
_, _, Vt = np.linalg.svd(X.to_numpy(), full_matrices=False)
pc1 = Vt[0]
resid = X.to_numpy() - np.outer(X.to_numpy() @ pc1, pc1)
neutral = enb(np.corrcoef(resid, rowvar=False))

print(f"  {len(px.columns)} mega-cap stocks, {px.index[0].date()}..{px.index[-1].date()}")
print(f"    long-only (market factor intact) : {raw:6.1f} effective bets")
print(f"    market-neutral (PC1 removed)     : {neutral:6.1f} effective bets")
print(f"    10-ETF universe (measured earlier):   2.65")
print(f"\n  Shorting roughly {neutral/raw:.1f}x's your breadth. That is why funds hedge --")
print("  and it is unavailable in a cash account.")

scale = raw / len(MEGACAP)  # effective bets per name, long-only, empirical
print(f"\n  Empirical rule: ~{scale:.2f} effective bets per name added (long-only).")


def net_alpha(n, r, acct):
    eff = scale * n
    gross = IC * np.sqrt(eff * r) * TARGET_VOL
    commission = MIN_ORDER * n * r / acct     # shrinks with account size
    friction = 2 * r * SPREAD_SLIP            # never shrinks
    return gross - commission - friction


print("\n\nNET EXPECTED ALPHA, now including spread/slippage")
print(f"{'account':>12} {'names':>7} {'rebalance':>11} {'net alpha':>11}")
for acct in [5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 1_000_000]:
    best = (-9, None, None)
    for n in range(5, 301, 5):
        for r, lab in [(1,"annual"),(4,"quarterly"),(12,"monthly"),(26,"biweekly"),(52,"weekly")]:
            v = net_alpha(n, r, acct)
            if v > best[0]: best = (v, n, lab)
    print(f"{'$'+format(acct,','):>12} {best[1]:>7} {best[2]:>11} {best[0]*100:>10.2f}%")

print("\n\nAT A FIXED MONTHLY REBALANCE -- how many names can each account afford?")
print(f"{'account':>12}  " + "  ".join(f"{n:>6}" for n in [10,25,50,100,200]))
for acct in [5_000, 10_000, 25_000, 50_000, 100_000]:
    cells = "  ".join(f"{net_alpha(n,12,acct)*100:>+6.2f}" for n in [10,25,50,100,200])
    print(f"{'$'+format(acct,','):>12}  {cells}")
