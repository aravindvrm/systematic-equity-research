"""How much breadth can a long-only US equity book actually reach?

Tests the natural intuition: pick blue chips across DIFFERENT sectors and they
should be less correlated than a handful of index ETFs.

NOTE ON SURVIVORSHIP: these are today's blue chips, so the sample excludes the
ones that stopped being blue chips (GE, Lehman, Kodak). That biases the
correlation structure toward names that co-survived, which if anything makes
these numbers OPTIMISTIC. Fine for an upper bound, which is what we want.
"""
import numpy as np, pandas as pd
from algo import data, research

# One large, liquid name per GICS sector -- the most diversified 11-stock
# long-only US equity book you could reasonably construct.
ONE_PER_SECTOR = {
    "Technology": "MSFT", "Financials": "JPM", "Health Care": "JNJ",
    "Energy": "XOM", "Cons. Staples": "PG", "Cons. Discretionary": "HD",
    "Industrials": "CAT", "Utilities": "NEE", "Materials": "LIN",
    "Comm. Services": "VZ", "Real Estate": "AMT",
}
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLP", "XLY", "XLI", "XLU", "XLB"]
MEGACAP50 = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","BRK-B","JPM","V",
             "UNH","XOM","JNJ","WMT","PG","MA","HD","CVX","ABBV","MRK",
             "KO","PEP","BAC","AVGO","COST","TMO","MCD","CSCO","ACN","ABT",
             "LIN","ADBE","DHR","VZ","TXN","NEE","NKE","PM","RTX","UNP",
             "LOW","INTC","IBM","CAT","GS","QCOM","SPGI","HON","BA","DE"]

def load(tickers):
    return data.load_panel(tickers, start="2012-01-01", end="2026-09-01").dropna(how="any")

def pc1_share(rets):
    """Fraction of total variance explained by the first principal component --
    i.e. how much of everything is just 'the market'."""
    c = rets.corr().to_numpy()
    lam = np.linalg.eigvalsh(c)[::-1]
    return lam[0] / lam.sum()

sets = {
    "10 broad ETFs":            load(data.ETF_UNIVERSE),
    "9 sector ETFs":            load(SECTOR_ETFS),
    "11 blue chips, 1/sector":  load(list(ONE_PER_SECTOR.values())),
    "50 mega-caps":             load(MEGACAP50),
}

print("EFFECTIVE BETS BY UNIVERSE  (long-only)\n")
print(f"{'universe':<28} {'names':>6} {'eff.bets':>9} {'per name':>9} {'PC1 var':>9} {'IR ceiling':>11}")
for name, p in sets.items():
    r = p.pct_change().dropna()
    eb = research.effective_bets(r)
    print(f"{name:<28} {p.shape[1]:>6} {eb:>9.2f} {eb/p.shape[1]:>9.2f}"
          f" {pc1_share(r)*100:>8.0f}% {0.03*np.sqrt(eb*252):>11.2f}")

print("\n\nSATURATION: does adding names keep helping?  (mega-cap universe)\n")
mc = sets["50 mega-caps"]; r_all = mc.pct_change().dropna()
rng = np.random.default_rng(0)
print(f"{'names':>6} {'eff.bets':>9} {'marginal gain':>14} {'IR ceiling':>11}")
prev = 0.0
for n in [2, 5, 10, 20, 30, 40, 50]:
    ebs = [research.effective_bets(r_all[list(rng.choice(mc.columns, n, replace=False))])
           for _ in range(20)]
    eb = float(np.mean(ebs))
    print(f"{n:>6} {eb:>9.2f} {eb-prev:>14.2f} {0.03*np.sqrt(eb*252):>11.2f}")
    prev = eb

print("\n\nWHAT IF YOU COULD SHORT?  (remove PC1 = the market factor)\n")
print(f"{'universe':<28} {'long-only':>10} {'mkt-neutral':>13} {'multiple':>9}")
for name, p in sets.items():
    r = p.pct_change().dropna()
    X = ((r - r.mean()) / r.std()).to_numpy()
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    resid = X - np.outer(X @ Vt[0], Vt[0])
    lo = research.effective_bets(r)
    mn = research.effective_bets(pd.DataFrame(resid, columns=p.columns))
    print(f"{name:<28} {lo:>10.2f} {mn:>13.2f} {mn/lo:>8.1f}x")
