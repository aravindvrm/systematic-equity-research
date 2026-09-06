"""Deflate correctly: measure residual correlation instead of assuming it.

sqrt(events/episodes) assumes same-day events are PERFECTLY correlated. True for
a market-wide trigger. Wrong for insider buys at different companies after market
adjustment -- those residuals are largely independent.

Correct effective sample size for a clustered mean:
    n_eff = n / (1 + (m - 1) * rho)
where m = mean cluster size and rho = mean within-cluster correlation of the
market-adjusted returns. rho ~ 1 recovers the pessimistic deflation; rho ~ 0
means clustering costs nothing.
"""
import numpy as np, pandas as pd
from algo import form4, data, research

buys = form4.open_market_buys(form4.load())
buys = buys[buys.trans_date >= "2018-01-01"]
px = data.load_panel(sorted(buys.ticker.unique()), start="2017-06-01", end="2026-09-01")
px = px[[c for c in px.columns if px[c].notna().mean() > 0.90]]
spy = data.load_panel(["SPY"], start="2017-06-01", end="2026-09-01")["SPY"].reindex(px.index).ffill()
buys = buys[buys.ticker.isin(px.columns)]


def collect(events, h):
    fwd = px.shift(-h) / px - 1.0
    mkt = spy.shift(-h) / spy - 1.0
    rows = []
    for _, e in events.iterrows():
        if e.ticker not in fwd.columns:
            continue
        i = fwd.index.searchsorted(e.trans_date)
        if i >= len(fwd):
            continue
        r, m = fwd[e.ticker].iloc[i], mkt.iloc[i]
        if pd.notna(r) and pd.notna(m):
            rows.append((fwd.index[i], e.ticker, r - m))
    return pd.DataFrame(rows, columns=["dt", "ticker", "excess"])


def within_day_rho(df):
    """Mean pairwise correlation of market-adjusted returns among same-day events.

    Estimated via the ANOVA identity: for clusters of a common mean,
    rho ~= between-cluster variance / total variance.
    """
    g = df.groupby(df.dt.dt.date).excess
    sizes = g.size()
    multi = sizes[sizes >= 2].index
    sub = df[df.dt.dt.date.isin(multi)]
    if sub.empty:
        return 0.0, 1.0
    gm = sub.groupby(sub.dt.dt.date).excess
    between = gm.mean().var(ddof=1)
    total = sub.excess.var(ddof=1)
    rho = float(np.clip(between / total, 0, 1)) if total > 0 else 0.0
    return rho, float(sizes.mean())


print(f"{'events':<34} {'h':>3} {'n':>7} {'m':>5} {'rho':>6} {'n_eff':>8} "
      f"{'excess%':>8} {'t_naive':>8} {'t_correct':>10}")
CASES = [("ALL buys", buys),
         ("buys >= $100k", buys[buys.value >= 1e5]),
         ("buys >= $1M", buys[buys.value >= 1e6]),
         ("officers only", buys[buys.relationship.fillna('').str.lower().str.contains('officer')])]
res = []
for lab, ev in CASES:
    for h in (5, 21, 63):
        df = collect(ev, h)
        if len(df) < 30:
            continue
        rho, m = within_day_rho(df)
        n = len(df)
        n_eff = n / (1 + (m - 1) * rho)
        # overlap correction still applies: h-day windows on consecutive days
        n_eff = n_eff / h
        t_naive = df.excess.mean() / (df.excess.std(ddof=1) / np.sqrt(n))
        t_corr = df.excess.mean() / (df.excess.std(ddof=1) / np.sqrt(max(n_eff, 1)))
        res.append((lab, h, t_corr))
        print(f"{lab:<34} {h:>3} {n:>7,} {m:>5.1f} {rho:>6.3f} {n_eff:>8.0f} "
              f"{df.excess.mean()*100:>8.2f} {t_naive:>8.2f} {t_corr:>10.2f}")

bar = research.bonferroni_t(len(res))
print(f"\n  Bonferroni bar for {len(res)} tests: |t| > {bar:.2f}")
surv = [r for r in res if abs(r[2]) > bar]
print(f"  survivors: {len(surv)} of {len(res)}")
for lab, h, t in surv:
    print(f"    {lab}  h={h}  t={t:.2f}")
