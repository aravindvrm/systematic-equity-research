"""Do insider open-market purchases predict returns?

Applies every lesson from the price-signal work:
  - deflate t for OVERLAP (h-day forward returns on consecutive days share h-1)
  - deflate t for CLUSTERING (insiders buy together on macro down days)
  - compare against the MARKET on the same dates, not against zero -- the
    mean-reversion tangent died because ~90% of its "edge" was beta
"""
import numpy as np, pandas as pd
from algo import form4, universe as U, data, research

buys = form4.open_market_buys(form4.load())
buys = buys[buys.trans_date >= "2018-01-01"]
tickers = sorted(buys.ticker.unique())
px = data.load_panel(tickers, start="2017-06-01", end="2026-09-01")
px = px[[c for c in px.columns if px[c].notna().mean() > 0.90]]
spy = data.load_panel(["SPY"], start="2017-06-01", end="2026-09-01")["SPY"].reindex(px.index).ffill()
buys = buys[buys.ticker.isin(px.columns)]
print(f"{len(buys):,} buys across {buys.ticker.nunique()} names with price data\n")


def study(events, label, horizons=(5, 21, 63)):
    """Event study with market adjustment and both deflations."""
    out = []
    for h in horizons:
        fwd = px.shift(-h) / px - 1.0
        mkt = (spy.shift(-h) / spy - 1.0)
        rows = []
        for _, e in events.iterrows():
            t, d = e.ticker, e.trans_date
            if t not in fwd.columns:
                continue
            idx = fwd.index.searchsorted(d)
            if idx >= len(fwd):
                continue
            r, m = fwd[t].iloc[idx], mkt.iloc[idx]
            if pd.notna(r) and pd.notna(m):
                rows.append((r, m, fwd.index[idx]))
        if len(rows) < 30:
            continue
        arr = pd.DataFrame(rows, columns=["r", "m", "dt"])
        excess = arr.r - arr.m                       # market-adjusted
        # clustering: how many DISTINCT days did these events land on?
        episodes = arr.dt.dt.date.nunique()
        per_ep = len(arr) / max(episodes, 1)
        t_naive = excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess)))
        t_adj = t_naive / np.sqrt(h) / np.sqrt(max(per_ep, 1.0))
        out.append(dict(h=h, n=len(arr), episodes=episodes,
                        raw=arr.r.mean(), mkt=arr.m.mean(), excess=excess.mean(),
                        t_naive=t_naive, t=t_adj, win=(excess > 0).mean()))
    df = pd.DataFrame(out)
    print(f"--- {label} ---")
    if df.empty:
        print("  too few events\n"); return df
    print(f"  {'h':>4} {'n':>7} {'episodes':>9} {'raw%':>7} {'mkt%':>7} {'excess%':>8} "
          f"{'win%':>6} {'t_naive':>8} {'t_adj':>7}")
    for _, r in df.iterrows():
        print(f"  {int(r.h):>4} {int(r.n):>7,} {int(r.episodes):>9,} {r.raw*100:>7.2f} "
              f"{r.mkt*100:>7.2f} {r.excess*100:>8.2f} {r.win*100:>5.1f}% "
              f"{r.t_naive:>8.2f} {r.t:>7.2f}")
    print()
    return df


study(buys, "ALL open-market buys")

big = buys[buys.value >= 100_000]
study(big, "buys >= $100k")

huge = buys[buys.value >= 1_000_000]
study(huge, "buys >= $1M")

roles = buys.relationship.fillna("").str.lower()
study(buys[roles.str.contains("officer")], "officers only")
study(buys[roles.str.contains("director")], "directors only")

# Cluster buying: multiple distinct insiders buying the same name within 5 days
buys = buys.sort_values("trans_date")
key = buys.assign(wk=buys.trans_date.dt.to_period("W"))
grp = key.groupby(["ticker", "wk"]).owner_cik.nunique()
multi = grp[grp >= 2].reset_index()[["ticker", "wk"]]
cl = key.merge(multi, on=["ticker", "wk"]).drop_duplicates(["ticker", "wk"])
study(cl, "CLUSTER buys (>=2 insiders, same week)")
