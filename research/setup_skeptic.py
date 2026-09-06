"""Stress the setup results before believing them.

Three inflation sources, in increasing order of seriousness:

  1. OVERLAP    -- 21-day forward returns on consecutive days share 20 of 21
                   days. ic_stats() corrects for this; event_study() did not.
  2. CLUSTERING -- when the market drops, MANY names trigger at once. 429
                   "events" may be a dozen market-wide episodes.
  3. REGIME     -- buying dips in a 14-year bull market on survivor blue chips
                   is not obviously an edge.
"""
import numpy as np, pandas as pd
from algo import data, research

BLUE = ["MSFT","JPM","JNJ","XOM","PG","HD","CAT","NEE","LIN","VZ","AMT",
        "UNH","WMT","CVX","KO","MCD","HON","TXN","DE","GS"]
px = data.load_panel(BLUE + ["TLT","IEF","GLD","DBC"], start="2012-01-01",
                     end="2026-09-01").dropna(how="any")

CASES = [(21, -1.5, 21), (21, -2.0, 21), (21, -2.5, 21), (21, -3.0, 21), (5, -2.0, 21)]

print("1. EVENT CLUSTERING -- how many INDEPENDENT episodes?\n")
print(f"{'setup':<20} {'raw events':>11} {'distinct days':>14} {'episodes':>10} "
      f"{'names/episode':>14}")
for lb, th, h in CASES:
    z = research.zscore(px, lookback=lb)
    cond = z <= th
    days = cond.any(axis=1)
    # an "episode" = a run of consecutive trigger days, allowing 5-day gaps
    d = days[days].index
    episodes = 1 + int((np.diff(d.values).astype("timedelta64[D]").astype(int) > 5).sum()) if len(d) > 1 else len(d)
    print(f"{str(lb)+'d <= '+str(th)+'sd':<20} {int(cond.sum().sum()):>11} "
          f"{int(days.sum()):>14} {episodes:>10} {cond.sum().sum()/max(episodes,1):>14.1f}")

print("\n  Effective sample size is EPISODES, not events. The t-stats above")
print("  were computed as if every name-day were independent.\n")

print("\n2. OVERLAP-CORRECTED t-STATS\n")
print(f"{'setup':<20} {'naive t':>9} {'overlap-adj':>12} {'episode-adj':>13} {'bar':>7}")
bar = research.bonferroni_t(72)
for lb, th, h in CASES:
    z = research.zscore(px, lookback=lb)
    cond = z <= th
    es = research.event_study(px, cond, (h,)).iloc[0]
    days = cond.any(axis=1)
    d = days[days].index
    episodes = 1 + int((np.diff(d.values).astype("timedelta64[D]").astype(int) > 5).sum())
    t_overlap = es["t"] / np.sqrt(h)
    # deflate further by the ratio of raw events to independent episodes
    t_episode = es["t"] / np.sqrt(es["n_events"] / max(episodes, 1)) / np.sqrt(h)
    print(f"{str(lb)+'d <= '+str(th)+'sd':<20} {es['t']:>9.2f} {t_overlap:>12.2f} "
          f"{t_episode:>13.2f} {bar:>7.2f}")

print("\n\n3. REGIME DEPENDENCE -- split the sample\n")
mid = len(px) // 2
print(f"  first half  {px.index[0].date()}..{px.index[mid].date()}")
print(f"  second half {px.index[mid].date()}..{px.index[-1].date()}\n")
print(f"{'setup':<20} {'1st half edge':>14} {'2nd half edge':>14} {'retention':>11}")
for lb, th, h in CASES:
    z = research.zscore(px, lookback=lb)
    cond = z <= th
    e1 = research.event_study(px.iloc[:mid], cond.iloc[:mid], (h,)).iloc[0]["ann_edge"]
    e2 = research.event_study(px.iloc[mid:], cond.iloc[mid:], (h,)).iloc[0]["ann_edge"]
    ret = e2 / e1 if e1 and not np.isnan(e1) else np.nan
    print(f"{str(lb)+'d <= '+str(th)+'sd':<20} {e1*100:>13.1f}% {e2*100:>13.1f}% "
          f"{ret*100:>10.0f}%")

print("\n\n4. IS IT JUST MARKET BETA? edge vs the equal-weight index on the same days\n")
eq = px.pct_change().mean(axis=1)
print(f"{'setup':<20} {'name fwd ret':>13} {'index fwd ret':>14} {'excess':>9}")
for lb, th, h in CASES:
    z = research.zscore(px, lookback=lb)
    cond = z <= th
    fwd = research.forward_returns(px, h)
    idx_fwd = (1 + eq).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
    hit = fwd.where(cond).stack().dropna()
    # index return over the same dates the events occurred
    dates = fwd.where(cond).stack().dropna().index.get_level_values(0)
    idx_matched = idx_fwd.reindex(dates).dropna()
    print(f"{str(lb)+'d <= '+str(th)+'sd':<20} {hit.mean()*100:>12.2f}% "
          f"{idx_matched.mean()*100:>13.2f}% {(hit.mean()-idx_matched.mean())*100:>8.2f}%")
