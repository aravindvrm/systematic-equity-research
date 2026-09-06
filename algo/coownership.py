"""Common-institutional-ownership features (Anton & Polk 2014).

THE INDEX-FUND PROBLEM
----------------------
The naive connectedness measure is useless because Vanguard and BlackRock hold
essentially every name in the universe. If you include them, every stock is
maximally "connected" to every other stock, the matrix goes flat, and the peer
selection becomes arbitrary. Worse, it is mechanistically wrong: an index fund's
flows are proportional and predictable, and it does not liquidate a concentrated
book under redemption pressure. The price-pressure story requires ACTIVE managers
with concentrated portfolios.

So filers are filtered to those holding between MIN_POS and MAX_POS names of the
universe -- enough positions to define a network, few enough to be discretionary.

POINT-IN-TIME
-------------
13F has a 45-DAY reporting deadline. A quarter ending 31 Dec is public in
mid-February. Every connectedness matrix is therefore effective from
period_end + 45 days, and is held until the NEXT quarter's data becomes public.
Anchoring to period_end instead would be the same error that manufactured a
Form 4 signal earlier in this project -- it vanished once re-anchored to the
filing date.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "thirteenf"
REPORTING_LAG = pd.Timedelta(days=45)

MIN_POS = 5      # fewer than this cannot define a network
MAX_POS = 100    # more than this is an indexer, not a discretionary manager


def load_holdings(min_filers: int = 500) -> pd.DataFrame:
    """All quarters, with obviously-partial ones dropped.

    2013q2 is SEC's first structured quarter and contains ~69 filers against
    ~3,000 in every subsequent period. Treating it as real would say
    institutions held almost nothing in mid-2013.
    """
    frames = []
    for f in sorted(DATA_DIR.glob("holdings_*.parquet")):
        d = pd.read_parquet(f)
        if d["filer_cik"].nunique() < min_filers:
            continue
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def period_end(period: str) -> pd.Timestamp:
    y, q = int(period[:4]), int(period[5])
    return pd.Timestamp(year=y, month=3 * q, day=1) + pd.offsets.MonthEnd(0)


def connectedness(holdings_q: pd.DataFrame, cusips: list[str]) -> pd.DataFrame:
    """C[i,j] = sum_f w[f,i] * w[f,j], over ACTIVE filers only.

    w[f,i] is filer f's share of the total institutional value held in stock i,
    so C[i,j] reads as: the probability that a randomly chosen ownership dollar
    of i and one of j belong to the same manager. Diagonal is zeroed.
    """
    d = holdings_q[holdings_q["cusip"].isin(cusips)]
    n_pos = d.groupby("filer_cik")["cusip"].nunique()
    active = n_pos[(n_pos >= MIN_POS) & (n_pos <= MAX_POS)].index
    d = d[d["filer_cik"].isin(active)]
    if d.empty:
        return pd.DataFrame(0.0, index=cusips, columns=cusips)

    w = d.pivot_table(index="filer_cik", columns="cusip", values="value",
                      aggfunc="sum", fill_value=0.0)
    w = w.reindex(columns=cusips, fill_value=0.0)
    tot = w.sum(axis=0).replace(0, np.nan)
    w = w.div(tot, axis=1).fillna(0.0)          # each column sums to 1
    c = w.T.to_numpy() @ w.to_numpy()
    np.fill_diagonal(c, 0.0)
    return pd.DataFrame(c, index=cusips, columns=cusips)


def build_panels(prices: pd.DataFrame, cusip_map: pd.DataFrame,
                 holdings: pd.DataFrame):
    """Peer map, connectedness centrality -- both on the daily price index.

    Each quarter's matrix takes effect at period_end + 45 days and is held until
    the following quarter's becomes public.
    """
    cm = cusip_map.dropna(subset=["cusip"]).drop_duplicates("ticker")
    cm = cm[cm["ticker"].isin(prices.columns)]
    c2t = dict(zip(cm["cusip"], cm["ticker"]))
    cusips = list(cm["cusip"])

    peers = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns,
                         dtype=object)
    central = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)

    periods = sorted(holdings["period"].unique())
    for k, p in enumerate(periods):
        start = period_end(p) + REPORTING_LAG
        end = (period_end(periods[k + 1]) + REPORTING_LAG
               if k + 1 < len(periods) else prices.index[-1] + pd.Timedelta(days=1))
        mask = (prices.index >= start) & (prices.index < end)
        if not mask.any():
            continue
        c = connectedness(holdings[holdings["period"] == p], cusips)
        if c.to_numpy().sum() == 0:
            continue
        best = c.idxmax(axis=1)
        cen = c.sum(axis=1)
        for cu, t in c2t.items():
            if t not in peers.columns:
                continue
            bp = best.get(cu)
            peers.loc[mask, t] = c2t.get(bp, np.nan)
            central.loc[mask, t] = cen.get(cu, np.nan)
    return peers, central
