"""Post-earnings announcement drift.

The most replicated anomaly in finance (Ball & Brown 1968 onward): stocks that
surprise on earnings keep drifting in the direction of the surprise for weeks.

MEASURING SURPRISE WITHOUT PAYING FOR ESTIMATES
-----------------------------------------------
The textbook construction needs analyst consensus, which costs money. The price
REACTION is a free proxy and is arguably better: it is the market's own
assessment of the surprise, and it already nets out whatever was expected. This
is "earnings momentum" and it is separately documented.

THE ANNOUNCEMENT-TIMING TRAP
----------------------------
An 8-K filed on day 0 may report earnings released BEFORE the open that day (so
the reaction is day 0) or AFTER the close (reaction is day +1). The filing date
alone does not say which. A [0, +1] window captures both cases; using day 0 alone
would miss half the reactions entirely, and using day 0 as though it were always
the reaction day would mix pre- and post-announcement returns.

Consequence for point-in-time: the signal is only fully known at the CLOSE of
day +1, so it is tradeable from day +2. Anything earlier is lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def abnormal_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Return minus the cross-sectional mean -- a cheap market adjustment.

    Beta-adjusting would be more precise, but a rolling beta estimated on the
    days around an earnings jump is itself contaminated by the jump.
    """
    r = prices.pct_change()
    return r.sub(r.mean(axis=1), axis=0)


def reactions(prices: pd.DataFrame, events: pd.DataFrame,
              window: int = 2, vol_window: int = 252) -> pd.DataFrame:
    """Cumulative abnormal return over [0, +1] for each announcement.

    Returns one row per event with:
        car  -- raw cumulative abnormal return across the window
        sue  -- car standardized by the stock's own trailing abnormal-return
                volatility. This is what makes events comparable ACROSS names:
                a 3% jump means something different for a utility than for a
                biotech, and ranking raw CARs cross-sectionally would just rank
                volatility.
        available -- the first date the signal may be traded on (day +2)
    """
    ar = abnormal_returns(prices)
    sd = ar.rolling(vol_window, min_periods=126).std()
    idx = prices.index
    out = []
    for e in events.itertuples():
        t = e.ticker
        if t not in prices.columns:
            continue
        pos = idx.searchsorted(e.filing_date)      # first trading day >= filing
        if pos >= len(idx) - (window + 1):
            continue
        seg = ar[t].iloc[pos:pos + window]
        if seg.isna().any():
            continue
        car = float(seg.sum())
        s = sd[t].iloc[pos - 1] if pos > 0 else np.nan
        out.append(dict(ticker=t, filing_date=e.filing_date,
                        event_date=idx[pos],
                        available=idx[min(pos + window, len(idx) - 1)],
                        car=car,
                        sue=car / s if s and np.isfinite(s) and s > 0 else np.nan))
    return pd.DataFrame(out)


def to_daily(react: pd.DataFrame, index: pd.DatetimeIndex,
             columns, col: str = "sue", hold_days: int = 63) -> pd.DataFrame:
    """Spread event values onto a daily panel.

    Effective from `available` (day +2), held for hold_days or until this name's
    next announcement, whichever comes first.
    """
    out = pd.DataFrame(np.nan, index=index, columns=list(columns))
    for t, g in react.groupby("ticker"):
        if t not in out.columns:
            continue
        g = g.sort_values("available")
        av = g["available"].to_numpy()
        vals = g[col].to_numpy()
        for j in range(len(g)):
            start = av[j]
            nxt = av[j + 1] if j + 1 < len(g) else index[-1] + pd.Timedelta(days=1)
            end = min(nxt, start + pd.Timedelta(days=hold_days))
            m = (index >= start) & (index < end)
            if m.any():
                out.loc[m, t] = vals[j]
    return out
