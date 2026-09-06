"""State-shaped insider signals: the CONDITION of insider behaviour, not events.

The event formulation ("a buy happened, what follows?") died on disclosure
latency -- the whole effect was priced within a day of filing. A state signal
sidesteps that entirely: you are not racing a filing, you are tracking an
accumulating condition. Being a day late costs nothing.

It also fixes breadth. Events gave 42/year in a 30-name universe. A state score
exists for every name on every day.

POINT-IN-TIME DISCIPLINE
------------------------
Every feature is built from filings AVAILABLE BY THAT DATE -- keyed on
filing_date, never trans_date. Using the transaction date would rebuild the same
lookahead that inflated the event study from t=1.45 to t=5.89.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_of_panel(events: pd.DataFrame, dates: pd.DatetimeIndex,
                 tickers: list[str], window_days: int,
                 agg: str = "sum", col: str = "value") -> pd.DataFrame:
    """Rolling aggregate of `col` over a trailing window, keyed on FILING date."""
    ev = events.dropna(subset=["filing_date"]).copy()
    ev["d"] = ev["filing_date"].dt.normalize()
    if agg == "nunique":
        g = ev.groupby(["d", "ticker"]).owner_cik.nunique()
    else:
        g = ev.groupby(["d", "ticker"])[col].sum()
    wide = g.unstack("ticker").reindex(index=dates, columns=tickers).fillna(0.0)
    return wide.rolling(window_days, min_periods=1).sum()


def build(form4: pd.DataFrame, prices: pd.DataFrame,
          window_days: int = 126) -> dict[str, pd.DataFrame]:
    """Construct state features on the price panel's index.

    window_days=126 is ~6 months, the horizon at which aggregate insider activity
    is documented to matter (Lakonishok & Lee), and long enough that a single
    filing barely moves the score.
    """
    dates, tickers = prices.index, list(prices.columns)
    f = form4[form4.ticker.isin(tickers)].copy()

    buys = f[(f.code == "P") & (f.acq_disp == "A")]
    sells = f[(f.code == "S") & (f.acq_disp == "D")]

    buy_val = _as_of_panel(buys, dates, tickers, window_days)
    sell_val = _as_of_panel(sells, dates, tickers, window_days)
    n_buyers = _as_of_panel(buys, dates, tickers, window_days, agg="nunique")
    n_sellers = _as_of_panel(sells, dates, tickers, window_days, agg="nunique")

    # Scale by dollar volume so a $1M buy in a small name outranks one in a mega-cap.
    dollar_vol = prices.rolling(60).mean() * 1e6   # crude scale; ranks are what matter
    denom = (buy_val + sell_val).replace(0, np.nan)

    out = {
        # net buying pressure, scale-normalized
        "ins_net_value": ((buy_val - sell_val) / dollar_vol).fillna(0.0),
        # what fraction of insider ACTIVITY was buying -- direction, not size
        "ins_buy_share": (buy_val / denom).fillna(0.5) - 0.5,
        # breadth of conviction: how many DIFFERENT insiders bought
        "ins_n_buyers": n_buyers,
        # buyer/seller balance by headcount
        "ins_buyer_ratio": (n_buyers / (n_buyers + n_sellers).replace(0, np.nan)).fillna(0.5) - 0.5,
        # any buying at all -- the sparse binary version, for comparison
        "ins_any_buy": (buy_val > 0).astype(float),
        # net selling pressure, sign-flipped so higher = more attractive
        "ins_net_sell": (-(sell_val) / dollar_vol).fillna(0.0),
    }
    return out
