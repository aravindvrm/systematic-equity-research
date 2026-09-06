"""The value / profitability / investment anomalies — the missing three categories.

Built from SEC filing facts (algo/fundamentals.py). Each is a documented
cross-sectional predictor with a canonical reference:

    book_to_market   value              Fama & French (1992)
    earnings_yield   value              Basu (1977)
    roe              profitability      Haugen & Baker (1996)
    gross_profit     profitability      Novy-Marx (2013) -- (rev-cogs)/assets
    asset_growth     investment         Cooper, Gulen & Schill (2008), NEGATED
    share_issuance   investment         Pontiff & Woodgate (2008), NEGATED
    roa              profitability
    leverage         intangibles/risk   NEGATED

SIGN CONVENTION: every feature is oriented so that HIGH = expected
outperformance, matching the rest of the codebase. Asset growth, share issuance
and leverage are therefore negated -- firms that grow assets fast, issue shares,
and lever up have historically UNDERperformed.

POINT-IN-TIME
-------------
Everything is keyed on the FILING date, never the fiscal period end. A quarter
ending 31 December is not public until the 10-K is filed 30-75 days later.
Values take effect at filed + 1 day and are held until the next filing.

TRAILING TWELVE MONTHS
----------------------
Flow items (revenue, income) are reported either quarterly (qtrs=1) or annually
(qtrs=4). Mixing them makes a ratio meaningless -- a quarterly income over
annual equity is a quarter of the true ROE. `_ttm` reconstructs a consistent
twelve-month figure per company before any ratio is formed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ttm(g: pd.DataFrame, col: str) -> pd.Series:
    """Trailing-twelve-month flow, from whichever periodicity was reported."""
    v = g[col]
    # A row already covering 4 quarters is TTM as-is; quarterly rows are summed
    # over the trailing four filings.
    is_annual = g["form"].eq("10-K")
    q = v.where(~is_annual)
    roll = q.rolling(4, min_periods=3).sum()
    return v.where(is_annual, roll)


def build_facts(facts: pd.DataFrame) -> pd.DataFrame:
    """Add TTM flows and year-ago balances, per company, in filing order."""
    out = []
    for cik, g in facts.sort_values("filed").groupby("cik", sort=False):
        g = g.copy()
        for c in ("revenue", "cogs", "net_income", "op_income", "cfo"):
            if c in g:
                g[f"{c}_ttm"] = _ttm(g, c)
        # Year-ago balance-sheet values: 4 filings back if quarterly.
        for c in ("assets", "equity", "shares"):
            if c in g:
                g[f"{c}_lag4"] = g[c].shift(4)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def compute(facts: pd.DataFrame) -> pd.DataFrame:
    """Per-filing anomaly values. Ratios only where the denominator is sane."""
    f = build_facts(facts).copy()

    def safe(num, den, pos_den=True):
        d = den.replace(0, np.nan)
        if pos_den:
            d = d.where(d > 0)
        return num / d

    f["roe"] = safe(f["net_income_ttm"], f["equity"])
    f["roa"] = safe(f["net_income_ttm"], f["assets"])
    f["gross_profit"] = safe(f["revenue_ttm"] - f["cogs_ttm"], f["assets"])
    f["asset_growth"] = -(safe(f["assets"], f["assets_lag4"]) - 1.0)
    f["share_issuance"] = -(safe(f["shares"], f["shares_lag4"]) - 1.0)
    f["leverage"] = -safe(f["liabilities"], f["assets"])
    # book_to_market and earnings_yield need a market cap, added in to_panel
    # where prices are available.
    f["book_equity"] = f["equity"].where(f["equity"] > 0)
    f["earnings_ttm"] = f["net_income_ttm"]
    f["shares_out"] = f["shares"].where(f["shares"] > 0)
    return f


FEATURES = ["roe", "roa", "gross_profit", "asset_growth", "share_issuance",
            "leverage"]
PRICE_FEATURES = ["book_to_market", "earnings_yield"]


def to_panel(vals: pd.DataFrame, index: pd.DatetimeIndex, cik_to_ticker: dict,
             col: str, lag_days: int = 1) -> pd.DataFrame:
    """Spread per-filing values onto a daily panel, effective at filed+lag."""
    cols = sorted(set(cik_to_ticker.values()))
    out = pd.DataFrame(np.nan, index=index, columns=cols)
    for cik, g in vals.groupby("cik", sort=False):
        t = cik_to_ticker.get(str(cik))
        if t is None or t not in out.columns:
            continue
        g = g.sort_values("filed")
        d = g["filed"].to_numpy()
        v = g[col].to_numpy()
        for j in range(len(g)):
            if not np.isfinite(v[j]):
                continue
            start = pd.Timestamp(d[j]) + pd.Timedelta(days=lag_days)
            end = (pd.Timestamp(d[j + 1]) + pd.Timedelta(days=lag_days)
                   if j + 1 < len(g) else index[-1] + pd.Timedelta(days=1))
            m = (index >= start) & (index < end)
            if m.any():
                out.loc[m, t] = v[j]
    return out


def market_ratios(vals: pd.DataFrame, prices: pd.DataFrame,
                  cik_to_ticker: dict) -> dict[str, pd.DataFrame]:
    """book_to_market and earnings_yield, which need a live market cap.

    Book equity and share count are as-of the last filing (stale between
    filings, which is correct and unavoidable); the PRICE is current, so the
    ratio moves daily as it should.
    """
    be = to_panel(vals, prices.index, cik_to_ticker, "book_equity")
    ea = to_panel(vals, prices.index, cik_to_ticker, "earnings_ttm")
    sh = to_panel(vals, prices.index, cik_to_ticker, "shares_out")
    mcap = (prices.reindex(columns=sh.columns) * sh).replace(0, np.nan)
    return {"book_to_market": be / mcap.where(mcap > 0),
            "earnings_yield": ea / mcap.where(mcap > 0)}
