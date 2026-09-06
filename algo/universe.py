"""Universe definition — a RULE, not a list of names I liked.

Two distinct universes, and conflating them was the mistake:

  COLLECTION universe  -- deliberately WIDE. What we snapshot options for daily.
                          Collection is IRREVERSIBLE: a name not collected today
                          can never be backfilled. Narrowing later is free;
                          widening retroactively is impossible. So over-collect.

  TRADING universe     -- deliberately NARROW. Selected FROM the collection set
                          by the spec's rule (diversification-optimized, sector
                          capped, liquidity floored). Can be re-derived any time.

> The 24 names previously hardcoded in collect_options.py were hand-picked by
> Claude in a single message with no rule applied. Pick dispersion was measured
> at 0.75-1.41 Sharpe across random draws, and that hand-picked set landed at the
> 44th percentile. This module exists so that never silently happens again.

SURVIVORSHIP: CANDIDATE_POOL is today's large-cap membership. Names that fell out
of the index (or failed) are absent, which biases any backtest optimistic --
measured earlier at roughly +0.23 Sharpe. Stated here so it is never implicit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Non-equity sleeves. Chosen for ASSET-CLASS COVERAGE, not performance -- the
# only picks in this project with a clean justification.
DIVERSIFIERS = {
    "TLT": "20+yr Treasuries -- duration",
    "IEF": "7-10yr Treasuries -- duration, lower vol",
    "SHY": "1-3yr Treasuries -- near-cash",
    "TIP": "inflation-linked -- real rates",
    "LQD": "investment grade -- credit",
    "HYG": "high yield -- credit, equity-correlated",
    "GLD": "gold -- real asset / crisis hedge",
    "SLV": "silver -- real asset, higher beta",
    "DBC": "broad commodities -- inflation",
    "VNQ": "US REITs -- real assets / rates",
    "EFA": "developed ex-US -- geography",
    "EEM": "emerging markets -- geography",
}

CONSTITUENTS_PATH = Path(__file__).resolve().parent.parent / "data" / "sp500_constituents.parquet"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_constituents(refresh: bool = False) -> pd.DataFrame:
    """Current S&P 500 membership: symbol, name, GICS sector, CIK, date added.

    Sourced from Wikipedia, which tracks the index reliably and -- usefully --
    carries the SEC CIK for each company, which is exactly what Form 4 lookups
    need. Cached to parquet so a universe decision is reproducible from a dated
    snapshot rather than from whatever the page says today.

    STILL SURVIVORSHIP-BIASED: this is TODAY's membership. Names removed after
    failing are absent. Point-in-time membership needs the change history plus
    prices for delisted names, and the latter is the part that costs money.
    """
    import io

    import requests

    if CONSTITUENTS_PATH.exists() and not refresh:
        return pd.read_parquet(CONSTITUENTS_PATH)

    resp = requests.get(WIKI_URL, headers={"User-Agent": "algo-research/0.1"}, timeout=30)
    resp.raise_for_status()
    df = pd.read_html(io.StringIO(resp.text))[0]
    df = df.rename(columns={"Symbol": "ticker", "Security": "name",
                            "GICS Sector": "sector", "GICS Sub-Industry": "industry",
                            "CIK": "cik", "Date added": "date_added"})
    # yfinance uses '-' where the index uses '.' (BRK.B -> BRK-B)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df["cik"] = df["cik"].astype("int64")
    df["fetched"] = pd.Timestamp.utcnow().tz_localize(None)
    CONSTITUENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CONSTITUENTS_PATH, index=False)
    return df


def screen_liquidity(tickers: list[str], min_dollar_volume: float = 5e7,
                     min_history_days: int = 1000,
                     lookback: str = "2020-01-01") -> pd.DataFrame:
    """Rank names by median daily dollar volume, dropping thin or short-history ones.

    Liquidity is the right screen for a COLLECTION universe: options on illiquid
    underlyings have spreads wide enough that any signal derived from them is
    unusable, so collecting them wastes time and adds noise.
    """
    import yfinance as yf

    raw = yf.download(tickers, start=lookback, progress=False,
                      auto_adjust=True, threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    vol = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]]

    rows = []
    for t in close.columns:
        px, vl = close[t].dropna(), vol[t].dropna()
        if len(px) < min_history_days:
            continue
        # Median over the FULL window, not the trailing 120 days. A recent-volume
        # screen tilts the universe toward whatever is currently hot (LITE, PLTR
        # and MRVL outranked most mega-caps on a 120-day median), which smuggles
        # a momentum tilt into what is supposed to be a neutral liquidity filter.
        dv = float((px * vl).median())
        if not np.isfinite(dv) or dv < min_dollar_volume:
            continue
        rows.append(dict(ticker=t, dollar_volume=dv, history_days=len(px)))
    return pd.DataFrame(rows).sort_values("dollar_volume", ascending=False)


SELECTION_PATH = Path(__file__).resolve().parent.parent / "data" / "collection_universe.parquet"


def build_collection_universe(n_equities: int = 250, per_sector_cap: int = 30,
                              refresh: bool = False) -> pd.DataFrame:
    """Select and CACHE the wide set we snapshot options for.

    Rule: S&P 500 members -> liquidity screen -> top `n_equities` by full-window
    median dollar volume, capped at `per_sector_cap` per GICS sector -> plus every
    diversifier.

    The sector cap matters. Ranking on dollar volume alone hands the universe to
    Information Technology, because turnover concentrates there. The cap enforces
    breadth across the economy instead of my guesswork about sector weights.

    Cached, because re-screening 503 tickers on every daily collector run would
    hammer the data source for a list that changes a few times a year.
    """
    cons = fetch_constituents(refresh=refresh)
    liq = screen_liquidity(cons["ticker"].tolist())
    merged = liq.merge(cons[["ticker", "sector", "cik", "name"]], on="ticker", how="left")

    picked, counts = [], {}
    for _, r in merged.iterrows():
        sec = r["sector"]
        if counts.get(sec, 0) >= per_sector_cap:
            continue
        picked.append(r)
        counts[sec] = counts.get(sec, 0) + 1
        if len(picked) >= n_equities:
            break

    eq = pd.DataFrame(picked)[["ticker", "name", "sector", "cik", "dollar_volume"]]
    div = pd.DataFrame([{"ticker": t, "name": d, "sector": "Diversifier",
                         "cik": pd.NA, "dollar_volume": pd.NA}
                        for t, d in DIVERSIFIERS.items()])
    out = pd.concat([eq, div], ignore_index=True)
    out["selected_on"] = pd.Timestamp.utcnow().tz_localize(None)
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SELECTION_PATH, index=False)
    return out


def collection_universe(rebuild: bool = False) -> list[str]:
    """Tickers to snapshot options for. Reads the cached selection."""
    if rebuild or not SELECTION_PATH.exists():
        build_collection_universe()
    return sorted(pd.read_parquet(SELECTION_PATH)["ticker"].tolist())


def collection_table() -> pd.DataFrame:
    """The cached selection with sector, CIK and the liquidity that earned it."""
    if not SELECTION_PATH.exists():
        build_collection_universe()
    return pd.read_parquet(SELECTION_PATH)


def market_proxy(returns: pd.DataFrame, cap_weighted: bool = True) -> pd.Series:
    """The series everything else is measured 'uncorrelated to'.

    Use SPY -- an actual CAP-WEIGHTED index. The equal-weighted average of the
    candidate pool tilts the proxy toward mid-caps, which makes mega-caps look
    artificially uncorrelated and drags selection toward the less liquid half of
    the pool. If SPY is unavailable, fall back to the equal-weight average and
    accept that distortion.
    """
    if cap_weighted and "SPY" in returns.columns:
        return returns["SPY"]
    return returns.mean(axis=1)


def _effective_bets(returns: pd.DataFrame) -> float:
    corr = returns.corr().to_numpy()
    lam = np.linalg.eigvalsh(corr)
    lam = lam[lam > 0]
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def trading_universe(prices: pd.DataFrame, n: int = 30,
                     must_include: list[str] | None = None,
                     method: str = "cluster",
                     sector_floor: int = 1,
                     sectors: pd.Series | None = None) -> list[str]:
    """Select `n` names by correlation STRUCTURE. Never looks at returns.

    method="cluster" (default): hierarchical clustering on correlation distance,
    taking the least-correlated member of each cluster. Beat greedy in 4/5
    out-of-sample splits, and -- more importantly -- IMPROVED out of sample
    (+2.15 mean) where greedy DEGRADED (-0.91). Greedy fits the training
    correlation matrix tightly and that fit does not transfer; clustering finds
    coarser structure that does.

    method="greedy": the previous sequential maximizer. Kept for comparison.

    sector_floor: minimum names per GICS sector. Costs ~0.23 effective bets and
    takes coverage from 9 sectors to all 11 -- the cheapest constraint tested.
    Beta and liquidity floors were far more expensive (a $1B/day floor cost 6.2
    effective bets) and are deliberately not offered here.
    """
    if method == "cluster":
        return _cluster_select(prices, n, must_include, sector_floor, sectors)
    return _greedy_select(prices, n, must_include)


def _cluster_select(prices, n, must_include=None, sector_floor=0, sectors=None):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    returns = prices.pct_change().dropna()
    forced = [t for t in (must_include or []) if t in returns.columns]
    chosen = list(forced)

    if sector_floor and sectors is not None:
        for sec in sorted(set(sectors.dropna()) - {"Diversifier"}):
            cands = [c for c in returns.columns
                     if c not in chosen and sectors.get(c) == sec]
            if not cands:
                continue
            # least market-entangled representative of the sector
            pick = returns[cands].corrwith(returns.mean(axis=1)).idxmin()
            chosen.append(pick)
            if len(chosen) >= n:
                break

    remaining = [c for c in returns.columns if c not in chosen]
    slots = n - len(chosen)
    if slots > 0 and remaining:
        sub = returns[remaining]
        corr = sub.corr().to_numpy()
        dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
        np.fill_diagonal(dist, 0.0)
        link = linkage(squareform(dist, checks=False), method="average")
        labels = fcluster(link, t=slots, criterion="maxclust")
        for k in np.unique(labels):
            members = sub.columns[labels == k]
            chosen.append(sub[members].corr().mean().idxmin())
    return sorted(chosen[:n])


def _greedy_select(prices: pd.DataFrame, n: int = 30,
                   must_include: list[str] | None = None) -> list[str]:
    """Select `n` names from `prices` by greedily maximizing effective bets.

    Diversification is the ONLY selection objective that survived out-of-sample
    testing: it delivered the breadth it targeted in 8/8 splits at the 94th
    percentile. Selecting on past RETURNS did not -- it was no better than random
    across the same eight splits. So this never looks at performance.

    Note what this buys and what it does not: more effective bets raise the IR
    CEILING (IR = IC x sqrt(BR)). They do not create IC. With no signal, a better
    container holds the same nothing.
    """
    returns = prices.pct_change().dropna()
    chosen = [t for t in (must_include or []) if t in returns.columns]
    remaining = [c for c in returns.columns if c not in chosen]

    while len(chosen) < n and remaining:
        best, best_v = remaining[0], -np.inf
        for t in remaining:
            v = _effective_bets(returns[chosen + [t]])
            if np.isfinite(v) and v > best_v:
                best, best_v = t, v
        chosen.append(best)
        remaining.remove(best)
    return sorted(chosen)
