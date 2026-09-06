"""Historical bar loading, with an on-disk parquet cache.

yfinance is the default source for daily bars: free, decades of history, and
split/dividend adjusted. Polygon's free tier only reaches back 2 years, which is
too short for a walk-forward test, so it is not the default here.

SURVIVORSHIP BIAS WARNING
------------------------
yfinance only knows about tickers that still exist. Backtesting a universe you
selected *today* (e.g. "current S&P 500 members") silently deletes every company
that went to zero, and will inflate your results enormously. Use `ETF_UNIVERSE`
or another fixed, long-lived universe until you have a point-in-time constituent
source.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

# Broad, liquid, long-lived ETFs. No survivorship problem: these all still exist
# and none were selected because they did well.
ETF_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "XLE", "XLF"]

# Wider universe for breadth. All launched well before 2010 and still trading,
# so no survivorship problem -- but note these were chosen for ASSET-CLASS
# COVERAGE, not performance, which is what keeps the selection honest.
WIDE_UNIVERSE = [
    # US equity beta / size / style
    "SPY", "QQQ", "IWM", "MDY", "IWD", "IWF",
    # US sectors
    "XLE", "XLF", "XLK", "XLV", "XLP", "XLY", "XLI", "XLU", "XLB",
    # international
    "EFA", "EEM", "VGK", "EWJ", "FXI", "EWZ", "EWY",
    # fixed income
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "EMB",
    # real assets
    "GLD", "SLV", "DBC", "VNQ",
]

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(ticker: str, interval: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}_{interval}.parquet"


def load_bars(ticker: str, start: str = "2005-01-01", end: str | None = None,
              interval: str = "1d", refresh: bool = False) -> pd.DataFrame:
    """Load OHLCV bars for one ticker. Cached to parquet; pass refresh=True to refetch.

    Returns a DataFrame indexed by tz-naive date with columns open/high/low/close/volume.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, interval)

    if path.exists() and not refresh:
        df = pd.read_parquet(path)
    else:
        log.info("fetching %s from yfinance", ticker)
        raw = yf.download(ticker, start=start, end=end, interval=interval,
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"no data returned for {ticker}")
        # yfinance returns a MultiIndex column frame when given a list; flatten.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
        df = raw[[c for c in BAR_COLUMNS if c in raw.columns]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"

        # MERGE with any existing cache rather than replacing it. A refresh over
        # a short window (latest_prices fetches ~30 days) would otherwise
        # OVERWRITE years of history with one month -- silently, and for every
        # ticker the daily collector touches. New rows win on overlap.
        if path.exists():
            try:
                old = pd.read_parquet(path)
                df = pd.concat([old, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            except Exception as exc:  # noqa: BLE001 - a corrupt cache must not block a fetch
                log.warning("could not merge cache for %s: %s", ticker, exc)
        df.to_parquet(path)

    if end is not None:
        df = df.loc[:end]
    return df.loc[start:]


def load_panel(tickers: list[str], start: str = "2005-01-01", end: str | None = None,
               field: str = "close", refresh: bool = False) -> pd.DataFrame:
    """Load one field across many tickers into a date x ticker frame.

    Columns are aligned on the union of dates and forward-filled *within* each
    series only where that series has already started -- no backfill, so a
    ticker that did not exist yet stays NaN rather than inheriting a later price.
    """
    series = {}
    for t in tickers:
        try:
            series[t] = load_bars(t, start=start, end=end, refresh=refresh)[field]
        except Exception as exc:  # noqa: BLE001 - one bad ticker shouldn't kill the panel
            log.warning("skipping %s: %s", t, exc)
    if not series:
        raise ValueError("no tickers loaded")
    panel = pd.DataFrame(series).sort_index()
    return panel.ffill().where(panel.ffill().notna() & panel.bfill().notna())


def latest_prices(tickers: list[str], lookback_days: int = 30,
                  refresh: bool = True) -> pd.Series:
    """Most recent POPULATED close per ticker.

    Never use `load_panel(...).iloc[-1]` for this. During the trading day the
    final index row is today's unfinished bar and is all NaN, so .iloc[-1]
    returns silent NaNs rather than raising.
    """
    start = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    panel = load_panel(tickers, start=start, refresh=refresh)
    populated = panel.dropna(how="all")
    if populated.empty:
        raise ValueError("no populated price rows in the lookback window")
    return populated.ffill().iloc[-1]
