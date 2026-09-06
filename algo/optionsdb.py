"""Build a local historical options database by snapshotting live chains.

THE LIMITATION, STATED UP FRONT
-------------------------------
This cannot recover the past. Starting today means zero coverage of 2020 or
2022, so the data will only ever describe the regime you collected in. For
walk-forward validation you need roughly 3 years. Polygon sells ~12 years of
history for about $79/month.

Collect anyway: the marginal cost is a cron job and a few hundred MB a year, and
the option value is real. Even if you later buy history, your own snapshots
validate the vendor's data and cover the present.

DATA QUALITY NOTES
------------------
- Use MID = (bid+ask)/2, never `lastPrice`. Options trade rarely; a last price
  can be hours or days stale. Observed on MSFT: last 191.58 against a
  178.40/182.40 market.
- Yahoo's `impliedVolatility` is their own calculation and quality varies.
  Bid/ask are stored so IV can be recomputed later.
- Snapshot at a CONSISTENT time each day. A 10:00 chain and a 15:45 chain are
  not comparable, and mixing them injects noise that looks like signal.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DB_DIR = Path(__file__).resolve().parent.parent / "data" / "options"
RAW_DIR = DB_DIR / "raw"
SUMMARY_PATH = DB_DIR / "daily_summary.parquet"

KEEP_COLS = ["contractSymbol", "strike", "bid", "ask", "lastPrice", "volume",
             "openInterest", "impliedVolatility", "inTheMoney"]


def _chain_for(ticker: str, max_expiries: int = 12) -> pd.DataFrame:
    import yfinance as yf

    t = yf.Ticker(ticker)
    exps = list(t.options)[:max_expiries]
    frames = []
    for e in exps:
        try:
            ch = t.option_chain(e)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s %s: %s", ticker, e, exc)
            continue
        for kind, df in (("C", ch.calls), ("P", ch.puts)):
            if df is None or df.empty:
                continue
            d = df[[c for c in KEEP_COLS if c in df.columns]].copy()
            d["type"] = kind
            d["expiry"] = pd.to_datetime(e)
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["underlying"] = ticker
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    return out


def market_session(ts: datetime | None = None) -> str:
    """Which session a snapshot was taken in, US Eastern.

    Matters for data quality: during RTH the chain carries live two-sided
    quotes. After the close, bids and asks go stale and wide, though volume and
    open interest are final. Tag every snapshot so the difference can be
    filtered later rather than silently averaged together.
    """
    from zoneinfo import ZoneInfo

    ts = ts or datetime.now(timezone.utc)
    et = ts.astimezone(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return "WEEKEND"
    mins = et.hour * 60 + et.minute
    if mins < 9 * 60 + 30:
        return "PRE"
    if mins <= 16 * 60:
        return "RTH"
    return "POST"


def have_good_snapshot_today(ts: datetime | None = None) -> bool:
    """True if today's file already exists AND was captured during RTH.

    Lets a catch-up run skip work when the scheduled run already succeeded, but
    still allows a PRE or POST snapshot to be replaced by a better RTH one.
    """
    ts = ts or datetime.now(timezone.utc)
    from zoneinfo import ZoneInfo
    date = ts.astimezone(ZoneInfo("America/New_York")).date()
    path = RAW_DIR / f"{date.isoformat()}.parquet"
    if not path.exists():
        return False
    try:
        existing = pd.read_parquet(path, columns=["session"])
    except Exception:  # noqa: BLE001 - older files predate the column
        return True
    return bool((existing["session"] == "RTH").any())


def snapshot(tickers: list[str], max_expiries: int = 12,
             spot: pd.Series | None = None) -> pd.DataFrame:
    """Fetch chains for `tickers` and write one dated parquet file.

    Idempotent: re-running on the same date overwrites that date's file rather
    than appending duplicates.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    from zoneinfo import ZoneInfo

    stamp = datetime.now(timezone.utc)
    # Use the EASTERN date. A 20:00 ET snapshot is already the next day in UTC,
    # which would file Monday evening's chain under Tuesday.
    date = stamp.astimezone(ZoneInfo("America/New_York")).date()

    frames = []
    for t in tickers:
        df = _chain_for(t, max_expiries)
        if df.empty:
            log.warning("no chain for %s", t)
            continue
        frames.append(df)
    if not frames:
        raise RuntimeError("no chains fetched for any ticker")

    all_df = pd.concat(frames, ignore_index=True)
    all_df["snapshot_utc"] = stamp
    all_df["session"] = market_session(stamp)
    all_df["date"] = pd.Timestamp(date)
    if spot is not None:
        # A price panel's LAST ROW is often today's unfinished bar: present in
        # the index, entirely NaN. Taking .iloc[-1] without checking silently
        # produces NaN spot -> NaN moneyness -> every moneyness-filtered metric
        # comes back empty, while unfiltered metrics still look fine. Fail loudly.
        missing = [t for t in all_df["underlying"].unique()
                   if t not in spot.index or pd.isna(spot.get(t))]
        if missing:
            raise ValueError(
                f"spot price missing/NaN for {len(missing)} underlyings: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
                "Use the last POPULATED row (dropna), not .iloc[-1].")
        all_df["spot"] = all_df["underlying"].map(spot)
        all_df["moneyness"] = all_df["strike"] / all_df["spot"]

    path = RAW_DIR / f"{date.isoformat()}.parquet"
    all_df.to_parquet(path, index=False)
    log.info("wrote %s rows to %s", len(all_df), path.name)
    return all_df


def summarize(chain: pd.DataFrame) -> pd.DataFrame:
    """Collapse a raw chain into per-underlying daily signal inputs.

    These are the quantities the published single-name options signals are built
    from: at-the-money IV, put/call skew, term structure, and flow ratios.
    """
    if "moneyness" not in chain.columns:
        raise ValueError("summarize needs `moneyness` -- pass spot to snapshot()")

    rows = []
    for (u, d), g in chain.groupby(["underlying", "date"]):
        near = g[(g.moneyness.between(0.95, 1.05)) & (g.mid > 0)]
        calls, puts = g[g.type == "C"], g[g.type == "P"]
        atm_c = near[near.type == "C"]["impliedVolatility"]
        atm_p = near[near.type == "P"]["impliedVolatility"]

        # 25-delta-ish skew proxy: OTM put IV minus OTM call IV
        otm_p = g[(g.type == "P") & (g.moneyness.between(0.88, 0.95))]["impliedVolatility"]
        otm_c = g[(g.type == "C") & (g.moneyness.between(1.05, 1.12))]["impliedVolatility"]

        exps = sorted(g.expiry.unique())
        front = g[g.expiry == exps[0]]["impliedVolatility"].median() if exps else np.nan
        back = g[g.expiry == exps[-1]]["impliedVolatility"].median() if len(exps) > 1 else np.nan

        rows.append(dict(
            underlying=u, date=d,
            atm_iv=float(pd.concat([atm_c, atm_p]).median()) if len(near) else np.nan,
            iv_spread=float(atm_c.median() - atm_p.median()) if len(atm_c) and len(atm_p) else np.nan,
            skew=float(otm_p.median() - otm_c.median()) if len(otm_p) and len(otm_c) else np.nan,
            term_slope=float(back - front) if np.isfinite(front) and np.isfinite(back) else np.nan,
            pc_volume=float(puts.volume.sum() / max(calls.volume.sum(), 1)),
            pc_oi=float(puts.openInterest.sum() / max(calls.openInterest.sum(), 1)),
            total_oi=float(g.openInterest.sum()),
            n_contracts=int(len(g)),
        ))
    return pd.DataFrame(rows)


def append_summary(summary: pd.DataFrame) -> None:
    """Append to the rolling summary table, de-duplicating on (underlying,date)."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    if SUMMARY_PATH.exists():
        prev = pd.read_parquet(SUMMARY_PATH)
        summary = pd.concat([prev, summary], ignore_index=True)
    summary = summary.drop_duplicates(subset=["underlying", "date"], keep="last")
    summary.to_parquet(SUMMARY_PATH, index=False)


def coverage() -> pd.DataFrame:
    """What has actually been collected, and where the gaps are.

    Gaps matter: a signal computed across a hole will silently compare
    non-adjacent dates. Check this before trusting the database.
    """
    if not SUMMARY_PATH.exists():
        return pd.DataFrame()
    s = pd.read_parquet(SUMMARY_PATH)
    out = s.groupby("underlying")["date"].agg(["min", "max", "count"])
    span = (out["max"] - out["min"]).dt.days
    out["expected_bars"] = (span / 7 * 5).round().astype(int).clip(lower=1)
    out["coverage_pct"] = (out["count"] / out["expected_bars"] * 100).clip(upper=100).round(1)
    return out
