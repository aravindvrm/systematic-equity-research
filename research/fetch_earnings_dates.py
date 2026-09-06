"""Earnings announcement dates from 8-K Item 2.02, for the whole universe.

No documents are fetched -- item codes come inline from the submissions
endpoint, so this is ~2 requests per name rather than ~90.
"""
import time
from pathlib import Path

import pandas as pd

from algo import edgar

OUT = Path("data/edgar/events")
OUT.mkdir(parents=True, exist_ok=True)

uni = pd.read_parquet("data/collection_universe.parquet").dropna(subset=["cik"])
print(f"{len(uni)} names", flush=True)

rows, fails = [], []
for i, r in enumerate(uni.itertuples(), 1):
    cik = str(int(r.cik))
    p = OUT / f"{cik}.parquet"
    try:
        if p.exists():
            e = pd.read_parquet(p)
        else:
            e = edgar.event_index(cik)
            if len(e):
                e.to_parquet(p)
            time.sleep(edgar.SLEEP)
        if e.empty:
            fails.append((r.ticker, "no 8-K filings"))
            continue
        ea = e[edgar.has_item(e["items"], "2.02")].copy()
        ea["ticker"] = r.ticker
        rows.append(ea)
        if i % 25 == 0:
            print(f"  [{i}/{len(uni)}] {r.ticker}: {len(ea)} earnings 8-Ks", flush=True)
    except Exception as ex:
        fails.append((r.ticker, f"{type(ex).__name__}: {ex}"))
        print(f"  [{i}/{len(uni)}] {r.ticker}: FAILED {ex}", flush=True)

E = pd.concat(rows, ignore_index=True)
E.to_parquet("data/edgar/earnings_dates.parquet")
print(f"\n{len(E):,} earnings announcements, {E.ticker.nunique()} names, "
      f"{E.filing_date.min().date()}..{E.filing_date.max().date()}")
print(f"failed: {len(fails)}")
for t, e in fails[:15]:
    print(f"   {t}: {e}")
