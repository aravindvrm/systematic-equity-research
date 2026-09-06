"""Bulk-fetch EDGAR filing text for the universe.

Downloading is the slow, rate-limited, failure-prone half; measurement is cheap
and will be redone many times. So this script does nothing but fill the text
cache, and every measure is computed separately from what it leaves on disk.

Usage:
    fetch_filings.py [n_names] [start_date]

Following the Form 4 lesson, per-name failures are COUNTED and reported rather
than silently swallowed -- a name with no filings and a name whose fetch failed
look identical downstream, and the second one poisons a year-over-year measure.
"""
import sys
import time
from pathlib import Path

import pandas as pd

from algo import edgar

N = int(sys.argv[1]) if len(sys.argv) > 1 else 0
START = sys.argv[2] if len(sys.argv) > 2 else "2005-01-01"

uni = pd.read_parquet("data/collection_universe.parquet")
uni = uni.dropna(subset=["cik"])
if N:
    uni = uni.head(N)
print(f"{len(uni)} names, filings from {START}", flush=True)

IDX = Path("data/edgar/index")
IDX.mkdir(parents=True, exist_ok=True)

rows, fails = [], []
for i, r in enumerate(uni.itertuples(), 1):
    cik = str(int(r.cik))
    ip = IDX / f"{cik}.parquet"
    try:
        if ip.exists():
            idx = pd.read_parquet(ip)
        else:
            idx = edgar.filing_index(cik)
            if len(idx):
                idx.to_parquet(ip)
            time.sleep(edgar.SLEEP)
        if idx.empty:
            fails.append((r.ticker, "no filings in index"))
            continue
        idx = idx[idx.filing_date >= START]
        got = miss = 0
        for f in idx.itertuples():
            p = edgar.text_path(cik, f.accession)
            if p.exists():
                got += 1
                continue
            t = edgar.fetch_filing_text(cik, f.accession, f.doc)
            time.sleep(edgar.SLEEP)
            if t is None:
                miss += 1
            else:
                got += 1
        rows.append(dict(ticker=r.ticker, cik=cik, n_index=len(idx),
                         n_text=got, n_missing=miss))
        if i % 10 == 0 or i == len(uni):
            print(f"  [{i}/{len(uni)}] {r.ticker}: {got} texts, {miss} missing",
                  flush=True)
    except Exception as e:
        fails.append((r.ticker, f"{type(e).__name__}: {e}"))
        print(f"  [{i}/{len(uni)}] {r.ticker}: FAILED {type(e).__name__}: {e}",
              flush=True)

st = pd.DataFrame(rows)
st.to_parquet("data/edgar/fetch_status.parquet")
print(f"\n{len(st)} names ok, {len(fails)} failed")
if len(st):
    print(f"total texts on disk: {st.n_text.sum():,}   missing: {st.n_missing.sum():,}")
if fails:
    print("\nFAILURES (these must not be read as 'this company files nothing'):")
    for t, e in fails[:20]:
        print(f"   {t}: {e}")
