"""Ingest 70 quarters of SEC Financial Statement Data Sets.

Each raw quarter is ~120MB compressed / ~630MB expanded and is DISCARDED after
filtering to the universe's CIKs and the tag whitelist. Only filing-level facts
survive to disk.

Following the Form 4 lesson: failed quarters are counted and reported, and the
script exits non-zero rather than silently reporting success on a partial
ingest. A missing quarter of fundamentals is indistinguishable from a company
having reported nothing.
"""
import sys
import time
from pathlib import Path

import pandas as pd

from algo import fundamentals as fu

OUT = Path("data/fundamentals")
OUT.mkdir(parents=True, exist_ok=True)

uni = pd.read_parquet("data/collection_universe.parquet").dropna(subset=["cik"])
ciks = set(uni["cik"].astype(int).astype(str))
qs = fu.quarters("2009q1", "2026q2")
print(f"{len(ciks)} CIKs, {len(qs)} quarters: {qs[0]}..{qs[-1]}", flush=True)

fails = []
for i, q in enumerate(qs, 1):
    dest = OUT / f"facts_{q}.parquet"
    if dest.exists():
        continue
    try:
        raw = fu.fetch_quarter(q, ciks)
        if raw.empty:
            fails.append((q, "no matching rows"))
            print(f"  [{i}/{len(qs)}] {q}: EMPTY", flush=True)
            continue
        facts = fu.to_facts(raw)
        facts.to_parquet(dest)
        if i % 5 == 0 or i == len(qs):
            print(f"  [{i}/{len(qs)}] {q}: {len(facts)} filings, "
                  f"{facts.cik.nunique()} companies", flush=True)
        del raw, facts
    except Exception as e:
        fails.append((q, f"{type(e).__name__}: {e}"))
        print(f"  [{i}/{len(qs)}] {q}: FAILED {e}", flush=True)
    time.sleep(0.4)

files = sorted(OUT.glob("facts_*.parquet"))
if files:
    F = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    F = F.sort_values("filed").reset_index(drop=True)
    F.to_parquet("data/fundamentals/all_facts.parquet")
    print(f"\n{len(F):,} filing-level rows, {F.cik.nunique()} companies, "
          f"{F.filed.min().date()}..{F.filed.max().date()}", flush=True)

if fails:
    print(f"\n{len(fails)} quarter(s) failed:", flush=True)
    for q, e in fails[:20]:
        print(f"   {q}: {e}", flush=True)
    sys.exit("REFUSING to report success on a partial ingest.")
print("\nall quarters ingested", flush=True)
