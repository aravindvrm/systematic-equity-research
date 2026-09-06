"""Download and compact every 13F quarter, 2013q2 onward.

Each raw quarter is ~73MB compressed / ~300MB expanded and is DISCARDED after
filtering to the universe's CUSIPs. Only the compact per-quarter holdings survive
to disk. Following the Form 4 lesson: a failed quarter RAISES rather than being
logged and skipped, because a silently missing period looks exactly like a period
in which nobody held anything.
"""
import sys
import time
from pathlib import Path

import pandas as pd

from algo import thirteenf as tf

OUT = Path("data/thirteenf")
OUT.mkdir(parents=True, exist_ok=True)

lc = pd.read_parquet("data/collection_universe.parquet")
sc = pd.read_parquet("data/sp600_constituents.parquet")
sc = sc.assign(cik=None, dollar_volume=None, selected_on=None)
uni = pd.concat([lc[["ticker", "name", "sector"]], sc[["ticker", "name", "sector"]]],
                ignore_index=True).drop_duplicates("ticker")
print(f"universe for mapping: {len(uni)} names", flush=True)

qs = tf.quarters()
print(f"{len(qs)} quarters: {qs[0]} .. {qs[-1]}\n", flush=True)

cmap = None
failures = []
for i, fn in enumerate(qs, 1):
    period = fn.split("_")[0]
    dest = OUT / f"holdings_{period}.parquet"
    if dest.exists():
        print(f"  [{i}/{len(qs)}] {period}: cached", flush=True)
        continue
    try:
        d = tf.fetch_quarter(fn)
        if cmap is None:
            cmap = tf.build_cusip_map(d["info"], uni)
            cmap = cmap.dropna(subset=["cusip"])
            cmap.to_parquet(OUT / "cusip_map.parquet")
            print(f"  cusip map: {len(cmap)}/{len(uni)} names matched\n", flush=True)
        cus = set(cmap["cusip"])
        h = tf.holdings_for_universe(d["info"], d["sub"], cus, period)
        h.to_parquet(dest)
        print(f"  [{i}/{len(qs)}] {period}: {len(h):>7,} filer-positions, "
              f"{h.filer_cik.nunique():>5,} filers, {h.cusip.nunique():>4} names",
              flush=True)
        del d
    except Exception as e:
        failures.append((period, f"{type(e).__name__}: {e}"))
        print(f"  [{i}/{len(qs)}] {period}: FAILED {type(e).__name__}: {e}", flush=True)
    time.sleep(0.5)

if failures:
    print(f"\n{len(failures)} quarter(s) failed:", flush=True)
    for p, e in failures:
        print(f"   {p}: {e}", flush=True)
    sys.exit(f"REFUSING to report success -- {len(failures)} quarters missing. "
             "A gap in ownership history is indistinguishable from zero ownership.")
print("\nall quarters ingested", flush=True)
