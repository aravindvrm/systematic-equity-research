"""Backfill the price cache to 2005.

The cache starts in 2017-2018 only because that was the start date of the first
fetch, not because the data stops there. data.load_bars MERGES rather than
replaces (see test_refresh_does_not_truncate_cache), so pulling an earlier start
extends each file backwards without disturbing what is already there.

Sample length is the binding constraint on every t-stat in this project. Going
from 7 years to ~20 is worth more than any number of additional feature variants.
"""
import json
import sys
import time

import pandas as pd
from algo import data

uni = pd.read_parquet("data/collection_universe.parquet")
sym_col = [c for c in uni.columns if c.lower() in ("symbol", "ticker")][0]
syms = sorted(set(uni[sym_col].tolist()) | set(json.load(open("data/trading_universe.json"))))
print(f"{len(syms)} symbols -> backfilling from 2005-01-01", flush=True)

ok = fail = 0
for i, s in enumerate(syms, 1):
    try:
        df = data.load_bars(s, start="2005-01-01", refresh=True)
        ok += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(syms)}] {s}: {len(df)} bars from {df.index.min().date()}",
                  flush=True)
    except Exception as e:
        fail += 1
        print(f"  FAIL {s}: {type(e).__name__}: {e}", flush=True)
    time.sleep(0.15)   # be polite to the endpoint
print(f"\ndone: {ok} ok, {fail} failed", flush=True)
