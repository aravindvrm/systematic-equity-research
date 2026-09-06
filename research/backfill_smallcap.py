"""Price history for the S&P 600, 2005-2026."""
import time
import pandas as pd
from algo import data

syms = pd.read_parquet("data/sp600_constituents.parquet")["ticker"].tolist()
print(f"{len(syms)} symbols", flush=True)
ok = fail = 0
for i, s in enumerate(syms, 1):
    try:
        data.load_bars(s, start="2005-01-01", refresh=True)
        ok += 1
    except Exception as e:
        fail += 1
    if i % 50 == 0:
        print(f"  [{i}/{len(syms)}]  ok={ok} fail={fail}", flush=True)
    time.sleep(0.12)
print(f"done: {ok} ok, {fail} failed", flush=True)
