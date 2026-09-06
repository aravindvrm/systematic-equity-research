"""S&P 600 SmallCap membership, for the break-even calculation.

SURVIVORSHIP WARNING -- read before using any number derived from this.
Small caps delist at several times the large-cap rate: acquisitions, bankruptcies,
and promotions up to the MidCap index. Today's S&P 600 carried back 20 years is a
FAR more biased sample than today's S&P 500 carried back, because the names that
left are disproportionately the ones that failed.

This matters differently for the two things we might compute:
  - ACHIEVABLE IC from a real feature: badly biased, do not trust.
  - BREAK-EVEN IC from SYNTHETIC forecasts: much more robust, because it asks
    "given this universe's vol/correlation structure and costs, how much IC is
    needed to beat its own benchmark?" Survivorship inflates the benchmark, which
    makes the computed bar CONSERVATIVE (too high) rather than flattering.
Only the second is computed here.
"""
import io
import sys

import pandas as pd
import requests

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
resp = requests.get(URL, headers={"User-Agent": "algo-research/0.1"}, timeout=30)
resp.raise_for_status()
tables = pd.read_html(io.StringIO(resp.text))
df = None
for t in tables:
    cols = {str(c).lower() for c in t.columns}
    if "symbol" in cols and any("sector" in c for c in cols):
        df = t
        break
if df is None:
    sys.exit("could not locate the constituents table")

ren = {}
for c in df.columns:
    lc = str(c).lower()
    if lc == "symbol":
        ren[c] = "ticker"
    elif "sector" in lc:
        ren[c] = "sector"
    elif "sub-industry" in lc or "industry" in lc:
        ren[c] = "industry"
    elif lc.startswith("company") or lc == "security":
        ren[c] = "name"
df = df.rename(columns=ren)[[c for c in ("ticker", "name", "sector", "industry")
                             if c in ren.values()]]
df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()
df = df.drop_duplicates("ticker").reset_index(drop=True)
df.to_parquet("data/sp600_constituents.parquet")
print(f"{len(df)} S&P 600 names across {df['sector'].nunique()} sectors")
print(df["sector"].value_counts().to_string())
