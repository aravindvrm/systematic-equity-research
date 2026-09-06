"""SEC Financial Statement Data Sets — the fundamentals we never fetched.

WHY THIS EXISTS
---------------
The standard anomaly taxonomy (Hou/Xue/Zhang and the replication literature)
has six categories: momentum, trading frictions, value-vs-growth, investment,
profitability, and intangibles. This project tested momentum and trading
frictions exhaustively and **never touched value, investment or profitability**
— three of the six, and three of the five Fama-French factors (HML, CMA, RMW).

The reason was mundane: those need balance-sheet and income-statement data,
which was never ingested. It is free, from the same DERA source already used
for Form 4 and 13F.

POINT-IN-TIME — THE TRAP THAT MATTERS MOST HERE
-----------------------------------------------
Fundamentals are the classic lookahead disaster. A fiscal quarter ending 31 Dec
is not public until the 10-K is FILED, typically 30-75 days later. Anchoring on
the period end would let you trade on figures nobody had. Worse, commercial
databases silently RESTATE history, so a value pulled today may not be the value
that was reported then.

The quarterly DERA files avoid both problems: each contains data AS FILED in
that quarter, and carries the filing date (`sub.filed`). Everything here is
keyed on `filed`, never on `ddate`.

SIZE
----
num.txt is ~514MB uncompressed PER QUARTER, ~36GB across 70 quarters. Each
quarter is therefore filtered to the universe's CIKs and a short tag whitelist,
written compactly, and the raw download discarded.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fundamentals"
BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"
UA = {"User-Agent": "algo-research aravindvrm@gmail.com"}

# Tag whitelist. XBRL is not consistent across filers or years, so each concept
# needs several aliases and the first available one wins.
TAGS = {
    "assets":        ["Assets"],
    "equity":        ["StockholdersEquity",
                      "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "revenue":       ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                      "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "cogs":          ["CostOfRevenue", "CostOfGoodsSold",
                      "CostOfGoodsAndServicesSold"],
    "net_income":    ["NetIncomeLoss", "ProfitLoss"],
    "op_income":     ["OperatingIncomeLoss"],
    "cash":          ["CashAndCashEquivalentsAtCarryingValue"],
    "liabilities":   ["Liabilities"],
    "cfo":           ["NetCashProvidedByUsedInOperatingActivities"],
    "shares":        ["CommonStockSharesOutstanding", "CommonStockSharesIssued",
                      "WeightedAverageNumberOfSharesOutstandingBasic"],
}
ALL_TAGS = {t for v in TAGS.values() for t in v}


def quarters(start: str = "2009q1", end: str = "2026q2") -> list[str]:
    sy, sq = int(start[:4]), int(start[5])
    ey, eq = int(end[:4]), int(end[5])
    out = []
    for y in range(sy, ey + 1):
        for q in range(1, 5):
            if (y == sy and q < sq) or (y == ey and q > eq):
                continue
            out.append(f"{y}q{q}")
    return out


def fetch_quarter(q: str, ciks: set[str]) -> pd.DataFrame:
    """One quarter, filtered to `ciks` and the tag whitelist.

    Returns one row per (cik, filing, concept) with the filing date attached.
    """
    r = requests.get(f"{BASE}/{q}.zip", headers=UA, timeout=600)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))

    sub = pd.read_csv(z.open("sub.txt"), sep="\t", low_memory=False,
                      usecols=["adsh", "cik", "form", "period", "filed", "fy", "fp"],
                      dtype={"adsh": str, "cik": str})
    sub["cik"] = sub["cik"].astype(str).str.lstrip("0")
    sub = sub[sub["cik"].isin(ciks) & sub["form"].isin(["10-K", "10-Q"])]
    if sub.empty:
        return pd.DataFrame()
    keep = set(sub["adsh"])

    chunks = []
    for ch in pd.read_csv(z.open("num.txt"), sep="\t", low_memory=False,
                          usecols=["adsh", "tag", "ddate", "qtrs", "uom", "value",
                                   "segments", "coreg"],
                          dtype={"adsh": str, "tag": str}, chunksize=1_000_000):
        ch = ch[ch["adsh"].isin(keep) & ch["tag"].isin(ALL_TAGS)]
        # Consolidated totals only: a segment or co-registrant breakdown is a
        # SLICE of the company, and mixing slices with totals silently corrupts
        # every ratio built from them.
        ch = ch[ch["segments"].isna() & ch["coreg"].isna()]
        if len(ch):
            chunks.append(ch.drop(columns=["segments", "coreg"]))
    if not chunks:
        return pd.DataFrame()

    num = pd.concat(chunks, ignore_index=True)
    df = num.merge(sub, on="adsh", how="inner")
    df["filed"] = pd.to_datetime(df["filed"].astype(str), format="%Y%m%d", errors="coerce")
    df["ddate"] = pd.to_datetime(df["ddate"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["filed", "ddate", "value"])
    df["quarter"] = q
    return df[["cik", "adsh", "form", "tag", "ddate", "qtrs", "uom",
               "value", "filed", "quarter"]]


def _pick(g: pd.DataFrame, aliases: list[str], want_qtrs) -> float:
    """First available alias, most recent period, matching the duration."""
    for a in aliases:
        s = g[(g["tag"] == a) & (g["qtrs"].isin(want_qtrs))]
        if len(s):
            return float(s.sort_values("ddate").iloc[-1]["value"])
    return np.nan


def to_facts(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw tag rows into one row of concepts per filing.

    Balance-sheet items are instants (qtrs=0); flow items are durations
    (qtrs=1 for a quarter, 4 for a year). Mixing them is the other classic way
    to corrupt a fundamental ratio.
    """
    rows = []
    for (cik, adsh), g in raw.groupby(["cik", "adsh"], sort=False):
        rec = {"cik": cik, "adsh": adsh,
               "filed": g["filed"].iloc[0], "form": g["form"].iloc[0],
               "ddate": g["ddate"].max()}
        for concept, aliases in TAGS.items():
            want = [0] if concept in ("assets", "equity", "cash", "liabilities",
                                      "shares") else [1, 4]
            rec[concept] = _pick(g, aliases, want)
        rows.append(rec)
    return pd.DataFrame(rows)
