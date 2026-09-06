"""SEC Form 13F ingestion: institutional ownership, quarterly.

WHY THIS DATA
-------------
This is the first NON-PRICE relational source in the project. Every relational
feature tested so far -- beta to basket, correlation to basket, pair spreads,
network centrality -- is a transformation of the return series, and that space
was measured at IC ~0.012 against a requirement of ~0.036. Common ownership is
different in kind: it says WHO HOLDS WHAT, which cannot be recovered from prices.

MECHANISM (Anton & Polk 2014, "Connected Stocks", Journal of Finance)
Two stocks held by the same institutions are linked by that institution's flows.
When a fund takes redemptions it sells its whole book, pushing both names down
together for reasons that have nothing to do with either company. That is
non-fundamental price pressure, and it reverses. The connection is only visible
in ownership data.

SOURCE AND ITS LIMITS
---------------------
SEC publishes 13F as structured quarterly TSVs, 2013q2 onward (~13 years).
Hard constraints to respect:
  - QUARTERLY, with a 45-DAY reporting lag. A Q4 position is public in mid
    February. Anything built from it must be lagged accordingly or it is
    lookahead. `period_end + 45 days` is the earliest usable date.
  - LONG POSITIONS ONLY. 13F does not report shorts, so ownership is one-sided.
  - $100M+ managers only. Small funds are invisible.
  - Options positions appear with PUTCALL set; those are excluded here so the
    holdings matrix means "shares held".

SIZE
----
INFOTABLE is ~2.9M rows and ~300MB uncompressed PER QUARTER. Fifty quarters will
not fit anywhere convenient, so each quarter is filtered to the universe's CUSIPs
and written out compactly, then the raw download is discarded.
"""
from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "thirteenf"
BASE = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/"
UA = {"User-Agent": "algo-research aravindvrm@gmail.com"}

USECOLS = ["ACCESSION_NUMBER", "NAMEOFISSUER", "CUSIP", "VALUE", "SSHPRNAMT", "PUTCALL"]


def quarters(start_year: int = 2013, start_q: int = 2, end_year: int = 2023,
             end_q: int = 4) -> list[str]:
    """Filenames for the quarterly-format datasets (2013q2 .. 2023q4)."""
    out = []
    for y in range(start_year, end_year + 1):
        for q in range(1, 5):
            if (y == start_year and q < start_q) or (y == end_year and q > end_q):
                continue
            out.append(f"{y}q{q}_form13f.zip")
    return out


def _norm(s: str) -> str:
    """Normalize a company name for matching across two very messy sources."""
    s = str(s).upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    drop = {"INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC",
            "HOLDINGS", "HOLDING", "GROUP", "THE", "CLASS", "COM", "NEW", "LLC",
            "LP", "SA", "NV", "AG", "TRUST", "REIT", "INTERNATIONAL", "INTL"}
    toks = [t for t in s.split() if t and t not in drop]
    return " ".join(toks)


def fetch_quarter(fname: str, cache_raw: Path | None = None) -> dict[str, pd.DataFrame]:
    """Download one quarterly zip and return its INFOTABLE and SUBMISSION frames."""
    url = BASE + fname
    r = requests.get(url, headers=UA, timeout=300)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    info = pd.read_csv(z.open("INFOTABLE.tsv"), sep="\t", usecols=USECOLS,
                       dtype={"CUSIP": str, "PUTCALL": str, "NAMEOFISSUER": str},
                       low_memory=False)
    sub = pd.read_csv(z.open("SUBMISSION.tsv"), sep="\t",
                      dtype={"CIK": str, "ACCESSION_NUMBER": str})
    return {"info": info, "sub": sub}


def build_cusip_map(info: pd.DataFrame, universe: pd.DataFrame,
                    name_col: str = "name", ticker_col: str = "ticker") -> pd.DataFrame:
    """Map each universe ticker to its CUSIP via issuer name.

    13F carries CUSIP and issuer name but no ticker, and no free CUSIP->ticker
    table exists. For a FIXED universe the name match is tractable: normalize
    both sides, then where a name matches several CUSIPs (share classes, stale
    identifiers) keep the one with the largest total reported value -- the real
    common stock dominates institutional holdings by orders of magnitude.
    """
    inf = info.copy()
    inf["key"] = inf["NAMEOFISSUER"].map(_norm)
    agg = (inf.groupby(["key", "CUSIP"], observed=True)["VALUE"].sum()
           .reset_index().sort_values("VALUE", ascending=False))
    best = agg.drop_duplicates("key")

    uni = universe.copy()
    uni["key"] = uni[name_col].map(_norm)
    m = uni.merge(best[["key", "CUSIP", "VALUE"]], on="key", how="left")

    # Fallback for names the two sources spell differently ("General Electric
    # Company" vs "GENERAL ELECTRIC CO NEW"). Require the FULL universe key to be
    # a prefix of the 13F key -- a token-overlap match would happily pair
    # "AMERICAN AIRLINES" with "AMERICAN EXPRESS".
    miss = m["CUSIP"].isna()
    if miss.any():
        for i in m.index[miss]:
            k = m.at[i, "key"]
            if not k:
                continue
            cand = best[best["key"].str.startswith(k + " ", na=False) |
                        (best["key"] == k)]
            if len(cand):
                top = cand.iloc[0]
                m.at[i, "CUSIP"], m.at[i, "VALUE"] = top["CUSIP"], top["VALUE"]

    return m[[ticker_col, name_col, "key", "CUSIP", "VALUE"]].rename(
        columns={"CUSIP": "cusip", "VALUE": "match_value"})


def holdings_for_universe(info: pd.DataFrame, sub: pd.DataFrame,
                          cusips: set[str], period: str) -> pd.DataFrame:
    """Compact per-quarter holdings: one row per (filer, cusip).

    Options positions are dropped -- PUTCALL non-empty means the manager reported
    a derivative, not shares, and mixing the two makes "ownership" meaningless.
    """
    inf = info[info["PUTCALL"].isna() | (info["PUTCALL"].astype(str).str.strip() == "")]
    inf = inf[inf["CUSIP"].isin(cusips)]
    if inf.empty:
        return pd.DataFrame(columns=["period", "filer_cik", "cusip", "value", "shares"])
    j = inf.merge(sub[["ACCESSION_NUMBER", "CIK"]], on="ACCESSION_NUMBER", how="left")
    out = (j.groupby(["CIK", "CUSIP"], observed=True)
           .agg(value=("VALUE", "sum"), shares=("SSHPRNAMT", "sum"))
           .reset_index()
           .rename(columns={"CIK": "filer_cik", "CUSIP": "cusip"}))
    out["period"] = period
    return out[["period", "filer_cik", "cusip", "value", "shares"]]
