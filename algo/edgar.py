"""EDGAR filing retrieval and text extraction.

Infrastructure for every text hypothesis, not just the first one. Fetching is the
expensive, rate-limited, failure-prone half; measurement is cheap and will be
redone many times. So cleaned text is CACHED to disk and the two are kept
separate -- a new measure must never require re-downloading twenty years of
filings.

RATE LIMITS
-----------
SEC requires a declared User-Agent with contact details and caps traffic at ~10
requests/second. We stay well under. Ignoring either gets the IP blocked, and the
block is not obviously distinguishable from "this company has no filings".

WHAT COUNTS AS THE FILING
-------------------------
Only the PRIMARY document. A 10-K submission also carries exhibits, XBRL, and
occasionally the entire prior year's report as an exhibit -- including those makes
year-over-year similarity meaningless because the exhibit set changes for
unrelated reasons.
"""
from __future__ import annotations

import gzip
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "edgar"
TEXT_DIR = DATA_DIR / "text"
UA = {"User-Agent": "algo-research aravindvrm@gmail.com"}
SLEEP = 0.12          # ~8 req/s, comfortably inside SEC's limit
FORMS = ("10-K", "10-Q")


def _get(url: str, tries: int = 3):
    for i in range(tries):
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        time.sleep(1.5 * (i + 1))       # 429/503: back off rather than hammer
    r.raise_for_status()


def filing_index(cik: str) -> pd.DataFrame:
    """Every 10-K / 10-Q for a CIK, oldest first.

    The submissions endpoint returns only the most recent ~1000 filings inline;
    older ones live in paginated files that must be fetched separately. Missing
    that pagination silently truncates history to a few years for any active
    filer -- which for a large company means losing everything before ~2015.
    """
    cik10 = str(cik).zfill(10)
    r = _get(f"https://data.sec.gov/submissions/CIK{cik10}.json")
    if r is None:
        return pd.DataFrame()
    d = r.json()
    chunks = [pd.DataFrame(d["filings"]["recent"])]
    for f in d["filings"].get("files", []):
        time.sleep(SLEEP)
        r2 = _get(f"https://data.sec.gov/submissions/{f['name']}")
        if r2 is not None:
            chunks.append(pd.DataFrame(r2.json()))
    df = pd.concat(chunks, ignore_index=True)
    df = df[df["form"].isin(FORMS)].copy()
    if df.empty:
        return df
    df["cik"] = str(int(cik))
    df["filing_date"] = pd.to_datetime(df["filingDate"], errors="coerce")
    df["report_date"] = pd.to_datetime(df["reportDate"], errors="coerce")
    df = df.dropna(subset=["filing_date"])
    return (df[["cik", "form", "filing_date", "report_date",
                "accessionNumber", "primaryDocument"]]
            .rename(columns={"accessionNumber": "accession",
                             "primaryDocument": "doc"})
            .sort_values("filing_date").reset_index(drop=True))


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_SCRIPT = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)

# Tokens that appear only as inline-XBRL scaffolding, never as prose.
_XBRL_STOP = {"xbrli", "xbrl", "iso", "utr", "dei", "srt", "fasb", "gaap",
              "http", "https", "www", "org", "xsd", "xsi", "linkbase",
              "instant", "duration", "unitref", "contextref", "decimals",
              "scale", "sign", "format", "ix", "nonfraction", "nonnumeric"}


def clean_html(raw: str) -> str:
    """HTML -> normalized lowercase word stream.

    Numbers are stripped. Year-over-year similarity should measure whether the
    LANGUAGE changed, not whether the figures did -- the figures change every
    quarter by construction, and leaving them in makes every filing look equally
    different from its predecessor.
    """
    s = _SCRIPT.sub(" ", raw)
    s = _TAG.sub(" ", s)
    s = (s.replace("&nbsp;", " ").replace("&amp;", "&")
         .replace("&#160;", " ").replace("&quot;", '"').replace("&#8217;", "'"))
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = _WS.sub(" ", s).strip()

    # Strip inline-XBRL taxonomy residue. Modern filings embed tag references
    # ("http fasb org us gaap marketablesecuritiescurrent") that survive tag
    # removal as ordinary words. They are boilerplate that changes with
    # accounting-standard updates rather than with what the company said, so
    # leaving them in adds a common-mode signal to every filing at once.
    toks = [t for t in s.split() if len(t) <= 20 and t not in _XBRL_STOP]
    s = " ".join(toks)

    # The inline-XBRL header sits before the document proper. Cut to the first
    # cover-page marker so the text starts where the filing does. If no marker is
    # found (older plain-text filings), keep everything rather than risk cutting
    # real content.
    for marker in ("securities and exchange commission", "table of contents",
                   "part i item"):
        i = s.find(marker)
        if 0 < i < 20000:
            return s[i:]
    return s


def text_path(cik: str, accession: str) -> Path:
    return TEXT_DIR / str(cik) / f"{accession.replace('-', '')}.txt.gz"


def fetch_filing_text(cik: str, accession: str, doc: str,
                      force: bool = False) -> str | None:
    """Cleaned primary-document text, cached gzipped on disk."""
    p = text_path(cik, accession)
    if p.exists() and not force:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return fh.read()
    acc = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
    r = _get(url)
    if r is None:
        return None
    txt = clean_html(r.text)
    if len(txt) < 2000:            # a stub, a redirect page, or a fetch failure
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write(txt)
    return txt


def event_index(cik: str, forms: tuple[str, ...] = ("8-K",)) -> pd.DataFrame:
    """All filings of the given forms, WITH their item codes.

    The submissions endpoint carries 8-K item codes inline (the `items` field),
    so earnings-announcement dates can be identified without fetching a single
    document. Item 2.02 is "Results of Operations and Financial Condition" --
    that IS the earnings release.

    Same pagination trap as filing_index: older filings live in separate files
    and skipping them silently truncates history for any active filer.
    """
    cik10 = str(cik).zfill(10)
    r = _get(f"https://data.sec.gov/submissions/CIK{cik10}.json")
    if r is None:
        return pd.DataFrame()
    d = r.json()
    chunks = [pd.DataFrame(d["filings"]["recent"])]
    for f in d["filings"].get("files", []):
        time.sleep(SLEEP)
        r2 = _get(f"https://data.sec.gov/submissions/{f['name']}")
        if r2 is not None:
            chunks.append(pd.DataFrame(r2.json()))
    df = pd.concat(chunks, ignore_index=True)
    df = df[df["form"].isin(forms)].copy()
    if df.empty:
        return df
    df["cik"] = str(int(cik))
    df["filing_date"] = pd.to_datetime(df["filingDate"], errors="coerce")
    df["items"] = df.get("items", "").fillna("").astype(str)
    df = df.dropna(subset=["filing_date"])
    return (df[["cik", "form", "filing_date", "items", "accessionNumber"]]
            .rename(columns={"accessionNumber": "accession"})
            .sort_values("filing_date").reset_index(drop=True))


def has_item(items: pd.Series, code: str) -> pd.Series:
    """Whether an item-code string contains a specific code.

    Codes appear comma-separated ("2.02,9.01"). A plain substring test would
    match 2.02 inside 12.02 and similar, so boundaries are enforced.
    """
    pat = rf"(?:^|,)\s*{re.escape(code)}\s*(?:,|$)"
    return items.fillna("").astype(str).str.contains(pat, regex=True)
