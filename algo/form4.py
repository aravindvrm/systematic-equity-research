"""SEC Form 4 (insider transactions) ingestion.

WHY THIS DATA
-------------
Corporate insiders are the one group with a *legal* informational advantage.
Under the asymmetry taxonomy this is the strongest free source available:
genuine information asymmetry, idiosyncratic by construction (a CFO buying says
something about that company, not about the market), and disclosed within two
business days.

SOURCE
------
SEC DERA publishes complete Form 3/4/5 data quarterly as structured TSVs. One
download per quarter beats fetching ~130,000 individual XML filings, and it is
the same data.

THE FILTERS THAT MATTER
-----------------------
Most Form 4 rows are noise. Transaction codes:

    P  open-market PURCHASE      <- the signal. Discretionary, uses own cash.
    S  open-market SALE          <- weak; insiders sell for diversification,
                                    tax, tuition. Far less informative than buys.
    A  grant / award             <- noise. Compensation, not a view.
    M  option exercise           <- noise. Expiry-driven.
    F  tax withholding           <- noise. Mechanical.
    G  gift                      <- noise.

And AFF10B5ONE flags trades made under a pre-scheduled 10b5-1 plan. Those were
decided months earlier and carry no timely information. Excluding them is the
difference between measuring a decision and measuring a calendar.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "form4"
RAW_DIR = DATA_DIR / "raw"
TRANS_PATH = DATA_DIR / "transactions.parquet"

BASE_URL = ("https://www.sec.gov/files/structureddata/data/"
            "insider-transactions-data-sets/{q}_form345.zip")
# SEC requires a descriptive User-Agent with contact info, and rate-limits to 10/s.
HEADERS = {"User-Agent": "algo-research aravindvrm@gmail.com"}

SIGNAL_CODES = {"P", "S"}


def quarters(start_year: int = 2018, end: str = "2026q1") -> list[str]:
    ey, eq = int(end[:4]), int(end[-1])
    out = []
    for y in range(start_year, ey + 1):
        for q in range(1, 5):
            if y == ey and q > eq:
                break
            out.append(f"{y}q{q}")
    return out


def download_quarter(q: str, refresh: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{q}.zip"
    if path.exists() and not refresh:
        return path
    resp = requests.get(BASE_URL.format(q=q), headers=HEADERS, timeout=180)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    log.info("downloaded %s (%.1f MB)", q, len(resp.content) / 1e6)
    return path


def parse_quarter(q: str, tickers: set[str] | None = None) -> pd.DataFrame:
    """Join submission + transaction + owner tables for one quarter."""
    zf = zipfile.ZipFile(download_quarter(q))

    # AFF10B5ONE (the pre-scheduled-plan checkbox) only exists from 2023 onward --
    # it was added to Form 4 by SEC rule amendment. Request it optionally so
    # earlier quarters parse instead of failing wholesale.
    sub_cols = ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT",
                "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL"]
    header = pd.read_csv(zf.open("SUBMISSION.tsv"), sep="\t", nrows=0).columns
    has_10b5 = "AFF10B5ONE" in header
    sub = pd.read_csv(zf.open("SUBMISSION.tsv"), sep="\t", dtype=str,
                      usecols=sub_cols + (["AFF10B5ONE"] if has_10b5 else []))
    if not has_10b5:
        sub["AFF10B5ONE"] = pd.NA
    sub = sub[sub.DOCUMENT_TYPE == "4"]
    sub["ticker"] = sub.ISSUERTRADINGSYMBOL.str.upper().str.strip()
    if tickers:
        sub = sub[sub.ticker.isin(tickers)]
    if sub.empty:
        return pd.DataFrame()

    tx = pd.read_csv(zf.open("NONDERIV_TRANS.tsv"), sep="\t", dtype=str,
                     usecols=["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE",
                              "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                              "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS",
                              "DIRECT_INDIRECT_OWNERSHIP"])
    tx = tx[tx.ACCESSION_NUMBER.isin(sub.ACCESSION_NUMBER)]

    own = pd.read_csv(zf.open("REPORTINGOWNER.tsv"), sep="\t", dtype=str,
                      usecols=["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                               "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
    own = own[own.ACCESSION_NUMBER.isin(sub.ACCESSION_NUMBER)]
    # One filing can list multiple owners (joint filings); keep the first.
    own = own.drop_duplicates("ACCESSION_NUMBER")

    df = tx.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")

    for c, dt in [("TRANS_DATE", None), ("FILING_DATE", None), ("PERIOD_OF_REPORT", None)]:
        df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
    for c in ["TRANS_SHARES", "TRANS_PRICEPERSHARE", "SHRS_OWND_FOLWNG_TRANS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.rename(columns={
        "TRANS_DATE": "trans_date", "FILING_DATE": "filing_date",
        "TRANS_CODE": "code", "TRANS_SHARES": "shares",
        "TRANS_PRICEPERSHARE": "price", "TRANS_ACQUIRED_DISP_CD": "acq_disp",
        "SHRS_OWND_FOLWNG_TRANS": "shares_after", "RPTOWNERCIK": "owner_cik",
        "RPTOWNERNAME": "owner_name", "RPTOWNER_RELATIONSHIP": "relationship",
        "RPTOWNER_TITLE": "title", "ISSUERCIK": "issuer_cik",
        "DIRECT_INDIRECT_OWNERSHIP": "direct_indirect",
    })
    df["is_10b5_1"] = df["AFF10B5ONE"].fillna("0").isin(["1", "true", "TRUE", "Y"])
    df["has_10b5_flag"] = has_10b5   # False pre-2023: unknown, not "not scheduled"

    # SEC filings contain genuine typos -- observed dates in year 0024 and 2028,
    # i.e. filers mistyping 2024. A Form 4 is due within two business days, so a
    # transaction cannot post-date its filing and should not precede it by years.
    bad = (df.trans_date.isna() | df.filing_date.isna()
           | (df.trans_date > df.filing_date)
           | (df.filing_date - df.trans_date > pd.Timedelta(days=365)))
    if bad.any():
        log.info("%s: dropped %d rows with implausible dates", q, int(bad.sum()))
    df = df[~bad]
    df["value"] = df["shares"] * df["price"]
    df["quarter"] = q

    keep = ["ticker", "issuer_cik", "trans_date", "filing_date", "code", "acq_disp",
            "shares", "price", "value", "shares_after", "owner_cik", "owner_name",
            "relationship", "title", "direct_indirect", "is_10b5_1",
            "has_10b5_flag", "quarter", "ACCESSION_NUMBER"]
    return df[[c for c in keep if c in df.columns]]


def ingest(tickers: set[str] | None = None, start_year: int = 2018,
           end: str = "2026q1") -> pd.DataFrame:
    """Download and parse every quarter, writing one combined parquet."""
    frames, failures = [], []
    for q in quarters(start_year, end):
        try:
            d = parse_quarter(q, tickers)
            if not d.empty:
                frames.append(d)
            log.info("%s: %d rows", q, len(d))
        except Exception as exc:  # noqa: BLE001 - one bad quarter must not kill the run
            log.error("%s FAILED: %s", q, exc)
            failures.append(q)
    if failures:
        # Silent partial failure is how 5 years of history went missing while the
        # run still reported success. Make it impossible to overlook.
        raise RuntimeError(
            f"{len(failures)} quarter(s) failed to parse: {failures}. "
            "Fix them or pass them explicitly -- do not ingest a partial history.")
    if not frames:
        raise RuntimeError("no quarters parsed")
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(TRANS_PATH, index=False)
    return out


def load() -> pd.DataFrame:
    return pd.read_parquet(TRANS_PATH)


def open_market_buys(df: pd.DataFrame, exclude_10b5_1: bool = True,
                     min_value: float = 0.0) -> pd.DataFrame:
    """The signal subset: discretionary open-market purchases.

    Code 'P' with acquisition, optionally excluding pre-scheduled 10b5-1 plans.
    An insider buying on the open market with their own money is the cleanest
    bullish disclosure in the free data universe.
    """
    out = df[(df.code == "P") & (df.acq_disp == "A")]
    if exclude_10b5_1:
        out = out[~out.is_10b5_1]
    if min_value:
        out = out[out.value >= min_value]
    return out
