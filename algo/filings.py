"""Year-over-year filing similarity -- the "Lazy Prices" measures.

Cohen, Malloy & Nguyen (2020, JF): firms that CHANGE their 10-K/10-Q language
relative to the same filing a year earlier subsequently underperform. Investors
anchor on boilerplate and do not read the diffs, so a material change is
incorporated slowly. The feature is therefore SIMILARITY (high = unchanged =
good), and the prediction is that low-similarity firms underperform.

PAIRING
-------
Same form, same fiscal quarter, one year apart. A 10-Q must be compared to the
10-Q of the same quarter last year, never to last quarter's: a Q1-vs-Q4
comparison measures seasonal boilerplate, not change.

POINT-IN-TIME
-------------
Everything is anchored to FILING_DATE, never report_date. The report period ends
weeks before anyone can read the document. Anchoring to report_date would repeat
the exact error that manufactured a Form 4 signal earlier in this project.

LENGTH DRIFT
------------
Document composition has changed structurally over twenty years (inline XBRL,
exhibit conventions). Cosine on L2-normalized term-frequency vectors is largely
length-invariant; Jaccard on word SETS is not, and will drift with vocabulary
size. Both are computed so the two can be compared -- if they disagree, length
drift is the first suspect.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from . import edgar


def load_text(cik: str, accession: str) -> str | None:
    p = edgar.text_path(cik, accession)
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return fh.read()


def _pair_index(idx: pd.DataFrame, tol_days: int = 75) -> pd.DataFrame:
    """Attach to each filing its same-form, same-quarter, prior-year counterpart."""
    idx = idx.sort_values("filing_date").reset_index(drop=True)
    anchor = idx["report_date"].fillna(idx["filing_date"])
    idx = idx.assign(anchor=anchor)
    prev_acc, prev_dt = [], []
    for f in idx.itertuples():
        target = f.anchor - pd.Timedelta(days=365)
        cand = idx[(idx.form == f.form) & (idx.anchor < f.anchor - pd.Timedelta(days=180))]
        if cand.empty:
            prev_acc.append(None); prev_dt.append(pd.NaT); continue
        d = (cand.anchor - target).abs()
        best = cand.loc[d.idxmin()]
        if abs((best.anchor - target).days) > tol_days:
            prev_acc.append(None); prev_dt.append(pd.NaT)
        else:
            prev_acc.append(best.accession); prev_dt.append(best.filing_date)
    return idx.assign(prev_accession=prev_acc, prev_filing_date=prev_dt)


def similarity_measures(cik: str, idx: pd.DataFrame) -> pd.DataFrame:
    """Cosine and Jaccard similarity for every filing with a valid predecessor."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    idx = _pair_index(idx)
    rows = []
    for f in idx.itertuples():
        if not isinstance(f.prev_accession, str):
            continue
        a, b = load_text(cik, f.accession), load_text(cik, f.prev_accession)
        if a is None or b is None:
            continue
        try:
            v = CountVectorizer(min_df=1).fit_transform([a, b]).toarray().astype(float)
            # Raw-count cosine SATURATES on documents this long: common words are
            # identical in every filing, so the measure sits at ~0.995 with an sd
            # of 0.01 and can barely discriminate. Stopword removal plus tf-idf
            # weights the terms that actually differ.
            vs = TfidfVectorizer(stop_words="english", min_df=1, sublinear_tf=True)
            w = vs.fit_transform([a, b]).toarray().astype(float)
        except ValueError:
            continue
        n = np.linalg.norm(v, axis=1)
        cos = float(v[0] @ v[1] / (n[0] * n[1])) if n.min() > 0 else np.nan
        nw = np.linalg.norm(w, axis=1)
        cos_tfidf = float(w[0] @ w[1] / (nw[0] * nw[1])) if nw.min() > 0 else np.nan
        sa, sb = set(a.split()), set(b.split())
        jac = len(sa & sb) / len(sa | sb) if (sa | sb) else np.nan
        rows.append(dict(cik=cik, form=f.form, filing_date=f.filing_date,
                         report_date=f.report_date, accession=f.accession,
                         sim_cosine=cos, sim_cosine_tfidf=cos_tfidf,
                         sim_jaccard=jac,
                         n_words=len(a.split()), n_words_prev=len(b.split())))
    return pd.DataFrame(rows)


def to_daily(measures: pd.DataFrame, index: pd.DatetimeIndex,
             tickers: dict[str, str], col: str = "sim_cosine",
             lag_days: int = 1, hold_days: int = 63) -> pd.DataFrame:
    """Spread per-filing measures onto a daily panel.

    Effective from filing_date + `lag_days` (a filing that appears after the
    close is not tradeable the same day), held for `hold_days` or until the next
    filing, whichever comes first.
    """
    out = pd.DataFrame(np.nan, index=index, columns=sorted(set(tickers.values())))
    for cik, g in measures.groupby("cik"):
        t = tickers.get(str(cik))
        if t is None or t not in out.columns:
            continue
        g = g.sort_values("filing_date")
        for j, f in enumerate(g.itertuples()):
            start = f.filing_date + pd.Timedelta(days=lag_days)
            nxt = (g.iloc[j + 1].filing_date + pd.Timedelta(days=lag_days)
                   if j + 1 < len(g) else index[-1] + pd.Timedelta(days=1))
            end = min(nxt, start + pd.Timedelta(days=hold_days))
            m = (index >= start) & (index < end)
            if m.any():
                out.loc[m, t] = getattr(f, col)
    return out
