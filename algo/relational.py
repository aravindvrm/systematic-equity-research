"""Pairwise relational features: a name measured against its closest PEER.

Why this is not the same as the relational features in features2.py
-------------------------------------------------------------------
Everything in features2 is name-vs-UNIVERSE: beta to the basket, correlation to
the basket, strength relative to the basket average. Those aggregate away the
specific relationship that makes relative value work. A well-matched pair runs
0.7-0.9 correlation; the same name against a 30-name basket runs ~0.4. The
residual you are trading is far cleaner in the first case.

Point-in-time discipline
------------------------
Peer selection is the lookahead trap here. Choosing "the name most correlated
with EQT" using the full sample tells you which pair held together *in the end*,
which is precisely the pair whose spread mean-reverted. Peers are therefore
re-chosen every `refit_every` bars using ONLY the trailing `window` bars, and
the peer map is held fixed until the next refit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def peer_map(prices: pd.DataFrame, window: int = 252, refit_every: int = 63,
             min_train: int = 252) -> pd.DataFrame:
    """For each date and name, the name it was most correlated with recently.

    Returns a frame of the same shape as `prices` holding peer TICKERS (object
    dtype), NaN before `min_train` bars have accumulated.
    """
    rets = prices.pct_change()
    cols = list(prices.columns)
    out = pd.DataFrame(np.nan, index=prices.index, columns=cols, dtype=object)

    fit_points = range(min_train, len(prices), refit_every)
    for i in fit_points:
        w = rets.iloc[max(0, i - window):i]
        # A name needs enough history in the window to be a credible peer.
        w = w.dropna(axis=1, thresh=int(0.8 * len(w)))
        if w.shape[1] < 2:
            continue
        c = w.corr()
        cv = c.to_numpy(copy=True)
        np.fill_diagonal(cv, -np.inf)   # never pair a name with itself
        c = pd.DataFrame(cv, index=c.index, columns=c.columns)
        best = c.idxmax(axis=1)
        end = min(i + refit_every, len(prices))
        for sym, pr in best.items():
            out.iloc[i:end, cols.index(sym)] = pr
    return out


def _peer_aligned(series_by_name: pd.DataFrame, peers: pd.DataFrame) -> pd.DataFrame:
    """Reindex a per-name quantity onto each name's peer.

    Result[t, i] is the value that quantity took at t for whichever name was i's
    peer at t.
    """
    vals = series_by_name.to_numpy()
    cols = list(series_by_name.columns)
    idx = {c: j for j, c in enumerate(cols)}
    out = np.full(vals.shape, np.nan)
    pv = peers.to_numpy()
    for t in range(vals.shape[0]):
        for i in range(vals.shape[1]):
            p = pv[t, i]
            if isinstance(p, str):
                out[t, i] = vals[t, idx[p]]
    return pd.DataFrame(out, index=series_by_name.index, columns=cols)


def pair_spread_z(prices: pd.DataFrame, peers: pd.DataFrame,
                  window: int = 126) -> pd.DataFrame:
    """The pairs-trading signal proper, in rankable form.

    log(P_i / P_peer) z-scored against its own trailing mean. Sign is FLIPPED so
    that a cheap name (spread far below its normal level) scores high, matching
    the convention that high feature value = expected outperformance.
    """
    lp = np.log(prices)
    peer_lp = _peer_aligned(lp, peers)
    spread = lp - peer_lp
    mu = spread.rolling(window, min_periods=window // 2).mean()
    sd = spread.rolling(window, min_periods=window // 2).std()
    return -((spread - mu) / sd.replace(0, np.nan))


def pair_lead_lag(prices: pd.DataFrame, peers: pd.DataFrame,
                  lookback: int = 5) -> pd.DataFrame:
    """The peer's recent return. Tests information diffusion between linked names.

    Positive sign assumes the lagging name FOLLOWS its peer. If diffusion runs
    the other way this simply shows up as a negative IC, which is still a result.
    """
    peer_ret = _peer_aligned(prices.pct_change(lookback), peers)
    return peer_ret


def pair_corr_break(prices: pd.DataFrame, peers: pd.DataFrame,
                    fast: int = 21, slow: int = 252) -> pd.DataFrame:
    """Recent correlation to the peer, minus its longer-run level.

    A name decoupling from the peer it normally tracks has usually had something
    name-specific happen to it. Negated so that DECOUPLING scores high.
    """
    rets = prices.pct_change()
    cols = list(prices.columns)
    idx = {c: j for j, c in enumerate(cols)}
    rv = rets.to_numpy()
    pv = peers.to_numpy()
    out = np.full(rv.shape, np.nan)

    # Rolling pairwise correlation, computed only for the (t, i) cells we need.
    for i, sym in enumerate(cols):
        s = pd.Series(rv[:, i], index=rets.index)
        # Peers change only at refit points; group by peer to vectorize.
        peer_col = pd.Series(pv[:, i], index=rets.index)
        for p, block in peer_col.groupby(peer_col):
            if not isinstance(p, str):
                continue
            o = pd.Series(rv[:, idx[p]], index=rets.index)
            cf = s.rolling(fast, min_periods=fast // 2).corr(o)
            cs = s.rolling(slow, min_periods=slow // 2).corr(o)
            d = -(cf - cs)
            out[rets.index.get_indexer(block.index), i] = d.reindex(block.index).to_numpy()
    return pd.DataFrame(out, index=rets.index, columns=cols)


REGISTRY = {
    "pair_spread_z": pair_spread_z,
    "pair_lead_lag": pair_lead_lag,
    "pair_corr_break": pair_corr_break,
}


# ---------------------------------------------------------------------------
# Cointegration-based pairs
#
# Correlation is the WRONG statistic for relative value. Correlation asks whether
# two names move together day to day; the trade needs the SPREAD between them to
# be stationary. Two names can be 0.9 correlated while their ratio drifts apart
# forever, and there is no money in that. Cointegration tests the property the
# trade actually depends on.
#
# The spread is also hedge-ratio adjusted: log(Pi) - beta*log(Pj), with beta from
# a trailing regression. The earlier pair_spread_z implicitly assumed beta == 1,
# which loads the "spread" with a directional exposure unrelated to relative value.
# ---------------------------------------------------------------------------

def coint_peer_map(prices: pd.DataFrame, sectors: pd.Series, window: int = 504,
                   refit_every: int = 252, n_candidates: int = 5):
    """Pick each name's peer by ADF stationarity of the hedged spread.

    Candidates are the `n_candidates` most-correlated same-sector names (a cheap
    prefilter -- testing every pair is O(N^2) ADF calls). Among those, the peer is
    whichever gives the most stationary spread. Returns (peers, betas).
    """
    from statsmodels.tsa.stattools import adfuller

    lp = np.log(prices)
    rets = prices.pct_change()
    cols = list(prices.columns)
    peers = pd.DataFrame(np.nan, index=prices.index, columns=cols, dtype=object)
    betas = pd.DataFrame(np.nan, index=prices.index, columns=cols, dtype="float64")

    for i in range(window, len(prices), refit_every):
        wr = rets.iloc[i - window:i]
        wl = lp.iloc[i - window:i]
        end = min(i + refit_every, len(prices))
        for sec, grp in sectors.groupby(sectors):
            sc = [c for c in grp.index if c in cols]
            if len(sc) < 3:
                continue
            sub = wr[sc].dropna(axis=1, thresh=int(0.8 * window))
            if sub.shape[1] < 3:
                continue
            corr = sub.corr()
            for sym in sub.columns:
                cand = corr[sym].drop(sym).nlargest(n_candidates).index
                best_p, best_peer, best_b = 1.1, None, np.nan
                y = wl[sym]
                for p in cand:
                    x = wl[p]
                    ok = y.notna() & x.notna()
                    if ok.sum() < window // 2:
                        continue
                    b = np.polyfit(x[ok], y[ok], 1)[0]
                    spread = (y - b * x).dropna()
                    try:
                        pv = adfuller(spread, maxlag=1, regression="c",
                                      autolag=None)[1]
                    except Exception:
                        continue
                    if pv < best_p:
                        best_p, best_peer, best_b = pv, p, b
                if best_peer is not None:
                    j = cols.index(sym)
                    peers.iloc[i:end, j] = best_peer
                    betas.iloc[i:end, j] = best_b
    return peers, betas


def hedged_spread_z(prices: pd.DataFrame, peers: pd.DataFrame, betas: pd.DataFrame,
                    window: int = 126) -> pd.DataFrame:
    """log(Pi) - beta*log(Ppeer), z-scored on its own trailing window.

    Sign flipped so a CHEAP name (spread below normal) scores high.
    """
    lp = np.log(prices)
    peer_lp = _peer_aligned(lp, peers)
    spread = lp - betas * peer_lp
    mu = spread.rolling(window, min_periods=window // 2).mean()
    sd = spread.rolling(window, min_periods=window // 2).std()
    return -((spread - mu) / sd.replace(0, np.nan))
