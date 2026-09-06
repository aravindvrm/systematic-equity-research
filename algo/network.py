"""Relational features that are NOT name-vs-basket and NOT simple pair correlation.

features2.py compares each name to the universe AGGREGATE. relational.py compares
it to its single most-correlated peer. Both leave large parts of the relational
space untouched:

    net_centrality   position in the correlation GRAPH, weighted by whom you are
                     correlated to -- distinct from correlation_to_universe, which
                     treats every counterparty as interchangeable
    size_lead_lag    DIRECTIONAL diffusion: large firms' returns leading small
                     firms' within the same industry (Hou 2007)
    down_corr_asym   correlation on down days minus up days (Ang/Chen/Xing);
                     symmetric measures average this away entirely
    beta_instability how unstable this name's market exposure has been
    corr_dispersion  spread of a name's correlations across the cross-section

All are point-in-time: rolling windows only, refit forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _refit_points(n, min_train, every):
    return range(min_train, n, every)


def net_centrality(prices: pd.DataFrame, window: int = 252, refit_every: int = 63,
                   min_train: int = 252) -> pd.DataFrame:
    """Eigenvector centrality in the |correlation| graph.

    Principal eigenvector of the absolute correlation matrix. High = this name
    sits at the centre of the co-movement structure; low = it is peripheral and
    doing its own thing. Negated so PERIPHERAL scores high, matching the
    idiosyncratic-premium prior.
    """
    rets = prices.pct_change()
    out = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for i in _refit_points(len(prices), min_train, refit_every):
        w = rets.iloc[max(0, i - window):i].dropna(axis=1, thresh=int(0.8 * window))
        if w.shape[1] < 5:
            continue
        c = w.corr().abs().fillna(0.0).to_numpy()
        vals, vecs = np.linalg.eigh(c)
        v = np.abs(vecs[:, -1])          # principal eigenvector
        end = min(i + refit_every, len(prices))
        out.iloc[i:end, [out.columns.get_loc(s) for s in w.columns]] = -v
    return out


def size_lead_lag(prices: pd.DataFrame, size_proxy: pd.DataFrame,
                  sectors: pd.Series, lookback: int = 5, top_frac: float = 0.3,
                  window: int = 252, refit_every: int = 63) -> pd.DataFrame:
    """Recent return of the LARGE names in this name's own sector.

    Hou (2007): information diffuses from large firms to small firms within an
    industry, so big-firm returns lead small-firm returns. Each name receives the
    lagged return of the large cohort of its sector, EXCLUDING itself (otherwise a
    large name is handed its own return and the feature becomes momentum).
    """
    lagged = prices.pct_change(lookback)
    out = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    sz = size_proxy.rolling(window, min_periods=window // 2).median()

    for sec, grp in sectors.groupby(sectors):
        cols = [c for c in grp.index if c in prices.columns]
        if len(cols) < 4:
            continue
        s_sz, s_ret = sz[cols], lagged[cols]
        rank = s_sz.rank(axis=1, pct=True, ascending=False)
        is_big = rank <= top_frac
        big_ret = s_ret.where(is_big)
        tot, cnt = big_ret.sum(axis=1), is_big.sum(axis=1)
        for c in cols:
            # leave-one-out: a big name must not be fed its own return
            own = s_ret[c].where(is_big[c], 0.0)
            k = cnt - is_big[c].astype(int)
            out[c] = ((tot - own) / k.where(k > 0)).where(k > 0)
    return out


def down_corr_asym(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Correlation to the basket on DOWN days minus on UP days.

    Negated: a name that decouples when the market falls is the desirable one, so
    LOW asymmetry scores high.
    """
    r = prices.pct_change()
    mkt = r.mean(axis=1)
    down, up = mkt < 0, mkt > 0
    rd, ru = r.where(down), r.where(up)
    md, mu = mkt.where(down), mkt.where(up)
    cd = rd.rolling(window, min_periods=window // 4).corr(md)
    cu = ru.rolling(window, min_periods=window // 4).corr(mu)
    return -(cd - cu)


def beta_instability(prices: pd.DataFrame, beta_window: int = 126,
                     std_window: int = 252) -> pd.DataFrame:
    """Volatility of the name's own rolling market beta. Negated: stable = high."""
    r = prices.pct_change()
    mkt = r.mean(axis=1)
    cov = r.rolling(beta_window, min_periods=beta_window // 2).cov(mkt)
    var = mkt.rolling(beta_window, min_periods=beta_window // 2).var()
    beta = cov.div(var, axis=0)
    return -beta.rolling(std_window, min_periods=std_window // 2).std()


def corr_dispersion(prices: pd.DataFrame, window: int = 252, refit_every: int = 63,
                    min_train: int = 252) -> pd.DataFrame:
    """Spread of this name's correlations to every other name.

    High dispersion = the name is tightly linked to some and unlinked to others,
    i.e. it has real structure. Low = it relates to everything equally.
    """
    rets = prices.pct_change()
    out = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    for i in _refit_points(len(prices), min_train, refit_every):
        w = rets.iloc[max(0, i - window):i].dropna(axis=1, thresh=int(0.8 * window))
        if w.shape[1] < 5:
            continue
        c = w.corr()
        np.fill_diagonal(c.values.copy(), np.nan)
        d = c.where(~np.eye(len(c), dtype=bool)).std()
        end = min(i + refit_every, len(prices))
        out.iloc[i:end, [out.columns.get_loc(s) for s in w.columns]] = d.to_numpy()
    return out
