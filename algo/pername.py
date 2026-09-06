"""Per-name (idiosyncratic) state labels.

The point of these is BREADTH. A market-wide regime label puts every name in the
same state on the same day, so a decade yields 7-12 independent episodes and no
amount of sophistication in the labeller fixes that. A per-name label puts each
name in its own state, so on any given day both states are populated and you can
measure a cross-sectional IC *within each state simultaneously*. The unit of
observation becomes the day, not the episode.

Every label is a comparison of a name to ITS OWN trailing history, using a
rolling median as the threshold. Rolling, not expanding-full-sample: the
threshold at time t must be computable at time t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_REF = 504          # ~2y of history defining "normal" for this name
_MIN = 252


def _above_own_median(x: pd.DataFrame, ref: int = _REF) -> pd.DataFrame:
    """Binary state: is this name's metric above its own trailing median?

    Returns float 1.0/0.0 with NaN where undefined, so it can be masked directly.
    """
    med = x.rolling(ref, min_periods=_MIN).median()
    out = (x > med).astype("float64")
    return out.where(x.notna() & med.notna())


def own_vol_state(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """1 = this name is more volatile than it usually is."""
    v = prices.pct_change().rolling(lookback).std()
    return _above_own_median(v)


def own_drawdown_state(prices: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """1 = this name is closer to its own high than it usually is."""
    dd = prices / prices.rolling(lookback, min_periods=lookback // 2).max() - 1.0
    return _above_own_median(dd)


def own_corr_state(prices: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    """1 = this name is tracking the basket more closely than it usually does.

    The interesting side is 0: a name that has DECOUPLED is being driven by
    something of its own, which is where idiosyncratic signal should live.
    """
    r = prices.pct_change()
    mkt = r.mean(axis=1)
    c = r.rolling(lookback, min_periods=lookback // 2).corr(mkt)
    return _above_own_median(c)


def own_volume_state(volume: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """1 = this name is trading more actively than it usually does."""
    v = volume.rolling(lookback).mean()
    return _above_own_median(v)


def own_trend_state(prices: pd.DataFrame, lookback: int = 200) -> pd.DataFrame:
    """1 = above its own long moving average (by more than it usually is)."""
    d = prices / prices.rolling(lookback, min_periods=lookback // 2).mean() - 1.0
    return _above_own_median(d)


REGISTRY = {
    "own_vol": own_vol_state,
    "own_drawdown": own_drawdown_state,
    "own_corr": own_corr_state,
    "own_trend": own_trend_state,
}
