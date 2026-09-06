"""Candidate features.

Every function takes a date x ticker price panel and returns a date x ticker
frame of feature values computed from data available UP TO AND INCLUDING each
bar. No function here may look forward -- the forward-looking half lives only in
research.forward_returns(), so the two can never be confused.

These are deliberately ordinary. The point is not clever features; it is a
population of plausible ones to measure honestly, so we learn which (if any)
carry signal before building anything on top.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- momentum --
def momentum(prices, lookback=252, skip=0):
    """Trailing return, optionally skipping the most recent `skip` bars."""
    return prices.shift(skip).pct_change(lookback - skip)


def momentum_12_1(prices):
    """The canonical 12-month-minus-1-month momentum."""
    return momentum(prices, 252, 21)


def reversal(prices, lookback=5):
    """Short-horizon reversal: recent losers tend to bounce. Sign flipped so
    that a HIGHER value means a more attractive asset, like every other feature
    here -- consistency matters when features get combined."""
    return -prices.pct_change(lookback)


# -------------------------------------------------------------- volatility --
def realized_vol(prices, lookback=60):
    return prices.pct_change().rolling(lookback).std() * np.sqrt(252)


def low_vol(prices, lookback=60):
    """Inverse volatility -- the low-volatility anomaly, as a rankable score."""
    return -realized_vol(prices, lookback)


def vol_change(prices, fast=20, slow=60):
    """Rising vol is typically a bad sign, so negate it."""
    return -(realized_vol(prices, fast) - realized_vol(prices, slow))


# ------------------------------------------------------------ trend quality --
def trend_strength(prices, lookback=126):
    """R-squared of a linear fit of log price on time.

    Distinguishes a smooth grind up from a volatile one that ended in the same
    place. Two assets can share a 12-month return and be very different bets.
    """
    logp = np.log(prices)

    def r2(y):
        if np.isnan(y).any():
            return np.nan
        x = np.arange(len(y))
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum() * ((y - ym) ** 2).sum()
        if denom <= 0:
            return np.nan
        return ((x - xm) @ (y - ym)) ** 2 / denom

    return logp.rolling(lookback).apply(r2, raw=True)


def ma_distance(prices, window=200):
    """Percent above/below the moving average."""
    return prices / prices.rolling(window).mean() - 1.0


def drawdown(prices, lookback=252):
    """Distance below the trailing high (negative). Less drawdown = higher."""
    return prices / prices.rolling(lookback).max() - 1.0


def near_high(prices, lookback=252):
    """52-week-high proximity -- a documented and distinct effect from momentum."""
    return prices / prices.rolling(lookback).max()


# ------------------------------------------------------------------ shapes --
def skewness(prices, lookback=126):
    """Investors overpay for lottery-like payoffs, so NEGATIVE skew has tended
    to be rewarded. Sign flipped accordingly."""
    return -prices.pct_change().rolling(lookback).skew()


def sharpe_trailing(prices, lookback=126):
    r = prices.pct_change()
    return r.rolling(lookback).mean() / r.rolling(lookback).std()


# ------------------------------------------------------------------ registry --
REGISTRY = {
    "mom_21":        lambda p: momentum(p, 21),
    "mom_63":        lambda p: momentum(p, 63),
    "mom_126":       lambda p: momentum(p, 126),
    "mom_252":       lambda p: momentum(p, 252),
    "mom_12_1":      momentum_12_1,
    "reversal_5":    lambda p: reversal(p, 5),
    "reversal_21":   lambda p: reversal(p, 21),
    "low_vol_60":    lambda p: low_vol(p, 60),
    "vol_change":    vol_change,
    "trend_r2_126":  lambda p: trend_strength(p, 126),
    "ma_dist_200":   lambda p: ma_distance(p, 200),
    "drawdown_252":  lambda p: drawdown(p, 252),
    "near_high_252": lambda p: near_high(p, 252),
    "skew_126":      skewness,
    "sharpe_126":    lambda p: sharpe_trailing(p, 126),
}


def compute_all(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: fn(prices) for name, fn in REGISTRY.items()}
