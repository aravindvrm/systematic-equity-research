"""Baseline strategies.

Deliberately simple and economically motivated -- no ML. The point of these is
to exercise the pipeline honestly and give you a reference number to beat. If a
learned model cannot beat time-series momentum after costs, it has learned noise.

Every function returns a date x ticker frame of TARGET WEIGHTS computed from
information available at that bar's close. Do not shift here; the engine does it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight(prices: pd.DataFrame) -> pd.DataFrame:
    """Hold every asset equally, rebalanced every bar. Turnover baseline."""
    n = prices.notna().sum(axis=1).replace(0, np.nan)
    return prices.notna().div(n, axis=0).fillna(0.0)


def time_series_momentum(prices: pd.DataFrame, lookback: int = 252,
                         long_only: bool = True) -> pd.DataFrame:
    """Hold assets whose trailing `lookback` return is positive.

    The oldest and most replicated anomaly in the literature. Long-only here
    because the target account is a cash account (no shorting).
    """
    trailing = prices.pct_change(lookback)
    signal = (trailing > 0).astype(float)
    if not long_only:
        signal = (trailing > 0).astype(float) - (trailing < 0).astype(float)
    n = signal.abs().sum(axis=1).replace(0, np.nan)
    return signal.div(n, axis=0).fillna(0.0)


def cross_sectional_momentum(prices: pd.DataFrame, lookback: int = 252,
                             skip: int = 21, top_n: int = 3) -> pd.DataFrame:
    """Equal-weight the `top_n` assets by trailing return, skipping the last month.

    The `skip` is not decoration: short-horizon reversal partially offsets
    momentum, so the standard formulation measures 12-month return excluding the
    most recent month.
    """
    trailing = prices.shift(skip).pct_change(lookback - skip)
    ranks = trailing.rank(axis=1, ascending=False)
    selected = (ranks <= top_n).astype(float)
    n = selected.sum(axis=1).replace(0, np.nan)
    return selected.div(n, axis=0).fillna(0.0)


def sma_crossover(prices: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """Classic trend filter: long while fast SMA is above slow SMA."""
    f = prices.rolling(fast).mean()
    s = prices.rolling(slow).mean()
    signal = (f > s).astype(float)
    n = signal.sum(axis=1).replace(0, np.nan)
    return signal.div(n, axis=0).fillna(0.0)


def inverse_volatility(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Risk parity-ish: weight inversely to trailing volatility.

    Usually improves Sharpe over equal weight without predicting anything, by
    stopping one high-vol asset from dominating portfolio risk.
    """
    vol = prices.pct_change().rolling(lookback).std()
    inv = 1.0 / vol.replace(0, np.nan)
    return inv.div(inv.sum(axis=1), axis=0).fillna(0.0)


def volatility_target(weights: pd.DataFrame, prices: pd.DataFrame,
                      target_vol: float = 0.10, lookback: int = 60,
                      max_leverage: float = 1.0,
                      periods_per_year: int = 252) -> pd.DataFrame:
    """Scale an existing weight scheme so realized portfolio vol tracks target_vol.

    This forecasts NOTHING. It is a risk transformation, not a signal: it scales
    exposure down when recent volatility is high and up when it is low. It tends
    to raise Sharpe because volatility is strongly autocorrelated (calm begets
    calm) while returns are not.

    max_leverage=1.0 keeps it cash-account legal -- exposure is only ever cut,
    never levered up.
    """
    port_ret = (weights * prices.pct_change()).sum(axis=1)
    realized = port_ret.rolling(lookback).std() * (periods_per_year ** 0.5)
    scale = (target_vol / realized).clip(upper=max_leverage).fillna(0.0)
    return weights.mul(scale, axis=0)


def ensemble(*weight_frames: pd.DataFrame) -> pd.DataFrame:
    """Average several weight schemes.

    The standard defence against parameter overfitting: instead of picking the
    lookback that backtested best, hold all of them. You give up the (illusory)
    peak and keep far more of the edge out of sample.
    """
    total = sum(weight_frames)
    return total / len(weight_frames)
