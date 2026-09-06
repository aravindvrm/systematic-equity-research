"""Estimate effective bid-ask spreads from daily OHLC.

WHY THIS EXISTS
---------------
The conclusion "small caps do not help because costs cancel the larger premia"
rested on spread numbers I ASSUMED (12bp central for the S&P 600) and never
measured. That is a weak foundation for a load-bearing claim: at 12bp the
required IC is 0.091, at 6bp it is 0.037 -- the difference between "2.5x harder
than large caps" and "a wash".

Corwin & Schultz (2012, Journal of Finance) estimate the effective spread from
daily HIGH-LOW ratios. The intuition: the high-low range reflects both true
volatility and the spread, but volatility scales with the time interval while
the spread does not. Comparing a single day's range to a two-day range separates
them.

Abdi & Ranaldo (2017) is included as an independent cross-check using close
prices and high-low midpoints. Two estimators agreeing is worth far more than
either alone -- both are noisy at the daily level.

CAVEATS
-------
Both estimate the EFFECTIVE spread on the quoted market. A retail order of a few
hundred dollars often does better (price improvement, midpoint fills) and never
pays market impact. So these are upper bounds on what a small order pays, which
is the conservative direction for our purpose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_K = 3.0 - 2.0 * np.sqrt(2.0)


def corwin_schultz(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Daily effective spread estimate, as a FRACTION of price.

    Negative estimates are set to zero, per the paper: the estimator is noisy
    and negative values are not economically meaningful.
    """
    h, l = np.log(high), np.log(low)
    hl = (h - l) ** 2
    beta = hl + hl.shift(-1)

    h2 = np.log(pd.concat([high, high.shift(-1)]).groupby(level=0).max())
    l2 = np.log(pd.concat([low, low.shift(-1)]).groupby(level=0).min())
    gamma = (h2 - l2) ** 2

    alpha = ((np.sqrt(2 * beta) - np.sqrt(beta)) / _K) - np.sqrt(gamma / _K)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.where(s > 0, 0.0)


def abdi_ranaldo(high: pd.DataFrame, low: pd.DataFrame,
                 close: pd.DataFrame) -> pd.DataFrame:
    """Independent estimator: 2*sqrt(E[(c_t - eta_t)(c_t - eta_{t+1})]).

    eta is the log high-low midpoint, a proxy for the efficient price.
    """
    c = np.log(close)
    eta = (np.log(high) + np.log(low)) / 2.0
    x = (c - eta) * (c - eta.shift(-1))
    s = 2 * np.sqrt(x.clip(lower=0))
    return s.where(np.isfinite(s), np.nan)


def summarize(high, low, close, window: int = 252) -> pd.DataFrame:
    """Per-name median spread in basis points, from both estimators."""
    cs = corwin_schultz(high, low).rolling(window, min_periods=window // 2).median()
    ar = abdi_ranaldo(high, low, close).rolling(window, min_periods=window // 2).median()
    return pd.DataFrame({
        "corwin_schultz_bps": cs.median() * 1e4,
        "abdi_ranaldo_bps": ar.median() * 1e4,
    })
