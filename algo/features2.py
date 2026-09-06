"""Second feature batch: the categories the first screen never touched.

Batch one was entirely PER-ASSET and used only closing prices. This adds:

  VOLUME   -- present in the data since day one and never used
  RANGE    -- high/low/open carry information the close does not
  RELATIONAL -- features that only exist BETWEEN assets. A per-asset view
                cannot see relative strength, dispersion regimes, lead-lag,
                or what a name does once its market beta is removed.

The relational block is the interesting one: because the market factor explains
~40-55% of variance in these universes, a per-asset feature is mostly measuring
the market. Stripping that out is the cheapest way to look somewhere new.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ volume --
def volume_trend(volume, fast=5, slow=60):
    return volume.rolling(fast).mean() / volume.rolling(slow).mean() - 1.0


def volume_shock(volume, lookback=60):
    v = np.log1p(volume)
    return (v - v.rolling(lookback).mean()) / v.rolling(lookback).std()


def price_volume_divergence(prices, volume, lookback=21):
    """Price rising on falling volume is classically read as weak."""
    pr = prices.pct_change(lookback)
    vr = volume.rolling(lookback).mean() / volume.rolling(lookback * 3).mean() - 1.0
    return np.sign(pr) * vr


def dollar_volume(prices, volume, lookback=21):
    return np.log1p((prices * volume).rolling(lookback).mean())


# ------------------------------------------------------------------- range --
def garman_klass_vol(o, h, l, c, lookback=21):
    """Range-based volatility estimator -- far more efficient than close-to-close
    because it uses the whole bar. Negated so lower vol scores higher."""
    hl = (np.log(h / l)) ** 2
    co = (np.log(c / o)) ** 2
    gk = 0.5 * hl - (2 * np.log(2) - 1) * co
    return -np.sqrt(gk.rolling(lookback).mean().clip(lower=0) * 252)


def close_position_in_range(h, l, c, lookback=21):
    """Where the close sits within the recent high-low band (0=low, 1=high)."""
    hi = h.rolling(lookback).max(); lo = l.rolling(lookback).min()
    return (c - lo) / (hi - lo).replace(0, np.nan)


def intraday_vs_overnight(o, c):
    """Overnight drift and intraday drift behave very differently; this is the
    trailing balance between them."""
    intraday = (c / o - 1.0).rolling(21).mean()
    overnight = (o / c.shift(1) - 1.0).rolling(21).mean()
    return overnight - intraday


def true_range_pct(h, l, c, lookback=21):
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()]).groupby(level=0).max()
    return -(tr / c).rolling(lookback).mean()


# -------------------------------------------------------------- relational --
def relative_strength(prices, lookback=126):
    """Own return minus the universe's equal-weighted return."""
    r = prices.pct_change(lookback)
    return r.sub(r.mean(axis=1), axis=0)


def residual_momentum(prices, lookback=126, beta_window=252):
    """Momentum AFTER removing exposure to the equal-weighted universe.

    The single most promising relational idea: if the market factor explains
    half the variance, then plain momentum is mostly market timing. This asks
    what a name did on its own account.
    """
    r = prices.pct_change()
    mkt = r.mean(axis=1)
    cov = r.rolling(beta_window).cov(mkt)
    var = mkt.rolling(beta_window).var()
    beta = cov.div(var, axis=0)
    resid = r - beta.mul(mkt, axis=0)
    return resid.rolling(lookback).sum()


def beta_to_universe(prices, window=252):
    """Low-beta names have historically outperformed risk-adjusted. Negated."""
    r = prices.pct_change(); mkt = r.mean(axis=1)
    return -r.rolling(window).cov(mkt).div(mkt.rolling(window).var(), axis=0)


def correlation_to_universe(prices, window=126):
    """How much a name moves with everything else. Less is more diversifying."""
    r = prices.pct_change(); mkt = r.mean(axis=1)
    return -r.rolling(window).corr(mkt)


def dispersion_regime(prices, window=21):
    """Cross-sectional dispersion -- a REGIME signal, identical for every asset.

    High dispersion means stock selection has more to work with. Broadcast
    across columns so it can be screened alongside per-asset features.
    """
    disp = prices.pct_change().std(axis=1).rolling(window).mean()
    return pd.DataFrame({c: disp for c in prices.columns})


def lead_lag(prices, leader_lookback=5, window=252):
    """Does the universe's recent move predict this name's next one?

    Names that systematically lag the aggregate are the classic lead-lag effect.
    Measured as trailing correlation of own return to the universe's LAGGED
    return, times that lagged return.
    """
    r = prices.pct_change()
    mkt_lag = r.mean(axis=1).shift(1).rolling(leader_lookback).sum()
    sens = r.rolling(window).corr(mkt_lag)
    return sens.mul(mkt_lag, axis=0)


def idio_vol_share(prices, window=126):
    """Fraction of variance NOT explained by the universe. High = more
    independent, which is where cross-sectional signal can live."""
    r = prices.pct_change(); mkt = r.mean(axis=1)
    corr = r.rolling(window).corr(mkt)
    return 1.0 - corr ** 2
