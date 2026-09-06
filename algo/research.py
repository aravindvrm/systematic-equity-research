"""Signal research: measure whether a feature predicts anything.

This is the only module allowed to look forward. Keeping the forward-return
construction in exactly one place is deliberate -- lookahead bugs happen when
"what I knew" and "what happened next" get computed side by side.

The headline number is the INFORMATION COEFFICIENT: the cross-sectional rank
correlation between a feature today and returns over the next h bars. Recall the
calibration: professional signals run IC 0.02-0.05. Anything above ~0.10 on
daily data should be treated as a bug until proven otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as _features


def forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Return from bar t to bar t+horizon, indexed at t.

    THE ONLY forward-looking function in the codebase.
    """
    return prices.shift(-horizon) / prices - 1.0


def cross_sectional_ic(feature: pd.DataFrame, fwd: pd.DataFrame,
                       min_names: int = 5) -> pd.Series:
    """Per-date Spearman rank IC across assets.

    Rank correlation rather than Pearson: returns are fat-tailed and one
    outlier can manufacture or destroy an apparent linear relationship.
    """
    f = feature.rank(axis=1)
    r = fwd.rank(axis=1)
    valid = f.notna() & r.notna()
    n = valid.sum(axis=1)

    fm = f.where(valid); rm = r.where(valid)
    fc = fm.sub(fm.mean(axis=1), axis=0)
    rc = rm.sub(rm.mean(axis=1), axis=0)
    num = (fc * rc).sum(axis=1)
    den = np.sqrt((fc**2).sum(axis=1) * (rc**2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    return ic.where(n >= min_names).dropna()


def ic_stats(ic: pd.Series, horizon: int = 1) -> dict:
    """Summarize an IC series.

    For horizon > 1 the daily IC observations overlap, which inflates the
    t-statistic. We deflate by sqrt(horizon) -- a standard, slightly
    conservative correction (Newey-West would be more precise).
    """
    if ic.empty:
        return dict(ic_mean=0.0, ic_std=0.0, ic_t=0.0, ic_ir=0.0, hit_rate=0.0, n=0)
    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    t = mean / (std / np.sqrt(len(ic))) if std > 0 else 0.0
    return dict(
        ic_mean=mean,
        ic_std=std,
        ic_t=float(t / np.sqrt(horizon)),      # overlap-adjusted
        ic_ir=float(mean / std) if std > 0 else 0.0,
        hit_rate=float((ic > 0).mean()),       # fraction of days pointing the right way
        n=int(len(ic)),
    )


def evaluate(prices: pd.DataFrame, feature: pd.DataFrame, horizon: int = 1) -> dict:
    fwd = forward_returns(prices, horizon)
    ic = cross_sectional_ic(feature, fwd)
    out = ic_stats(ic, horizon)
    out["ic_series"] = ic
    return out


def screen(prices: pd.DataFrame, horizons=(1, 5, 21),
           feature_dict: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Measure every feature at every horizon. Returns a tidy frame.

    Includes a SHUFFLED control for each feature: the same values with the
    cross-section randomly permuted each day, which destroys any real signal
    while preserving the distribution. If a real feature's IC is not clearly
    above its shuffled twin, you have found noise.
    """
    feats = feature_dict or _features.compute_all(prices)
    rng = np.random.default_rng(0)
    rows = []
    for name, f in feats.items():
        shuffled = f.apply(lambda row: pd.Series(
            rng.permutation(row.values), index=row.index), axis=1)
        for h in horizons:
            for label, ff in (("real", f), ("shuffled", shuffled)):
                s = evaluate(prices, ff, h)
                rows.append(dict(feature=name, kind=label, horizon=h,
                                 ic=s["ic_mean"], ic_t=s["ic_t"],
                                 ic_ir=s["ic_ir"], hit=s["hit_rate"], n=s["n"]))
    return pd.DataFrame(rows)


def ic_by_year(prices: pd.DataFrame, feature: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """IC per calendar year. A feature that only worked in one regime shows up
    here and nowhere else."""
    ic = cross_sectional_ic(feature, forward_returns(prices, horizon))
    return ic.groupby(ic.index.year).mean()


def bonferroni_t(n_features: int, alpha: float = 0.05) -> float:
    """The |t| a feature must clear when you tested `n_features` of them.

    Testing 15 features at the usual t>2 means you expect ~0.75 false positives
    by chance. This is the corrected bar.
    """
    from scipy.stats import norm
    return float(norm.ppf(1 - alpha / (2 * n_features)))


# ------------------------------------------------- time-series (per asset) --
def time_series_ic(feature: pd.DataFrame, fwd: pd.DataFrame,
                   min_obs: int = 100) -> pd.Series:
    """Per-ASSET rank IC through time.

    Cross-sectional IC asks "can I rank these assets against each other today?"
    Time-series IC asks "can I time each asset against its own history?" A
    long-only trend portfolio that goes to cash is doing the second one, so this
    is the number that describes it.
    """
    out = {}
    for col in feature.columns:
        f, r = feature[col], fwd[col]
        ok = f.notna() & r.notna()
        if ok.sum() < min_obs:
            continue
        out[col] = f[ok].rank().corr(r[ok].rank())
    return pd.Series(out)


def binary_signal_test(feature: pd.DataFrame, fwd: pd.DataFrame,
                       threshold: float = 0.0, periods_per_year: int = 252) -> dict:
    """Test the ACTUAL decision rule: hold when feature > threshold, else cash.

    A correlation can be nonzero while the specific binary cut the strategy uses
    captures none of it. This measures the thing the code really does: the gap
    between forward returns when the signal is on versus off.
    """
    on = feature > threshold
    ok = feature.notna() & fwd.notna()
    r_on = fwd.where(on & ok).stack().dropna()
    r_off = fwd.where(~on & ok).stack().dropna()
    if r_on.empty or r_off.empty:
        return dict(n_on=len(r_on), n_off=len(r_off), spread_ann=0.0, t_stat=0.0)

    diff = float(r_on.mean() - r_off.mean())
    se = np.sqrt(r_on.var(ddof=1) / len(r_on) + r_off.var(ddof=1) / len(r_off))
    return dict(
        n_on=int(len(r_on)), n_off=int(len(r_off)),
        mean_on_ann=float(r_on.mean() * periods_per_year),
        mean_off_ann=float(r_off.mean() * periods_per_year),
        spread_ann=float(diff * periods_per_year),
        t_stat=float(diff / se) if se > 0 else 0.0,
        pct_on=float(on.where(ok).stack().mean()),
    )


def effective_bets(returns: pd.DataFrame) -> float:
    """Participation ratio of the correlation-matrix eigenvalues."""
    corr = returns.corr().to_numpy()
    lam = np.linalg.eigvalsh(corr)
    lam = lam[lam > 0]
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def static_vs_dynamic(prices: pd.DataFrame, feature: pd.DataFrame,
                      horizon: int = 1) -> dict:
    """Separate a feature's STATIC tilt from its TIME-VARYING signal.

    A feature whose cross-sectional ordering barely changes is not forecasting;
    it is a fixed bet. Cross-sectional IC cannot see the difference, because
    every day's observation is nearly the same one repeated -- which inflates
    the t-statistic enormously while the effective sample size is closer to 1.

    Subtracting each asset's own long-run mean removes the fixed ranking and
    leaves only variation through time. A real signal survives; a tilt does not.

    ALWAYS run this before believing a cross-sectional IC. In a small universe
    with persistent asset-class differences, it is the default failure mode.

    !! DIAGNOSTIC ONLY -- NEVER TRADE THE DEMEANED SERIES !!
    The demeaning subtracts each asset's FULL-SAMPLE mean, which is future
    information: at time t you do not know a company's average feature level over
    the years that follow. It is valid for answering "is there time variation
    here"; it is a lookahead if used to build weights.

    This exact mistake produced an insider signal with IC 0.0275 (t=3.14) that
    fell to 0.0083 (t=0.75) under expanding-window demeaning and showed a
    walk-forward edge of +0.03 across four windows. The BACKTEST barely moved
    (+0.15 -> +0.12), because ranks and the risk stack absorbed the difference --
    so only the walk-forward caught it.

    For a tradeable version use `dynamic_pit()` below.
    """
    fwd = forward_returns(prices, horizon)
    raw = ic_stats(cross_sectional_ic(feature, fwd), horizon)
    demeaned = feature.sub(feature.mean(axis=0), axis=1)
    dyn = ic_stats(cross_sectional_ic(demeaned, fwd), horizon)

    ranks = feature.rank(axis=1)
    stability = float(ranks.corrwith(ranks.shift(252), axis=1).mean())

    retained = (abs(dyn["ic_mean"]) / abs(raw["ic_mean"])) if raw["ic_mean"] else 0.0
    return dict(
        rank_stability=stability,
        ic_raw=raw["ic_mean"], t_raw=raw["ic_t"],
        ic_dynamic=dyn["ic_mean"], t_dynamic=dyn["ic_t"],
        fraction_retained=float(retained),
        is_static=bool(stability > 0.5 or retained < 0.5),
    )


# ------------------------------------------------------------ event studies --
def count_episodes(condition: pd.DataFrame, gap_days: int = 5) -> int:
    """Number of INDEPENDENT episodes in a boolean setup frame.

    When the market falls, dozens of names trigger on the same day. Counting
    each name-day as an observation overstates the sample by an order of
    magnitude. An episode is a run of trigger days separated by more than
    `gap_days` from the next.
    """
    days = condition.any(axis=1)
    d = days[days].index
    if len(d) == 0:
        return 0
    if len(d) == 1:
        return 1
    gaps = np.diff(d.values).astype("timedelta64[D]").astype(int)
    return 1 + int((gaps > gap_days).sum())


def event_study(prices: pd.DataFrame, condition: pd.DataFrame,
                horizons=(1, 5, 21), periods_per_year: int = 252) -> pd.DataFrame:
    """Conditional forward returns when a boolean setup fires.

    IC is the wrong lens for setups. It averages over every observation, so a
    rule that fires 2% of the time and works is diluted into invisibility -- and
    rank IC only sees MONOTONE relationships, so "extremes in either direction
    mean revert" scores zero by construction.

    This instead asks the question the strategy actually poses: given the setup
    fired, what happened next, versus what normally happens?

    `condition` is a date x ticker boolean frame aligned to `prices`.
    """
    out = []
    cond = condition.reindex_like(prices).fillna(False).astype(bool)
    for h in horizons:
        fwd = forward_returns(prices, h)
        ok = fwd.notna()
        hit = fwd.where(cond & ok).stack().dropna()
        base = fwd.where(~cond & ok).stack().dropna()
        if len(hit) < 30 or base.empty:
            out.append(dict(horizon=h, n_events=len(hit), n_episodes=0,
                            mean=np.nan, base=np.nan, edge=np.nan,
                            t_naive=np.nan, t=np.nan, win=np.nan, ann_edge=np.nan))
            continue
        se = np.sqrt(hit.var(ddof=1) / len(hit) + base.var(ddof=1) / len(base))
        edge = float(hit.mean() - base.mean())
        t_naive = float(edge / se) if se > 0 else 0.0

        # TWO deflations, both mandatory. Reporting t_naive alone is how a
        # correlated sample produces a t-stat of 10 on an effect of zero.
        #   overlap:  consecutive h-day forward returns share h-1 days
        #   clustering: many names trigger on the same market-wide day
        episodes = count_episodes(cond)
        per_episode = len(hit) / max(episodes, 1)
        t_adj = t_naive / np.sqrt(h) / np.sqrt(max(per_episode, 1.0))

        out.append(dict(
            horizon=h, n_events=int(len(hit)), n_episodes=episodes,
            mean=float(hit.mean()), base=float(base.mean()), edge=edge,
            t_naive=t_naive, t=t_adj,
            win=float((hit > 0).mean()),
            ann_edge=float(edge * (periods_per_year / h)),
        ))
    return pd.DataFrame(out)


def zscore(prices: pd.DataFrame, lookback: int = 21, window: int = 252) -> pd.DataFrame:
    """Trailing return over `lookback`, standardized by its own history."""
    r = prices.pct_change(lookback)
    return (r - r.rolling(window).mean()) / r.rolling(window).std()


def dynamic_pit(feature: pd.DataFrame, method: str = "expanding",
                window: int = 504, min_periods: int = 252) -> pd.DataFrame:
    """Point-in-time version of the static/dynamic decomposition.

    Same idea as `static_vs_dynamic` -- strip each asset's own baseline so only
    time variation remains -- but using ONLY past data at every point. This is
    the form that may be traded.

    method="expanding": mean of everything up to t (stable, slow to adapt)
    method="rolling":   trailing `window` bars (adapts to regime, noisier)
    """
    if method == "rolling":
        base = feature.rolling(window, min_periods=min_periods).mean()
    else:
        base = feature.expanding(min_periods=min_periods).mean()
    return feature - base
