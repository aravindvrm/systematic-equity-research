"""Evaluate a strategy the way an investor with an EXISTING portfolio should.

THE METHODOLOGICAL ERROR THIS FIXES
-----------------------------------
Every result in this project was judged on STANDALONE Sharpe against a
vol-targeted equal-weight equity benchmark. Two things are wrong with that.

1. WRONG BAR. That benchmark ran at Sharpe ~1.05 over 2016-2026. Long-run US
   equity Sharpe is ~0.4-0.5. We demanded that every signal beat one of the best
   equity decades in history, measured over that same decade.

2. WRONG QUESTION. The stated goal was a SEPARATE sleeve alongside an existing
   balanced portfolio. For that purpose standalone Sharpe is close to
   irrelevant; what matters is MARGINAL CONTRIBUTION, which is driven by
   CORRELATION. A Sharpe-0.6 strategy uncorrelated with your core adds more than
   a Sharpe-0.9 strategy that is 90% equity beta -- because you already own
   equity beta.

The right test is: does adding this at a realistic weight improve the portfolio
I actually hold?

STATISTICS
----------
Alpha t-stats use Newey-West standard errors. Daily strategy returns are
autocorrelated (vol targeting, band rebalancing, overlapping signals) and OLS
standard errors understate the uncertainty, sometimes badly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return df["a"], df["b"]


def alpha_beta(strategy: pd.Series, core: pd.Series,
               periods_per_year: int = 252, lags: int = 21) -> dict:
    """Regress strategy on core. Newey-West t on the intercept."""
    import statsmodels.api as sm

    s, c = _align(strategy, core)
    if len(s) < 60:
        return dict(alpha_ann=np.nan, alpha_t=np.nan, beta=np.nan, r2=np.nan, n=len(s))
    X = sm.add_constant(c.to_numpy())
    fit = sm.OLS(s.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return dict(
        alpha_ann=float(fit.params[0]) * periods_per_year,
        alpha_t=float(fit.tvalues[0]),
        beta=float(fit.params[1]),
        r2=float(fit.rsquared),
        n=len(s),
    )


def blend_curve(strategy: pd.Series, core: pd.Series,
                weights=(0.0, 0.05, 0.10, 0.20, 0.30, 0.50),
                periods_per_year: int = 252) -> pd.DataFrame:
    """Sharpe / drawdown of core+strategy blends, rebalanced daily."""
    s, c = _align(strategy, core)
    rows = []
    for w in weights:
        r = (1 - w) * c + w * s
        eq = (1 + r).cumprod()
        rows.append(dict(
            w_strategy=w,
            sharpe=metrics.sharpe(r, periods_per_year=periods_per_year),
            cagr=metrics.cagr(eq, periods_per_year),
            vol=metrics.volatility(r, periods_per_year),
            max_dd=metrics.max_drawdown(eq),
            calmar=metrics.calmar(eq, periods_per_year),
        ))
    return pd.DataFrame(rows)


def optimal_weight(strategy: pd.Series, core: pd.Series) -> float:
    """Sharpe-maximising weight on the strategy in a two-asset blend.

    Closed form for two assets. Clipped to [0, 1] -- shorting the core or
    levering the sleeve is not available in a cash account, so an unconstrained
    optimum outside that range is not actionable.
    """
    s, c = _align(strategy, core)
    mu = np.array([s.mean(), c.mean()])
    S = np.cov(np.vstack([s.to_numpy(), c.to_numpy()]))
    try:
        w = np.linalg.solve(S, mu)
    except np.linalg.LinAlgError:
        return np.nan
    if w.sum() == 0:
        return np.nan
    w = w / w.sum()
    return float(np.clip(w[0], 0.0, 1.0))


def evaluate(strategy: pd.Series, core: pd.Series, label: str = "",
             periods_per_year: int = 252) -> dict:
    """The full marginal picture for one strategy against one core."""
    s, c = _align(strategy, core)
    ab = alpha_beta(s, c, periods_per_year)
    w_opt = optimal_weight(s, c)
    base = metrics.sharpe(c, periods_per_year=periods_per_year)
    b10 = metrics.sharpe(0.9 * c + 0.1 * s, periods_per_year=periods_per_year)
    b_opt = (metrics.sharpe((1 - w_opt) * c + w_opt * s,
                            periods_per_year=periods_per_year)
             if np.isfinite(w_opt) else np.nan)
    return dict(
        label=label,
        sharpe_standalone=metrics.sharpe(s, periods_per_year=periods_per_year),
        corr_to_core=float(s.corr(c)),
        beta=ab["beta"],
        alpha_ann=ab["alpha_ann"],
        alpha_t=ab["alpha_t"],
        core_sharpe=base,
        sharpe_at_10pct=b10,
        delta_at_10pct=b10 - base,
        w_optimal=w_opt,
        sharpe_at_opt=b_opt,
        delta_at_opt=b_opt - base,
        n=ab["n"],
    )


# ---------------------------------------------------------------------------
# THE NULL FLOOR -- the only control that actually works.
#
# A beta-matched core/cash mix is NOT a sufficient control for a vol-targeted
# sleeve. Vol targeting cuts equity exposure exactly when volatility spikes,
# which is exactly when a 60/40 core crashes, so the sleeve's true beta is
# time-varying and conditionally low when it matters most. A static full-sample
# beta cannot represent that, and the residual shows up as fake alpha.
#
# Measured (2026-09-05, 40 draws): random signals through a top-30%,
# inverse-vol, 10%-vol-targeted stack produced
#       genuine gain @10% = +0.029 mean, +0.035 p95
#       alpha Newey-West t = +3.14 mean, +3.95 max
# i.e. ZERO-information signals cleared an |t| > 3 bar on average.
#
# The fix is to hold the ENTIRE construction pipeline constant and vary only the
# information content: run N random signals through the identical stack and read
# the strategy's percentile against that distribution.
# ---------------------------------------------------------------------------

def null_floor(build_returns, n: int = 40, seed0: int = 1000) -> dict:
    """Distribution of a metric under zero-information signals.

    build_returns: callable(np.random.Generator) -> return Series, which MUST
        route the random signal through the same stack, universe, costs and
        rebalancing as the strategy under test. Anything held constant here is
        controlled for; anything not is not.
    """
    import numpy as _np

    out = []
    for i in range(n):
        out.append(build_returns(_np.random.default_rng(seed0 + i)))
    return {"returns": out, "n": n}


def percentile_vs_null(value: float, null_values) -> float:
    """Where a statistic falls in the null distribution, as a percentile."""
    a = np.asarray([v for v in null_values if np.isfinite(v)])
    if a.size == 0:
        return np.nan
    return float((a < value).mean() * 100.0)
