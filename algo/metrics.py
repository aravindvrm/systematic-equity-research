"""Performance metrics.

Adapted from traderlab-backend/app/metrics/core.py, decoupled from DuckDB so the
same evaluation layer works on backtest output and on real broker fills.

Everything here takes plain pandas objects. `periods_per_year` must match the
bar frequency of the returns you pass in -- 252 for daily equities, 365 for
daily crypto (it trades every calendar day, so a year has more bars).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
CRYPTO_DAYS = 365


def max_drawdown(equity: pd.Series) -> float:
    """Most negative peak-to-trough move, as a fraction (-0.20 == -20%)."""
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    return equity / equity.cummax() - 1.0


# Volatilities below this are treated as zero. A constant return series has a
# floating-point std around 1e-19 rather than exactly 0, which would otherwise
# produce a Sharpe of ~1e16 instead of the 0 it should be.
_VOL_EPS = 1e-12


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized Sharpe. `rf` is an annual rate."""
    if returns.empty:
        return 0.0
    vol = returns.std(ddof=0)
    if not np.isfinite(vol) or vol < _VOL_EPS:
        return 0.0
    excess = returns - rf / periods_per_year
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    if returns.empty:
        return 0.0
    excess = returns - rf / periods_per_year
    # Downside deviation is the RMS of the full series with upside set to zero,
    # not the std of the negative subset. Using the subset overstates Sortino
    # (measured 16% high against empyrical) because it both drops the zeros and
    # centres on the negative mean instead of on zero.
    downside = excess.clip(upper=0.0)
    if (downside < 0).sum() == 0:
        return 0.0
    dvol = float(np.sqrt((downside ** 2).mean()))
    if not np.isfinite(dvol) or dvol < _VOL_EPS:
        return 0.0
    return float(excess.mean() / dvol * np.sqrt(periods_per_year))


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    if equity.empty or equity.iloc[0] <= 0:
        return 0.0
    n_years = len(equity) / float(periods_per_year)
    if n_years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_years) - 1.0)


def volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def calmar(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr(equity, periods_per_year) / mdd)


def deflated_sharpe(observed_sharpe: float, n_trials: int, n_obs: int) -> float:
    """Haircut a Sharpe for the number of strategy variants you tried.

    Selecting the best of `n_trials` backtests inflates Sharpe even when every
    variant is pure noise. This returns the Sharpe you should actually believe.
    Simplified Bailey/Lopez de Prado: subtract the expected maximum of `n_trials`
    draws from a null distribution of zero-Sharpe strategies.

    If this number is near zero, the backtest found nothing -- regardless of how
    good the raw Sharpe looked.
    """
    if n_trials < 1 or n_obs < 2:
        return observed_sharpe
    if n_trials == 1:
        return observed_sharpe
    from scipy.stats import norm

    euler = 0.5772156649
    # Expected max of n_trials standard normals.
    e_max = (1 - euler) * norm.ppf(1 - 1.0 / n_trials) + euler * norm.ppf(
        1 - 1.0 / (n_trials * np.e)
    )
    # Standard error of an estimated Sharpe over n_obs observations.
    se = np.sqrt((1 + 0.5 * observed_sharpe**2) / n_obs)
    return float(observed_sharpe - e_max * se)


def trade_stats(trade_pnl: pd.Series) -> dict:
    """Win rate / profit factor / expectancy from a series of per-trade P&L."""
    if trade_pnl.empty:
        return dict(n_trades=0, win_rate=0.0, profit_factor=0.0,
                    avg_win=0.0, avg_loss=0.0, expectancy=0.0)
    wins = trade_pnl[trade_pnl > 0]
    losses = trade_pnl[trade_pnl < 0]
    win_rate = len(wins) / len(trade_pnl)
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    gross_loss = float(-losses.sum())
    return dict(
        n_trades=int(len(trade_pnl)),
        win_rate=float(win_rate),
        profit_factor=float(wins.sum() / gross_loss) if gross_loss > 0 else np.inf,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=float(win_rate * avg_win + (1 - win_rate) * avg_loss),
    )


def summary(equity: pd.Series, returns: pd.Series | None = None,
            trade_pnl: pd.Series | None = None,
            periods_per_year: int = TRADING_DAYS,
            rf: float = 0.0) -> dict:
    """Full performance summary for an equity curve."""
    if returns is None:
        returns = equity.pct_change().dropna()
    out = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) else 0.0,
        "cagr": cagr(equity, periods_per_year),
        "vol": volatility(returns, periods_per_year),
        "sharpe": sharpe(returns, rf, periods_per_year),
        "sortino": sortino(returns, rf, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(equity, periods_per_year),
        "n_periods": int(len(equity)),
    }
    if trade_pnl is not None:
        out.update(trade_stats(trade_pnl))
    return out


def format_summary(s: dict) -> str:
    """Human-readable one-block rendering of a summary dict."""
    pct = {"total_return", "cagr", "vol", "max_drawdown", "win_rate"}
    lines = []
    for k, v in s.items():
        if isinstance(v, float):
            lines.append(f"  {k:16s} {v*100:>9.2f}%" if k in pct else f"  {k:16s} {v:>10.3f}")
        else:
            lines.append(f"  {k:16s} {v:>10}")
    return "\n".join(lines)


def tail_stats(returns: pd.Series) -> dict:
    """Distribution shape. Sharpe assumes roughly Gaussian returns; when excess
    kurtosis is large, Sharpe systematically flatters strategies that are
    quietly selling tail risk."""
    from scipy import stats as _st

    if len(returns) < 3:
        return dict(skew=0.0, excess_kurtosis=0.0, worst=0.0, best=0.0)
    return dict(
        skew=float(_st.skew(returns)),
        excess_kurtosis=float(_st.kurtosis(returns)),
        worst=float(returns.min()),
        best=float(returns.max()),
    )


def split_stability(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Sharpe in the first vs second half of the sample.

    A strategy whose edge lives entirely in one half is describing a regime, not
    a rule. This is the cheapest and most informative robustness check there is.
    """
    if len(returns) < 20:
        return dict(first_half=0.0, second_half=0.0, retention=0.0)
    mid = len(returns) // 2
    s1 = sharpe(returns.iloc[:mid], periods_per_year=periods_per_year)
    s2 = sharpe(returns.iloc[mid:], periods_per_year=periods_per_year)
    return dict(
        first_half=s1,
        second_half=s2,
        retention=float(s2 / s1) if s1 > 0 else 0.0,
    )
