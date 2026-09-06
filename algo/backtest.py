"""Vectorized backtest engine with explicit lookahead protection.

The single most important line in this file is the `.shift(1)` in `run()`.

A signal computed from bar t's close cannot be traded until bar t+1. Forgetting
this is the most common way a backtest reports a Sharpe of 3 for a strategy that
loses money live, and it is invisible in the output -- the equity curve just
looks great. So the shift happens here, once, in the engine, and strategies are
forbidden from doing it themselves.

Convention: a strategy returns a *target weight* per asset per bar, computed
using data available up to and including that bar's close. The engine handles
the lag, the turnover, and the costs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics
from .costs import CostModel, IBKR_US_EQUITY


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    gross_returns: pd.Series
    periods_per_year: int = metrics.TRADING_DAYS
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        s = metrics.summary(self.equity, self.returns,
                            periods_per_year=self.periods_per_year)
        s["total_cost_drag"] = float(self.costs.sum())
        s["avg_turnover"] = float(self.turnover.mean())
        gross_eq = (1 + self.gross_returns).cumprod()
        s["gross_sharpe"] = metrics.sharpe(self.gross_returns,
                                           periods_per_year=self.periods_per_year)
        s["gross_cagr"] = metrics.cagr(gross_eq, self.periods_per_year)
        return s

    def diagnostics(self, benchmark: "BacktestResult | None" = None,
                    benchmark_name: str = "benchmark", n_trials: int = 1,
                    core=None):
        """Run the standard sanity checks. See algo/diagnostics.py.

        `core`: return series of the portfolio you ALREADY hold. Pass it. Without
        it the report answers "is this a good standalone strategy", which is not
        the question for a sleeve held alongside something else.
        """
        from . import diagnostics as _diag
        return _diag.analyze(self, benchmark=benchmark,
                             benchmark_name=benchmark_name, n_trials=n_trials,
                             core=core)

    def report(self, benchmark: "BacktestResult | None" = None,
               benchmark_name: str = "benchmark", n_trials: int = 1,
               core=None) -> str:
        """Full report: performance plus automatic sanity checks.

        A bare metrics dump is what lets a bad strategy look fine, so `report`
        always runs the diagnostics rather than offering them as an extra.
        """
        d = self.diagnostics(benchmark, benchmark_name, n_trials, core)
        return str(d)


def run_legacy(prices: pd.DataFrame,
        weights: pd.DataFrame,
        cost_model: CostModel = IBKR_US_EQUITY,
        initial_capital: float = 100_000.0,
        periods_per_year: int = metrics.TRADING_DAYS) -> BacktestResult:
    """DEPRECATED -- target-change turnover. Use `run` (delegates to v2).

    Kept only to reproduce numbers generated before 2026-09-05. It computes
    turnover from changes in the TARGET and assumes the book always sits exactly
    on target, so a constant-target strategy pays nothing while being rebalanced
    for free. Measured effect was ~5.8bp/yr on the benchmark and 0.003 Sharpe on
    comparisons -- real, but small. See algo/backtest2.py.

    prices:  date x ticker frame of prices (adjusted closes).
    weights: date x ticker frame of target portfolio weights, computed from
             information available at that bar's close. THE ENGINE LAGS THESE
             BY ONE BAR -- do not pre-shift them yourself.

    Weights need not sum to 1; leftover is treated as uninvested cash earning
    nothing. Sum > 1 implies leverage and is not validated here.
    """
    prices = prices.sort_index()
    weights = weights.reindex(prices.index).reindex(columns=prices.columns).fillna(0.0)

    # Coerce to float. A single pd.NA anywhere upstream promotes a frame to
    # object dtype, which propagates silently through every arithmetic op and
    # only surfaces much later as an opaque error inside scipy. Fail here, or
    # better, fix it here.
    prices = prices.astype("float64")
    weights = weights.astype("float64")

    asset_returns = prices.pct_change().fillna(0.0)

    # --- the lookahead guard -------------------------------------------------
    # Weights decided at close of bar t are held over bar t+1's return.
    held = weights.shift(1).fillna(0.0)

    gross = (held * asset_returns).sum(axis=1)

    # Turnover: how much of the portfolio changed hands entering this bar.
    turnover = (weights.shift(1) - weights.shift(2)).abs().sum(axis=1).fillna(0.0)

    # Cost as a fraction of portfolio value. one_way_bps is per side, and
    # turnover already counts both the exit and the entry legs of a switch.
    cost_frac = turnover * cost_model.one_way_bps * 1e-4

    net = gross - cost_frac
    equity = initial_capital * (1 + net).cumprod()

    return BacktestResult(
        equity=equity,
        returns=net,
        weights=held,
        turnover=turnover,
        costs=cost_frac,
        gross_returns=gross,
        periods_per_year=periods_per_year,
        meta={"cost_model": cost_model.name, "initial_capital": initial_capital},
    )


def buy_and_hold(prices: pd.DataFrame, ticker: str,
                 cost_model: CostModel = IBKR_US_EQUITY,
                 **kw) -> BacktestResult:
    """Benchmark: 100% in one asset from the first bar, never traded again."""
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[ticker] = 1.0
    return run(prices, w, cost_model=cost_model, **kw)


def walk_forward(prices: pd.DataFrame,
                 strategy_fn,
                 train_periods: int = 756,
                 test_periods: int = 252,
                 cost_model: CostModel = IBKR_US_EQUITY,
                 periods_per_year: int = metrics.TRADING_DAYS,
                 **strategy_kw) -> BacktestResult:
    """Anchored-window walk-forward.

    `strategy_fn(train_prices, test_prices, **kw) -> weights for the test window`.

    Each test window is scored using only a strategy fitted on data strictly
    before it. This is the only honest way to evaluate a strategy that has any
    fitted parameters. The stitched-together out-of-sample equity curve is what
    you should judge, never the in-sample fit.
    """
    segments = []
    start = 0
    while start + train_periods + test_periods <= len(prices):
        train = prices.iloc[start:start + train_periods]
        test = prices.iloc[start + train_periods:start + train_periods + test_periods]
        segments.append(strategy_fn(train, test, **strategy_kw))
        start += test_periods

    if not segments:
        raise ValueError(
            f"not enough data: need >= {train_periods + test_periods} bars, got {len(prices)}"
        )

    oos_weights = pd.concat(segments).sort_index()
    oos_prices = prices.loc[oos_weights.index[0]:oos_weights.index[-1]]
    return run(oos_prices, oos_weights, cost_model=cost_model,
               periods_per_year=periods_per_year)


def run(prices: pd.DataFrame, weights: pd.DataFrame, **kw) -> BacktestResult:
    """Run a backtest. Delegates to the v2 engine (true holdings drift).

    v1 (`run_legacy`) charged turnover on TARGET changes only, which let a
    constant-target portfolio rebalance for free. v2 tracks actual drifting
    holdings and charges for the trade back to target.
    """
    from . import backtest2
    return backtest2.run(prices, weights, **kw)
