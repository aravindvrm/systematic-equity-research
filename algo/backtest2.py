"""Backtest engine v2 -- correct holdings accounting.

WHAT WAS WRONG IN v1
--------------------
v1 computed turnover as the change in TARGET weights:

    turnover = (weights.shift(1) - weights.shift(2)).abs().sum(axis=1)

and set held = weights.shift(1), i.e. it assumed the portfolio always sat
exactly on target. Those two assumptions are mutually inconsistent. Holding a
constant target requires trading as prices drift, so:

  * a constant-target strategy (equal weight, the no-signal BENCHMARK) was
    charged ZERO cost while actually being rebalanced to target for free;
  * a signal strategy paid for every target change.

The bias runs one way: it flatters the benchmark and penalises any strategy that
changes its mind. Every "signal loses to benchmark" comparison in this project
was measured on that tilted field.

WHAT v2 DOES
------------
Tracks the actual portfolio. Between rebalances holdings DRIFT with returns; at
a rebalance we pay to move from the drifted holdings to the new target:

    w_drift[t] = w_held[t-1] * (1 + r[t]) / (1 + portfolio_return[t])
    turnover[t] = sum |w_target[t-1] - w_drift[t]|

This is path-dependent, so it is a loop rather than a vectorised expression.
That is the price of being correct.

The lookahead guard is unchanged and still the most important thing here: a
target formed at the close of bar t is traded into at bar t+1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics
from .backtest import BacktestResult
from .costs import CostModel, IBKR_US_EQUITY


def run(prices: pd.DataFrame,
        weights: pd.DataFrame,
        cost_model: CostModel = IBKR_US_EQUITY,
        initial_capital: float = 100_000.0,
        periods_per_year: int = metrics.TRADING_DAYS,
        rebalance_tol: float = 0.0) -> BacktestResult:
    """Backtest with true holdings drift.

    rebalance_tol: skip rebalancing when total drift from target is below this
        (a no-trade band on the WHOLE portfolio). 0.0 rebalances every bar.
    """
    prices = prices.sort_index()
    weights = (weights.reindex(prices.index)
               .reindex(columns=prices.columns).fillna(0.0).astype("float64"))
    prices = prices.astype("float64")
    r = prices.pct_change().fillna(0.0).to_numpy()
    tgt = weights.to_numpy()
    n = len(prices)

    held = np.zeros_like(tgt)      # weights actually held over each bar
    gross = np.zeros(n)
    turn = np.zeros(n)
    cur = np.zeros(tgt.shape[1])   # current (drifted) weights

    one_way = cost_model.one_way_bps * 1e-4
    for t in range(1, n):
        # Target formed at close of t-1 is what we trade into for bar t.
        want = tgt[t - 1]
        d = np.abs(want - cur).sum()
        if d > rebalance_tol:
            turn[t] = d
            cur = want.copy()
        held[t] = cur
        gross[t] = float(cur @ r[t])
        # Drift: holdings grow with their own returns, renormalised by the
        # portfolio's return so the weights still describe the same book.
        denom = 1.0 + gross[t]
        if abs(denom) > 1e-12:
            cur = cur * (1.0 + r[t]) / denom

    cost_frac = turn * one_way
    net = gross - cost_frac
    equity = initial_capital * np.cumprod(1.0 + net)
    idx = prices.index
    return BacktestResult(
        equity=pd.Series(equity, index=idx),
        returns=pd.Series(net, index=idx),
        weights=pd.DataFrame(held, index=idx, columns=prices.columns),
        turnover=pd.Series(turn, index=idx),
        costs=pd.Series(cost_frac, index=idx),
        gross_returns=pd.Series(gross, index=idx),
        periods_per_year=periods_per_year,
        meta={"cost_model": cost_model.name, "initial_capital": initial_capital,
              "engine": "v2-drift"},
    )
