"""Order-level backtest: simulates actual share counts and per-order commissions.

The fractional cost model in costs.py expresses everything as bps of notional,
which cannot represent a per-ORDER minimum -- the charge that dominates a small
account. This module trades real share quantities so the $0.35 floor bites the
way it actually will.

IBKR Pro Tiered US equities:
    commission = clamp(0.0035 x shares, min $0.35, max 1% of trade value)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics

PER_SHARE, MIN_ORDER, MAX_PCT = 0.0035, 0.35, 0.01
SPREAD_SLIP_BPS = 1.5


def _commission(shares: float, notional: float) -> float:
    if shares <= 0 or notional <= 0:
        return 0.0
    return min(max(PER_SHARE * shares, MIN_ORDER), MAX_PCT * notional)


def run(prices: pd.DataFrame, weights: pd.DataFrame, capital: float,
        rebalance: str | None = None, min_trade: float = 50.0,
        spread_slip_bps: float = SPREAD_SLIP_BPS,
        periods_per_year: int = 252):
    """Simulate the portfolio at a specific account size.

    rebalance: None = act on every bar the weights change (what the vectorized
               engine implicitly assumes); 'W'/'ME'/'QE'/'YE' = only rebalance on
               those period ends, holding weights fixed in between.
    min_trade: skip trades below this dollar value -- a real and important
               optimization, since a $12 trade still costs $0.35 (292bp).
    """
    prices = prices.sort_index().astype("float64")
    weights = weights.reindex(prices.index).reindex(columns=prices.columns).fillna(0.0).astype("float64")
    weights = weights.shift(1).fillna(0.0)          # same lookahead guard as the engine

    if rebalance is not None:
        # to_period uses M/Q/Y/W, not the resample ME/QE/YE aliases.
        alias = {"ME": "M", "QE": "Q", "YE": "Y", "W": "W"}.get(rebalance, rebalance)
        period = prices.index.to_period(alias)
        is_rebal = np.r_[True, period[1:] != period[:-1]]   # first bar of each period
        weights = weights.where(pd.Series(is_rebal, index=prices.index), other=np.nan).ffill().fillna(0.0)

    cash = float(capital)
    shares = pd.Series(0.0, index=prices.columns, dtype="float64")
    eq_curve, costs, n_orders = [], [], 0

    for t in prices.index:
        px_row = prices.loc[t]
        equity = cash + float(shares @ px_row)

        target_shares = (weights.loc[t] * equity / px_row).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        delta = target_shares - shares
        trade_value = delta.abs() * px_row
        do = trade_value >= min_trade

        bar_cost = 0.0
        for tick in prices.columns[do.to_numpy()]:
            d = float(delta[tick]); notion = float(trade_value[tick])
            c = _commission(abs(d), notion) + notion * spread_slip_bps * 1e-4
            cash -= d * float(px_row[tick])     # buying spends cash, selling raises it
            cash -= c
            shares[tick] = float(target_shares[tick])
            bar_cost += c
            n_orders += 1

        costs.append(bar_cost)
        eq_curve.append(cash + float(shares @ px_row))

    eq = pd.Series(eq_curve, index=prices.index)
    rets = eq.pct_change().fillna(0.0)
    return dict(
        equity=eq, returns=rets,
        cagr=metrics.cagr(eq, periods_per_year),
        sharpe=metrics.sharpe(rets, periods_per_year=periods_per_year),
        max_drawdown=metrics.max_drawdown(eq),
        total_cost=float(np.sum(costs)),
        cost_pct_of_capital=float(np.sum(costs) / capital),
        n_orders=n_orders,
    )
