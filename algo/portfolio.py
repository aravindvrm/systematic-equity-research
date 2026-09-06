"""A combined multi-premia portfolio for a small long-only cash account.

Three ingredients, each chosen because it survives the constraints established
earlier -- small size, no shorting, low turnover, no forecasting skill assumed:

  1. TREND (time-series momentum), ensembled across lookbacks rather than
     tuned. Parameter selection was shown to be actively harmful out of sample.
  2. RISK PREMIA weighting via inverse volatility, so no single high-vol asset
     dominates portfolio risk.
  3. VOLATILITY TARGETING as an overlay, capped at 1.0x so it only ever cuts
     exposure. In a cash account this buys drawdown reduction, not Sharpe.

Nothing here forecasts anything beyond "trends persist a bit". That is the
point: the edge is in construction and cost discipline, not prediction.
"""
from __future__ import annotations

import pandas as pd

from . import strategies

DEFAULT_LOOKBACKS = (63, 126, 189, 252, 315)


def trend_sleeve(prices: pd.DataFrame,
                 lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS) -> pd.DataFrame:
    """Ensembled long-only time-series momentum.

    Each lookback votes on each asset; assets with more positive-trend votes get
    more weight. Averaging the specs is the defence against picking one.
    """
    return strategies.ensemble(*[
        strategies.time_series_momentum(prices, lookback=lb, long_only=True)
        for lb in lookbacks
    ])


def build(prices: pd.DataFrame,
          lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
          target_vol: float = 0.10,
          vol_lookback: int = 60,
          risk_weight: bool = True,
          max_leverage: float = 1.0,
          periods_per_year: int = 252) -> pd.DataFrame:
    """Full portfolio weights: trend selection, risk weighting, vol overlay."""
    trend = trend_sleeve(prices, lookbacks)

    if risk_weight:
        # Tilt the trend-selected names toward the lower-volatility ones.
        inv_vol = strategies.inverse_volatility(prices, lookback=vol_lookback)
        combined = trend * inv_vol
        total = combined.sum(axis=1)
        combined = combined.div(total.where(total > 0), axis=0).fillna(0.0)
        # Preserve the trend sleeve's total exposure (it may be < 1 when few
        # assets are trending); risk weighting should redistribute, not lever.
        combined = combined.mul(trend.sum(axis=1), axis=0)
    else:
        combined = trend

    return strategies.volatility_target(
        combined, prices, target_vol=target_vol, lookback=vol_lookback,
        max_leverage=max_leverage, periods_per_year=periods_per_year)


def select_best_lookback(train_prices: pd.DataFrame, test_prices: pd.DataFrame,
                         lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
                         cost_model=None, **kw) -> pd.DataFrame:
    """Walk-forward variant that FITS the lookback on the training window.

    Provided for comparison against the ensemble, not as a recommendation. If
    this underperforms `build`, that is evidence parameter selection is noise
    -- which is the expected result.
    """
    from . import backtest, metrics
    from .costs import IBKR_US_EQUITY

    cost_model = cost_model or IBKR_US_EQUITY
    best, best_sharpe = lookbacks[0], -1e9
    for lb in lookbacks:
        w = strategies.time_series_momentum(train_prices, lookback=lb)
        r = backtest.run(train_prices, w, cost_model)
        s = metrics.sharpe(r.returns)
        if s > best_sharpe:
            best, best_sharpe = lb, s
    # Apply the winner to the test window.
    full = pd.concat([train_prices, test_prices])
    w = build(full, lookbacks=(best,), **kw)
    return w.loc[test_prices.index]


def walk_forward_ensemble(train_prices: pd.DataFrame, test_prices: pd.DataFrame,
                          **kw) -> pd.DataFrame:
    """Walk-forward wrapper for `build`. Fits nothing -- included so the two
    approaches are evaluated over identical out-of-sample windows."""
    full = pd.concat([train_prices, test_prices])
    return build(full, **kw).loc[test_prices.index]
