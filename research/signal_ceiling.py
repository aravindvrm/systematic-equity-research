"""How much is better signal actually worth on this universe?

Instead of guessing, inject synthetic forecasts with a KNOWN information
coefficient and measure the Sharpe they produce through the real risk stack.

If an oracle-grade signal only buys a little, then signal engineering is capped
by universe structure and the effort belongs elsewhere.
"""
import numpy as np, pandas as pd
from algo import backtest, data, metrics, strategies, portfolio
from algo.costs import IBKR_US_EQUITY

rng = np.random.default_rng(11)
px = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")


def synthetic_weights(prices, ic, long_only=True, seed=0):
    """Forecast correlating with NEXT bar's return at exactly `ic`."""
    r = rng_local = np.random.default_rng(seed)
    fut = prices.pct_change().shift(-1)
    z = (fut - fut.mean()) / fut.std()
    noise = pd.DataFrame(r.normal(size=z.shape), index=z.index, columns=z.columns)
    fc = ic * z + np.sqrt(1 - ic**2) * noise
    sig = (fc > 0).astype(float) if long_only else np.sign(fc)
    n = sig.abs().sum(axis=1).replace(0, np.nan)
    return sig.div(n, axis=0).fillna(0.0)


def through_stack(prices, raw_w, target_vol=0.10):
    """Apply the same inverse-vol + vol-target layers the real portfolio uses."""
    iv = strategies.inverse_volatility(prices, lookback=60)
    c = raw_w * iv
    tot = c.sum(axis=1)
    c = c.div(tot.where(tot > 0), axis=0).fillna(0.0).mul(raw_w.sum(axis=1), axis=0)
    return strategies.volatility_target(c, prices, target_vol=target_vol, max_leverage=1.0)


print("SIGNAL QUALITY -> REALIZED SHARPE  (10-ETF universe, 2.65 effective bets)")
print("all run through the identical risk stack, identical costs\n")
print(f"{'IC':>7} {'interpretation':<28} {'raw signal':>11} {'+risk stack':>12}")
for ic, label in [(0.00, "pure noise"),
                  (0.03, "professional quant"),
                  (0.05, "top-decile quant"),
                  (0.10, "2-3x better than pros"),
                  (0.20, "implausible"),
                  (0.40, "oracle-tier")]:
    w = synthetic_weights(px, ic, seed=int(ic*1000)+1)
    bare = backtest.run(px, w, IBKR_US_EQUITY)
    stacked = backtest.run(px, through_stack(px, w), IBKR_US_EQUITY)
    print(f"{ic:>7.2f} {label:<28} {metrics.sharpe(bare.returns):>11.2f} "
          f"{metrics.sharpe(stacked.returns):>12.2f}")

real = backtest.run(px, portfolio.build(px), IBKR_US_EQUITY)
trend_only = backtest.run(px, portfolio.trend_sleeve(px), IBKR_US_EQUITY)
print(f"\n  ACTUAL trend signal, bare:        {metrics.sharpe(trend_only.returns):.2f}")
print(f"  ACTUAL trend signal, +risk stack: {metrics.sharpe(real.returns):.2f}")
print(f"  SPY buy & hold:                   {metrics.sharpe(backtest.buy_and_hold(px,'SPY',cost_model=IBKR_US_EQUITY).returns):.2f}")
