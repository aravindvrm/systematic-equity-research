"""Does crypto market structure actually favour systematic trading?

Compares the equity-ETF universe against the crypto universe IBKR offers, on the
things that determine whether a systematic strategy has anything to work with:
breadth, cost drag, and history length.
"""
import numpy as np
import pandas as pd

from algo import backtest, data, metrics, strategies
from algo.costs import IBKR_US_EQUITY, IBKR_CRYPTO, ZERO_COST

CRYPTO = ["BTC-USD", "ETH-USD", "LTC-USD", "BCH-USD", "SOL-USD",
          "ADA-USD", "DOGE-USD", "AVAX-USD", "LINK-USD", "XRP-USD"]


def effective_bets(returns: pd.DataFrame) -> float:
    """Participation ratio of the correlation matrix eigenvalues.

    N independent assets -> N. N perfectly correlated assets -> 1. This is how
    many genuinely distinct bets a cross-sectional strategy can actually place.
    """
    corr = returns.corr().to_numpy()
    eig = np.linalg.eigvalsh(corr)
    eig = eig[eig > 0]
    return float(eig.sum() ** 2 / (eig**2).sum())


def profile(name, prices, cost_model, ppy):
    rets = prices.pct_change().dropna()
    n = prices.shape[1]
    corr = rets.corr()
    avg_corr = corr.values[np.triu_indices(n, k=1)].mean()

    print(f"\n=== {name} ===")
    print(f"  assets                {n}")
    print(f"  common history        {prices.index[0].date()} .. {prices.index[-1].date()}"
          f"  ({len(prices)} bars, {len(prices)/ppy:.1f} yrs)")
    print(f"  avg pairwise corr     {avg_corr:6.2f}")
    print(f"  effective # of bets   {effective_bets(rets):6.2f}  (out of {n})")
    print(f"  ann. vol (equal-wt)   {rets.mean(axis=1).std()*np.sqrt(ppy)*100:6.1f}%")
    print(f"  round-trip cost       {cost_model.round_trip_bps:6.1f} bp")

    w = strategies.cross_sectional_momentum(prices, lookback=180, skip=15, top_n=3)
    gross = backtest.run(prices, w, ZERO_COST, periods_per_year=ppy)
    net = backtest.run(prices, w, cost_model, periods_per_year=ppy)
    print(f"  XS momentum Sharpe    {metrics.sharpe(gross.returns, periods_per_year=ppy):6.2f} gross"
          f"  ->{metrics.sharpe(net.returns, periods_per_year=ppy):6.2f} net")
    print(f"  cost drag on Sharpe   {metrics.sharpe(gross.returns,periods_per_year=ppy)-metrics.sharpe(net.returns,periods_per_year=ppy):6.2f}")


eq = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")
profile("US equity ETFs (IBKR Pro)", eq, IBKR_US_EQUITY, 252)

cr = data.load_panel(CRYPTO, start="2015-01-01", end="2026-09-01").dropna(how="any")
profile("Crypto (IBKR ZEROHASH)", cr, IBKR_CRYPTO, 365)
