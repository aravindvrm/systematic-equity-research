"""Actually backtest the middle ground rather than inferring it from IC."""
import numpy as np, pandas as pd
from algo import backtest, data, features, metrics, research, strategies
from algo.costs import CostModel

BLUE = ["MSFT","JPM","JNJ","XOM","PG","HD","CAT","NEE","LIN","VZ","AMT",
        "UNH","WMT","CVX","KO","MCD","HON","TXN","DE","GS"]
DIV = ["TLT","IEF","GLD","DBC"]
px = data.load_panel(BLUE + DIV, start="2012-01-01", end="2026-09-01").dropna(how="any")
ALPACA = CostModel(name="Alpaca", commission_bps=0.0, spread_bps=1.5, slippage_bps=1.5)

def stack(p, raw):
    iv = strategies.inverse_volatility(p, 60); c = raw * iv; t = c.sum(axis=1)
    c = c.div(t.where(t > 0), axis=0).fillna(0.0).mul(raw.sum(axis=1), axis=0)
    return strategies.volatility_target(c, p, target_vol=0.10, max_leverage=1.0)

def hold_every(w, n):
    m = pd.Series(False, index=w.index); m.iloc[::n] = True
    return w.where(m).ffill().fillna(0.0)

def xs_long(feat, top_frac=0.4):
    """Long the most attractive fraction by the feature, equal weight."""
    n_top = max(3, int(feat.shape[1] * top_frac))
    ranks = feat.rank(axis=1, ascending=False)
    sel = (ranks <= n_top).astype(float)
    n = sel.sum(axis=1).replace(0, np.nan)
    return sel.div(n, axis=0).fillna(0.0)

bench = backtest.buy_and_hold(px, "SPY", cost_model=ALPACA) if "SPY" in px.columns else None
eqw = backtest.run(px, strategies.equal_weight(px), ALPACA)

print("MIDDLE-GROUND BACKTESTS (20 blue chips + 4 diversifiers, Alpaca costs)\n")
print(f"{'signal':<18} {'rebal':>7} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'turn':>7} "
      f"{'corr(eqw)':>10}")
rows = []
for fname, feat in [("reversal_21", features.reversal(px, 21)),
                    ("reversal_10", features.reversal(px, 10)),
                    ("reversal_5",  features.reversal(px, 5)),
                    ("mom_12_1",    features.momentum_12_1(px))]:
    raw = xs_long(feat)
    for n, lab in [(5, "weekly"), (10, "2 weeks"), (21, "monthly"), (63, "quarterly")]:
        w = stack(px, hold_every(raw, n))
        r = backtest.run(px, w, ALPACA)
        c = r.returns.corr(eqw.returns)
        print(f"{fname:<18} {lab:>7} {metrics.sharpe(r.returns):>8.2f} "
              f"{metrics.cagr(r.equity)*100:>7.2f}% {metrics.max_drawdown(r.equity)*100:>7.1f}%"
              f" {r.turnover.mean():>7.3f} {c:>10.2f}")
        rows.append((fname, lab, r, c))
print(f"\n{'equal-weight (no signal)':<26} {metrics.sharpe(eqw.returns):>8.2f} "
      f"{metrics.cagr(eqw.equity)*100:>7.2f}% {metrics.max_drawdown(eqw.equity)*100:>7.1f}%")

best = max(rows, key=lambda x: metrics.sharpe(x[2].returns))
print(f"\n\nDIAGNOSTICS ON THE BEST CELL: {best[0]} / {best[1]}")
print("=" * 70)
print(best[2].report(benchmark=eqw, benchmark_name="equal-weight", n_trials=16))
