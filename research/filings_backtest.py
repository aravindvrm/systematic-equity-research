"""Does the price+filing composite beat the no-signal benchmark, out of sample?

WHY IC IS NOT ENOUGH
--------------------
The combined IC of 0.0323 at h=126 looks close to the 0.036 target, but that
target was derived from a synthetic forecast REGENERATED DAILY. A filing signal
refreshes only when a company files and predicts six months out, so its breadth
is far lower and its true requirement is correspondingly HIGHER. IR ~ IC*sqrt(BR)
cuts both ways. The only honest resolution is to run it.

The price composite had t = 5.98 and lost to holding everything. That is the
standard this has to clear.

DESIGN
------
  select 2006-2015 : fix signs and weights, never look at the rest
  trade  2016-2026 : the reported number
Benchmark is the no-signal version of the identical stack at the same rebalance
frequency -- not buy-and-hold, which would flatter a de-risked portfolio.
"""
import numpy as np
import pandas as pd

from algo import (backtest, data, features, features2 as f2, filings,
                  metrics, research, strategies)
from algo.costs import IBKR_US_EQUITY

pd.set_option("display.width", 200)
SPLIT = "2016-01-01"

M = pd.read_parquet("data/edgar/measures.parquet")
uni = pd.read_parquet("data/collection_universe.parquet").dropna(subset=["cik"])
uni["cik"] = uni["cik"].astype(int).astype(str)
c = data.load_panel(sorted(M.ticker.unique()), start="2005-01-01", end="2026-09-01",
                    field="close", refresh=False)
c = c.dropna(axis=1, thresh=int(0.9 * len(c))).ffill(limit=5)
M = M[M.ticker.isin(c.columns)]
tick = dict(zip(M.cik.astype(str), M.ticker))
P = {f: data.load_panel(sorted(M.ticker.unique()), start="2005-01-01",
                        end="2026-09-01", field=f, refresh=False)
     .reindex(index=c.index, columns=c.columns).ffill(limit=5)
     for f in ("open", "high", "low", "volume")}
o, h_, l, v = P["open"], P["high"], P["low"], P["volume"]
print(f"{c.shape[1]} names, {len(c)} bars\n")

jac = filings.to_daily(M, c.index, tick, col="sim_jaccard", lag_days=1, hold_days=126)
PRICE = {
    "reversal_5":       features.reversal(c, 5),
    "mom_12_1":         features.momentum_12_1(c),
    "vol_shock":        f2.volume_shock(v),
    "close_in_range":   f2.close_position_in_range(h_, l, c),
    "resid_mom_252":    f2.residual_momentum(c, 252),
    "mom_252":          features.momentum(c, 252),
    "overnight_vs_day": f2.intraday_vs_overnight(o, c),
}

def z(f):
    zz = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1).replace(0, np.nan), axis=0)
    return zz.clip(-3, 3)

train = c.index < SPLIT
fwd63 = research.forward_returns(c, 63)
signs = {n: float(np.sign(research.cross_sectional_ic(f[train], fwd63[train]).mean()))
         for n, f in PRICE.items()}
sj = float(np.sign(research.cross_sectional_ic(jac[train], fwd63[train]).mean()))
print(f"signs fixed on 2006-2015; jaccard sign = {sj:+.0f}")

price_comp = sum(z(PRICE[n]) * signs[n] for n in PRICE) / len(PRICE)
jac_z = z(jac) * sj

def stack(score, prices, top_frac=0.3, rebal=21):
    s = score.copy()
    mask = np.zeros(len(s), dtype=bool); mask[::rebal] = True
    s = s.where(pd.Series(mask, index=s.index), np.nan).ffill()
    rank = s.rank(axis=1, pct=True, ascending=False)
    raw = (rank <= top_frac).astype(float)
    w = raw * strategies.inverse_volatility(prices, 60)
    tot = w.sum(axis=1)
    w = w.div(tot.where(tot > 0), axis=0).fillna(0.0)
    return strategies.volatility_target(w, prices, target_vol=0.10, max_leverage=1.0)

test = c.index >= SPLIT
px_t = c[test]
flat = pd.DataFrame(0.0, index=c.index, columns=c.columns)

print(f"\n{'='*94}")
print(f"OUT OF SAMPLE {px_t.index[0].date()}..{px_t.index[-1].date()} "
      f"({len(px_t)/252:.1f} years), long-only, IBKR costs")
print(f"{'='*94}\n")

for rebal, tag in [(21, "monthly"), (63, "quarterly")]:
    b = backtest.run(px_t, stack(flat, c, 1.0, rebal)[test],
                     cost_model=IBKR_US_EQUITY).summary()
    print(f"--- rebalance {tag} ---")
    print(f"{'strategy':<26} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7} "
          f"{'vs bench':>9} {'grossSh':>8}")
    print(f"{'benchmark (no signal)':<26} {b['sharpe']:>8.2f} {b['cagr']*100:>7.2f}% "
          f"{b['max_drawdown']*100:>7.1f}% {b['calmar']:>7.2f} {'--':>9} {'--':>8}")
    for lbl, sc in [("price only", price_comp),
                    ("filings only", jac_z),
                    ("price + filings", (price_comp + jac_z) / 2)]:
        r = backtest.run(px_t, stack(sc, c, 0.3, rebal)[test], cost_model=IBKR_US_EQUITY)
        s = r.summary()
        print(f"{lbl:<26} {s['sharpe']:>8.2f} {s['cagr']*100:>7.2f}% "
              f"{s['max_drawdown']*100:>7.1f}% {s['calmar']:>7.2f} "
              f"{s['sharpe']-b['sharpe']:>+9.2f} {metrics.sharpe(r.gross_returns):>8.2f}")
    print()

spy = data.load_panel(["SPY"], start=str(px_t.index[0].date()), end="2026-09-01",
                      refresh=False).reindex(px_t.index).ffill()
ss = backtest.buy_and_hold(spy, "SPY", cost_model=IBKR_US_EQUITY).summary()
print(f"SPY reference: Sharpe {ss['sharpe']:.2f}  CAGR {ss['cagr']*100:.2f}%  "
      f"MaxDD {ss['max_drawdown']*100:.1f}%")
