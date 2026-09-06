"""Out-of-sample test of the features that survived the long-sample screen.

THE SELECTION PROBLEM
---------------------
Those seven features were chosen by looking at IC over 2005-2026. Backtesting
them on 2005-2026 would report the selection, not a forecast. So:

    SELECT on 2005-2015   (rank features by IC, pick the top k, fix the signs)
    TRADE  on 2016-2026   (never examined during selection)

The selection half is discarded from the performance record entirely. Whatever
the second half shows is the honest number.

WHAT IS BEING TESTED
--------------------
Long-only, cash account, no leverage -- the actual constraint set. Equal-weighted
composite of z-scored features, through the same inverse-vol + vol-target stack
the live portfolio uses, with IBKR costs. Rebalance frequency is swept because
h=1 signals imply daily turnover and that is where this most likely dies.

Benchmark is the no-signal version of the identical stack, NOT plain buy-and-hold
-- otherwise a de-risked portfolio flatters itself.
"""
import numpy as np
import pandas as pd

from algo import backtest, data, features, features2 as f2, metrics, research, strategies
from algo.costs import IBKR_US_EQUITY

pd.set_option("display.width", 220)

SPLIT = "2016-01-01"
uni = pd.read_parquet("data/collection_universe.parquet")
sym_col = [c for c in uni.columns if c.lower() in ("symbol", "ticker")][0]
SYMS = uni[sym_col].tolist()

close = data.load_panel(SYMS, start="2005-01-01", end="2026-09-01", field="close",
                        refresh=False)
close = close.dropna(axis=1, thresh=int(0.9 * len(close))).ffill(limit=5)
P = {"close": close}
for f in ("open", "high", "low", "volume"):
    P[f] = (data.load_panel(SYMS, start="2005-01-01", end="2026-09-01", field=f,
                            refresh=False)
            .reindex(index=close.index, columns=close.columns).ffill(limit=5))
c, o, h, l, v = P["close"], P["open"], P["high"], P["low"], P["volume"]
print(f"{c.shape[1]} names, {len(c)} bars, {c.index[0].date()}..{c.index[-1].date()}")

CANDIDATES = {
    "reversal_5":       features.reversal(c, 5),
    "mom_12_1":         features.momentum_12_1(c),
    "vol_shock":        f2.volume_shock(v),
    "close_in_range":   f2.close_position_in_range(h, l, c),
    "resid_mom_252":    f2.residual_momentum(c, 252),
    "mom_252":          features.momentum(c, 252),
    "overnight_vs_day": f2.intraday_vs_overnight(o, c),
}

# ---- SELECT on the first half only -----------------------------------------
train = c.index < SPLIT
fwd1 = research.forward_returns(c, 1)
print(f"\nSELECTION on {c.index[0].date()}..{c.index[train][-1].date()} "
      f"({train.sum()} bars) -- the trading half is not touched here\n")
sel = []
for name, f in CANDIDATES.items():
    ic = research.cross_sectional_ic(f[train], fwd1[train]).dropna()
    t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
    sel.append((name, float(ic.mean()), float(t)))
sel = pd.DataFrame(sel, columns=["feature", "ic_train", "t_train"]).sort_values(
    "t_train", key=abs, ascending=False)
print(sel.to_string(index=False))

KEEP = sel[sel.t_train.abs() > 2.0]
print(f"\nkept {len(KEEP)} features with |t| > 2.0 in-sample: "
      f"{', '.join(KEEP.feature)}")
signs = dict(zip(KEEP.feature, np.sign(KEEP.t_train)))


def zscore(f):
    """Cross-sectional z-score, clipped -- one outlier must not own the book."""
    z = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1).replace(0, np.nan), axis=0)
    return z.clip(-3, 3)


composite = sum(zscore(CANDIDATES[n]) * s for n, s in signs.items()) / len(signs)


def long_only_weights(score, prices, top_frac=0.3, rebal=1):
    """Long the top `top_frac` by score, inverse-vol weighted, vol-targeted.

    Rebalancing every `rebal` bars: the score is held constant between dates, so
    turnover falls roughly linearly with the period.
    """
    s = score.copy()
    if rebal > 1:
        mask = np.zeros(len(s), dtype=bool)
        mask[::rebal] = True
        s = s.where(pd.Series(mask, index=s.index), np.nan).ffill()
    rank = s.rank(axis=1, pct=True, ascending=False)
    raw = (rank <= top_frac).astype(float)
    iv = strategies.inverse_volatility(prices, lookback=60)
    w = raw * iv
    tot = w.sum(axis=1)
    w = w.div(tot.where(tot > 0), axis=0).fillna(0.0)
    return strategies.volatility_target(w, prices, target_vol=0.10, max_leverage=1.0)


test = c.index >= SPLIT
px_test = c[test]
print(f"\n{'='*100}")
print(f"OUT OF SAMPLE: {px_test.index[0].date()}..{px_test.index[-1].date()} "
      f"({len(px_test)} bars, {len(px_test)/252:.1f} years)")
print(f"{'='*100}\n")

# no-signal benchmark: same stack, flat score -> holds everything
flat = pd.DataFrame(0.0, index=c.index, columns=c.columns)
bench_w = long_only_weights(flat, c, top_frac=1.0, rebal=21)[test]
bench = backtest.run(px_test, bench_w, cost_model=IBKR_US_EQUITY)
bs = bench.summary()
print(f"{'benchmark (no signal, same stack)':<40} Sharpe {bs['sharpe']:>5.2f}  "
      f"CAGR {bs['cagr']*100:>6.2f}%  MaxDD {bs['max_drawdown']*100:>6.1f}%  "
      f"Calmar {bs['calmar']:>5.2f}")
print()
print(f"{'rebalance':<12} {'turnover':>9} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} "
      f"{'Calmar':>8} {'vs bench':>9} {'gross Sh':>9}")
for rebal in (1, 5, 10, 21, 63):
    w = long_only_weights(composite, c, top_frac=0.3, rebal=rebal)[test]
    res = backtest.run(px_test, w, cost_model=IBKR_US_EQUITY)
    s = res.summary()
    gross = metrics.sharpe(res.gross_returns)
    tag = {1: "daily", 5: "weekly", 10: "biweekly", 21: "monthly", 63: "quarterly"}[rebal]
    print(f"{tag:<12} {s['avg_turnover']:>9.3f} {s['sharpe']:>8.2f} "
          f"{s['cagr']*100:>7.2f}% {s['max_drawdown']*100:>7.1f}% {s['calmar']:>8.2f} "
          f"{s['sharpe']-bs['sharpe']:>+9.2f} {gross:>9.2f}")

print("\nSPY over the same window, for reference")
spy = data.load_panel(["SPY"], start=str(px_test.index[0].date()),
                      end="2026-09-01", refresh=False).reindex(px_test.index).ffill()
sres = backtest.buy_and_hold(spy, "SPY", cost_model=IBKR_US_EQUITY)
ss = sres.summary()
print(f"  SPY  Sharpe {ss['sharpe']:.2f}  CAGR {ss['cagr']*100:.2f}%  "
      f"MaxDD {ss['max_drawdown']*100:.1f}%  Calmar {ss['calmar']:.2f}")


# ---------------------------------------------------------------------------
# DIAGNOSIS: is the ceiling the SIGNAL, or the LONG-ONLY CONSTRAINT?
#
# The long-short book is NOT tradeable in a cash account. It is run here purely
# to locate the binding constraint. If long-short works and long-only does not,
# the signal is real and the account type is the problem. If neither works, the
# signal is simply too small and no account type rescues it.
# ---------------------------------------------------------------------------
print(f"\n{'='*100}\nDIAGNOSIS -- where is the ceiling?\n{'='*100}\n")


def ls_weights(score, prices, frac=0.3, rebal=5):
    s = score.copy()
    mask = np.zeros(len(s), dtype=bool); mask[::rebal] = True
    s = s.where(pd.Series(mask, index=s.index), np.nan).ffill()
    up = s.rank(axis=1, pct=True, ascending=False)
    dn = s.rank(axis=1, pct=True, ascending=True)
    raw = (up <= frac).astype(float) - (dn <= frac).astype(float)
    n = raw.abs().sum(axis=1).replace(0, np.nan)
    return raw.div(n, axis=0).fillna(0.0)


print("A. LONG-SHORT (diagnostic only -- not legal in a cash account)\n")
print(f"{'top/bottom':<12} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'gross Sh':>9}")
for frac in (0.1, 0.2, 0.3):
    w = ls_weights(composite, c, frac=frac, rebal=5)[test]
    r = backtest.run(px_test, w, cost_model=IBKR_US_EQUITY)
    s = r.summary()
    print(f"{frac*100:>10.0f}% {s['sharpe']:>8.2f} {s['cagr']*100:>7.2f}% "
          f"{s['max_drawdown']*100:>7.1f}% {metrics.sharpe(r.gross_returns):>9.2f}")

print("\nB. LONG-ONLY, varying concentration")
print("   (concentrating recovers signal but gives up diversification -- "
      "there is an optimum)\n")
print(f"{'top frac':<12} {'n names':>8} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'vs bench':>9}")
for frac in (0.05, 0.10, 0.20, 0.30, 0.50):
    w = long_only_weights(composite, c, top_frac=frac, rebal=5)[test]
    r = backtest.run(px_test, w, cost_model=IBKR_US_EQUITY)
    s = r.summary()
    print(f"{frac*100:>10.0f}% {int(frac*c.shape[1]):>8} {s['sharpe']:>8.2f} "
          f"{s['cagr']*100:>7.2f}% {s['max_drawdown']*100:>7.1f}% "
          f"{s['sharpe']-bs['sharpe']:>+9.2f}")

print("\nC. WHAT IC WOULD IT TAKE?  (synthetic forecast, known IC, same stack)")
print("   answers: is 0.012 just too small, or is the stack itself the problem?\n")
rng = np.random.default_rng(11)
fut = c.pct_change().shift(-1)
z = fut.sub(fut.mean(axis=1), axis=0).div(fut.std(axis=1).replace(0, np.nan), axis=0)
noise = pd.DataFrame(rng.normal(size=z.shape), index=z.index, columns=z.columns)
print(f"{'true IC':>9} {'long-only Sh':>13} {'vs bench':>9}")
for ic in (0.00, 0.012, 0.03, 0.05, 0.10):
    fc = ic * z + np.sqrt(1 - ic**2) * noise
    w = long_only_weights(fc, c, top_frac=0.3, rebal=5)[test]
    r = backtest.run(px_test, w, cost_model=IBKR_US_EQUITY)
    s = r.summary()
    print(f"{ic:>9.3f} {s['sharpe']:>13.2f} {s['sharpe']-bs['sharpe']:>+9.2f}")
