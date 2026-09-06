"""Time-series trend following — the risk premium we never tested.

WHY IT WAS MISSED
-----------------
We tested CROSS-SECTIONAL momentum exhaustively (which stock beats which) and
never tested TIME-SERIES trend (is this asset going up or down). They are
different premia with different mechanisms and different literatures:

  cross-sectional momentum : Jegadeesh & Titman (1993), a stock-selection effect
  time-series trend        : Moskowitz, Ooi & Pedersen (2012), an asset-level
                             directional effect, the classic CTA strategy

Ilmanen singles out trend as a SAFE HAVEN in severe drawdowns — the property the
volatility risk premium conspicuously lacks, since selling vol loses exactly when
markets crash. If both are real they are complements, not substitutes.

UNIVERSE
--------
Diversified across asset classes, in ETF form, because that is what is reachable
from a cash account. Real CTAs use ~50 futures markets; this is a coarse proxy
with maybe a third of the breadth, and the result should be read as a lower bound
on what the premium offers rather than as an implementation.

LONG-ONLY CONSTRAINT: a real trend follower goes short. In a cash account the
short leg becomes "hold cash", which discards roughly half the strategy. Both
versions are shown so the cost of the constraint is explicit.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from algo import backtest, data, evaluation, factors, metrics, strategies
from algo.costs import IBKR_US_EQUITY

pd.set_option("display.width", 220)

# Multi-asset, ETF form. Chosen for ASSET-CLASS COVERAGE, not performance.
UNIV = ["SPY", "EFA", "EEM", "IWM",          # equities
        "TLT", "IEF", "SHY",                  # rates
        "LQD", "HYG",                         # credit
        "GLD", "SLV", "DBC", "USO",           # commodities
        "VNQ", "TIP"]                         # real assets
px = data.load_panel(UNIV, start="2005-01-01", end="2026-09-01", refresh=False)
px = px.dropna(axis=1, thresh=int(0.85 * len(px))).ffill(limit=5).dropna()
print(f"{px.shape[1]} assets, {len(px)} bars, {px.index[0].date()}..{px.index[-1].date()}\n")

r1 = px.pct_change().fillna(0.0)
vol = r1.rolling(60).std() * np.sqrt(252)


def trend_weights(lookbacks=(63, 126, 252), long_only=False, target_vol=0.10):
    """Sign of trailing return, averaged across lookbacks, risk-parity scaled.

    Each asset is sized INVERSELY to its own volatility so a bond position and a
    commodity position carry comparable risk — the defining feature of the CTA
    construction, and the reason it diversifies.
    """
    sig = sum(np.sign(px.pct_change(lb)) for lb in lookbacks) / len(lookbacks)
    if long_only:
        sig = sig.clip(lower=0.0)
    w = sig / vol.replace(0, np.nan)
    gross = w.abs().sum(axis=1)
    w = w.div(gross.where(gross > 0), axis=0).fillna(0.0)
    return strategies.volatility_target(w, px, target_vol=target_vol, max_leverage=1.0)


def run(w, rebal=21):
    m = np.zeros(len(w), dtype=bool); m[::rebal] = True
    w = w.where(pd.Series(m, index=w.index), np.nan).ffill()
    return backtest.run(px, w, cost_model=IBKR_US_EQUITY)


print("=" * 100)
print("TIME-SERIES TREND, multi-asset")
print("=" * 100 + "\n")
print(f"{'variant':<28} {'Sharpe':>7} {'CAGR':>8} {'vol':>7} {'MaxDD':>8} {'Calmar':>7} {'skew':>7}")
from scipy import stats as sst
results = {}
for lbl, lo in (("long/short (needs margin)", False), ("long-only (cash account)", True)):
    res = run(trend_weights(long_only=lo))
    r = res.returns
    eq = res.equity
    results[lbl] = r
    print(f"{lbl:<28} {metrics.sharpe(r):>7.2f} {metrics.cagr(eq)*100:>7.2f}% "
          f"{metrics.volatility(r)*100:>6.2f}% {metrics.max_drawdown(eq)*100:>7.1f}% "
          f"{metrics.calmar(eq):>7.2f} {sst.skew(r):>7.2f}")

bh = backtest.run(px, pd.DataFrame(1.0 / px.shape[1], index=px.index, columns=px.columns),
                  cost_model=IBKR_US_EQUITY)
print(f"{'equal-weight buy&hold':<28} {metrics.sharpe(bh.returns):>7.2f} "
      f"{metrics.cagr(bh.equity)*100:>7.2f}% {metrics.volatility(bh.returns)*100:>6.2f}% "
      f"{metrics.max_drawdown(bh.equity)*100:>7.1f}% {metrics.calmar(bh.equity):>7.2f} "
      f"{sst.skew(bh.returns):>7.2f}")

print("\n" + "=" * 100)
print("CRISIS BEHAVIOUR — the property trend is supposed to have, and the wheel lacks")
print("=" * 100 + "\n")
spy = data.load_panel(["SPY"], start="2005-01-01", end="2026-09-01",
                      refresh=False).reindex(px.index).ffill()["SPY"]
CRISES = [("GFC 2008-09", "2007-10-09", "2009-03-09"),
          ("COVID 2020", "2020-02-19", "2020-03-23"),
          ("2022 bear", "2022-01-03", "2022-10-12")]
print(f"{'period':<18} {'SPY':>9} {'trend L/S':>11} {'trend LO':>10} {'buy&hold':>10}")
for lbl, a, b in CRISES:
    m = (px.index >= a) & (px.index <= b)
    if m.sum() < 5:
        continue
    def cum(s): return ((1 + s[m]).prod() - 1) * 100
    print(f"{lbl:<18} {(spy[m].iloc[-1]/spy[m].iloc[0]-1)*100:>8.1f}% "
          f"{cum(results['long/short (needs margin)']):>10.1f}% "
          f"{cum(results['long-only (cash account)']):>9.1f}% "
          f"{cum(bh.returns):>9.1f}%")

print("\n" + "=" * 100)
print("AS A SLEEVE beside a 60/40 core")
print("=" * 100 + "\n")
core_px = data.load_panel(["SPY", "AGG"], start="2005-01-01", end="2026-09-01",
                          refresh=False).ffill().dropna()
cw = pd.DataFrame({"SPY": 0.6, "AGG": 0.4}, index=core_px.index)
m = np.zeros(len(cw), dtype=bool); m[::21] = True
core = backtest.run(core_px, cw.where(pd.Series(m, index=cw.index), np.nan).ffill(),
                    cost_model=IBKR_US_EQUITY).returns
ff = factors.load()
print(f"{'sleeve':<28} {'corr':>7} {'beta':>7} {'alpha%':>8} {'NW t':>7} "
      f"{'FF6 a%':>8} {'FF6 t':>7} {'Sh@20%':>8} {'delta':>8}")
for lbl, r in results.items():
    ev = evaluation.evaluate(r, core, label=lbl)
    a6 = factors.attribution(r, ff)
    a, c = evaluation._align(r, core)
    b20 = metrics.sharpe(0.8 * c + 0.2 * a)
    print(f"{lbl:<28} {ev['corr_to_core']:>7.2f} {ev['beta']:>7.2f} "
          f"{ev['alpha_ann']*100:>7.2f}% {ev['alpha_t']:>7.2f} "
          f"{a6['alpha_ann']*100:>7.2f}% {a6['alpha_t']:>7.2f} "
          f"{b20:>8.2f} {b20-ev['core_sharpe']:>+8.3f}")
print(f"\n  core 60/40 Sharpe: {evaluation.evaluate(results['long-only (cash account)'], core)['core_sharpe']:.3f}")


# ---------------------------------------------------------------------------
# NULL FLOOR. Random SIGNS instead of trend signs, same construction.
#
# This is the right null: a random-sign multi-asset book is also roughly
# uncorrelated with a 60/40 core, so it isolates whether the TREND SIGNAL adds
# anything beyond "hold a low-correlation multi-asset book with random tilts".
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("NULL FLOOR — random signs, identical construction (40 draws)")
print("=" * 100)

def random_trend(rng, long_only):
    sig = pd.DataFrame(rng.choice([-1.0, 1.0], size=px.shape),
                       index=px.index, columns=px.columns)
    # match the trend signal's persistence: signs change monthly, not daily
    m = np.zeros(len(sig), dtype=bool); m[::21] = True
    sig = sig.where(pd.Series(m, index=sig.index), np.nan).ffill()
    if long_only:
        sig = sig.clip(lower=0.0)
    w = sig / vol.replace(0, np.nan)
    g = w.abs().sum(axis=1)
    w = w.div(g.where(g > 0), axis=0).fillna(0.0)
    return run(strategies.volatility_target(w, px, target_vol=0.10, max_leverage=1.0)).returns

for lbl, lo in (("long/short", False), ("long-only", True)):
    deltas, sharpes = [], []
    for i in range(40):
        r = random_trend(np.random.default_rng(4000 + i), lo)
        a, c = evaluation._align(r, core)
        deltas.append(metrics.sharpe(0.8 * c + 0.2 * a) - metrics.sharpe(c))
        sharpes.append(metrics.sharpe(r))
    d = np.array(deltas); s = np.array(sharpes)
    key = "long/short (needs margin)" if lo is False else "long-only (cash account)"
    ar, cr = evaluation._align(results[key], core)
    real_d = metrics.sharpe(0.8 * cr + 0.2 * ar) - metrics.sharpe(cr)
    print(f"\n  {lbl}:")
    print(f"    random-sign delta@20%: mean {d.mean():+.4f}  sd {d.std():.4f}  "
          f"p95 {np.percentile(d,95):+.4f}  max {d.max():+.4f}")
    print(f"    random-sign Sharpe:    mean {s.mean():+.3f}  max {s.max():+.3f}")
    print(f"    REAL trend delta@20%:  {real_d:+.4f}  -> "
          f"{evaluation.percentile_vs_null(real_d, d):.0f}th percentile")
    print(f"    REAL trend Sharpe:     {metrics.sharpe(results[key]):+.3f}  -> "
          f"{evaluation.percentile_vs_null(metrics.sharpe(results[key]), s):.0f}th percentile")
