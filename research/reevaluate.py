"""Re-evaluate everything under the CORRECT question.

Old question: does this beat a vol-targeted equal-weight equity benchmark on
standalone Sharpe, over 2016-2026?

New question: added at a realistic weight to the balanced portfolio the user
already owns, does this make that portfolio better?

The difference is not cosmetic. The old bar was one of the best equity decades
in history; the new test is driven by CORRELATION, which we never once measured.

CORE PORTFOLIO: 60/40 SPY/AGG, monthly rebalanced. That is the thing a separate
sleeve has to improve on.

Sample runs from 2010 (or as early as each strategy's data allows), NOT just
2016-2026, so the answer is not an artifact of one bull market. Where a strategy
needs a fitted sign, signs were fixed on 2006-2015 in the original scripts and
are reused here unchanged.
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from algo import (backtest2, data, evaluation, features, features2 as f2,
                  filings, metrics, research, strategies)
from algo.costs import IBKR_US_EQUITY

pd.set_option("display.width", 250)
START, SPLIT = "2010-01-01", "2016-01-01"

# ---- the core portfolio the user already owns -------------------------------
core_px = data.load_panel(["SPY", "AGG"], start=START, end="2026-09-01", refresh=False)
core_px = core_px.ffill().dropna()
cw = pd.DataFrame({"SPY": 0.6, "AGG": 0.4}, index=core_px.index)
mask = np.zeros(len(cw), dtype=bool); mask[::21] = True
cw = cw.where(pd.Series(mask, index=cw.index), np.nan).ffill()
core = backtest2.run(core_px, cw, cost_model=IBKR_US_EQUITY).returns
print(f"CORE 60/40 SPY/AGG: {core_px.index[0].date()}..{core_px.index[-1].date()}  "
      f"Sharpe {metrics.sharpe(core):.2f}  "
      f"CAGR {metrics.cagr((1+core).cumprod())*100:.2f}%  "
      f"MaxDD {metrics.max_drawdown((1+core).cumprod())*100:.1f}%\n")

def stack(score, prices, top_frac=0.3, rebal=21):
    s = score.copy()
    m = np.zeros(len(s), dtype=bool); m[::rebal] = True
    s = s.where(pd.Series(m, index=s.index), np.nan).ffill()
    raw = (s.rank(axis=1, pct=True, ascending=False) <= top_frac).astype(float)
    w = raw * strategies.inverse_volatility(prices, 60)
    tot = w.sum(axis=1)
    w = w.div(tot.where(tot > 0), axis=0).fillna(0.0)
    return strategies.volatility_target(w, prices, target_vol=0.10, max_leverage=1.0)

def z(f):
    zz = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1).replace(0, np.nan), axis=0)
    return zz.clip(-3, 3)

STRATS = {}

# 1. the 30-name low-correlation book, no signal
tk = json.load(open("data/trading_universe.json"))
p30 = data.load_panel(tk, start=START, end="2026-09-01", refresh=False)
p30 = p30.dropna(axis=1, thresh=int(0.9 * len(p30))).ffill(limit=5)
flat30 = pd.DataFrame(0.0, index=p30.index, columns=p30.columns)
STRATS["30-name book (no signal)"] = backtest2.run(
    p30, stack(flat30, p30, 1.0, 21), cost_model=IBKR_US_EQUITY).returns

# 2/3. price composite and filings, on the large-cap universe
M = pd.read_parquet("data/edgar/measures.parquet")
cl = data.load_panel(sorted(M.ticker.unique()), start=START, end="2026-09-01",
                     field="close", refresh=False)
cl = cl.dropna(axis=1, thresh=int(0.9 * len(cl))).ffill(limit=5)
M = M[M.ticker.isin(cl.columns)]
P = {f: data.load_panel(sorted(M.ticker.unique()), start=START, end="2026-09-01",
                        field=f, refresh=False)
     .reindex(index=cl.index, columns=cl.columns).ffill(limit=5)
     for f in ("open", "high", "low", "volume")}
o, hi, lo, vo = P["open"], P["high"], P["low"], P["volume"]

PRICE = {"reversal_5": features.reversal(cl, 5),
         "mom_12_1": features.momentum_12_1(cl),
         "vol_shock": f2.volume_shock(vo),
         "close_in_range": f2.close_position_in_range(hi, lo, cl),
         "resid_mom_252": f2.residual_momentum(cl, 252),
         "mom_252": features.momentum(cl, 252),
         "overnight_vs_day": f2.intraday_vs_overnight(o, cl)}
tr = cl.index < SPLIT
fwd = research.forward_returns(cl, 63)
sg = {n: float(np.sign(research.cross_sectional_ic(f[tr], fwd[tr]).mean()))
      for n, f in PRICE.items()}
price_comp = sum(z(PRICE[n]) * sg[n] for n in PRICE) / len(PRICE)
jac = filings.to_daily(M, cl.index, dict(zip(M.cik.astype(str), M.ticker)),
                       col="sim_jaccard", lag_days=1, hold_days=126)
sj = float(np.sign(research.cross_sectional_ic(jac[tr], fwd[tr]).mean()))

flatl = pd.DataFrame(0.0, index=cl.index, columns=cl.columns)
STRATS["large-cap EW (no signal)"] = backtest2.run(
    cl, stack(flatl, cl, 1.0, 21), cost_model=IBKR_US_EQUITY).returns
STRATS["price composite"] = backtest2.run(
    cl, stack(price_comp, cl, 0.3, 21), cost_model=IBKR_US_EQUITY).returns
STRATS["filings (Lazy Prices)"] = backtest2.run(
    cl, stack(z(jac) * sj, cl, 0.3, 21), cost_model=IBKR_US_EQUITY).returns
STRATS["SPY (reference)"] = core_px["SPY"].pct_change().dropna()

print("=" * 132)
print("MARGINAL CONTRIBUTION TO A 60/40 CORE   (the question we should have been asking)")
print("=" * 132)
rows = [evaluation.evaluate(s, core, label=k) for k, s in STRATS.items()]
D = pd.DataFrame(rows)
print(f"\n{'strategy':<27} {'stand Sh':>9} {'corr':>7} {'beta':>7} "
      f"{'alpha%':>8} {'alpha t':>8} {'Sh@10%':>8} {'delta':>8} "
      f"{'w*':>6} {'Sh@w*':>7} {'delta':>8}")
for r in rows:
    print(f"{r['label']:<27} {r['sharpe_standalone']:>9.2f} {r['corr_to_core']:>7.2f} "
          f"{r['beta']:>7.2f} {r['alpha_ann']*100:>7.2f}% {r['alpha_t']:>8.2f} "
          f"{r['sharpe_at_10pct']:>8.2f} {r['delta_at_10pct']:>+8.3f} "
          f"{r['w_optimal']:>6.2f} {r['sharpe_at_opt']:>7.2f} {r['delta_at_opt']:>+8.3f}")
print(f"\ncore Sharpe (60/40) = {rows[0]['core_sharpe']:.3f}   "
      f"alpha t uses Newey-West (21 lags)")

print(f"\n\n{'-'*132}\nBLEND CURVE for the two most promising sleeves\n{'-'*132}")
for k in ("30-name book (no signal)", "filings (Lazy Prices)"):
    print(f"\n{k}")
    bc = evaluation.blend_curve(STRATS[k], core)
    bc["cagr"] *= 100; bc["vol"] *= 100; bc["max_dd"] *= 100
    print(bc.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# THE CONTROL THAT MATTERS: is the Sharpe gain just DE-RISKING?
#
# These sleeves have beta 0.54-0.75 to the core and correlation ~0.80. Adding
# them lowers portfolio volatility, and lowering volatility raises Sharpe if
# return falls by less. But you can lower volatility for FREE by simply holding
# less equity. So the honest benchmark for a sleeve with beta b is not the core
# -- it is a b-weighted mix of core and cash, which requires no strategy, no
# data, and no trading.
#
# If a sleeve cannot beat "hold less equity", it has produced nothing.
# ---------------------------------------------------------------------------
print(f"\n\n{'='*118}")
print("DE-RISKING CONTROL -- each sleeve vs a beta-matched core/cash mix")
print(f"{'='*118}\n")

CASH_ANN = 0.02          # ~2% average short rate over the period, daily-ised
cash = pd.Series(CASH_ANN / 252, index=core.index)

print(f"{'strategy':<27} {'beta':>6} {'sleeve Sh':>10} {'matched Sh':>11} "
      f"{'sleeve@10%':>11} {'matched@10%':>12} {'genuine gain':>13}")
for k, s in STRATS.items():
    a, c = evaluation._align(s, core)
    b = evaluation.alpha_beta(a, c)["beta"]
    if not np.isfinite(b):
        continue
    # A beta-matched passive alternative: b in core, rest in cash.
    matched = b * c + (1 - b) * cash.reindex(c.index).fillna(CASH_ANN / 252)
    p_sleeve = 0.9 * c + 0.1 * a
    p_match = 0.9 * c + 0.1 * matched
    print(f"{k:<27} {b:>6.2f} {metrics.sharpe(a):>10.2f} "
          f"{metrics.sharpe(matched):>11.2f} {metrics.sharpe(p_sleeve):>11.3f} "
          f"{metrics.sharpe(p_match):>12.3f} "
          f"{metrics.sharpe(p_sleeve)-metrics.sharpe(p_match):>+13.3f}")

print("\n'genuine gain' is the Sharpe the sleeve adds OVER simply holding less")
print("equity at the same beta. Near zero = the sleeve is de-risking, not alpha.")

print(f"\n\n{'='*118}")
print("MULTIPLE TESTING -- how many sleeves were tried before these?")
print(f"{'='*118}")
print("""
  ~12 signal families, ~200+ individual feature/horizon cells, and several
  portfolio constructions were examined before these four sleeves. An alpha
  t-statistic of 1.98 is NOT significant against that search. The Bonferroni
  bar for even 20 independent sleeve candidates is |t| > 2.87; for 200 it is
  |t| > 3.48. Treat every alpha t below ~3 here as consistent with noise.""")
