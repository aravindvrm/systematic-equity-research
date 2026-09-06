"""Re-evaluate every family under the corrected framework -- WITH A NULL FLOOR.

THE DESIGN POINT
----------------
Re-running the twelve families against a 60/40 core would produce twelve numbers
and no way to read them. "+0.036 genuine gain" means nothing in isolation.

So this also runs N RANDOM signals through the identical stack. A random signal
has zero information by construction, so the distribution of its genuine gain IS
the noise floor. Any real family must clear that distribution, not merely be
positive.

This is the control I should have had from the start: it converts "is +0.036
good?" from a judgement call into a percentile.

Signals with IC ~ 0 are included deliberately. They are not expected to work --
they are extra draws from the null, and they tell us whether the framework
manufactures gains out of nothing.
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from algo import (backtest, data, evaluation, features, features2 as f2,
                  filings, metrics, pead, research, strategies)
from algo.costs import IBKR_US_EQUITY

pd.set_option("display.width", 250)
START, SPLIT = "2005-01-01", "2016-01-01"
N_RANDOM = 40

# ---- core the user already owns ---------------------------------------------
core_px = data.load_panel(["SPY", "AGG"], start=START, end="2026-09-01",
                          refresh=False).ffill().dropna()
cw = pd.DataFrame({"SPY": 0.6, "AGG": 0.4}, index=core_px.index)
m = np.zeros(len(cw), dtype=bool); m[::21] = True
cw = cw.where(pd.Series(m, index=cw.index), np.nan).ffill()
core = backtest.run(core_px, cw, cost_model=IBKR_US_EQUITY).returns
print(f"core 60/40: {core_px.index[0].date()}..{core_px.index[-1].date()} "
      f"Sharpe {metrics.sharpe(core):.3f}\n")

# ---- universe ---------------------------------------------------------------
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
print(f"universe: {cl.shape[1]} names, {len(cl)} bars", flush=True)


def z(f):
    zz = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1).replace(0, np.nan), axis=0)
    return zz.clip(-3, 3)


def stack(score, prices=None, top_frac=0.3, rebal=21):
    prices = cl if prices is None else prices
    s = score.copy()
    mm = np.zeros(len(s), dtype=bool); mm[::rebal] = True
    s = s.where(pd.Series(mm, index=s.index), np.nan).ffill()
    raw = (s.rank(axis=1, pct=True, ascending=False) <= top_frac).astype(float)
    w = raw * strategies.inverse_volatility(prices, 60)
    tot = w.sum(axis=1)
    w = w.div(tot.where(tot > 0), axis=0).fillna(0.0)
    return strategies.volatility_target(w, prices, target_vol=0.10, max_leverage=1.0)


def rets(score, prices=None, top_frac=0.3):
    prices = cl if prices is None else prices
    return backtest.run(prices, stack(score, prices, top_frac),
                        cost_model=IBKR_US_EQUITY).returns


tr = cl.index < SPLIT
fwd = research.forward_returns(cl, 63)

def fix_sign(f):
    """Sign fitted on the pre-2016 half only."""
    return float(np.sign(research.cross_sectional_ic(f[tr], fwd[tr]).mean()) or 1.0)


STRATS = {}

# --- family 1: price features -------------------------------------------------
PRICE = {"reversal_5": features.reversal(cl, 5),
         "mom_12_1": features.momentum_12_1(cl),
         "vol_shock": f2.volume_shock(vo),
         "close_in_range": f2.close_position_in_range(hi, lo, cl),
         "resid_mom_252": f2.residual_momentum(cl, 252),
         "mom_252": features.momentum(cl, 252),
         "overnight_vs_day": f2.intraday_vs_overnight(o, cl)}
STRATS["1 price composite"] = rets(
    sum(z(f) * fix_sign(f) for f in PRICE.values()) / len(PRICE))

# --- family 11: filing text ---------------------------------------------------
jac = filings.to_daily(M, cl.index, dict(zip(M.cik.astype(str), M.ticker)),
                       col="sim_jaccard", lag_days=1, hold_days=126)
STRATS["11 filings (jaccard)"] = rets(z(jac) * fix_sign(jac))

# --- family 12: PEAD ----------------------------------------------------------
R = pd.read_parquet("data/edgar/pead_reactions.parquet")
sue = pead.to_daily(R, cl.index, cl.columns, col="sue", hold_days=63)
STRATS["12 PEAD (sue)"] = rets(z(sue) * fix_sign(sue))

# --- families 6/7: conditioners, best cell as a standalone tilt ----------------
from algo import pername
vol_state = pername.own_vol_state(cl)
STRATS["7 per-name low-vol state"] = rets(z(-vol_state.fillna(0.5)))

# --- family 8/9: relational ---------------------------------------------------
STRATS["9 low beta to universe"] = rets(z(f2.beta_to_universe(cl)))
STRATS["9 idio vol share"] = rets(z(f2.idio_vol_share(cl)))

# --- no-signal books ----------------------------------------------------------
flat = pd.DataFrame(0.0, index=cl.index, columns=cl.columns)
STRATS["0 large-cap EW (no signal)"] = rets(flat, top_frac=1.0)
tk = json.load(open("data/trading_universe.json"))
p30 = data.load_panel(tk, start=START, end="2026-09-01", refresh=False)
p30 = p30.dropna(axis=1, thresh=int(0.9 * len(p30))).ffill(limit=5)
STRATS["0 30-name book (no signal)"] = rets(
    pd.DataFrame(0.0, index=p30.index, columns=p30.columns), p30, top_frac=1.0)

print(f"built {len(STRATS)} strategies; now {N_RANDOM} random controls...", flush=True)

# --- THE NULL FLOOR -----------------------------------------------------------
null_gains, null_alphas = [], []
cash = pd.Series(0.02 / 252, index=core.index)
for seed in range(N_RANDOM):
    rng = np.random.default_rng(1000 + seed)
    noise = pd.DataFrame(rng.normal(size=cl.shape), index=cl.index, columns=cl.columns)
    r = rets(noise)
    a, c = evaluation._align(r, core)
    ab = evaluation.alpha_beta(a, c)
    b = ab["beta"]
    matched = b * c + (1 - b) * cash.reindex(c.index).fillna(0.02 / 252)
    null_gains.append(metrics.sharpe(0.9 * c + 0.1 * a)
                      - metrics.sharpe(0.9 * c + 0.1 * matched))
    null_alphas.append(ab["alpha_t"])
null_gains = np.array(null_gains); null_alphas = np.array(null_alphas)

print(f"\n{'='*118}")
print("NULL FLOOR -- 40 RANDOM signals through the identical stack")
print(f"{'='*118}")
print(f"  genuine gain @10%:  mean {null_gains.mean():+.4f}  sd {null_gains.std():.4f}  "
      f"median {np.median(null_gains):+.4f}")
print(f"                      p90 {np.percentile(null_gains,90):+.4f}   "
      f"p95 {np.percentile(null_gains,95):+.4f}   max {null_gains.max():+.4f}")
print(f"  alpha NW t:         mean {null_alphas.mean():+.2f}  sd {null_alphas.std():.2f}  "
      f"max {null_alphas.max():+.2f}")
print("\n  A random signal has ZERO information. Whatever it scores here is the")
print("  noise floor. Real families must clear this distribution, not merely be positive.")

print(f"\n{'='*118}")
print("EVERY FAMILY vs the null floor")
print(f"{'='*118}\n")
print(f"{'strategy':<30} {'corr':>6} {'beta':>6} {'alpha%':>8} {'NW t':>7} "
      f"{'gain@10%':>9} {'null pctile':>12} {'verdict':>10}")
rows = []
for k, s in STRATS.items():
    a, c = evaluation._align(s, core)
    ab = evaluation.alpha_beta(a, c)
    b = ab["beta"]
    matched = b * c + (1 - b) * cash.reindex(c.index).fillna(0.02 / 252)
    g = (metrics.sharpe(0.9 * c + 0.1 * a)
         - metrics.sharpe(0.9 * c + 0.1 * matched))
    pct = (null_gains < g).mean() * 100
    verdict = "clears" if pct >= 95 else ("noise" if pct < 90 else "marginal")
    rows.append((k, ab, g, pct))
    print(f"{k:<30} {a.corr(c):>6.2f} {b:>6.2f} {ab['alpha_ann']*100:>7.2f}% "
          f"{ab['alpha_t']:>7.2f} {g:>+9.4f} {pct:>11.0f}% {verdict:>10}")

print(f"\n{'-'*118}")
n_clear = sum(1 for _, _, _, p in rows if p >= 95)
print(f"families clearing the 95th percentile of the null: {n_clear} of {len(rows)}")
print("expected by chance at 5%: {:.1f}".format(0.05 * len(rows)))
