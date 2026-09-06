"""The evaluation this project should have started with.

THREE CONTROLS, all standard, none of which were run until now:

1. FAMA-FRENCH 6-FACTOR ATTRIBUTION. Regress excess returns on MKT, SMB, HML,
   RMW, CMA, MOM. Alpha that disappears here is not alpha -- it is a factor tilt
   purchasable through cheap ETFs. This is the single most standard test in
   asset pricing and this project ran twelve families without it.

2. NULL FLOOR. 40 random signals through the identical stack. Holds the whole
   construction pipeline constant and varies only information content. Measured
   earlier: random signals score alpha t = 3.14 against a 60/40 core.

3. HARVEY-LIU-ZHU BAR. With hundreds of published factors already mined, |t|>2
   is far too lenient; they argue |t|>3.0 for a NEW factor. That is for one
   PRE-SPECIFIED test. This project ran ~200 cells, so the honest bar is higher
   still -- the null floor is what actually calibrates it.

References: Fama & French (2015); Carhart (1997); Harvey, Liu & Zhu (2016, RFS);
Bailey, Borwein, Lopez de Prado & Zhu (PBO/CSCV); Ferson & Schadt (1996).
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from algo import (backtest, data, evaluation, factors, features,
                  features2 as f2, filings, metrics, pead, research, strategies)
from algo.costs import IBKR_US_EQUITY

pd.set_option("display.width", 260)
START, SPLIT, N_RANDOM = "2005-01-01", "2016-01-01", 40

ff = factors.load()
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
print(f"universe {cl.shape[1]} names, {len(cl)} bars\n", flush=True)

def z(f):
    zz = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1).replace(0, np.nan), axis=0)
    return zz.clip(-3, 3)

def stack(score, prices, top_frac=0.3, rebal=21):
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
sign = lambda f: float(np.sign(research.cross_sectional_ic(f[tr], fwd[tr]).mean()) or 1.0)

S = {}
PRICE = {"reversal_5": features.reversal(cl, 5), "mom_12_1": features.momentum_12_1(cl),
         "vol_shock": f2.volume_shock(vo),
         "close_in_range": f2.close_position_in_range(hi, lo, cl),
         "resid_mom_252": f2.residual_momentum(cl, 252),
         "mom_252": features.momentum(cl, 252),
         "overnight_vs_day": f2.intraday_vs_overnight(o, cl)}
S["price composite"] = rets(sum(z(f) * sign(f) for f in PRICE.values()) / len(PRICE))
jac = filings.to_daily(M, cl.index, dict(zip(M.cik.astype(str), M.ticker)),
                       col="sim_jaccard", lag_days=1, hold_days=126)
S["filings (jaccard)"] = rets(z(jac) * sign(jac))
R = pd.read_parquet("data/edgar/pead_reactions.parquet")
sue = pead.to_daily(R, cl.index, cl.columns, col="sue", hold_days=63)
S["PEAD (sue)"] = rets(z(sue) * sign(sue))
S["idio vol share"] = rets(z(f2.idio_vol_share(cl)))
S["low beta to universe"] = rets(z(f2.beta_to_universe(cl)))
S["large-cap EW (NO SIGNAL)"] = rets(pd.DataFrame(0.0, index=cl.index,
                                                  columns=cl.columns), top_frac=1.0)
tk = json.load(open("data/trading_universe.json"))
p30 = data.load_panel(tk, start=START, end="2026-09-01", refresh=False)
p30 = p30.dropna(axis=1, thresh=int(0.9 * len(p30))).ffill(limit=5)
S["30-name book (NO SIGNAL)"] = rets(pd.DataFrame(0.0, index=p30.index,
                                                  columns=p30.columns), p30, 1.0)

print(f"running {N_RANDOM} random controls...", flush=True)
null_a = []
for i in range(N_RANDOM):
    rng = np.random.default_rng(2000 + i)
    noise = pd.DataFrame(rng.normal(size=cl.shape), index=cl.index, columns=cl.columns)
    null_a.append(factors.attribution(rets(noise), ff)["alpha_t"])
null_a = np.array([x for x in null_a if np.isfinite(x)])

print(f"\n{'='*126}")
print("NULL FLOOR -- FF6 alpha t of 40 ZERO-INFORMATION signals through the identical stack")
print(f"{'='*126}")
print(f"  mean {null_a.mean():+.2f}   sd {null_a.std():.2f}   "
      f"p95 {np.percentile(null_a,95):+.2f}   max {null_a.max():+.2f}")

print(f"\n{'='*126}")
print("FAMA-FRENCH 6-FACTOR ATTRIBUTION")
print(f"{'='*126}\n")
print(f"{'strategy':<28} {'alpha%':>8} {'t':>7} {'pctile':>7} | "
      f"{'MKT':>6} {'SMB':>6} {'HML':>6} {'RMW':>6} {'CMA':>6} {'MOM':>6} {'R2':>6}")
res = []
for k, s in S.items():
    a = factors.attribution(s, ff)
    pct = evaluation.percentile_vs_null(a["alpha_t"], null_a)
    res.append((k, a, pct))
    print(f"{k:<28} {a['alpha_ann']*100:>7.2f}% {a['alpha_t']:>7.2f} {pct:>6.0f}% | "
          + " ".join(f"{a['b_'+f]:>6.2f}" for f in factors.FACTORS)
          + f" {a['r2']:>6.2f}")

print(f"\n{'-'*126}")
hlz = sum(1 for _, a, _ in res if abs(a["alpha_t"]) > 3.0)
beat = sum(1 for _, _, p in res if p >= 95)
print(f"clearing Harvey-Liu-Zhu |t| > 3.0 (single pre-specified test): {hlz} of {len(res)}")
print(f"clearing the 95th percentile of the NULL FLOOR:                {beat} of {len(res)}")
print("\nThe second number is the honest one. HLZ's bar assumes ONE pre-specified")
print("test; the null floor is calibrated to THIS stack and THIS search.")


# ---------------------------------------------------------------------------
# FERSON-SCHADT CONDITIONAL MODEL -- the literature's fix for exactly this bug.
# ---------------------------------------------------------------------------
print(f"\n\n{'='*110}")
print("FERSON-SCHADT (1996) CONDITIONAL ATTRIBUTION -- beta allowed to vary with public info")
print(f"{'='*110}")
cnull = []
for i in range(N_RANDOM):
    rng = np.random.default_rng(2000 + i)
    noise = pd.DataFrame(rng.normal(size=cl.shape), index=cl.index, columns=cl.columns)
    cnull.append(factors.conditional_attribution(rets(noise), ff)["alpha_t"])
cnull = np.array([x for x in cnull if np.isfinite(x)])
print(f"\nnull floor, CONDITIONAL alpha t: mean {cnull.mean():+.2f}  sd {cnull.std():.2f}  "
      f"p95 {np.percentile(cnull,95):+.2f}")
print(f"(unconditional was mean {null_a.mean():+.2f}, p95 {np.percentile(null_a,95):+.2f})\n")
print(f"{'strategy':<28} {'uncond t':>9} {'cond t':>8} {'timing b':>10} {'timing t':>9} {'pctile':>7}")
for k, s in S.items():
    u = factors.attribution(s, ff)
    c = factors.conditional_attribution(s, ff)
    print(f"{k:<28} {u['alpha_t']:>9.2f} {c['alpha_t']:>8.2f} "
          f"{c['b_timing']:>10.3f} {c['t_timing']:>9.2f} "
          f"{evaluation.percentile_vs_null(c['alpha_t'], cnull):>6.0f}%")
print("\nA large NEGATIVE timing loading means beta falls when volatility rises --")
print("i.e. the vol-target overlay, which is what was masquerading as alpha.")
