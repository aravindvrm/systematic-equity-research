"""EVERY family, under the full control stack. Nothing exempted.

Controls applied to all of them:
  1. Fama-French 6-factor attribution (MKT SMB HML RMW CMA MOM), Newey-West
  2. Ferson-Schadt conditional model (beta allowed to vary with lagged mkt vol)
  3. NULL FLOOR -- 40 zero-information signals through the IDENTICAL stack

The null floor is what calibrates everything. Since all cross-sectional
strategies here share one universe, one stack, one cost model and one rebalance
schedule, a single null floor is valid for all of them. Any strategy that
changes those (different universe, different top_frac) needs its OWN null and is
marked accordingly.

Signs, where a signal needs one, are fitted on the pre-2016 half only.
"""
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from algo import (backtest, data, evaluation, factors, features,
                  features2 as f2, filings, insider_state, metrics, network,
                  pead, pername, regimes, research, relational, strategies)
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
print(f"universe {cl.shape[1]} names, {len(cl)} bars, "
      f"{cl.index[0].date()}..{cl.index[-1].date()}\n", flush=True)

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
def sgn(f):
    v = research.cross_sectional_ic(f[tr], fwd[tr]).mean()
    return float(np.sign(v)) if np.isfinite(v) and v != 0 else 1.0

S, notes = {}, {}

# ---- FAMILY 1: price features, individually and as a composite ---------------
PRICE = {"reversal_5": features.reversal(cl, 5), "mom_12_1": features.momentum_12_1(cl),
         "mom_252": features.momentum(cl, 252), "mom_63": features.momentum(cl, 63),
         "low_vol_60": features.low_vol(cl, 60), "near_high_252": features.near_high(cl, 252),
         "skew_126": features.skewness(cl, 126), "ma_dist_200": features.ma_distance(cl, 200),
         "vol_shock": f2.volume_shock(vo), "vol_trend": f2.volume_trend(vo),
         "close_in_range": f2.close_position_in_range(hi, lo, cl),
         "overnight_vs_day": f2.intraday_vs_overnight(o, cl),
         "garman_klass": f2.garman_klass_vol(o, hi, lo, cl),
         "resid_mom_252": f2.residual_momentum(cl, 252)}
for k, f in PRICE.items():
    S[f"1 px:{k}"] = rets(z(f) * sgn(f))
S["1 px:COMPOSITE"] = rets(sum(z(f) * sgn(f) for f in PRICE.values()) / len(PRICE))
print("family 1 built", flush=True)

# ---- FAMILY 3: mean-reversion setups ----------------------------------------
r1 = cl.pct_change()
sd20 = r1.rolling(20).std()
S["3 setup:2sd drop"] = rets(z(-(r1 / sd20.replace(0, np.nan))))
S["3 setup:5d oversold"] = rets(z(-cl.pct_change(5) / sd20.replace(0, np.nan)))
print("family 3 built", flush=True)

# ---- FAMILY 4/5: insider ------------------------------------------------------
try:
    f4 = pd.read_parquet("data/form4/transactions.parquet")
    ins = insider_state.build(f4, cl)
    for k in ("ins_net_value", "ins_buy_share", "ins_n_buyers", "ins_any_buy"):
        if k in ins:
            S[f"4 insider:{k}"] = rets(z(ins[k].fillna(0.0)) * sgn(ins[k].fillna(0.0)))
    notes["4 insider"] = "form4 data starts 2018 -- shorter sample"
    print("family 4/5 built", flush=True)
except Exception as e:
    print("family 4/5 SKIPPED:", e, flush=True)

# ---- FAMILY 7: per-name conditioning -----------------------------------------
for k, fn in (("own_vol", pername.own_vol_state), ("own_drawdown", pername.own_drawdown_state),
              ("own_corr", pername.own_corr_state), ("own_trend", pername.own_trend_state)):
    st = fn(cl)
    S[f"7 state:{k}"] = rets(z(st.fillna(0.5)) * sgn(st.fillna(0.5)))
print("family 7 built", flush=True)

# ---- FAMILY 8/9: relational ---------------------------------------------------
for k, f in (("beta_to_univ", f2.beta_to_universe(cl)),
             ("corr_to_univ", f2.correlation_to_universe(cl)),
             ("idio_vol_share", f2.idio_vol_share(cl)),
             ("lead_lag", f2.lead_lag(cl)),
             ("rel_strength_126", f2.relative_strength(cl, 126))):
    S[f"9 rel:{k}"] = rets(z(f) * sgn(f))
for k, fn in (("net_centrality", network.net_centrality),
              ("down_corr_asym", network.down_corr_asym),
              ("beta_instability", network.beta_instability),
              ("corr_dispersion", network.corr_dispersion)):
    f = fn(cl)
    S[f"9 net:{k}"] = rets(z(f) * sgn(f))
print("family 8/9 built", flush=True)

# ---- FAMILY 11/12: filings, PEAD ---------------------------------------------
jac = filings.to_daily(M, cl.index, dict(zip(M.cik.astype(str), M.ticker)),
                       col="sim_jaccard", lag_days=1, hold_days=126)
S["11 filings:jaccard"] = rets(z(jac) * sgn(jac))
tf = filings.to_daily(M, cl.index, dict(zip(M.cik.astype(str), M.ticker)),
                      col="sim_cosine_tfidf", lag_days=1, hold_days=126)
S["11 filings:tfidf"] = rets(z(tf) * sgn(tf))
R = pd.read_parquet("data/edgar/pead_reactions.parquet")
for c in ("sue", "car"):
    d = pead.to_daily(R, cl.index, cl.columns, col=c, hold_days=63)
    S[f"12 pead:{c}"] = rets(z(d) * sgn(d))
print("family 11/12 built", flush=True)

# ---- controls: no-signal -------------------------------------------------------
S["0 NO-SIGNAL large-cap EW"] = rets(pd.DataFrame(0.0, index=cl.index,
                                                  columns=cl.columns), top_frac=1.0)
notes["0 NO-SIGNAL large-cap EW"] = "top_frac=1.0 -- own null, not comparable"

print(f"\n{len(S)} strategies built. running {N_RANDOM} null draws...", flush=True)
nu, nc = [], []
for i in range(N_RANDOM):
    rng = np.random.default_rng(3000 + i)
    noise = pd.DataFrame(rng.normal(size=cl.shape), index=cl.index, columns=cl.columns)
    r = rets(noise)
    nu.append(factors.attribution(r, ff)["alpha_t"])
    nc.append(factors.conditional_attribution(r, ff)["alpha_t"])
nu = np.array([x for x in nu if np.isfinite(x)])
nc = np.array([x for x in nc if np.isfinite(x)])

print(f"\n{'='*128}")
print(f"NULL FLOOR (n={len(nu)}, top_frac=0.3, monthly, IBKR costs)")
print(f"  FF6 uncond alpha t : mean {nu.mean():+.2f}  sd {nu.std():.2f}  "
      f"p95 {np.percentile(nu,95):+.2f}  max {nu.max():+.2f}")
print(f"  Ferson-Schadt cond : mean {nc.mean():+.2f}  sd {nc.std():.2f}  "
      f"p95 {np.percentile(nc,95):+.2f}  max {nc.max():+.2f}")
print(f"{'='*128}\n")

rows = []
for k, s in S.items():
    u = factors.attribution(s, ff)
    c = factors.conditional_attribution(s, ff)
    rows.append(dict(strategy=k, alpha=u["alpha_ann"], t_u=u["alpha_t"],
                     t_c=c["alpha_t"], r2=u["r2"],
                     pct_u=evaluation.percentile_vs_null(u["alpha_t"], nu),
                     pct_c=evaluation.percentile_vs_null(c["alpha_t"], nc),
                     sharpe=metrics.sharpe(s)))
D = pd.DataFrame(rows).sort_values("pct_c", ascending=False)
print(f"{'strategy':<26} {'Sharpe':>7} {'alpha%':>8} {'uncond t':>9} {'pct':>5} "
      f"{'cond t':>7} {'pct':>5} {'R2':>6} {'note':<10}")
for r in D.itertuples():
    n = "own null" if r.strategy in notes and "own null" in notes[r.strategy] else \
        ("2018+" if r.strategy in notes else "")
    print(f"{r.strategy:<26} {r.sharpe:>7.2f} {r.alpha*100:>7.2f}% {r.t_u:>9.2f} "
          f"{r.pct_u:>4.0f}% {r.t_c:>7.2f} {r.pct_c:>4.0f}% {r.r2:>6.2f} {n:<10}")

comp = D[~D.strategy.isin([k for k in notes if "own null" in notes.get(k, "")])]
print(f"\n{'-'*128}")
print(f"comparable strategies: {len(comp)}")
print(f"clearing null p95 UNCONDITIONAL: {(comp.pct_u >= 95).sum()}")
print(f"clearing null p95 CONDITIONAL:   {(comp.pct_c >= 95).sum()}")
print(f"expected by chance at 5%:        {0.05*len(comp):.1f}")
D.to_csv("reassess_all.csv", index=False)
