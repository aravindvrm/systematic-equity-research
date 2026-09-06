"""Do non-linear methods extract more than a linear composite?

WHY THIS IS THE BEST-MOTIVATED TEST LEFT
----------------------------------------
We have direct evidence our COMBINATION method was destroying information: the
7-signal linear composite ranked at the 2nd percentile of the null while single
features reached the 70th. Averaging z-scores of signals that are 97% correlated
averages one signal with itself and adds noise from the rest. Interactions are
exactly what a linear blend cannot represent.

Gu, Kelly & Xiu (2020) find trees and neural nets roughly DOUBLE out-of-sample
R2 versus linear methods on comparable characteristic sets.

METHOD -- WALK-FORWARD, NO LOOKAHEAD
------------------------------------
For each test year Y: train on everything before Y-1, validate on Y-1, predict Y.
Roll forward one year at a time. The model never sees its own test period, and
the feature set is identical for every model so the comparison isolates the
FUNCTIONAL FORM rather than the information.

Target is the cross-sectionally demeaned forward return: we are ranking names
against each other, not forecasting the market.

Observations are monthly, matching the rebalance frequency. Using daily
observations would inflate the sample 21x with overlapping, near-duplicate rows
and make every standard error meaningless.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from scipy import stats as sst

from algo import (anomalies, data, features, features2 as f2, filings,
                  pead, research)

pd.set_option("display.width", 220)

M = pd.read_parquet("data/edgar/measures.parquet")
cl = data.load_panel(sorted(M.ticker.unique()), start="2009-01-01", end="2026-09-01",
                     field="close", refresh=False)
cl = cl.dropna(axis=1, thresh=int(0.9 * len(cl))).ffill(limit=5)
M = M[M.ticker.isin(cl.columns)]
P = {f: data.load_panel(sorted(M.ticker.unique()), start="2009-01-01",
                        end="2026-09-01", field=f, refresh=False)
     .reindex(index=cl.index, columns=cl.columns).ffill(limit=5)
     for f in ("open", "high", "low", "volume")}
o, hi, lo, vo = P["open"], P["high"], P["low"], P["volume"]
print(f"{cl.shape[1]} names, {len(cl)} bars", flush=True)

FEATS = {
    "mom_12_1": features.momentum_12_1(cl), "mom_252": features.momentum(cl, 252),
    "mom_63": features.momentum(cl, 63), "mom_21": features.momentum(cl, 21),
    "reversal_5": features.reversal(cl, 5), "reversal_21": features.reversal(cl, 21),
    "low_vol_60": features.low_vol(cl, 60), "vol_change": features.vol_change(cl),
    "near_high_252": features.near_high(cl, 252), "drawdown_252": features.drawdown(cl, 252),
    "skew_126": features.skewness(cl, 126), "ma_dist_200": features.ma_distance(cl, 200),
    "trend_r2": features.trend_strength(cl, 126),
    "vol_shock": f2.volume_shock(vo), "vol_trend": f2.volume_trend(vo),
    "dollar_volume": f2.dollar_volume(cl, vo),
    "px_vol_diverge": f2.price_volume_divergence(cl, vo),
    "garman_klass": f2.garman_klass_vol(o, hi, lo, cl),
    "close_in_range": f2.close_position_in_range(hi, lo, cl),
    "overnight_vs_day": f2.intraday_vs_overnight(o, cl),
    "true_range": f2.true_range_pct(hi, lo, cl),
    "resid_mom_252": f2.residual_momentum(cl, 252),
    "beta_to_univ": f2.beta_to_universe(cl), "corr_to_univ": f2.correlation_to_universe(cl),
    "idio_vol_share": f2.idio_vol_share(cl), "lead_lag": f2.lead_lag(cl),
}
# fundamentals
F = pd.read_parquet("data/fundamentals/all_facts.parquet")
uni = pd.read_parquet("data/collection_universe.parquet").dropna(subset=["cik"])
uni["cik"] = uni["cik"].astype(int).astype(str)
c2t = {k: v for k, v in zip(uni["cik"], uni["ticker"]) if v in cl.columns}
V = anomalies.compute(F)
for f in anomalies.FEATURES:
    FEATS[f] = anomalies.to_panel(V, cl.index, c2t, f)
FEATS.update(anomalies.market_ratios(V, cl, c2t))
# text + events
FEATS["filing_sim"] = filings.to_daily(M, cl.index, dict(zip(M.cik.astype(str), M.ticker)),
                                       col="sim_jaccard", lag_days=1, hold_days=126)
R = pd.read_parquet("data/edgar/pead_reactions.parquet")
FEATS["pead_sue"] = pead.to_daily(R, cl.index, cl.columns, col="sue", hold_days=63)
print(f"{len(FEATS)} features", flush=True)

# ---- long panel, monthly observations ---------------------------------------
H = 21
fwd = research.forward_returns(cl, H)
fwd_x = fwd.sub(fwd.mean(axis=1), axis=0)          # cross-sectionally demeaned
dates = cl.index[::H]

def xs_rank(df):
    """Cross-sectional percentile rank -- scale-free, outlier-robust, and the
    form the models should see since we only ever act on relative ordering."""
    return df.rank(axis=1, pct=True)

Xr = {k: xs_rank(v).loc[dates] for k, v in FEATS.items()}
rows = []
for d in dates:
    r = pd.DataFrame({k: Xr[k].loc[d] for k in FEATS})
    r["y"] = fwd_x.loc[d]
    r["date"] = d
    rows.append(r.reset_index().rename(columns={"index": "ticker"}))
L = pd.concat(rows, ignore_index=True).dropna(subset=["y"])
L = L[L[list(FEATS)].notna().mean(axis=1) > 0.7]
print(f"panel: {len(L):,} rows, {L.date.nunique()} dates, "
      f"{L.date.min().date()}..{L.date.max().date()}\n", flush=True)

cols = list(FEATS)
L[cols] = L[cols].fillna(0.5)          # rank space: 0.5 == median, a neutral fill

# ---- walk-forward -------------------------------------------------------------
years = sorted(L.date.dt.year.unique())
test_years = [y for y in years if y >= 2015]
MODELS = {
    "ridge (linear)": lambda: Ridge(alpha=10.0),
    "hist grad boost": lambda: HistGradientBoostingRegressor(
        max_iter=200, max_depth=4, learning_rate=0.05, l2_regularization=1.0,
        min_samples_leaf=40, random_state=0),
    "random forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=40, n_jobs=-1, random_state=0),
}
preds = {k: [] for k in MODELS}
for y in test_years:
    tr = L[L.date.dt.year < y]
    te = L[L.date.dt.year == y]
    if len(tr) < 2000 or te.empty:
        continue
    for name, mk in MODELS.items():
        m = mk()
        m.fit(tr[cols].to_numpy(), tr["y"].to_numpy())
        p = te[["date", "ticker"]].copy()
        p["pred"] = m.predict(te[cols].to_numpy())
        p["y"] = te["y"].to_numpy()
        preds[name].append(p)
    print(f"  {y}: train {len(tr):,} test {len(te):,}", flush=True)

print(f"\n{'='*78}")
print("OUT-OF-SAMPLE IC  (walk-forward, model never sees its test year)")
print(f"{'='*78}\n")
print(f"{'model':<20} {'IC':>9} {'t':>8} {'months':>8} {'hit rate':>10}")
res = {}
for name, ps in preds.items():
    D = pd.concat(ps, ignore_index=True)
    ic = D.groupby("date").apply(
        lambda g: sst.spearmanr(g["pred"], g["y"]).statistic if len(g) > 5 else np.nan
    ).dropna()
    res[name] = (D, ic)
    t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
    print(f"{name:<20} {ic.mean():>9.4f} {t:>8.2f} {len(ic):>8} {(ic>0).mean()*100:>9.0f}%")

print("\nlinear is the baseline. If the trees do not clearly beat it, the")
print("combination method was not what was holding the linear composite back.")


# ---------------------------------------------------------------------------
# PORTFOLIO LEVEL vs the NULL FLOOR. IC is not the deliverable.
# ---------------------------------------------------------------------------
import json
from algo import backtest, evaluation, factors, metrics, strategies
from algo.costs import IBKR_US_EQUITY

ff = factors.load()
N_RANDOM = 40

def to_panel(D):
    """Model predictions back onto a daily panel, held between rebalances."""
    w = D.pivot(index="date", columns="ticker", values="pred")
    return w.reindex(cl.index).ffill().reindex(columns=cl.columns)

def stack(score, top_frac=0.3, rebal=21):
    s = score.copy()
    mm = np.zeros(len(s), dtype=bool); mm[::rebal] = True
    s = s.where(pd.Series(mm, index=s.index), np.nan).ffill()
    raw = (s.rank(axis=1, pct=True, ascending=False) <= top_frac).astype(float)
    w = raw * strategies.inverse_volatility(cl, 60)
    tot = w.sum(axis=1)
    w = w.div(tot.where(tot > 0), axis=0).fillna(0.0)
    return strategies.volatility_target(w, cl, target_vol=0.10, max_leverage=1.0)

def rets(score, top_frac=0.3):
    return backtest.run(cl, stack(score, top_frac), cost_model=IBKR_US_EQUITY).returns

first_test = pd.Timestamp(f"{test_years[0]}-01-01")
print(f"\nrunning {N_RANDOM} null draws (OOS window {first_test.date()}+)...", flush=True)
nu, nc = [], []
for i in range(N_RANDOM):
    rng = np.random.default_rng(9000 + i)
    noise = pd.DataFrame(rng.normal(size=cl.shape), index=cl.index, columns=cl.columns)
    r = rets(noise)[cl.index >= first_test]
    nu.append(factors.attribution(r, ff)["alpha_t"])
    nc.append(factors.conditional_attribution(r, ff)["alpha_t"])
nu = np.array([x for x in nu if np.isfinite(x)])
nc = np.array([x for x in nc if np.isfinite(x)])

print(f"\n{'='*100}")
print(f"NULL FLOOR  uncond t: mean {nu.mean():+.2f} p95 {np.percentile(nu,95):+.2f}   "
      f"cond t: mean {nc.mean():+.2f} p95 {np.percentile(nc,95):+.2f}")
print(f"{'='*100}\n")
print(f"{'model':<22} {'Sharpe':>7} {'alpha%':>8} {'uncond t':>9} {'pct':>5} "
      f"{'cond t':>7} {'pct':>5}")

# equal-weight composite of the SAME features, as the reference point
eq = sum((Xr[k].reindex(cl.index).ffill() - 0.5) for k in cols) / len(cols)
for lbl, sc in [("equal-wt composite", eq)] + \
               [(k, to_panel(res[k][0])) for k in MODELS]:
    r = rets(sc)[cl.index >= first_test]
    u = factors.attribution(r, ff); c = factors.conditional_attribution(r, ff)
    print(f"{lbl:<22} {metrics.sharpe(r):>7.2f} {u['alpha_ann']*100:>7.2f}% "
          f"{u['alpha_t']:>9.2f} {evaluation.percentile_vs_null(u['alpha_t'], nu):>4.0f}% "
          f"{c['alpha_t']:>7.2f} {evaluation.percentile_vs_null(c['alpha_t'], nc):>4.0f}%")

print("\nThe equal-weight row is what we ran before. Compare the fitted models to it.")
