"""The volatility overlay, held to account: run the headline with and without it.

WHY
---
The paper's claim has two halves that are easy to conflate:

  (a) there is no tradeable cross-sectional alpha in this universe, and
  (b) the portfolio overlay makes conventional alpha statistics unreliable.

Every number in the study is produced by a stack that includes a 10% volatility
target. So a reader is entitled to ask whether the null floor of t = 2.75 is a
property of the SIGNALS or a property of the OVERLAY -- and, separately, whether
the strategies fail because they have no edge or because the overlay removed it.

This separates them. Identical universe, identical signal ranking, identical
top-30% truncation, identical inverse-volatility weighting, identical monthly
rebalance, identical costs. The ONLY difference is whether the volatility target
is applied. Each arm gets its own recalibrated null, because a floor computed
for one construction does not transfer to another -- that is the project's
standing rule and it applies to its own ablation.

WHAT TO EXPECT IF THE PAPER IS RIGHT
------------------------------------
  * the null floor should COLLAPSE without the overlay -- no time-varying beta,
    nothing for a constant-beta factor model to misread as alpha
  * the strategies should still fail against their own null in both arms, since
    the claim is that the signals carry nothing either way

If the first happens and the second does not, the overlay was manufacturing the
phantom alpha AND hiding a real signal, which would be a genuinely new result.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from algo import backtest, data, evaluation, factors, features, metrics, research, strategies
from algo.costs import IBKR_US_EQUITY

pd.set_option("display.width", 200)

START = "2005-01-01"
TOP_FRAC = 0.30
REBAL = 21
N_NULL = 200

M = pd.read_parquet("data/edgar/measures.parquet")
cl = data.load_panel(sorted(M.ticker.unique()), start=START, end="2026-09-01",
                     field="close", refresh=False)
cl = cl.dropna(axis=1, thresh=int(0.9 * len(cl))).ffill(limit=5)
print(f"universe {cl.shape[1]} names, {len(cl)} bars, "
      f"{cl.index[0].date()}..{cl.index[-1].date()}", flush=True)

ff = factors.load()


def weights(score, vol_target: bool):
    """Identical construction in both arms except the overlay."""
    s = score.copy()
    mm = np.zeros(len(s), dtype=bool)
    mm[::REBAL] = True
    s = s.where(pd.Series(mm, index=s.index), np.nan).ffill()
    raw = (s.rank(axis=1, pct=True, ascending=False) <= TOP_FRAC).astype(float)
    w = raw * strategies.inverse_volatility(cl, 60)
    tot = w.sum(axis=1)
    w = w.div(tot.where(tot > 0), axis=0).fillna(0.0)
    if not vol_target:
        return w                      # fully invested, long-only, no overlay
    return strategies.volatility_target(w, cl, target_vol=0.10, max_leverage=1.0)


def run(score, vol_target: bool) -> pd.Series:
    return backtest.run(cl, weights(score, vol_target),
                        cost_model=IBKR_US_EQUITY).returns


def null_stats(vol_target: bool, n: int = N_NULL):
    """Zero-information signals through this exact arm."""
    ts, sharpes = [], []
    rng_cols, rng_idx = cl.columns, cl.index
    for i in range(n):
        rng = np.random.default_rng(7000 + i)
        noise = pd.DataFrame(rng.normal(size=cl.shape), index=rng_idx, columns=rng_cols)
        r = run(noise, vol_target)
        ts.append(factors.attribution(r, ff).get("alpha_t", np.nan))
        sharpes.append(metrics.sharpe(r))
    ts = np.array([t for t in ts if np.isfinite(t)])
    sh = np.array([s for s in sharpes if np.isfinite(s)])
    return ts, sh


# ---- the real signals, same set the headline battery ranks on -----------------
SIGNALS = {
    "reversal_5":     features.reversal(cl, 5),
    "mom_12_1":       features.momentum_12_1(cl),
    "mom_252":        features.momentum(cl, 252),
    "mom_63":         features.momentum(cl, 63),
    "low_vol_60":     features.low_vol(cl, 60),
    "near_high_252":  features.near_high(cl, 252),
}

print(f"\nRunning {N_NULL}-draw nulls for each arm...", flush=True)
arms = {}
for label, vt in (("with vol target", True), ("no overlay", False)):
    ts, sh = null_stats(vt)
    arms[label] = {"ts": ts, "sh": sh, "vt": vt}
    print(f"  {label:<18} alpha-t null: mean {ts.mean():+.2f}  sd {ts.std():.2f}  "
          f"p95 {np.percentile(ts,95):+.2f}  max {ts.max():+.2f}", flush=True)

print("\n" + "=" * 104)
print("THE OVERLAY'S EFFECT ON THE NULL  (zero-information signals, identical stack otherwise)")
print("=" * 104)
print(f"{'arm':<20}{'alpha-t mean':>14}{'sd':>8}{'p95':>9}{'max':>9}"
      f"{'Sharpe mean':>14}{'Sharpe p95':>12}")
for label, a in arms.items():
    print(f"{label:<20}{a['ts'].mean():>+14.2f}{a['ts'].std():>8.2f}"
          f"{np.percentile(a['ts'],95):>+9.2f}{a['ts'].max():>+9.2f}"
          f"{a['sh'].mean():>+14.3f}{np.percentile(a['sh'],95):>+12.3f}")

print("\n" + "=" * 104)
print("THE SAME SIGNALS IN BOTH ARMS, each against its own arm's null")
print("=" * 104)
print(f"{'signal':<16}{'arm':<18}{'Sharpe':>8}{'alpha ann':>11}{'alpha t':>9}"
      f"{'t pctile':>10}{'clears p95?':>12}")
rows = []
for name, feat in SIGNALS.items():
    for label, a in arms.items():
        r = run(feat, a["vt"])
        att = factors.attribution(r, ff)
        t = att.get("alpha_t", np.nan)
        pct = evaluation.percentile_vs_null(t, a["ts"])
        clears = "yes" if t > np.percentile(a["ts"], 95) else "no"
        print(f"{name:<16}{label:<18}{metrics.sharpe(r):>8.2f}"
              f"{att.get('alpha_ann', np.nan)*100:>10.2f}%{t:>9.2f}{pct:>9.0f}%{clears:>12}")
        rows.append({"signal": name, "arm": label, "sharpe": metrics.sharpe(r),
                     "alpha_ann": att.get("alpha_ann", np.nan), "alpha_t": t,
                     "t_pctile": pct, "clears_p95": clears})

out = pd.DataFrame(rows)
out.to_csv("results/voltarget_ablation.csv", index=False)

summary = pd.DataFrame([
    {"arm": k, "null_t_mean": v["ts"].mean(), "null_t_sd": v["ts"].std(),
     "null_t_p95": np.percentile(v["ts"], 95), "null_t_max": v["ts"].max(),
     "null_sharpe_mean": v["sh"].mean(), "n_draws": len(v["ts"])}
    for k, v in arms.items()])
summary.to_csv("results/voltarget_ablation_null.csv", index=False)
print("\nwritten: results/voltarget_ablation.csv, results/voltarget_ablation_null.csv")
