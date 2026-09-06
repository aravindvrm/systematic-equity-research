"""Regime-conditional features -- PRE-SPECIFIED regimes only.

Three regimes, each chosen for a documented economic reason, defined BEFORE
looking at results:

  VOL      -- momentum crashes after high-volatility drawdowns (Barroso &
              Santa-Clara). Mean reversion is stronger when vol is high.
  DISPERSION -- cross-sectional stock selection has more to work with when names
              are moving differently from each other.
  TREND    -- momentum is documented to reverse sharply after market drawdowns
              (the "momentum crash").

Regime boundaries are MEDIAN splits computed on a TRAILING window -- no
in-sample fitting of thresholds, and no lookahead in the regime label itself.
"""
import numpy as np, pandas as pd
from algo import data, features, research, universe as U

tab = U.collection_table()
tk = [t for t in tab.ticker if t not in U.DIVERSIFIERS]
px = data.load_panel(tk + ["SPY"], start="2018-01-01", end="2026-09-01")
px = px[[c for c in px.columns if px[c].notna().mean() > 0.95]].ffill().dropna()
spy = px["SPY"]; eq = px[[c for c in px.columns if c != "SPY"]]
R = eq.pct_change()

# --- regimes, all point-in-time -------------------------------------------
vol = spy.pct_change().rolling(21).std()
disp = R.std(axis=1).rolling(21).mean()
trend = spy / spy.rolling(200).mean() - 1.0
REGIMES = {
    "high vol":   vol > vol.rolling(504, min_periods=252).median(),
    "low vol":    vol <= vol.rolling(504, min_periods=252).median(),
    "high disp":  disp > disp.rolling(504, min_periods=252).median(),
    "low disp":   disp <= disp.rolling(504, min_periods=252).median(),
    "uptrend":    trend > 0,
    "downtrend":  trend <= 0,
}
FEATS = {n: features.REGISTRY[n](eq) for n in
         ["mom_12_1", "mom_126", "mom_21", "reversal_5", "reversal_21", "low_vol_60"]}
H = 21

print(f"{'feature':<14} {'regime':<12} {'coverage':>9} {'IC':>8} {'t':>7} "
      f"{'breadth-adj':>12}")
rows = []
for fn, F in FEATS.items():
    # POINT-IN-TIME demeaning -- the lookahead that killed the insider signal
    d = research.dynamic_pit(F, method="expanding")
    fwd = research.forward_returns(eq, H)
    ic_all = research.cross_sectional_ic(d, fwd)
    st_all = research.ic_stats(ic_all, H)
    rows.append((fn, "UNCONDITIONAL", 1.0, st_all["ic_mean"], st_all["ic_t"], st_all["ic_t"]))
    for rn, mask in REGIMES.items():
        m = mask.reindex(ic_all.index).fillna(False)
        sub = ic_all[m]
        if len(sub) < 100:
            continue
        cov = float(m.mean())
        st = research.ic_stats(sub, H)
        # breadth penalty: fewer periods -> sqrt(coverage) haircut on comparable IR
        adj = st["ic_t"] * np.sqrt(cov)
        rows.append((fn, rn, cov, st["ic_mean"], st["ic_t"], adj))

for fn, rn, cov, ic, t, adj in rows:
    mark = "" if rn == "UNCONDITIONAL" else "  "
    print(f"{fn:<14} {mark}{rn:<12} {cov*100:>7.0f}% {ic:>8.4f} {t:>7.2f} {adj:>12.2f}")

n_tests = len([r for r in rows if r[1] != "UNCONDITIONAL"])
bar = research.bonferroni_t(n_tests)
print(f"\n  {n_tests} conditional tests -> Bonferroni bar |t| > {bar:.2f}")
surv = [r for r in rows if r[1] != "UNCONDITIONAL" and abs(r[4]) > bar]
print(f"  survivors on raw t: {len(surv)}")
for r in surv:
    print(f"    {r[0]} in {r[1]}: IC {r[3]:+.4f} t {r[4]:+.2f} "
          f"(breadth-adjusted {r[5]:+.2f})")
surv_adj = [r for r in surv if abs(r[5]) > bar]
print(f"  survivors AFTER breadth penalty: {len(surv_adj)}")
