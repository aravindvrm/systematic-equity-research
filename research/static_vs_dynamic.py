"""Is a 'signal' actually time-varying, or just a static ranking in disguise?

A feature that barely changes its cross-sectional ORDER over time is not
predicting anything -- it is a fixed bet. Over 2010-2026 a fixed bet on the big
equity ETFs looks fantastic, and cross-sectional IC cannot tell the difference,
because every day's observation is nearly the same one repeated. The t-stat
inflates accordingly.

Test: de-mean each feature by its own per-asset time average. That removes the
static ranking and leaves only time variation. If IC survives, the feature is a
real signal. If it collapses, it was a static tilt.
"""
import numpy as np, pandas as pd
from algo import data, features, features2 as f2, research

U, S, E = data.ETF_UNIVERSE, "2010-01-01", "2026-09-01"
close = data.load_panel(U, start=S, end=E, field="close").dropna(how="any")
op = data.load_panel(U, start=S, end=E, field="open").reindex(close.index)
hi = data.load_panel(U, start=S, end=E, field="high").reindex(close.index)
lo = data.load_panel(U, start=S, end=E, field="low").reindex(close.index)
vol = data.load_panel(U, start=S, end=E, field="volume").reindex(close.index)

CANDIDATES = {
    "dollar_volume":  f2.dollar_volume(close, vol),
    "idio_vol_share": f2.idio_vol_share(close),
    "low_corr":       f2.correlation_to_universe(close),
    "garman_klass":   f2.garman_klass_vol(op, hi, lo, close),
    "low_beta":       f2.beta_to_universe(close),
    "rel_str_252":    f2.relative_strength(close, 252),
    "mom_12_1":       features.momentum_12_1(close),
    "resid_mom_252":  f2.residual_momentum(close, 252),
    "reversal_5":     features.reversal(close, 5),
}

fwd = research.forward_returns(close, 1)
print("STATIC vs TIME-VARYING DECOMPOSITION\n")
print("rank stability = avg correlation of today's cross-sectional ranking with")
print("its own ranking 1 year ago. Near 1.0 means the feature never reorders.\n")
print(f"{'feature':<16} {'rank stab':>10} {'IC raw':>9} {'t raw':>8} "
      f"{'IC demeaned':>12} {'t demeaned':>11} {'verdict':>10}")
for name, f in CANDIDATES.items():
    ranks = f.rank(axis=1)
    lagged = ranks.shift(252)
    stab = ranks.corrwith(lagged, axis=1).mean()

    raw = research.ic_stats(research.cross_sectional_ic(f, fwd), 1)
    # remove each asset's own long-run average level -> only time variation left
    dyn = f.sub(f.mean(axis=0), axis=1)
    dm = research.ic_stats(research.cross_sectional_ic(dyn, fwd), 1)

    keep = abs(dm["ic_t"]) > 2.0 and abs(dm["ic_mean"]) > 0.5 * abs(raw["ic_mean"])
    verdict = "REAL" if keep else "static"
    print(f"{name:<16} {stab:>10.2f} {raw['ic_mean']:>9.4f} {raw['ic_t']:>8.2f} "
          f"{dm['ic_mean']:>12.4f} {dm['ic_t']:>11.2f} {verdict:>10}")

print("\n\nWHAT THE STATIC ONES ARE ACTUALLY BETTING ON")
print("(average cross-sectional rank, 1 = most attractive per the feature)\n")
for name in ["dollar_volume", "idio_vol_share", "low_corr"]:
    r = CANDIDATES[name].rank(axis=1, ascending=False).mean().sort_values()
    print(f"  {name:<16} top 3: {', '.join(r.index[:3])}   bottom 3: {', '.join(r.index[-3:])}")
