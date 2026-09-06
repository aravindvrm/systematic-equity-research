"""Screen batch two: volume, range, and RELATIONAL features."""
import numpy as np, pandas as pd
from algo import data, features, features2 as f2, research

pd.set_option("display.width", 220)
U = data.ETF_UNIVERSE
S, E = "2010-01-01", "2026-09-01"
close = data.load_panel(U, start=S, end=E, field="close").dropna(how="any")
op = data.load_panel(U, start=S, end=E, field="open").reindex(close.index)
hi = data.load_panel(U, start=S, end=E, field="high").reindex(close.index)
lo = data.load_panel(U, start=S, end=E, field="low").reindex(close.index)
vol = data.load_panel(U, start=S, end=E, field="volume").reindex(close.index)
print(f"{close.shape[1]} ETFs, {len(close)} bars, OHLCV loaded\n")

BATCH2 = {
    # volume
    "vol_trend":        f2.volume_trend(vol),
    "vol_shock":        f2.volume_shock(vol),
    "px_vol_diverge":   f2.price_volume_divergence(close, vol),
    "dollar_volume":    f2.dollar_volume(close, vol),
    # range / intraday
    "garman_klass":     f2.garman_klass_vol(op, hi, lo, close),
    "close_in_range":   f2.close_position_in_range(hi, lo, close),
    "overnight_vs_day": f2.intraday_vs_overnight(op, close),
    "true_range":       f2.true_range_pct(hi, lo, close),
    # relational
    "rel_strength_126": f2.relative_strength(close, 126),
    "rel_strength_252": f2.relative_strength(close, 252),
    "resid_mom_126":    f2.residual_momentum(close, 126),
    "resid_mom_252":    f2.residual_momentum(close, 252),
    "low_beta":         f2.beta_to_universe(close),
    "low_corr":         f2.correlation_to_universe(close),
    "dispersion":       f2.dispersion_regime(close),
    "lead_lag":         f2.lead_lag(close),
    "idio_vol_share":   f2.idio_vol_share(close),
}
# carry the batch-1 winner as a reference point
BATCH2["[ref] mom_12_1"] = features.momentum_12_1(close)

df = research.screen(close, horizons=(1, 5, 21), feature_dict=BATCH2)
n_tests = (len(BATCH2) - 1) * 3
bar = research.bonferroni_t(n_tests)

real = df[df.kind == "real"].pivot(index="feature", columns="horizon", values=["ic", "ic_t"])
fwd1 = research.forward_returns(close, 1)

rows = []
for f in real.index:
    ts = research.time_series_ic(BATCH2[f], fwd1).mean()
    rows.append(dict(feature=f, ic_xs=real.loc[f, ("ic", 1)], t_xs=real.loc[f, ("ic_t", 1)],
                     ic_xs_21=real.loc[f, ("ic", 21)], t_xs_21=real.loc[f, ("ic_t", 21)],
                     ic_ts=ts))
out = pd.DataFrame(rows).set_index("feature").sort_values("t_xs", key=abs, ascending=False)
print("BATCH 2: cross-sectional IC (h=1, h=21) and time-series IC (h=1)\n")
print(out.round(4).to_string())
print(f"\n\nBonferroni bar for {n_tests} tests: |t| > {bar:.2f}\n")
surv = out[(out.t_xs.abs() > bar) | (out.t_xs_21.abs() > bar)]
if len(surv):
    print("  SURVIVORS:")
    for f, r in surv.iterrows():
        print(f"    {f:20s} xsec IC {r.ic_xs:+.4f} (t {r.t_xs:+.2f})   ts IC {r.ic_ts:+.4f}")
else:
    print("  nothing survives")
