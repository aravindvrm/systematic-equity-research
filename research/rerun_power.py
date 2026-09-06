"""Re-run every price-feature battery on 2005-2026 and compare to 2018-2026.

WHY THIS IS NOT P-HACKING
-------------------------
Every feature and horizon here was specified BEFORE the long sample existed --
they are literally the same dicts from run_screen.py / run_screen2.py. No new
variants are introduced, and the test count is declared up front. The question
being asked is not "does anything work" (we already answered that on the short
sample) but "were the earlier nulls real, or merely underpowered?"

THE POWER TEST
--------------
If a feature has a true effect, tripling the sample scales its t-statistic by
sqrt(n_long / n_short) ~ 1.55. So for each cell we report:

    t_expected = t_short * sqrt(n_long / n_short)

A feature whose long-sample t lands near t_expected has a stable effect that was
simply too small to see. A feature whose long-sample t collapses toward zero was
noise. This is the same test that killed pair_spread_z.

SURVIVORSHIP BIAS -- READ THIS
------------------------------
The universe is TODAY's S&P 500 membership carried back to 2005, so it excludes
every name that was deleted from the index. That inflates long-only return
levels. It matters much less for cross-sectional rank IC (we rank survivors
against survivors), but it is not zero: names on their way out of the index are
exactly the distressed names where reversal and low-vol effects are strongest.
Read every number below as an upper bound.
"""
import json
import numpy as np
import pandas as pd

from algo import data, features, features2 as f2, research

pd.set_option("display.width", 240)

SHORT = ("2018-01-01", "2026-09-01")
LONG = ("2005-01-01", "2026-09-01")
HORIZONS = (1, 5, 21)

uni = pd.read_parquet("data/collection_universe.parquet")
sym_col = [c for c in uni.columns if c.lower() in ("symbol", "ticker")][0]
SYMS = uni[sym_col].tolist()


def panels(start, end):
    """OHLCV panels on a common index, names with >=90% coverage kept."""
    close = data.load_panel(SYMS, start=start, end=end, field="close", refresh=False)
    close = close.dropna(axis=1, thresh=int(0.9 * len(close))).ffill(limit=5)
    out = {"close": close}
    for f in ("open", "high", "low", "volume"):
        out[f] = (data.load_panel(SYMS, start=start, end=end, field=f, refresh=False)
                  .reindex(index=close.index, columns=close.columns).ffill(limit=5))
    return out


def build(p):
    """The two original batteries, verbatim in composition."""
    c, o, h, l, v = p["close"], p["open"], p["high"], p["low"], p["volume"]
    d = {name: fn(c) for name, fn in features.REGISTRY.items()}
    d.update({
        "vol_trend":        f2.volume_trend(v),
        "vol_shock":        f2.volume_shock(v),
        "px_vol_diverge":   f2.price_volume_divergence(c, v),
        "dollar_volume":    f2.dollar_volume(c, v),
        "garman_klass":     f2.garman_klass_vol(o, h, l, c),
        "close_in_range":   f2.close_position_in_range(h, l, c),
        "overnight_vs_day": f2.intraday_vs_overnight(o, c),
        "true_range":       f2.true_range_pct(h, l, c),
        "rel_strength_126": f2.relative_strength(c, 126),
        "rel_strength_252": f2.relative_strength(c, 252),
        "resid_mom_126":    f2.residual_momentum(c, 126),
        "resid_mom_252":    f2.residual_momentum(c, 252),
        "low_beta":         f2.beta_to_universe(c),
        "low_corr":         f2.correlation_to_universe(c),
        "dispersion":       f2.dispersion_regime(c),
        "lead_lag":         f2.lead_lag(c),
        "idio_vol_share":   f2.idio_vol_share(c),
    })
    return d


def ic_table(p, feats):
    c = p["close"]
    rows = []
    for name, f in feats.items():
        for hz in HORIZONS:
            ic = research.cross_sectional_ic(f, research.forward_returns(c, hz)).dropna()
            if len(ic) < 100:
                continue
            m = float(ic.mean())
            t = (m / (ic.std(ddof=1) / np.sqrt(len(ic)))) / np.sqrt(hz)
            rows.append(dict(feature=name, h=hz, ic=m, t=t, n=len(ic)))
    return pd.DataFrame(rows)


print("loading panels...", flush=True)
ps, pl = panels(*SHORT), panels(*LONG)
print(f"short: {ps['close'].shape[1]} names, {len(ps['close'])} bars, "
      f"{ps['close'].index[0].date()}..{ps['close'].index[-1].date()}")
print(f"long : {pl['close'].shape[1]} names, {len(pl['close'])} bars, "
      f"{pl['close'].index[0].date()}..{pl['close'].index[-1].date()}\n", flush=True)

fs, fl = build(ps), build(pl)
ts, tl = ic_table(ps, fs), ic_table(pl, fl)

m = ts.merge(tl, on=["feature", "h"], suffixes=("_s", "_l"))
m["scale"] = np.sqrt(m.n_l / m.n_s)
m["t_exp"] = m.t_s * m.scale
# How much of the expected t actually materialized. ~1.0 = stable effect that was
# underpowered before; ~0 = it was noise; negative = it reversed sign.
m["realized"] = m.t_l / m.t_exp.where(m.t_exp.abs() > 0.05)

K = len(m)
bar = research.bonferroni_t(K)
print("=" * 108)
print(f"{K} pre-specified tests (features x horizons)  ->  Bonferroni bar |t| > {bar:.2f}")
print("=" * 108)

print("\nFULL TABLE, sorted by |t| on the LONG sample\n")
show = m.reindex(m.t_l.abs().sort_values(ascending=False).index)
print(f"{'feature':<18} {'h':>3} {'IC short':>9} {'t short':>8} | "
      f"{'IC long':>9} {'t long':>8} {'t exp':>7} {'realized':>9}")
for _, r in show.iterrows():
    rz = "  n/a" if not np.isfinite(r.realized) else f"{r.realized:>9.2f}"
    print(f"{r.feature:<18} {int(r.h):>3} {r.ic_s:>9.4f} {r.t_s:>8.2f} | "
          f"{r.ic_l:>9.4f} {r.t_l:>8.2f} {r.t_exp:>7.2f} {rz}")

surv = show[show.t_l.abs() > bar]
print(f"\n{'-'*108}\nSURVIVORS on the long sample: {len(surv)}")
for _, r in surv.iterrows():
    print(f"   {r.feature:<18} h={int(r.h):<3} IC {r.ic_l:+.4f}  t {r.t_l:+.2f}  "
          f"(short-sample t was {r.t_s:+.2f})")

print(f"\n{'-'*108}\nPOWER VERDICT -- was anything merely underpowered?\n")
und = m[(m.t_s.abs() < 2.0) & (m.t_l.abs() > 2.5)]
print(f"  cells that were <2.0 on 7yr and >2.5 on 20yr: {len(und)}")
for _, r in und.iterrows():
    print(f"     {r.feature:<18} h={int(r.h):<3} t {r.t_s:+.2f} -> {r.t_l:+.2f}")
died = m[(m.t_s.abs() > 2.0) & (m.t_l.abs() < 1.0)]
print(f"\n  cells that were >2.0 on 7yr and collapsed below 1.0 on 20yr: {len(died)}")
for _, r in died.iterrows():
    print(f"     {r.feature:<18} h={int(r.h):<3} t {r.t_s:+.2f} -> {r.t_l:+.2f}")

print(f"\n  median realized fraction of expected t: {m.realized.median():.2f}")
print("  (near 1.0 = effects are stable and were underpowered; near 0 = they were noise)")
m.to_csv("rerun_power.csv", index=False)
print("\nwritten: rerun_power.csv")
