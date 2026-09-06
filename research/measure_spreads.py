"""Measure actual spreads: S&P 500 vs S&P 600. The assumption under review.

The small-cap conclusion assumed 12bp spreads for the S&P 600. This measures
them instead, with two independent estimators, and recomputes the break-even IC
at whatever the data says.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from algo import data, spreads

pd.set_option("display.width", 200)


def load(syms, label):
    P = {}
    for f in ("high", "low", "close", "volume"):
        P[f] = data.load_panel(syms, start="2015-01-01", end="2026-09-01",
                               field=f, refresh=False)
    px = P["close"].dropna(axis=1, thresh=int(0.8 * len(P["close"])))
    for k in P:
        P[k] = P[k].reindex(columns=px.columns).ffill(limit=3)
    print(f"{label:<12} {px.shape[1]:>4} names, {len(px)} bars")
    return P


lc = pd.read_parquet("data/collection_universe.parquet")["ticker"].tolist()
sc = pd.read_parquet("data/sp600_constituents.parquet")["ticker"].tolist()
L, S = load(lc, "large cap"), load(sc, "small cap")

print("\n" + "=" * 84)
print("MEASURED EFFECTIVE SPREADS (bps, one way)")
print("=" * 84)
print(f"\n{'universe':<12} {'estimator':<18} {'p25':>8} {'median':>8} {'p75':>8} {'p90':>8}")
out = {}
for lbl, P in (("large cap", L), ("small cap", S)):
    s = spreads.summarize(P["high"], P["low"], P["close"])
    out[lbl] = s
    for est in ("corwin_schultz_bps", "abdi_ranaldo_bps"):
        v = s[est].dropna()
        print(f"{lbl:<12} {est.replace('_bps',''):<18} {v.quantile(.25):>8.1f} "
              f"{v.median():>8.1f} {v.quantile(.75):>8.1f} {v.quantile(.90):>8.1f}")

lc_med = out["large cap"][["corwin_schultz_bps", "abdi_ranaldo_bps"]].median().mean()
sc_med = out["small cap"][["corwin_schultz_bps", "abdi_ranaldo_bps"]].median().mean()
print(f"\n  large-cap median across both estimators: {lc_med:.1f}bp one way")
print(f"  small-cap median across both estimators: {sc_med:.1f}bp one way")
print(f"  ratio: {sc_med/lc_med:.2f}x")

print("\n  I ASSUMED: large cap 1.0bp spread, small cap 12.0bp -> a 12x ratio")
print(f"  MEASURED:  large cap {lc_med:.1f}bp,   small cap {sc_med:.1f}bp -> a "
      f"{sc_med/lc_med:.1f}x ratio")

# Liquidity-screened small caps: the ones we would actually trade.
dv = (S["close"] * S["volume"]).rolling(63).median().median()
liquid = dv[dv > 20e6].index
print(f"\n  small caps with >$20M/day median dollar volume: {len(liquid)} of "
      f"{S['close'].shape[1]}")
if len(liquid) > 20:
    ls = spreads.summarize(S["high"][liquid], S["low"][liquid], S["close"][liquid])
    lm = ls[["corwin_schultz_bps", "abdi_ranaldo_bps"]].median().mean()
    print(f"  their median spread: {lm:.1f}bp one way  "
          f"({lm/lc_med:.1f}x large cap)")
