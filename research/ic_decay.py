"""Did the surviving features decay, or were they always this small?

Distinguishes two very different worlds:
  - decay      -> the effects were arbitraged away; recent data is what counts
                  and the 20-year t-stats are backward-looking artifacts
  - stationary -> the effects are stable and simply too small to trade under
                  our constraints, which is a structural verdict, not a timing one
"""
import numpy as np
import pandas as pd
from algo import data, features, features2 as f2, research

pd.set_option("display.width", 200)
uni = pd.read_parquet("data/collection_universe.parquet")
sc = [c for c in uni.columns if c.lower() in ("symbol", "ticker")][0]
close = data.load_panel(uni[sc].tolist(), start="2005-01-01", end="2026-09-01",
                        field="close", refresh=False)
close = close.dropna(axis=1, thresh=int(0.9 * len(close))).ffill(limit=5)
P = {f: data.load_panel(uni[sc].tolist(), start="2005-01-01", end="2026-09-01",
                        field=f, refresh=False)
     .reindex(index=close.index, columns=close.columns).ffill(limit=5)
     for f in ("open", "high", "low", "volume")}
c, o, h, l, v = close, P["open"], P["high"], P["low"], P["volume"]

FEATS = {
    "reversal_5":       features.reversal(c, 5),
    "mom_12_1":         features.momentum_12_1(c),
    "vol_shock":        f2.volume_shock(v),
    "close_in_range":   f2.close_position_in_range(h, l, c),
    "resid_mom_252":    f2.residual_momentum(c, 252),
    "mom_252":          features.momentum(c, 252),
    "overnight_vs_day": f2.intraday_vs_overnight(o, c),
}
fwd = research.forward_returns(c, 1)
PERIODS = [("2005-2009", "2005", "2009"), ("2010-2014", "2010", "2014"),
           ("2015-2019", "2015", "2019"), ("2020-2026", "2020", "2026")]

print("CROSS-SECTIONAL IC BY ERA  (h=1)\n")
print(f"{'feature':<18} " + " ".join(f"{p[0]:>11}" for p in PERIODS) + f"{'trend':>10}")
for name, f in FEATS.items():
    row, vals = [], []
    for lbl, a, b in PERIODS:
        m = (c.index >= f"{a}-01-01") & (c.index <= f"{b}-12-31")
        ic = research.cross_sectional_ic(f[m], fwd[m]).dropna()
        val = float(ic.mean()) if len(ic) > 100 else np.nan
        t = val / (ic.std(ddof=1) / np.sqrt(len(ic))) if len(ic) > 100 else np.nan
        vals.append(val)
        row.append(f"{val:>+7.4f}({t:>3.1f})" if np.isfinite(val) else f"{'--':>11}")
    good = [x for x in vals if np.isfinite(x)]
    slope = np.polyfit(range(len(good)), good, 1)[0] if len(good) > 2 else np.nan
    print(f"{name:<18} " + " ".join(row) + f"{slope:>+10.5f}")

print("\ntrend = OLS slope of IC across the four eras.")
print("strongly negative -> arbitraged away.  near zero -> stable but small.")
