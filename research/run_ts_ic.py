"""Time-series IC: does the portfolio's ACTUAL decision rule predict anything?"""
import pandas as pd
from algo import data, features, research

pd.set_option("display.width", 200)
px = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")

print("TIME-SERIES IC vs CROSS-SECTIONAL IC, per feature (h=1)\n")
print(f"{'feature':<16} {'x-sec IC':>10} {'time-series IC':>16} {'ts spread':>11}")
fwd = research.forward_returns(px, 1)
rows = []
for name in ["mom_12_1", "mom_252", "mom_126", "mom_63", "mom_21", "ma_dist_200", "low_vol_60"]:
    f = features.REGISTRY[name](px)
    xs = research.cross_sectional_ic(f, fwd).mean()
    ts = research.time_series_ic(f, fwd)
    print(f"{name:<16} {xs:>10.4f} {ts.mean():>16.4f} {ts.max()-ts.min():>11.4f}")

print("\n\nTHE ACTUAL RULE: hold when trailing return > 0, else cash")
print("(this is what portfolio.trend_sleeve does, per lookback)\n")
print(f"{'lookback':>10} {'% time on':>11} {'ann. ret ON':>13} {'ann. ret OFF':>14} {'spread':>9} {'t':>7}")
for lb in [21, 63, 126, 189, 252, 315]:
    f = features.momentum(px, lb)
    r = research.binary_signal_test(f, fwd)
    print(f"{lb:>10} {r['pct_on']*100:>10.1f}% {r['mean_on_ann']*100:>12.2f}%"
          f" {r['mean_off_ann']*100:>13.2f}% {r['spread_ann']*100:>8.2f}% {r['t_stat']:>7.2f}")

print("\n\nPER-ASSET time-series IC for mom_12_1 (is it universal or a few names?)\n")
ts = research.time_series_ic(features.momentum_12_1(px), fwd).sort_values(ascending=False)
for k, v in ts.items():
    bar = "#" * max(0, int(abs(v) * 400))
    print(f"  {k:6s} {v:+.4f}  {bar}")
