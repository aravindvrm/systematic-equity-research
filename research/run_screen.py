"""Screen every candidate feature for predictive power."""
import pandas as pd
from algo import data, features, research

pd.set_option("display.width", 220)
px = data.load_panel(data.ETF_UNIVERSE, start="2010-01-01", end="2026-09-01").dropna(how="any")
print(f"{px.shape[1]} ETFs, {px.index[0].date()}..{px.index[-1].date()}, {len(px)} bars\n")

df = research.screen(px, horizons=(1, 5, 21))
n_feat = len(features.REGISTRY)
bar = research.bonferroni_t(n_feat * 3)   # features x horizons tested

real = df[df.kind == "real"].pivot(index="feature", columns="horizon",
                                   values=["ic", "ic_t"])
shuf = df[df.kind == "shuffled"].pivot(index="feature", columns="horizon", values="ic_t")

print("INFORMATION COEFFICIENT (cross-sectional rank IC)")
print("calibration: pro signals run IC 0.02-0.05; >0.10 on daily data is suspicious\n")
out = pd.DataFrame({
    "IC h=1":   real[("ic", 1)],   "t h=1":  real[("ic_t", 1)],
    "IC h=5":   real[("ic", 5)],   "t h=5":  real[("ic_t", 5)],
    "IC h=21":  real[("ic", 21)],  "t h=21": real[("ic_t", 21)],
}).sort_values("IC h=21", ascending=False)
print(out.round(3).to_string())

print(f"\n\nMULTIPLE-TESTING BAR: {n_feat} features x 3 horizons = {n_feat*3} tests")
print(f"  Bonferroni-corrected |t| threshold at alpha=0.05:  {bar:.2f}")
print(f"  (the naive t>2.0 bar would expect ~{n_feat*3*0.05:.1f} false positives)\n")

survivors = []
for f in out.index:
    for h in (1, 5, 21):
        t = out.loc[f, f"t h={h}"]
        if abs(t) > bar:
            survivors.append((f, h, out.loc[f, f"IC h={h}"], t))
if survivors:
    print("  SURVIVING features (|t| above corrected bar):")
    for f, h, ic, t in sorted(survivors, key=lambda x: -abs(x[3])):
        print(f"    {f:16s} h={h:<3} IC {ic:+.4f}  t {t:+.2f}")
else:
    print("  NOTHING survives the corrected bar.")

print("\n\nSHUFFLED CONTROL (same values, cross-section permuted -> signal destroyed)")
print("a real feature must clearly beat its shuffled twin\n")
cmp = pd.DataFrame({"real t h=21": real[("ic_t", 21)], "shuffled t h=21": shuf[21]})
print(cmp.round(2).sort_values("real t h=21", ascending=False).to_string())
