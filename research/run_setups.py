"""Do quantifiable setups work? Event study over z-score thresholds."""
import numpy as np, pandas as pd
from algo import data, research

pd.set_option("display.width", 220)
BLUE = ["MSFT","JPM","JNJ","XOM","PG","HD","CAT","NEE","LIN","VZ","AMT",
        "UNH","WMT","CVX","KO","MCD","HON","TXN","DE","GS"]
DIV = ["TLT","IEF","GLD","DBC"]
px = data.load_panel(BLUE + DIV, start="2012-01-01", end="2026-09-01").dropna(how="any")
print(f"{px.shape[1]} names, {len(px)} bars, {px.index[0].date()}..{px.index[-1].date()}\n")

LOOKBACKS = [1, 5, 21]
THRESHOLDS = [-3.0, -2.5, -2.0, -1.5, 1.5, 2.0, 2.5, 3.0]
HORIZONS = [1, 5, 21]
n_tests = len(LOOKBACKS) * len(THRESHOLDS) * len(HORIZONS)
bar = research.bonferroni_t(n_tests)

print(f"testing {n_tests} setups; Bonferroni |t| bar = {bar:.2f}\n")
print(f"{'setup':<22} {'events':>8} {'h':>4} {'edge/period':>12} {'ann. edge':>11} "
      f"{'win%':>7} {'t':>7}")
rows = []
for lb in LOOKBACKS:
    z = research.zscore(px, lookback=lb)
    for th in THRESHOLDS:
        cond = (z <= th) if th < 0 else (z >= th)
        es = research.event_study(px, cond, HORIZONS)
        for _, r in es.iterrows():
            if np.isnan(r["t"]):
                continue
            name = f"{lb}d move {'<=' if th<0 else '>='} {th:+.1f}sd"
            rows.append(dict(setup=name, lb=lb, th=th, **r.to_dict()))
            flag = "  <<<" if abs(r["t"]) > bar else ""
            if abs(r["t"]) > 2.0:
                print(f"{name:<22} {int(r['n_events']):>8} {int(r['horizon']):>4} "
                      f"{r['edge']*100:>11.3f}% {r['ann_edge']*100:>10.1f}% "
                      f"{r['win']*100:>6.1f}% {r['t']:>7.2f}{flag}")

df = pd.DataFrame(rows)
print(f"\n  (only |t| > 2.0 shown; <<< marks survivors of the corrected bar)\n")
surv = df[df.t.abs() > bar]
print(f"  {len(surv)} of {len(df)} setups clear |t| > {bar:.2f}")
if len(surv):
    print("\n  SURVIVORS, ranked by annualized edge:")
    for _, r in surv.reindex(surv.ann_edge.abs().sort_values(ascending=False).index).iterrows():
        direction = "BUY (reversion)" if r["edge"] > 0 else "AVOID/FADE"
        print(f"    {r['setup']:<22} h={int(r['horizon']):<3} "
              f"ann edge {r['ann_edge']*100:+7.1f}%  n={int(r['n_events']):<6} "
              f"t={r['t']:+.2f}  {direction}")
df.to_csv("setup_results.csv", index=False)
