"""Screen state-shaped insider features with the full apparatus."""
import numpy as np, pandas as pd
from algo import form4, insider_state, universe as U, data, research

f4 = form4.load()
tab = U.collection_table()
tk = [t for t in tab.ticker if t not in U.DIVERSIFIERS]
px = data.load_panel(tk, start="2018-01-01", end="2026-09-01")
px = px[[c for c in px.columns if px[c].notna().mean() > 0.95]].ffill().dropna()
print(f"{px.shape[1]} names, {len(px)} bars, {px.index[0].date()}..{px.index[-1].date()}")

feats = insider_state.build(f4, px, window_days=126)
cov = {k: float((v != 0).mean().mean()) for k, v in feats.items()}
print(f"feature coverage (non-zero fraction): "
      f"{ {k: round(c,3) for k,c in cov.items()} }\n")

fwd1 = research.forward_returns(px, 1)
rows = []
for name, F in feats.items():
    for h in (5, 21, 63):
        s = research.static_vs_dynamic(px, F, horizon=h)
        rows.append(dict(feature=name, h=h, ic_raw=s["ic_raw"], t_raw=s["t_raw"],
                         ic_dyn=s["ic_dynamic"], t_dyn=s["t_dynamic"],
                         stability=s["rank_stability"]))
df = pd.DataFrame(rows)
bar = research.bonferroni_t(len(df))
print("STATE-SHAPED INSIDER FEATURES  (dynamic = static tilt removed)\n")
print(f"{'feature':<18} {'h':>3} {'stab':>6} {'IC raw':>8} {'t raw':>7} "
      f"{'IC dyn':>8} {'t dyn':>7}")
for _, r in df.iterrows():
    flag = "  <<<" if abs(r.t_dyn) > bar else ""
    print(f"{r.feature:<18} {int(r.h):>3} {r.stability:>6.2f} {r.ic_raw:>8.4f} "
          f"{r.t_raw:>7.2f} {r.ic_dyn:>8.4f} {r.t_dyn:>7.2f}{flag}")
print(f"\nBonferroni bar for {len(df)} tests: |t| > {bar:.2f}")
surv = df[df.t_dyn.abs() > bar]
print(f"survivors: {len(surv)} of {len(df)}")
for _, r in surv.iterrows():
    print(f"  {r.feature} h={int(r.h)}  dynamic IC {r.ic_dyn:+.4f}  t {r.t_dyn:+.2f}")
