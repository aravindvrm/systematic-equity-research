"""PEAD: does the earnings-day reaction predict subsequent drift?

PRE-SPECIFIED: 2 measures (car, sue) x 3 horizons (5, 21, 63) = 6 tests.
Sign is POSITIVE -- a positive surprise should keep drifting up.
Everything is tradeable from day +2 (see algo/pead.py on announcement timing).
"""
import numpy as np
import pandas as pd
from scipy import stats as sst

from algo import data, pead, research

pd.set_option("display.width", 200)

E = pd.read_parquet("data/edgar/earnings_dates.parquet")
c = data.load_panel(sorted(E.ticker.unique()), start="2005-01-01", end="2026-09-01",
                    field="close", refresh=False)
c = c.dropna(axis=1, thresh=int(0.9 * len(c))).ffill(limit=5)
E = E[E.ticker.isin(c.columns) & (E.filing_date >= "2005-01-01")]
print(f"{c.shape[1]} names, {len(c)} bars, {len(E):,} announcements\n")

R = pead.reactions(c, E)
R.to_parquet("data/edgar/pead_reactions.parquet")
print(f"{len(R):,} usable events ({len(R)/len(E)*100:.0f}% of announcements)")
print(f"  CAR[0,+1]  mean {R.car.mean()*100:+.3f}%  sd {R.car.std()*100:.2f}%  "
      f"|CAR| median {R.car.abs().median()*100:.2f}%")
print(f"  SUE        mean {R.sue.mean():+.3f}  sd {R.sue.std():.2f}")
print(f"  events/name/yr: {len(R)/c.shape[1]/(len(c)/252):.1f}")

HZ = (5, 21, 63)
MEAS = ("car", "sue")
K = len(MEAS) * len(HZ)
bar = sst.norm.ppf(1 - 0.05 / (2 * K))
print(f"\n{'='*74}\n{K} pre-specified tests -> Bonferroni bar |t| > {bar:.2f}\n{'='*74}\n")
print(f"{'measure':<10} {'h':>4} {'IC':>9} {'t':>8} {'n days':>8} {'coverage':>9}")

rows, panels = [], {}
for meas in MEAS:
    f = pead.to_daily(R, c.index, c.columns, col=meas, hold_days=63)
    panels[meas] = f
    cov = f.notna().mean(axis=1).mean()
    for hz in HZ:
        ic = research.cross_sectional_ic(f, research.forward_returns(c, hz)).dropna()
        if len(ic) < 100:
            continue
        m = float(ic.mean())
        t = (m / (ic.std(ddof=1) / np.sqrt(len(ic)))) / np.sqrt(hz)
        rows.append((meas, hz, m, t))
        print(f"{meas:<10} {hz:>4} {m:>9.4f} {t:>8.2f} {len(ic):>8} {cov*100:>8.0f}%")

df = pd.DataFrame(rows, columns=["measure", "h", "ic", "t"])
surv = df[df.t.abs() > bar]
print(f"\nsurvivors: {len(surv)}")
for _, r in surv.iterrows():
    print(f"   {r.measure:<10} h={int(r.h):<4} IC {r.ic:+.4f}  t {r.t:+.2f}")
print(f"\nlargest |t|: {df.t.abs().max():.2f}  "
      f"(null expectation ~{sst.norm.ppf(1-1/(2*K)):.2f})")
print(f"largest |IC|: {df.ic.abs().max():.4f}")

print(f"\n{'-'*74}\nEVENT STUDY -- mean abnormal return by SUE quintile, days +2..+63\n{'-'*74}")
ar = pead.abnormal_returns(c)
q = R.copy().dropna(subset=["sue"])
q["quintile"] = pd.qcut(q.sue, 5, labels=[1, 2, 3, 4, 5])
idx = c.index
print(f"{'quintile':>9} {'n':>7} {'mean SUE':>10} {'CAR +2..+21':>13} {'CAR +2..+63':>13}")
for lab, g in q.groupby("quintile", observed=True):
    d21, d63 = [], []
    for e in g.itertuples():
        p = idx.searchsorted(e.event_date)
        s = ar[e.ticker]
        if p + 63 < len(idx):
            d21.append(s.iloc[p + 2:p + 21].sum())
            d63.append(s.iloc[p + 2:p + 63].sum())
    print(f"{str(lab):>9} {len(g):>7} {g.sue.mean():>10.2f} "
          f"{np.mean(d21)*100:>12.2f}% {np.mean(d63)*100:>12.2f}%")
df.to_csv("pead_test.csv", index=False)
