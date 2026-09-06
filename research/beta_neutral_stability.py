"""Is variant B's residual stable, or concentrated in one regime?

The RF's edge is documented as a low-volatility inversion that held over
2009-2026 and carries the opposite sign in the long-run literature. Removing
the beta tilt may not remove the regime bet. If the residual is real it should
appear in both halves; if it is a regime artifact it will be concentrated.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from research.beta_neutral import (_prelude, load_rf_predictions, rolling_betas,
                                   beta_orthogonal_neutral, beta_neutral,
                                   book_returns, realised_beta)
from algo import factors, metrics

ns = _prelude(); cl = ns["cl"]
r1 = cl.pct_change().fillna(0.0)
ff = factors.load(); mkt = ff["Mkt-RF"].reindex(cl.index).fillna(0.0)
rf = load_rf_predictions(cl, ns)
betas = rolling_betas(r1, mkt)

for label, builder in (("beta-neutral (A)", beta_neutral),
                       ("beta-orthogonal (B)", beta_orthogonal_neutral)):
    w = builder(rf, r1, betas)
    r, _ = book_returns(w, r1)
    r = r[cl.index >= pd.Timestamp("2015-01-01")]
    print(f"\n{'='*74}\n{label}\n{'='*74}")
    print(f"{'window':<16}{'n':>6}{'Sharpe':>9}{'alpha_ann':>11}{'alpha_t':>9}{'beta':>8}")
    halves = [("2015-2020", "2015-01-01", "2020-12-31"),
              ("2021-2026", "2021-01-01", "2026-12-31"),
              ("full", "2015-01-01", "2026-12-31")]
    for nm, a, b in halves:
        seg = r[(r.index >= a) & (r.index <= b)]
        if len(seg) < 250:
            print(f"{nm:<16}{len(seg):>6}  too short"); continue
        att = factors.attribution(seg, ff)
        print(f"{nm:<16}{len(seg):>6}{metrics.sharpe(seg):>9.3f}"
              f"{att.get('alpha_ann',np.nan)*100:>10.2f}%{att.get('alpha_t',np.nan):>9.2f}"
              f"{realised_beta(seg, ff['Mkt-RF']):>8.3f}")
    yr = r.groupby(r.index.year).apply(lambda s: metrics.sharpe(s))
    print("  by year:", "  ".join(f"{y}:{v:+.2f}" for y, v in yr.items()))
    print(f"  positive years: {(yr>0).sum()}/{len(yr)}")
