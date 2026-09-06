"""Sector-adjust instead of market-adjust.

Measured rho (residual correlation among same-day insider buys) ran 0.25-0.90
AFTER market adjustment, and rose with horizon. That says insiders are buying
SECTOR drawdowns, not just market ones -- subtracting the index leaves the sector
move behind, so 'independent' events still share a common factor.

Subtracting the sector should cut rho, raise effective sample size, and -- if the
signal is genuinely idiosyncratic -- raise the corrected t-statistic.
"""
import numpy as np, pandas as pd
from algo import form4, universe as U, data, research

buys = form4.open_market_buys(form4.load())
buys = buys[buys.trans_date >= "2018-01-01"]
cons = U.fetch_constituents().set_index("ticker")
px = data.load_panel(sorted(buys.ticker.unique()), start="2017-06-01", end="2026-09-01")
px = px[[c for c in px.columns if px[c].notna().mean() > 0.90]]
spy = data.load_panel(["SPY"], start="2017-06-01", end="2026-09-01")["SPY"].reindex(px.index).ffill()
buys = buys[buys.ticker.isin(px.columns)]

# equal-weight sector index from the pool itself
sec_of = cons.sector.reindex(px.columns)
ret = px.pct_change()
SEC = {s: ret[[c for c in px.columns if sec_of.get(c) == s]].mean(axis=1)
       for s in sec_of.dropna().unique()}
SECPX = {s: (1 + r.fillna(0)).cumprod() for s, r in SEC.items()}


def collect(events, h, mode):
    fwd = px.shift(-h) / px - 1.0
    mkt = spy.shift(-h) / spy - 1.0
    rows = []
    for _, e in events.iterrows():
        t = e.ticker
        if t not in fwd.columns:
            continue
        # ANCHOR ON FILING DATE, NOT TRANSACTION DATE.
        # A Form 4 is not public until filed -- median 1 day, p90 4 days, p99 219
        # days after the trade. Measuring forward returns from trans_date counts
        # the pre-disclosure window, when only the insider knew. That window is
        # not tradeable, and it is where the entire h=2 "signal" lived.
        # +1 TRADING DAY after the filing date. EDGAR accepts submissions until
        # 22:00 ET, so a filing dated T is frequently not visible during T's
        # session. Entering at T+1 is the conservative, unambiguously tradeable
        # assumption.
        i = fwd.index.searchsorted(e.filing_date) + 1
        if i >= len(fwd):
            continue
        r = fwd[t].iloc[i]
        if pd.isna(r):
            continue
        if mode == "market":
            b = mkt.iloc[i]
        else:
            s = sec_of.get(t)
            if s is None or s not in SECPX:
                continue
            sp = SECPX[s]
            b = (sp.shift(-h) / sp - 1.0).iloc[i]
        if pd.isna(b):
            continue
        rows.append((fwd.index[i], r - b))
    return pd.DataFrame(rows, columns=["dt", "excess"])


def corrected_t(df, h):
    g = df.groupby(df.dt.dt.date).excess
    sizes = g.size(); multi = sizes[sizes >= 2].index
    sub = df[df.dt.dt.date.isin(multi)]
    if sub.empty:
        rho, mbar = 0.0, 1.0
    else:
        gm = sub.groupby(sub.dt.dt.date).excess
        tot = sub.excess.var(ddof=1)
        rho = float(np.clip(gm.mean().var(ddof=1) / tot, 0, 1)) if tot > 0 else 0.0
        mbar = float(sizes.mean())
    n_eff = len(df) / (1 + (mbar - 1) * rho) / h
    t = df.excess.mean() / (df.excess.std(ddof=1) / np.sqrt(max(n_eff, 1)))
    return rho, n_eff, t


print("ANCHORED ON FILING DATE (tradeable)\n")
print(f"{'adjustment':<12} {'events':<22} {'h':>3} {'excess%':>8} {'rho':>6} "
      f"{'n_eff':>7} {'t':>7}")
CASES = [("ALL buys", buys), ("buys >= $100k", buys[buys.value >= 1e5]),
         ("officers", buys[buys.relationship.fillna('').str.lower().str.contains('officer')])]
res = []
for mode in ("market", "sector"):
    for lab, ev in CASES:
        for h in (2, 5, 10, 21):
            df = collect(ev, h, mode)
            if len(df) < 50:
                continue
            rho, ne, t = corrected_t(df, h)
            res.append((mode, lab, h, t))
            print(f"{mode:<12} {lab:<22} {h:>3} {df.excess.mean()*100:>8.2f} "
                  f"{rho:>6.3f} {ne:>7.0f} {t:>7.2f}")
    print()

bar = research.bonferroni_t(len(res))
print(f"Bonferroni bar for {len(res)} tests: |t| > {bar:.2f}")
surv = [r for r in res if abs(r[3]) > bar]
print(f"survivors: {len(surv)}")
for mode, lab, h, t in sorted(surv, key=lambda x: -abs(x[3])):
    print(f"  {mode:<8} {lab:<22} h={h:<3} t={t:+.2f}")
