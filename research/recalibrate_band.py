"""Re-run the band calibration with the CORRECTED relative implementation.

The original 10% default was set on 10-24 name books using an ABSOLUTE band.
With small weights that band never triggers, so those results were partly
accidental and the default is untrustworthy. Redo it properly, at two universe
sizes, against calendar rebalancing.
"""
import numpy as np, pandas as pd, json
from algo import (backtest, data, form4, insider_state, metrics, strategies,
                  universe as U)
from algo.costs import CostModel

ALPACA = CostModel(name="Alpaca", commission_bps=0.0, spread_bps=1.5, slippage_bps=1.5)
f4 = form4.load(); tab = U.collection_table()
DIVS = list(U.DIVERSIFIERS)
trading30 = [t for t in json.load(open("data/trading_universe.json")) if t not in DIVS]

def stack(p, raw):
    iv = strategies.inverse_volatility(p, 60)
    c = raw*iv; tot = c.sum(axis=1)
    c = c.div(tot.where(tot>0), axis=0).fillna(0.0).mul(raw.sum(axis=1), axis=0)
    return strategies.volatility_target(c, p, target_vol=0.10, max_leverage=1.0)

def rel_band(w, b):
    cur = np.zeros(w.shape[1]); rows=[]
    for _, t in w.iterrows():
        t = t.to_numpy()
        mv = ((cur==0)&(t>0)) | (np.abs(t-cur) > b*np.maximum(t,1e-9))
        cur = np.where(mv, t, cur); rows.append(cur.copy())
    return pd.DataFrame(rows, index=w.index, columns=w.columns)

def calendar(w, n):
    mask = pd.Series(False, index=w.index); mask.iloc[::n] = True
    return w.where(mask).ffill().fillna(0.0)

def sig_w(F, px, top_frac=0.4):
    d = F.sub(F.mean(axis=0), axis=1)
    k = max(3, int(F.shape[1]*top_frac))
    sel = (d.rank(axis=1, ascending=False) <= k).astype(float)
    n = sel.sum(axis=1).replace(0, np.nan)
    w = sel.div(n, axis=0).fillna(0.0).reindex(columns=px.columns).fillna(0.0)
    for dv in [c for c in px.columns if c in DIVS]:
        w[dv] = 1.0/len(px.columns)
    return w.div(w.sum(axis=1), axis=0).fillna(0.0)

for lab, tk in [("trading 30", trading30+DIVS),
                ("collection 262", [t for t in tab.ticker if t not in DIVS]+DIVS)]:
    px = data.load_panel(tk, start="2018-01-01", end="2026-09-01")
    px = px[[c for c in px.columns if px[c].notna().mean()>0.95]].ffill().dropna()
    eq = [c for c in px.columns if c not in DIVS]
    F = insider_state.build(f4, px[eq], window_days=126)["ins_buy_share"]
    base_raw = strategies.equal_weight(px)
    sig_raw = sig_w(F, px)

    print(f"\n=== {lab} ({len(px.columns)} names) ===")
    print(f"{'rebalance rule':<22} {'NO-SIG Sh':>10} {'SIGNAL Sh':>10} {'edge':>7} "
          f"{'sig CAGR':>9} {'turn':>7}")
    rows=[]
    for b in [0.05, 0.10, 0.25, 0.50, 1.00]:
        r0 = backtest.run(px, rel_band(stack(px, base_raw), b), ALPACA)
        r1 = backtest.run(px, rel_band(stack(px, sig_raw), b), ALPACA)
        rows.append((f"band {int(b*100)}% relative", r0, r1))
    for n, nm in [(1,"daily"), (5,"weekly"), (21,"monthly"), (63,"quarterly")]:
        r0 = backtest.run(px, calendar(stack(px, base_raw), n), ALPACA)
        r1 = backtest.run(px, calendar(stack(px, sig_raw), n), ALPACA)
        rows.append((f"calendar {nm}", r0, r1))
    for nm, r0, r1 in rows:
        s0, s1 = metrics.sharpe(r0.returns), metrics.sharpe(r1.returns)
        print(f"{nm:<22} {s0:>10.2f} {s1:>10.2f} {s1-s0:>+7.2f} "
              f"{metrics.cagr(r1.equity)*100:>8.2f}% {r1.turnover.mean():>7.3f}")
