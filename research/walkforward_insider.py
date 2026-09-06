"""Walk-forward, and first: is the demeaning itself a lookahead?

`static_vs_dynamic` subtracts each asset's FULL-SAMPLE mean to strip the static
tilt. That is fine as a DIAGNOSTIC -- it answers "is there time variation here".
It is NOT fine as a tradeable signal: at time t you do not know the asset's
average buy-share over the following years.

Test three demeanings:
  full     -- full-sample mean            (lookahead; what the screen used)
  expanding-- mean of everything so far   (point-in-time correct)
  rolling  -- trailing 2-year mean        (point-in-time, adapts to regime)
"""
import numpy as np, pandas as pd, json
from algo import (backtest, data, form4, insider_state, metrics, research,
                  strategies, universe as U)
from algo.costs import CostModel

ALPACA = CostModel(name="Alpaca", commission_bps=0.0, spread_bps=1.5, slippage_bps=1.5)
f4 = form4.load(); tab = U.collection_table()
DIVS = list(U.DIVERSIFIERS)
tk = [t for t in tab.ticker if t not in DIVS] + DIVS
px = data.load_panel(tk, start="2018-01-01", end="2026-09-01")
px = px[[c for c in px.columns if px[c].notna().mean() > 0.95]].ffill().dropna()
eq = [c for c in px.columns if c not in DIVS]
F = insider_state.build(f4, px[eq], window_days=126)["ins_buy_share"]

DEMEAN = {
    "full-sample (LOOKAHEAD)": lambda X: X.sub(X.mean(axis=0), axis=1),
    "expanding (point-in-time)": lambda X: X - X.expanding(min_periods=252).mean(),
    "rolling 2y (point-in-time)": lambda X: X - X.rolling(504, min_periods=252).mean(),
}

print("1. DOES THE SIGNAL SURVIVE POINT-IN-TIME DEMEANING?\n")
print(f"{'demeaning':<28} {'IC h=21':>9} {'t':>7}")
for lab, fn in DEMEAN.items():
    d = fn(F)
    ic = research.cross_sectional_ic(d, research.forward_returns(px[eq], 21))
    st = research.ic_stats(ic, 21)
    print(f"{lab:<28} {st['ic_mean']:>9.4f} {st['ic_t']:>7.2f}")

def stack(p, raw):
    iv = strategies.inverse_volatility(p, 60)
    c = raw*iv; tot = c.sum(axis=1)
    c = c.div(tot.where(tot>0), axis=0).fillna(0.0).mul(raw.sum(axis=1), axis=0)
    return strategies.volatility_target(c, p, target_vol=0.10, max_leverage=1.0)

def rel_band(w, b=0.25):
    cur = np.zeros(w.shape[1]); rows=[]
    for _, t in w.iterrows():
        t = t.to_numpy()
        mv = ((cur==0)&(t>0)) | (np.abs(t-cur) > b*np.maximum(t,1e-9))
        cur = np.where(mv, t, cur); rows.append(cur.copy())
    return pd.DataFrame(rows, index=w.index, columns=w.columns)

def weights(d, top_frac=0.4):
    k = max(3, int(d.shape[1]*top_frac))
    sel = (d.rank(axis=1, ascending=False) <= k).astype(float)
    n = sel.sum(axis=1).replace(0, np.nan)
    w = sel.div(n, axis=0).fillna(0.0).reindex(columns=px.columns).fillna(0.0)
    for dv in [c for c in px.columns if c in DIVS]:
        w[dv] = 1.0/len(px.columns)
    return w.div(w.sum(axis=1), axis=0).fillna(0.0)

print("\n\n2. BACKTEST UNDER EACH DEMEANING (Alpaca costs, 25% band)\n")
base = backtest.run(px, rel_band(stack(px, strategies.equal_weight(px))), ALPACA)
print(f"{'demeaning':<28} {'Sharpe':>7} {'CAGR':>8} {'vs no-signal':>13}")
print(f"{'NO SIGNAL':<28} {metrics.sharpe(base.returns):>7.2f} "
      f"{metrics.cagr(base.equity)*100:>7.2f}% {'--':>13}")
res={}
for lab, fn in DEMEAN.items():
    r = backtest.run(px, rel_band(stack(px, weights(fn(F)))), ALPACA)
    res[lab]=r
    print(f"{lab:<28} {metrics.sharpe(r.returns):>7.2f} "
          f"{metrics.cagr(r.equity)*100:>7.2f}% "
          f"{metrics.sharpe(r.returns)-metrics.sharpe(base.returns):>+13.2f}")

print("\n\n3. WALK-FORWARD (expanding demeaning, disjoint test windows)\n")
d = DEMEAN["expanding (point-in-time)"](F)
w = rel_band(stack(px, weights(d)))
bw = rel_band(stack(px, strategies.equal_weight(px)))
n = len(px)
print(f"{'test window':<26} {'signal Sh':>10} {'no-sig Sh':>10} {'edge':>7}")
edges=[]
for frac in [0.40, 0.55, 0.70, 0.85]:
    a, b = int(n*frac), min(int(n*(frac+0.15)), n)
    sl = px.index[a:b]
    if len(sl) < 200: continue
    rs = backtest.run(px.loc[sl], w.loc[sl], ALPACA)
    rb = backtest.run(px.loc[sl], bw.loc[sl], ALPACA)
    e = metrics.sharpe(rs.returns)-metrics.sharpe(rb.returns); edges.append(e)
    print(f"{str(sl[0].date())+'..'+str(sl[-1].date()):<26} "
          f"{metrics.sharpe(rs.returns):>10.2f} {metrics.sharpe(rb.returns):>10.2f} {e:>+7.2f}")
print(f"\n  positive edge in {sum(1 for e in edges if e>0)}/{len(edges)} windows, "
      f"mean {np.mean(edges):+.2f}")
