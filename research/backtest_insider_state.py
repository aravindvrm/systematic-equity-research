"""Does ins_buy_share survive a real backtest with costs?

The IC said yes. Every strategy this session has underperformed its implied IR
once long-only constraints, discrete rebalancing and costs bite. Benchmarked
against the VOL-TARGETED NO-SIGNAL book -- not plain equal weight, and not SPY.
"""
import numpy as np, pandas as pd, json
from algo import (backtest, data, form4, insider_state, metrics, research,
                  strategies, universe as U)
from algo.costs import CostModel

ALPACA = CostModel(name="Alpaca", commission_bps=0.0, spread_bps=1.5, slippage_bps=1.5)
f4 = form4.load(); tab = U.collection_table()
trading30 = [t for t in json.load(open("data/trading_universe.json"))
             if t not in U.DIVERSIFIERS]
DIVS = [d for d in U.DIVERSIFIERS]

def stack(p, raw):
    iv = strategies.inverse_volatility(p, 60)
    c = raw * iv; tot = c.sum(axis=1)
    c = c.div(tot.where(tot > 0), axis=0).fillna(0.0).mul(raw.sum(axis=1), axis=0)
    return strategies.volatility_target(c, p, target_vol=0.10, max_leverage=1.0)

def band(w, b=0.25):
    """Rebalance band RELATIVE to target weight, not absolute.

    An absolute 10pp band is meaningless once weights are small: across 247 names
    each target is ~0.004, so every position sits permanently inside the band and
    NOTHING EVER OPENS -- the book stays in cash while the metrics quietly report
    on a rounding error (0.18% vol, 0.000 turnover).

    Relative banding scales with position size, so the same parameter behaves
    sensibly at 10 names or 500. Cold start (cur==0, target>0) always trades,
    otherwise the portfolio can never be built.
    """
    cur = np.zeros(w.shape[1]); rows = []
    for _, t in w.iterrows():
        t = t.to_numpy()
        opening = (cur == 0) & (t > 0)
        drift = np.abs(t - cur) > b * np.maximum(t, 1e-9)
        mv = opening | drift
        cur = np.where(mv, t, cur); rows.append(cur.copy())
    return pd.DataFrame(rows, index=w.index, columns=w.columns)

def signal_weights(F, top_frac=0.4):
    """Long the top fraction by demeaned buy-share."""
    d = F.sub(F.mean(axis=0), axis=1)
    n_top = max(3, int(F.shape[1] * top_frac))
    sel = (d.rank(axis=1, ascending=False) <= n_top).astype(float)
    n = sel.sum(axis=1).replace(0, np.nan)
    return sel.div(n, axis=0).fillna(0.0)

print(f"{'universe':<22} {'names':>6} {'strategy':<18} {'Sharpe':>7} {'CAGR':>7} "
      f"{'MaxDD':>7} {'turn':>6}")
UNIVERSES = [
    ("trading 30", trading30 + DIVS),
    ("collection 262", [t for t in tab.ticker if t not in U.DIVERSIFIERS] + DIVS),
]
results = {}
for lab, tk in UNIVERSES:
    px = data.load_panel(tk, start="2018-01-01", end="2026-09-01")
    px = px[[c for c in px.columns if px[c].notna().mean() > 0.95]].ffill().dropna()
    eq = [c for c in px.columns if c not in U.DIVERSIFIERS]
    F = insider_state.build(f4, px[eq], window_days=126)["ins_buy_share"]

    # no-signal baseline: equal weight, same risk stack, same band
    base_raw = strategies.equal_weight(px)
    base = backtest.run(px, band(stack(px, base_raw)), ALPACA)

    sig_raw = signal_weights(F).reindex(columns=px.columns).fillna(0.0)
    # keep the diversifier sleeve at its equal-weight allocation
    for d in [c for c in px.columns if c in U.DIVERSIFIERS]:
        sig_raw[d] = 1.0 / len(px.columns)
    sig_raw = sig_raw.div(sig_raw.sum(axis=1), axis=0).fillna(0.0)
    sig = backtest.run(px, band(stack(px, sig_raw)), ALPACA)
    results[lab] = (sig, base)

    for nm, r in [("NO SIGNAL (base)", base), ("ins_buy_share", sig)]:
        print(f"{lab:<22} {len(px.columns):>6} {nm:<18} {metrics.sharpe(r.returns):>7.2f} "
              f"{metrics.cagr(r.equity)*100:>6.2f}% {metrics.max_drawdown(r.equity)*100:>6.1f}% "
              f"{r.turnover.mean():>6.3f}")
    print()

print("=" * 74)
print("DIAGNOSTICS — collection 262, benchmarked vs the NO-SIGNAL book")
print("=" * 74)
sig, base = results["collection 262"]
print(sig.report(benchmark=base, benchmark_name="no-signal", n_trials=60))
