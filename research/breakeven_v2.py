"""Small-cap break-even IC, recomputed with MEASURED cost ratios.

WHAT CHANGED
------------
The original computation assumed small-cap spreads of 12bp against large-cap
1.0bp -- a 12x penalty -- and concluded small caps required IC 0.091 versus
0.036, i.e. 2.5x harder. That assumption was never measured.

Corwin-Schultz and Abdi-Ranaldo, run on the actual OHLC of both universes,
agree the ratio is ~2.0x (1.6x for names above $20M/day dollar volume), not 12x.

The estimators' LEVELS are unreliable -- both are biased upward and disagree by
about 2x with each other -- so this anchors the large-cap cost at the known IBKR
schedule and scales small caps by the MEASURED RATIO. That uses the estimators
for the thing they are good at (relative comparison) and not for the thing they
are bad at (absolute level).
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from algo import backtest, data, metrics, research, strategies
from algo.costs import CostModel, IBKR_US_EQUITY

pd.set_option("display.width", 200)
SPLIT = "2016-01-01"

# Large cap: the known IBKR schedule. Small cap: same, scaled by the MEASURED
# ratio. A 3x variant is carried as a pessimistic bound since the estimators are
# noisy and illiquid names sit in the tail.
SC_MEASURED = CostModel("small-cap 2.0x measured", 0.7, 2.0, 1.0, 0.35)
SC_LIQUID   = CostModel("small-cap 1.6x (liquid)", 0.7, 1.6, 0.8, 0.35)
SC_PESSIM   = CostModel("small-cap 3.0x (bound)",  0.7, 3.0, 1.5, 0.35)
SC_OLD      = CostModel("small-cap 12bp (ASSUMED)", 0.7, 12.0, 6.0, 0.35)


def load(path_or_syms, label, min_dv=None):
    syms = (pd.read_parquet(path_or_syms)["ticker"].tolist()
            if isinstance(path_or_syms, str) else path_or_syms)
    px = data.load_panel(syms, start="2005-01-01", end="2026-09-01",
                         field="close", refresh=False)
    px = px.dropna(axis=1, thresh=int(0.9 * len(px))).ffill(limit=5)
    if min_dv:
        vol = data.load_panel(syms, start="2005-01-01", end="2026-09-01",
                              field="volume", refresh=False).reindex(
                                  index=px.index, columns=px.columns)
        dv = (px * vol).rolling(63).median().median()
        px = px[dv[dv > min_dv].index]
    print(f"{label:<26} {px.shape[1]:>4} names, {len(px)} bars")
    return px


def stack(score, prices, top_frac=0.3, rebal=21):
    s = score.copy()
    m = np.zeros(len(s), dtype=bool); m[::rebal] = True
    s = s.where(pd.Series(m, index=s.index), np.nan).ffill()
    raw = (s.rank(axis=1, pct=True, ascending=False) <= top_frac).astype(float)
    w = raw * strategies.inverse_volatility(prices, 60)
    tot = w.sum(axis=1)
    w = w.div(tot.where(tot > 0), axis=0).fillna(0.0)
    return strategies.volatility_target(w, prices, target_vol=0.10, max_leverage=1.0)


def synth(prices, ic, seed=11):
    rng = np.random.default_rng(seed)
    fut = prices.pct_change().shift(-1)
    z = fut.sub(fut.mean(axis=1), axis=0).div(fut.std(axis=1).replace(0, np.nan), axis=0)
    n = pd.DataFrame(rng.normal(size=z.shape), index=z.index, columns=z.columns)
    return ic * z + np.sqrt(1 - ic ** 2) * n


def breakeven(px, cost):
    test = px.index >= SPLIT
    pt = px[test]
    flat = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    b = backtest.run(pt, stack(flat, px, 1.0, 21)[test], cost_model=cost).summary()
    grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20]
    cur = [backtest.run(pt, stack(synth(px, ic), px)[test],
                        cost_model=cost).summary()["sharpe"] for ic in grid]
    be = np.nan
    for i in range(len(grid) - 1):
        if (cur[i] - b["sharpe"]) * (cur[i + 1] - b["sharpe"]) < 0:
            be = grid[i] + (b["sharpe"] - cur[i]) / (cur[i + 1] - cur[i]) * (grid[i + 1] - grid[i])
            break
    return b, be


lc_syms = pd.read_parquet("data/collection_universe.parquet")["ticker"].tolist()
lc = load(lc_syms, "large cap")
sc = load("data/sp600_constituents.parquet", "small cap (all)")
scl = load("data/sp600_constituents.parquet", "small cap (>$20M/day)", min_dv=20e6)

print("\nBREADTH")
for lbl, p in (("large cap", lc), ("small cap", sc), ("small cap liquid", scl)):
    eb = research.effective_bets(p.pct_change().dropna(how="all").dropna(axis=1))
    print(f"  {lbl:<20} {p.shape[1]:>4} names -> {eb:5.2f} effective bets")

print(f"\n{'='*94}")
print("BREAK-EVEN IC, monthly rebalance, out of sample 2016+")
print(f"{'='*94}\n")
print(f"{'universe':<26} {'cost model':<26} {'RT bp':>7} {'bench Sh':>9} {'break-even IC':>14}")
for lbl, px, cost in [
        ("large cap", lc, IBKR_US_EQUITY),
        ("small cap (all)", sc, SC_OLD),
        ("small cap (all)", sc, SC_MEASURED),
        ("small cap (all)", sc, SC_PESSIM),
        ("small cap (>$20M/day)", scl, SC_LIQUID),
        ("small cap (>$20M/day)", scl, SC_PESSIM)]:
    b, be = breakeven(px, cost)
    s = f"{be:.4f}" if np.isfinite(be) else ">0.20"
    print(f"{lbl:<26} {cost.name:<26} {cost.round_trip_bps:>7.0f} "
          f"{b['sharpe']:>9.2f} {s:>14}")

print("\n" + "=" * 94)
print("The 'small-cap 12bp (ASSUMED)' row is what the original conclusion rested on.")
print("Compare it to the measured rows.")
