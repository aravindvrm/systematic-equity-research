"""What IC would a small-cap strategy need to beat its own benchmark?

The 0.05 figure from the large-cap work does NOT transplant. It was derived
against that universe's benchmark (vol-targeted equal-weight S&P 500 at Sharpe
1.07), its correlation structure, and its ~2.5bp costs. A small-cap universe
differs on all three, and they push in OPPOSITE directions:

    more names, lower correlation  -> more effective bets -> LOWER bar
    weaker/more volatile benchmark -> LOWER bar
    much wider spreads             -> HIGHER bar

So the answer has to be computed, not assumed. Method is identical to the
large-cap version: inject synthetic forecasts of KNOWN IC through the same risk
stack and find where the curve crosses the benchmark.

Costs are bracketed rather than guessed at a single value, because small-cap
spreads are the single biggest uncertainty here and the answer is sensitive to
them.
"""
import numpy as np
import pandas as pd

from algo import backtest, data, metrics, research, strategies
from algo.costs import CostModel, IBKR_US_EQUITY

pd.set_option("display.width", 200)

# Retail small-cap execution. The binding term is the SPREAD, not commission:
# a $1-2bn name quotes far wider than a mega-cap, and a marketable order pays it.
SC_MID = CostModel(name="small-cap (12bp spread)", commission_bps=0.7,
                   spread_bps=12.0, slippage_bps=6.0, min_commission=0.35)
SC_OPTIMISTIC = CostModel(name="small-cap (6bp spread)", commission_bps=0.7,
                          spread_bps=6.0, slippage_bps=3.0, min_commission=0.35)
SC_PESSIMISTIC = CostModel(name="small-cap (25bp spread)", commission_bps=0.7,
                           spread_bps=25.0, slippage_bps=12.0, min_commission=0.35)

SPLIT = "2016-01-01"


def load(path_or_syms, label):
    syms = (pd.read_parquet(path_or_syms)["ticker"].tolist()
            if isinstance(path_or_syms, str) else path_or_syms)
    px = data.load_panel(syms, start="2005-01-01", end="2026-09-01",
                         field="close", refresh=False)
    px = px.dropna(axis=1, thresh=int(0.9 * len(px))).ffill(limit=5)
    print(f"{label:<12} {px.shape[1]:>4} names with full history, {len(px)} bars")
    return px


def stack(score, prices, top_frac=0.3, rebal=5):
    s = score.copy()
    mask = np.zeros(len(s), dtype=bool)
    mask[::rebal] = True
    s = s.where(pd.Series(mask, index=s.index), np.nan).ffill()
    rank = s.rank(axis=1, pct=True, ascending=False)
    raw = (rank <= top_frac).astype(float)
    iv = strategies.inverse_volatility(prices, lookback=60)
    w = raw * iv
    tot = w.sum(axis=1)
    w = w.div(tot.where(tot > 0), axis=0).fillna(0.0)
    return strategies.volatility_target(w, prices, target_vol=0.10, max_leverage=1.0)


def synth(prices, ic, seed=11):
    """Forecast with a known cross-sectional IC against the next bar."""
    rng = np.random.default_rng(seed)
    fut = prices.pct_change().shift(-1)
    z = fut.sub(fut.mean(axis=1), axis=0).div(fut.std(axis=1).replace(0, np.nan), axis=0)
    noise = pd.DataFrame(rng.normal(size=z.shape), index=z.index, columns=z.columns)
    return ic * z + np.sqrt(1 - ic**2) * noise


def breakeven(px, cost, label):
    test = px.index >= SPLIT
    px_t = px[test]
    flat = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    bench = backtest.run(px_t, stack(flat, px, top_frac=1.0, rebal=21)[test],
                         cost_model=cost)
    bs = bench.summary()

    grid, curve = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12], []
    for ic in grid:
        r = backtest.run(px_t, stack(synth(px, ic), px)[test], cost_model=cost)
        curve.append(r.summary()["sharpe"])

    # linear interpolation to the crossing point
    be = np.nan
    for i in range(len(grid) - 1):
        if (curve[i] - bs["sharpe"]) * (curve[i + 1] - bs["sharpe"]) < 0:
            f = (bs["sharpe"] - curve[i]) / (curve[i + 1] - curve[i])
            be = grid[i] + f * (grid[i + 1] - grid[i])
            break
    return bs, grid, curve, be


print("loading...\n")
lc_syms = pd.read_parquet("data/collection_universe.parquet")
lc_syms = lc_syms[[c for c in lc_syms.columns
                   if c.lower() in ("symbol", "ticker")][0]].tolist()
lc = load(lc_syms, "large cap")
sc = load("data/sp600_constituents.parquet", "small cap")

print("\nBREADTH (effective bets from the correlation eigenvalue spectrum)")
for lbl, p in [("large cap", lc), ("small cap", sc)]:
    eb = research.effective_bets(p.pct_change().dropna(how="all").dropna(axis=1))
    print(f"  {lbl:<12} {p.shape[1]:>4} names -> {eb:6.2f} effective bets")

print(f"\n{'='*88}")
print(f"BREAK-EVEN IC   (out of sample {SPLIT}+, long-only, vol-targeted, top 30%)")
print(f"{'='*88}")
for lbl, px, cost in [("LARGE CAP", lc, IBKR_US_EQUITY),
                      ("SMALL CAP  optimistic", sc, SC_OPTIMISTIC),
                      ("SMALL CAP  central", sc, SC_MID),
                      ("SMALL CAP  pessimistic", sc, SC_PESSIMISTIC)]:
    bs, grid, curve, be = breakeven(px, cost, lbl)
    print(f"\n{lbl}   [{cost.name}, {cost.round_trip_bps:.0f}bp round trip]")
    print(f"  benchmark (no signal): Sharpe {bs['sharpe']:.2f}  "
          f"CAGR {bs['cagr']*100:.2f}%  MaxDD {bs['max_drawdown']*100:.1f}%")
    print("   IC   " + " ".join(f"{g:>6.3f}" for g in grid))
    print("   Sh   " + " ".join(f"{s:>6.2f}" for s in curve))
    print(f"  -> BREAK-EVEN IC = {be:.4f}" if np.isfinite(be)
          else "  -> no crossing within the tested grid")


# ---------------------------------------------------------------------------
# FIX: the comparison above gave the strategy weekly rebalancing and the
# benchmark monthly. At 4bp that is immaterial; at 37bp it hands the benchmark a
# large unearned advantage. Redo with MATCHED rebalancing, and sweep the
# frequency -- if costs are what kills small caps, slowing down should rescue it.
# ---------------------------------------------------------------------------
print(f"\n\n{'='*88}")
print("MATCHED REBALANCING -- benchmark and strategy on the same schedule")
print(f"{'='*88}")

for lbl, px, cost in [("LARGE CAP", lc, IBKR_US_EQUITY),
                      ("SMALL CAP central", sc, SC_MID),
                      ("SMALL CAP optimistic", sc, SC_OPTIMISTIC)]:
    test = px.index >= SPLIT
    px_t = px[test]
    flat = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    print(f"\n{lbl}  [{cost.round_trip_bps:.0f}bp round trip]")
    print(f"{'rebal':<10} {'bench Sh':>9} {'IC=0':>7} {'IC=.02':>7} {'IC=.05':>7} "
          f"{'IC=.10':>7} {'break-even IC':>14}")
    for rebal, tag in [(5, "weekly"), (21, "monthly"), (63, "quarterly")]:
        b = backtest.run(px_t, stack(flat, px, top_frac=1.0, rebal=rebal)[test],
                         cost_model=cost).summary()["sharpe"]
        grid = [0.0, 0.02, 0.05, 0.10, 0.20]
        cur = [backtest.run(px_t, stack(synth(px, ic), px, rebal=rebal)[test],
                            cost_model=cost).summary()["sharpe"] for ic in grid]
        be = np.nan
        for i in range(len(grid) - 1):
            if (cur[i] - b) * (cur[i + 1] - b) < 0:
                be = grid[i] + (b - cur[i]) / (cur[i + 1] - cur[i]) * (grid[i + 1] - grid[i])
                break
        bes = f"{be:.4f}" if np.isfinite(be) else ">0.20"
        print(f"{tag:<10} {b:>9.2f} {cur[0]:>7.2f} {cur[1]:>7.2f} {cur[2]:>7.2f} "
              f"{cur[3]:>7.2f} {bes:>14}")

print("\n\nWHY -- decomposing the small-cap disadvantage")
print("(effective bets already reported above: 7.13 large vs 8.43 small)")
r_lc = lc.pct_change(); r_sc = sc.pct_change()
print(f"  mean pairwise correlation   large {r_lc.corr().values[np.triu_indices(len(lc.columns),1)].mean():.3f}"
      f"   small {r_sc.corr().values[np.triu_indices(len(sc.columns),1)].mean():.3f}")
print(f"  median annualised vol       large {(r_lc.std()*np.sqrt(252)).median()*100:.1f}%"
      f"   small {(r_sc.std()*np.sqrt(252)).median()*100:.1f}%")
print(f"  names surviving 2005-2026   large {lc.shape[1]}/211 of the index"
      f"   small {sc.shape[1]}/603 of the index  <- SURVIVORSHIP")
