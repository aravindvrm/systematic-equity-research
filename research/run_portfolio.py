"""The combined multi-premia portfolio, with full diagnostics.

Honest trial count: the specs below were reached after roughly 30 variants
across this whole session (6 baselines, 12 in the turnover sweep, 5 lookbacks,
several vol targets). n_trials is set accordingly -- inflating your own
confidence by under-counting is the easiest mistake to make.
"""
import pandas as pd
from algo import backtest, data, metrics, portfolio
from algo.costs import IBKR_US_EQUITY, IBKR_US_EQUITY_SMALL

N_TRIALS = 30
START, END = "2010-01-01", "2026-09-01"

px = data.load_panel(data.ETF_UNIVERSE, start=START, end=END).dropna(how="any")
print(f"universe: {list(px.columns)}")
print(f"period:   {px.index[0].date()} .. {px.index[-1].date()}  ({len(px)} bars)\n")

bench = backtest.buy_and_hold(px, "SPY", cost_model=IBKR_US_EQUITY)

print("=" * 74)
print("COMBINED PORTFOLIO  (ensembled trend + inverse-vol weighting + vol target)")
print("=" * 74)
w = portfolio.build(px, target_vol=0.10)
res = backtest.run(px, w, IBKR_US_EQUITY)
print(res.report(benchmark=bench, benchmark_name="SPY B&H", n_trials=N_TRIALS))

print("\n" + "=" * 74)
print("COMPONENT ATTRIBUTION  (what each ingredient actually contributes)")
print("=" * 74)
variants = {
    "SPY buy & hold":                     bench,
    "trend only (ensembled)":             backtest.run(px, portfolio.trend_sleeve(px), IBKR_US_EQUITY),
    "trend + inverse-vol weighting":      backtest.run(px, portfolio.build(px, target_vol=99, risk_weight=True), IBKR_US_EQUITY),
    "trend + vol target (no risk wt)":    backtest.run(px, portfolio.build(px, risk_weight=False, target_vol=0.10), IBKR_US_EQUITY),
    "FULL (trend + riskwt + vol target)": res,
}
rows = []
for name, r in variants.items():
    s = r.summary()
    rows.append({"variant": name, "Sharpe": s["sharpe"], "CAGR %": s["cagr"]*100,
                 "Vol %": s["vol"]*100, "MaxDD %": s["max_drawdown"]*100,
                 "Calmar": s["calmar"], "Turnover": s["avg_turnover"]})
print(pd.DataFrame(rows).set_index("variant").round(2).to_string())

print("\n" + "=" * 74)
print("ORDER-SIZE SENSITIVITY  (does it survive at the size you will trade it?)")
print("=" * 74)
for cm, label in [(IBKR_US_EQUITY, "$2,000+ orders"), (IBKR_US_EQUITY_SMALL, "~$1,000 orders")]:
    r = backtest.run(px, w, cm)
    print(f"  {label:18s} Sharpe {metrics.sharpe(r.returns):5.2f}   "
          f"CAGR {metrics.cagr(r.equity)*100:6.2f}%   "
          f"cost drag {r.costs.sum()*100:5.2f}% total")

print("\n" + "=" * 74)
print("WALK-FORWARD: ensemble vs fitting the lookback on each training window")
print("=" * 74)
ens = backtest.walk_forward(px, portfolio.walk_forward_ensemble,
                            train_periods=756, test_periods=252,
                            cost_model=IBKR_US_EQUITY)
fit = backtest.walk_forward(px, portfolio.select_best_lookback,
                            train_periods=756, test_periods=252,
                            cost_model=IBKR_US_EQUITY)
print(f"  {'ensemble (fits nothing)':<34} OOS Sharpe {metrics.sharpe(ens.returns):5.2f}"
      f"   CAGR {metrics.cagr(ens.equity)*100:6.2f}%   MaxDD {metrics.max_drawdown(ens.equity)*100:6.1f}%")
print(f"  {'best-lookback (fitted per window)':<34} OOS Sharpe {metrics.sharpe(fit.returns):5.2f}"
      f"   CAGR {metrics.cagr(fit.equity)*100:6.2f}%   MaxDD {metrics.max_drawdown(fit.equity)*100:6.1f}%")
