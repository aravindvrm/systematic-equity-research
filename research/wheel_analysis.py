"""The wheel (CSP -> assignment -> CC) as a strategy CLASS, measured over 30 years.

WHY THIS IS DIFFERENT FROM EVERYTHING ELSE IN THIS PROJECT
----------------------------------------------------------
Every signal we tested was ALPHA -- trying to predict which stocks outperform.
The wheel is not that. It harvests the VOLATILITY RISK PREMIUM: implied
volatility systematically exceeds subsequent realised volatility because people
pay up for protection. That premium is compensation for bearing tail risk, not
an oversight, so PUBLICATION DOES NOT COMPETE IT AWAY. That is the structural
difference from every anomaly in this project.

It is also the only thing here that works in a CASH ACCOUNT: cash-secured puts
are secured by definition, covered calls need only the shares.

WHAT WE CAN AND CANNOT MEASURE
------------------------------
CANNOT: the specific implementation -- which strikes, which deltas, when to
roll. That needs historical single-name option chains we do not have (our
collector has 1 day).

CAN: the strategy CLASS, via CBOE's own benchmark indices:
    ^BXM   BuyWrite -- at-the-money covered calls on SPX, rolled monthly
    ^PUT   PutWrite -- cash-secured at-the-money puts on SPX, rolled monthly
Both are total-return indices computed by CBOE, spanning 2000-02, 2008 and 2020.

THE THING TO LOOK FOR
---------------------
Not the average return. The TAIL. Selling volatility produces many small gains
and occasional large losses; the mean flatters it and Sharpe understates the
risk because the distribution is not Gaussian. Skew, kurtosis, worst months and
crisis-period behaviour are what matter.
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
from algo import evaluation, metrics

pd.set_option("display.width", 220)

px = {}
for t, lbl in (("^BXM", "BXM covered call"), ("^PUT", "PUT cash-sec put"),
               ("^SPX", "SPX price"), ("^VIX", "VIX")):
    d = yf.download(t, start="1996-08-01", progress=False, auto_adjust=False)
    px[lbl] = d["Close"].squeeze()
P = pd.DataFrame(px).dropna()
R = P.pct_change().dropna()
print(f"{len(P)} days, {P.index[0].date()}..{P.index[-1].date()}\n")

STRATS = ["BXM covered call", "PUT cash-sec put", "SPX price"]
print("=" * 104)
print("THE STRATEGY CLASS, 1996-2026 (includes 2000-02, 2008, 2020, 2022)")
print("=" * 104 + "\n")
print(f"{'strategy':<20} {'CAGR':>8} {'vol':>7} {'Sharpe':>7} {'MaxDD':>8} "
      f"{'Calmar':>7} {'skew':>7} {'ex kurt':>8} {'worst mo':>9}")
for s in STRATS:
    r = R[s]
    eq = (1 + r).cumprod()
    mo = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    from scipy import stats as sst
    print(f"{s:<20} {metrics.cagr(eq)*100:>7.2f}% {metrics.volatility(r)*100:>6.2f}% "
          f"{metrics.sharpe(r):>7.2f} {metrics.max_drawdown(eq)*100:>7.1f}% "
          f"{metrics.calmar(eq):>7.2f} {sst.skew(r):>7.2f} {sst.kurtosis(r):>8.1f} "
          f"{mo.min()*100:>8.1f}%")

print("\nNOTE: SPX here is PRICE only (no dividends), while BXM and PUT are total")
print("return. That understates SPX by ~1.8%/yr. Adjust mentally.\n")

print("=" * 104)
print("CRISIS BEHAVIOUR -- where selling volatility gets tested")
print("=" * 104 + "\n")
CRISES = [("dot-com 2000-02", "2000-03-01", "2002-10-09"),
          ("GFC 2008-09", "2007-10-09", "2009-03-09"),
          ("COVID 2020", "2020-02-19", "2020-03-23"),
          ("2022 bear", "2022-01-03", "2022-10-12"),
          ("recovery 2009-2019", "2009-03-10", "2019-12-31")]
print(f"{'period':<22} " + " ".join(f"{s.split()[0]:>10}" for s in STRATS))
for lbl, a, b in CRISES:
    m = (P.index >= a) & (P.index <= b)
    if m.sum() < 5:
        continue
    row = [f"{(P[s][m].iloc[-1]/P[s][m].iloc[0]-1)*100:>9.1f}%" for s in STRATS]
    print(f"{lbl:<22} " + " ".join(row))

print("\n" + "=" * 104)
print("AS A SLEEVE beside a 60/40 core -- our framework, applied")
print("=" * 104)
spy = yf.download("SPY", start="1996-08-01", progress=False, auto_adjust=True)["Close"].squeeze()
agg = yf.download("AGG", start="1996-08-01", progress=False, auto_adjust=True)["Close"].squeeze()
core = (0.6 * spy.pct_change() + 0.4 * agg.pct_change()).dropna()
print(f"\n{'sleeve':<20} {'stand Sh':>9} {'corr':>7} {'beta':>7} {'alpha%':>8} "
      f"{'NW t':>7} {'Sh@20%':>8} {'delta':>8}")
for s in ("BXM covered call", "PUT cash-sec put"):
    ev = evaluation.evaluate(R[s], core, label=s)
    a, c = evaluation._align(R[s], core)
    b20 = metrics.sharpe(0.8 * c + 0.2 * a)
    print(f"{s:<20} {ev['sharpe_standalone']:>9.2f} {ev['corr_to_core']:>7.2f} "
          f"{ev['beta']:>7.2f} {ev['alpha_ann']*100:>7.2f}% {ev['alpha_t']:>7.2f} "
          f"{b20:>8.2f} {b20-ev['core_sharpe']:>+8.3f}")
print(f"\n  core (60/40) Sharpe over the overlapping window: "
      f"{evaluation.evaluate(R['PUT cash-sec put'], core)['core_sharpe']:.3f}")

print("\n" + "=" * 104)
print("IS THE PREMIUM STILL THERE? (VIX minus subsequent realised vol, by era)")
print("=" * 104 + "\n")
spx_r = R["SPX price"]
fwd_rv = spx_r.rolling(21).std().shift(-21) * np.sqrt(252) * 100
vrp = (P["VIX"] - fwd_rv).dropna()
print(f"{'era':<14} {'mean VRP':>10} {'% positive':>12} {'n':>7}")
for a, b in [("1996", "2004"), ("2005", "2012"), ("2013", "2019"), ("2020", "2026")]:
    m = (vrp.index >= f"{a}-01-01") & (vrp.index <= f"{b}-12-31")
    v = vrp[m]
    if len(v) > 100:
        print(f"{a}-{b:<9} {v.mean():>9.2f}pp {(v>0).mean()*100:>11.0f}% {len(v):>7}")
print("\n  positive = implied exceeded subsequent realised = sellers were paid.")
