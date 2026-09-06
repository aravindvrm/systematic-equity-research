"""Does intraday data improve VOLATILITY ESTIMATION -- the one component that works?

WHY THIS AND NOT SIGNALS
------------------------
The null floor established that the vol-targeting overlay produces essentially
all of this project's results: zero-information signals routed through it scored
FF6 alpha t = 2.75. Everything we built on top of it was noise. So the highest-
value use of new data is improving the overlay, not adding a 14th signal.

Andersen, Bollerslev, Diebold & Labys: REALIZED volatility computed from
intraday returns forecasts future volatility far better than close-to-close
estimates, because it uses ~7 observations per day instead of 1 and is not
contaminated by the overnight gap in the same way.

Crucially, this needs only a ROLLING WINDOW -- no long history. Free hourly data
(~3 years) is enough to test it, unlike a signal backtest.

THE TEST
--------
Which estimator better predicts NEXT MONTH's realized volatility?
    close-to-close std (what we use everywhere)
    realized vol from hourly bars
    Garman-Klass from daily OHLC (a middle option, uses the daily range)
Scored by rank correlation with the subsequent realised value, and by regression
R-squared. Better vol forecasting means better risk targeting and better
inverse-vol weights.
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
from scipy import stats as sst

pd.set_option("display.width", 200)

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "JPM", "XOM", "JNJ", "TLT", "GLD",
           "NVDA", "WMT", "PG", "HD", "BAC", "CVX", "KO", "PFE", "T"]
print(f"{len(TICKERS)} names, hourly ~3y\n", flush=True)

hourly, daily = {}, {}
for t in TICKERS:
    h = yf.download(t, interval="1h", period="730d", progress=False, auto_adjust=True)
    d = yf.download(t, interval="1d", period="1000d", progress=False, auto_adjust=True)
    if len(h) > 2000 and len(d) > 400:
        hourly[t] = h["Close"].squeeze()
        daily[t] = d[["Open", "High", "Low", "Close"]].droplevel(1, axis=1) \
            if isinstance(d.columns, pd.MultiIndex) else d[["Open", "High", "Low", "Close"]]
print(f"loaded {len(hourly)} names\n", flush=True)

W = 21          # estimation window, trading days
ANN = np.sqrt(252)
rows = []
for t in hourly:
    h, d = hourly[t], daily[t]
    hr = np.log(h).diff().dropna()
    # Realized vol: sum of squared intraday returns per day, annualised.
    rv_daily = hr.groupby(hr.index.date).apply(lambda x: np.sqrt((x ** 2).sum()))
    rv_daily.index = pd.to_datetime(rv_daily.index)

    c = d["Close"]
    cc = c.pct_change()
    idx = rv_daily.index.intersection(c.index)
    if len(idx) < 300:
        continue

    est_cc = cc.reindex(idx).rolling(W).std() * ANN
    est_rv = rv_daily.reindex(idx).rolling(W).mean() * ANN
    lo, hi_, op = d["Low"].reindex(idx), d["High"].reindex(idx), d["Open"].reindex(idx)
    gk = np.sqrt((0.5 * np.log(hi_ / lo) ** 2
                  - (2 * np.log(2) - 1) * np.log(c.reindex(idx) / op) ** 2)
                 .rolling(W).mean()) * ANN
    # TARGET: the volatility that actually occurred over the NEXT window.
    fut = cc.reindex(idx).rolling(W).std().shift(-W) * ANN

    df = pd.DataFrame({"cc": est_cc, "rv": est_rv, "gk": gk, "fut": fut}).dropna()
    if len(df) < 200:
        continue
    for k in ("cc", "rv", "gk"):
        rows.append(dict(ticker=t, est=k,
                         rho=sst.spearmanr(df[k], df["fut"]).statistic,
                         r2=np.corrcoef(df[k], df["fut"])[0, 1] ** 2,
                         mae=float((df[k] - df["fut"]).abs().mean()),
                         n=len(df)))
R = pd.DataFrame(rows)

print("=" * 76)
print(f"PREDICTING NEXT {W}-DAY REALISED VOLATILITY  ({R.ticker.nunique()} names)")
print("=" * 76 + "\n")
print(f"{'estimator':<28} {'rank corr':>11} {'R2':>8} {'MAE':>9} {'wins':>7}")
NAMES = {"cc": "close-to-close (ours)", "rv": "realised vol (hourly)",
         "gk": "Garman-Klass (daily OHLC)"}
best = R.loc[R.groupby("ticker")["rho"].idxmax()]
for k in ("cc", "rv", "gk"):
    g = R[R.est == k]
    print(f"{NAMES[k]:<28} {g.rho.mean():>11.3f} {g.r2.mean():>8.3f} "
          f"{g.mae.mean()*100:>8.2f}% {(best.est == k).sum():>7}")

print(f"\n{'-'*76}")
piv = R.pivot(index="ticker", columns="est", values="rho")
print("per-name rank correlation (higher = better forecast)\n")
print(piv[["cc", "rv", "gk"]].round(3).to_string())
imp = (piv["rv"] - piv["cc"])
print(f"\nrealised-vol improvement over close-to-close: mean {imp.mean():+.3f}, "
      f"positive for {(imp > 0).sum()}/{len(imp)} names")
print(f"Garman-Klass improvement (FREE, daily data): {(piv['gk']-piv['cc']).mean():+.3f}, "
      f"positive for {(piv['gk'] > piv['cc']).sum()}/{len(piv)}")
