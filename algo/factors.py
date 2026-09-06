"""Fama-French factor attribution -- the control that should have come first.

WHY THIS IS THE STANDARD TEST
-----------------------------
Regressing a strategy on a 60/40 portfolio, as this project did, controls for
almost nothing. The strategy might simply be loading on size, value, profitability,
investment or momentum -- all of which are known, free, and buyable through cheap
ETFs. "Alpha" that vanishes once those are on the right-hand side is not alpha;
it is a factor tilt you could have bought.

Every published asset-pricing result runs this regression. This project ran
twelve signal families without it.

    r_p - rf = alpha + b1*MKT + b2*SMB + b3*HML + b4*RMW + b5*CMA + b6*MOM + e

THE BAR
-------
Harvey, Liu & Zhu (2016, RFS) show that with hundreds of factors already tested,
the conventional |t| > 2.0 is far too lenient; they argue for |t| > 3.0 as a
minimum for a NEW factor. That is a floor, not a target, and it applies to a
single pre-specified test -- not to the best of a 200-cell search.

Data: Ken French's data library (free). Daily 5-factor + momentum.
Units: French publishes PERCENT; converted to decimal here.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
CACHE = Path(__file__).resolve().parent.parent / "data" / "factors"
UA = {"User-Agent": "algo-research aravindvrm@gmail.com"}
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]


def _read_zip_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=UA, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = z.namelist()[0]
    raw = z.open(name).read().decode("latin-1")
    # French's files carry a prose header and a trailing annual section. Keep
    # only lines whose first field is an 8-digit date.
    rows = [ln for ln in raw.splitlines()
            if len(ln) > 8 and ln.strip()[:8].isdigit()]
    hdr = None
    for ln in raw.splitlines():
        if "Mkt-RF" in ln or "Mom" in ln:
            hdr = [c.strip() for c in ln.split(",") if c.strip()]
            break
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None)
    df.columns = ["date"] + (hdr if hdr and len(hdr) == df.shape[1] - 1
                             else [f"f{i}" for i in range(df.shape[1] - 1)])
    df["date"] = pd.to_datetime(df["date"].astype(int).astype(str), format="%Y%m%d")
    return df.set_index("date").astype(float) / 100.0     # percent -> decimal


def load(refresh: bool = False) -> pd.DataFrame:
    """Daily FF5 + momentum + RF, as decimals."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "ff5_mom_daily.parquet"
    if p.exists() and not refresh:
        return pd.read_parquet(p)
    ff = _read_zip_csv(f"{BASE}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    mom = _read_zip_csv(f"{BASE}/F-F_Momentum_Factor_daily_CSV.zip")
    mom.columns = ["MOM"]
    out = ff.join(mom, how="inner")
    out.to_parquet(p)
    return out


def attribution(strategy: pd.Series, ff: pd.DataFrame | None = None,
                factors: list[str] | None = None, lags: int = 21) -> dict:
    """Regress strategy EXCESS return on the factors. Newey-West t-stats.

    Returns annualised alpha, its t, every loading, and R-squared.
    """
    import statsmodels.api as sm

    ff = load() if ff is None else ff
    factors = FACTORS if factors is None else factors
    df = pd.concat([strategy.rename("r"), ff], axis=1).dropna()
    if len(df) < 250:
        return {"alpha_ann": np.nan, "alpha_t": np.nan, "n": len(df)}
    y = (df["r"] - df["RF"]).to_numpy()
    X = sm.add_constant(df[factors].to_numpy())
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {"alpha_ann": float(fit.params[0]) * 252,
           "alpha_t": float(fit.tvalues[0]),
           "r2": float(fit.rsquared), "n": len(df)}
    for i, f in enumerate(factors, start=1):
        out[f"b_{f}"] = float(fit.params[i])
        out[f"t_{f}"] = float(fit.tvalues[i])
    return out


def conditional_attribution(strategy: pd.Series, ff: pd.DataFrame | None = None,
                            factors_: list[str] | None = None,
                            lags: int = 21) -> dict:
    """Ferson-Schadt (1996) conditional performance evaluation.

    THE PROBLEM IT SOLVES
    ---------------------
    Unconditional alpha is BIASED when beta varies with public information.
    Volatility targeting does precisely that: it cuts market exposure when
    trailing volatility is high, and trailing volatility is public. A constant-
    beta model cannot represent a strategy that times its own exposure, so the
    unmodelled timing shows up as alpha.

    Measured here: 40 ZERO-INFORMATION signals through a vol-targeted stack
    produced unconditional FF6 alpha t = 2.61 on average -- above the |t| > 2
    convention and close to Harvey-Liu-Zhu's |t| > 3.

    THE FIX
    -------
    Interact the market factor with lagged, demeaned public information z:

        r - rf = alpha + b*MKT + B*(z_{t-1} x MKT) + other factors + e

    The interaction absorbs beta variation that is predictable from public data,
    leaving alpha to mean what it is supposed to mean. z here is lagged trailing
    market volatility -- the exact variable a vol-targeting rule responds to.
    """
    import statsmodels.api as sm

    ff = load() if ff is None else ff
    factors_ = FACTORS if factors_ is None else factors_
    df = pd.concat([strategy.rename("r"), ff], axis=1).dropna()
    if len(df) < 500:
        return {"alpha_ann": np.nan, "alpha_t": np.nan, "n": len(df)}

    # Public information: trailing market vol, LAGGED (known at t-1) and demeaned
    # so the alpha keeps its usual "average abnormal return" interpretation.
    vol = df["Mkt-RF"].rolling(60, min_periods=30).std().shift(1)
    z = (vol - vol.mean()) / vol.std()
    df = df.assign(_z=z).dropna()

    y = (df["r"] - df["RF"]).to_numpy()
    base = df[factors_].to_numpy()
    inter = (df["_z"].to_numpy()[:, None] * df[["Mkt-RF"]].to_numpy())
    X = sm.add_constant(np.hstack([base, inter]))
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"alpha_ann": float(fit.params[0]) * 252,
            "alpha_t": float(fit.tvalues[0]),
            "b_timing": float(fit.params[-1]),
            "t_timing": float(fit.tvalues[-1]),
            "r2": float(fit.rsquared), "n": len(df)}
