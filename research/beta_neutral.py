"""Beta-neutral cross-section: is anything left once beta is actually removed?

Pre-registered in notes/Beta-Neutral Pre-Registration.md. Read that first; the
predictions there were committed before this file existed.

WHY THIS RUN
------------
market_neutral.py tested a DOLLAR-neutral long/short book on the random forest:
Sharpe 0.735 vs null max 0.738, a fail. But it realised a market beta of 0.53,
so it was never beta-neutral and never answered the actual question. Equal
dollars is not equal risk when the signal is a volatility tilt -- the RF's
predictions correlate -0.472 with low-volatility, so its long leg holds
higher-beta names than its short leg.

DESIGN, FROM THE LITERATURE (not invented here)
-----------------------------------------------
beta estimate   daily returns, 1-year trailing, EXPONENTIALLY weighted, then
                Vasicek (1973) shrinkage toward 1.0. The market-neutrality
                forecasting literature finds daily/1yr minimises the ex-ante vs
                realised gap, and exponential weighting pulls realised betas of
                ex-ante-neutral books toward zero. Frazzini-Pedersen report a
                mean shrinkage weight near 0.51 on US equities.
leg scaling     each leg divided by its own portfolio beta, so EX-ANTE beta is
                zero (Frazzini-Pedersen; averages ~$1.52 long / $0.71 short).
held weights    portfolio beta is aggregated on the weights ACTUALLY HELD, not
                on rank weights. Novy-Marx & Velikov show rank-weighting is a
                backdoor to equal-weighting that concentrates in microcaps and
                OVERSTATES beta-neutral profitability. The 211-large-cap
                universe closes the microcap channel; the held-weight
                aggregation closes the other one.
realised beta   measured, never assumed. BAB's own realised loading is not
                zero, and a book targeting zero off 5yr monthly betas has been
                shown to realise ex-post beta above one. Assuming neutrality is
                precisely the error that produced the 0.53.
variant B       FF6-residual as well as market-beta-neutral. Blitz, Huij &
                Martens: momentum carries large time-varying FF exposures and
                ranking on residuals removes them. With FF6 R^2 of 0.66-0.75
                across this project, hedging market alone leaves the value,
                size and volatility exposure intact.
costs           50bp GC borrow, NO rebate on short proceeds (retail), margin
                interest on the debit balance (charged here for the first time
                in this project), and an explicit Reg T check.
null            recalibrated per construction. A floor computed for one
                construction does not transfer to another.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor

from algo import evaluation, factors, metrics

pd.set_option("display.width", 200)

CACHE = "results/rf_preds.parquet"
BORROW_ANN = 0.0050        # 50bp general collateral, S&P names
MARGIN_ANN = 0.058         # IBKR Reg T debit rate, retail tier
COST_BPS = 2.2             # one-way, IBKR US equity
SHRINK_W = 0.51            # Vasicek weight, Frazzini-Pedersen US mean
BETA_HALFLIFE = 63         # trading days, inside a 252-day window
REG_T_GROSS_MAX = 2.0      # 50% initial margin


# ----------------------------------------------------------------- signal
def load_rf_predictions(cl: pd.DataFrame, ns: dict) -> pd.DataFrame:
    """Walk-forward RF predictions, cached -- identical recipe to rf_stress.py."""
    from pathlib import Path
    if Path(CACHE).exists():
        rf = pd.read_parquet(CACHE)
        print(f"  RF predictions loaded from {CACHE}")
        return rf.reindex(cl.index).reindex(columns=cl.columns)

    L, cols = ns["L"], ns["cols"]
    test_years = [y for y in sorted(L.date.dt.year.unique()) if y >= 2015]
    preds = []
    for y in test_years:
        tr, te = L[L.date.dt.year < y], L[L.date.dt.year == y]
        if len(tr) < 2000 or te.empty:
            continue
        m = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=100,
                                  max_features=0.3, n_jobs=-1, random_state=0)
        m.fit(tr[cols].to_numpy(), tr["y"].to_numpy())
        p = te[["date", "ticker"]].copy()
        p["pred"] = m.predict(te[cols].to_numpy())
        preds.append(p)
    rf = (pd.concat(preds, ignore_index=True)
          .pivot(index="date", columns="ticker", values="pred")
          .reindex(cl.index).ffill().reindex(columns=cl.columns))
    rf.to_parquet(CACHE)
    print(f"  RF predictions trained and cached to {CACHE}")
    return rf


# ----------------------------------------------------------------- betas
def rolling_betas(r1: pd.DataFrame, mkt: pd.Series,
                  window: int = 252, halflife: int = BETA_HALFLIFE,
                  shrink: float = SHRINK_W) -> pd.DataFrame:
    """Point-in-time per-name betas: daily, 1yr, exponentially weighted, shrunk.

    Uses only data strictly BEFORE each date (shift(1)), so a beta applied on
    day t never sees day t. Vasicek shrinkage pulls toward a prior of 1.0.
    """
    m = mkt.reindex(r1.index).fillna(0.0)
    cov = r1.ewm(halflife=halflife, min_periods=window // 2).cov(m)
    var = m.ewm(halflife=halflife, min_periods=window // 2).var()
    raw = cov.div(var, axis=0)
    shrunk = shrink * raw + (1.0 - shrink) * 1.0
    return shrunk.shift(1).clip(lower=-1.0, upper=3.0)


# ----------------------------------------------------------------- books
def _legs(score: pd.DataFrame, frac: float):
    r = score.rank(axis=1, pct=True, ascending=False)
    lo = (r <= frac).astype(float)
    sh = (r >= 1 - frac).astype(float)
    wl = lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    ws = sh.div(sh.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return wl, ws


def _rebalance(w: pd.DataFrame, rebal: int) -> pd.DataFrame:
    m = np.zeros(len(w), dtype=bool)
    m[::rebal] = True
    return w.where(pd.Series(m, index=w.index), np.nan).ffill().fillna(0.0)


def apply_reg_t(w: pd.DataFrame, gross_max: float = REG_T_GROSS_MAX) -> pd.DataFrame:
    """Scale the whole book down on days it would exceed Reg T initial margin.

    Without this a beta-scaled book can run gross above 2:1, which is not a
    backtest assumption but a margin call.
    """
    gross = w.abs().sum(axis=1)
    scale = (gross_max / gross).clip(upper=1.0).replace([np.inf, -np.inf], 1.0)
    return w.mul(scale.fillna(1.0), axis=0)


def book_returns(w: pd.DataFrame, r1: pd.DataFrame,
                 borrow: float = BORROW_ANN, margin: float = MARGIN_ANN,
                 cost_bps: float = COST_BPS,
                 mask: pd.Series | None = None) -> tuple[pd.Series, dict]:
    """Net returns for a signed book, with borrow, margin interest and costs.

    `mask` restricts the DIAGNOSTICS to the evaluation window; without it the
    exposure and financing figures are diluted by dates where the book is empty.
    """
    w = apply_reg_t(w)
    held = w.shift(1).fillna(0.0)
    gross = (held * r1).sum(axis=1)
    turn = (w.shift(1) - w.shift(2)).abs().sum(axis=1).fillna(0.0)
    cost = turn * cost_bps * 1e-4

    short_notional = held.clip(upper=0).abs().sum(axis=1)
    long_notional = held.clip(lower=0).sum(axis=1)
    gross_expo = long_notional + short_notional

    fee = short_notional * borrow / 252.0
    # Debit balance: capital is 1.0; anything held long beyond that is borrowed.
    # Short proceeds earn no rebate (retail), so they do not offset the debit.
    debit = (long_notional - 1.0).clip(lower=0)
    interest = debit * margin / 252.0

    net = gross - cost - fee - interest
    m = slice(None) if mask is None else mask
    diag = {"gross_expo_mean": float(gross_expo[m].mean()),
            "gross_expo_max": float(gross_expo[m].max()),
            "reg_t_capped_days": int((gross_expo[m] > REG_T_GROSS_MAX + 1e-9).sum()),
            "long_mean": float(long_notional[m].mean()),
            "short_mean": float(short_notional[m].mean()),
            "borrow_drag_ann": float(fee[m].mean() * 252),
            "margin_drag_ann": float(interest[m].mean() * 252),
            "turnover_ann": float(turn[m].mean() * 252)}
    return net, diag


def dollar_neutral(score, r1, betas, frac=0.2, rebal=21):
    wl, ws = _legs(score, frac)
    return _rebalance(wl - ws, rebal)


def beta_neutral(score, r1, betas, frac=0.2, rebal=21):
    """Variant A: each leg scaled by its own portfolio beta -> ex-ante beta 0."""
    wl, ws = _legs(score, frac)
    bl = (wl * betas).sum(axis=1).replace(0, np.nan)
    bs = (ws * betas).sum(axis=1).replace(0, np.nan)
    # Scale so beta_long * k_l == beta_short * k_s, normalised to unit capital.
    kl = (1.0 / bl).replace([np.inf, -np.inf], np.nan)
    ks = (1.0 / bs).replace([np.inf, -np.inf], np.nan)
    scale = (kl + ks) / 2.0
    w = wl.mul(kl / scale, axis=0) - ws.mul(ks / scale, axis=0)
    return _rebalance(w.fillna(0.0), rebal)


def orthogonalise(score: pd.DataFrame, against: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional residual of `score` on `against`, per date.

    Removes the part of the signal that is a bet on the conditioning variable,
    leaving the part that is not. Done per date across names, so it uses no
    information from other dates.
    """
    x, y = against.to_numpy(dtype=float), score.to_numpy(dtype=float)
    out = np.full_like(y, np.nan)
    for i in range(y.shape[0]):
        m = np.isfinite(x[i]) & np.isfinite(y[i])
        if m.sum() < 30:
            continue
        xi, yi = x[i, m], y[i, m]
        xc, yc = xi - xi.mean(), yi - yi.mean()
        denom = float(xc @ xc)
        b = (xc @ yc) / denom if denom > 1e-12 else 0.0
        out[i, m] = yc - b * xc
    return pd.DataFrame(out, index=score.index, columns=score.columns)


def beta_orthogonal_neutral(score, r1, betas, frac=0.2, rebal=21):
    """Variant B: rank on the signal ORTHOGONALISED to beta, then beta-neutralise.

    Variant A removes the book's beta exposure but leaves the signal free to
    keep ranking on beta. B removes the tilt from the ranking itself, which is
    the practical form of the Blitz/Huij/Martens residual argument and targets
    the volatility tilt the RF is documented to carry directly.
    """
    return beta_neutral(orthogonalise(score, betas), r1, betas, frac, rebal)


# ----------------------------------------------------------------- report
def realised_beta(r: pd.Series, mkt: pd.Series) -> float:
    both = pd.concat([r.rename("r"), mkt.rename("m")], axis=1).dropna()
    if len(both) < 100:
        return float("nan")
    return float(np.polyfit(both["m"], both["r"], 1)[0])


def evaluate_construction(name, builder, score, r1, betas, mkt, oos, ff, n_null=40):
    w = apply_reg_t(builder(score, r1, betas))
    r, diag = book_returns(w, r1, mask=oos)
    r = r[oos]
    eq = (1 + r).cumprod()

    exante = float(((w.shift(1).fillna(0.0)) * betas).sum(axis=1)[oos].mean())
    real = realised_beta(r, ff["Mkt-RF"])

    nulls, null_ts = [], []
    for i in range(n_null):
        rng = np.random.default_rng(9000 + i)
        n = pd.DataFrame(rng.normal(size=score.shape),
                         index=score.index, columns=score.columns)
        nr, _ = book_returns(builder(n, r1, betas), r1, mask=oos)
        nr = nr[oos]
        nulls.append(metrics.sharpe(nr))
        null_ts.append(factors.attribution(nr, ff).get("alpha_t", np.nan))
    nulls = np.array(nulls)
    null_ts = np.array([t for t in null_ts if np.isfinite(t)])

    sh = metrics.sharpe(r)
    att = factors.attribution(r, ff)
    return {"name": name, "sharpe": sh, "cagr": metrics.cagr(eq),
            "vol": metrics.volatility(r), "maxdd": metrics.max_drawdown(eq),
            "beta_exante": exante, "beta_realised": real,
            "alpha_t": att.get("alpha_t", float("nan")),
            "beta_ff6": att.get("b_Mkt-RF", float("nan")),
            "null_mean": nulls.mean(), "null_p95": np.percentile(nulls, 95),
            "null_max": nulls.max(),
            "pctile": evaluation.percentile_vs_null(sh, nulls),
            "alpha_t_null_p95": np.percentile(null_ts, 95) if null_ts.size else np.nan,
            "alpha_t_null_max": null_ts.max() if null_ts.size else np.nan,
            "alpha_t_pctile": evaluation.percentile_vs_null(
                att.get("alpha_t", np.nan), null_ts) if null_ts.size else np.nan,
            **diag}


def _prelude():
    """Run rf_stress.py's data-building header to get the same panel and labels."""
    src = open("research/rf_stress.py").read()
    src = src.split("test_years =")[0].split('print(f"{len(L):,}')[0]
    ns: dict = {}
    exec(compile(src, "rf_stress_prelude", "exec"), ns)
    return ns


def main(n_null: int = 40):
    print("Building panel and labels (rf_stress.py prelude)...")
    ns = _prelude()
    cl = ns["cl"]
    r1 = cl.pct_change().fillna(0.0)
    ff = factors.load()
    mkt = ff["Mkt-RF"].reindex(cl.index).fillna(0.0)

    print("Building RF predictions...")
    rf = load_rf_predictions(cl, ns)

    print(f"Estimating betas (daily, 1yr, halflife {BETA_HALFLIFE}d, "
          f"Vasicek w={SHRINK_W})...")
    betas = rolling_betas(r1, mkt)
    print(f"  cross-sectional beta: mean {betas.stack().mean():.3f}  "
          f"sd {betas.stack().std():.3f}")

    oos = cl.index >= pd.Timestamp("2015-01-01")
    rows = []
    for name, builder in (("dollar-neutral (reproduce)", dollar_neutral),
                          ("beta-neutral (A)", beta_neutral),
                          ("beta-orthogonal (B)", beta_orthogonal_neutral)):
        print(f"Running {name} + its own {n_null}-draw null...")
        rows.append(evaluate_construction(name, builder, rf, r1, betas, mkt, oos, ff,
                                         n_null=n_null))

    out = pd.DataFrame(rows).set_index("name")
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print("\n" + "=" * 100)
    print("RESULT")
    print("=" * 100)
    print(out[["sharpe", "beta_exante", "beta_realised", "beta_ff6",
               "null_p95", "null_max", "pctile"]].to_string())
    print("\nFF6 alpha t against its OWN null (the pre-registered falsifier)")
    print(out[["alpha_t", "alpha_t_null_p95", "alpha_t_null_max",
               "alpha_t_pctile"]].to_string())
    print("\nExposure and financing")
    print(out[["long_mean", "short_mean", "gross_expo_mean", "reg_t_capped_days",
               "turnover_ann", "borrow_drag_ann", "margin_drag_ann"]].to_string())
    path = f"results/beta_neutral_n{n_null}.csv"
    out.to_csv(path)
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
