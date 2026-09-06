"""Options-derived signals: what the free data can and cannot tell you.

Everything free here (VIX complex, SKEW) is INDEX-LEVEL. Per the spec's new
clause, that means market-wide triggers -- one bet wearing many hats. These can
only ever be MARKET TIMING signals on a single series, which caps breadth at
roughly the number of independent regime episodes, not the number of days.

Tested honestly anyway, because market timing is a legitimate (if low-breadth)
game and the data costs nothing.
"""
import numpy as np, pandas as pd, warnings, yfinance as yf
from algo import metrics, research
warnings.filterwarnings("ignore")

S, E = "2012-01-01", "2026-09-01"
def grab(t):
    d = yf.download(t, start=S, end=E, progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d["Close"]

raw = {t: grab(t) for t in ["SPY", "^VIX", "^VIX3M", "^VIX9D", "^VVIX", "^SKEW"]}
df = pd.DataFrame(raw).dropna()
df.index = pd.to_datetime(df.index).tz_localize(None)
spy = df[["SPY"]]
print(f"{len(df)} common bars, {df.index[0].date()}..{df.index[-1].date()}\n")

def z(s, w=252):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()

SIGNALS = {
    "vix_term (VIX/VIX3M)": df["^VIX"] / df["^VIX3M"],
    "vix9d/vix":            df["^VIX9D"] / df["^VIX"],
    "vix_level_z":          z(df["^VIX"]),
    "vvix/vix":             df["^VVIX"] / df["^VIX"],
    "skew_z":               z(df["^SKEW"]),
    "vix - realized":       df["^VIX"] / (spy["SPY"].pct_change().rolling(21).std()
                                          * np.sqrt(252) * 100),
}

print("TIME-SERIES IC vs SPY FORWARD RETURNS (single asset -> low breadth)\n")
print(f"{'signal':<24} " + "".join(f"{'h='+str(h):>10}" for h in (1, 5, 21)))
for name, s in SIGNALS.items():
    sig = pd.DataFrame({"SPY": s}).reindex(spy.index)
    row = ""
    for h in (1, 5, 21):
        fwd = research.forward_returns(spy, h)
        ok = sig["SPY"].notna() & fwd["SPY"].notna()
        ic = sig["SPY"][ok].rank().corr(fwd["SPY"][ok].rank())
        row += f"{ic:>10.4f}"
    print(f"{name:<24} {row}")

print("\n\nAS CONDITIONAL SETUPS -- with episode + overlap deflation\n")
print(f"{'setup':<34} {'events':>8} {'episodes':>9} {'ann edge':>10} {'t_naive':>9} {'t_adj':>8}")
CONDS = {
    "VIX/VIX3M > 1.00 (backwardation)":  (df["^VIX"] / df["^VIX3M"]) > 1.00,
    "VIX/VIX3M > 1.05":                  (df["^VIX"] / df["^VIX3M"]) > 1.05,
    "VIX/VIX3M < 0.85 (steep contango)": (df["^VIX"] / df["^VIX3M"]) < 0.85,
    "VIX z > 2":                         z(df["^VIX"]) > 2,
    "VIX z < -1":                        z(df["^VIX"]) < -1,
    "SKEW z > 1.5":                      z(df["^SKEW"]) > 1.5,
    "SKEW z < -1.5":                     z(df["^SKEW"]) < -1.5,
    "VVIX/VIX > 8":                      (df["^VVIX"] / df["^VIX"]) > 8,
}
bar = research.bonferroni_t(len(CONDS) * 3)
res = []
for name, c in CONDS.items():
    cond = pd.DataFrame({"SPY": c}).reindex(spy.index).fillna(False)
    for h in (5, 21, 63):
        r = research.event_study(spy, cond, (h,)).iloc[0]
        if np.isnan(r["t"]):
            continue
        res.append((name, h, r))
        if abs(r["t_naive"]) > 2.0:
            mark = "  <<<" if abs(r["t"]) > bar else ""
            print(f"{name[:32]:<34} {int(r['n_events']):>8} {int(r['n_episodes']):>9} "
                  f"{r['ann_edge']*100:>9.1f}% {r['t_naive']:>9.2f} {r['t']:>8.2f}{mark}")

print(f"\n  Bonferroni bar for {len(CONDS)*3} tests: |t| > {bar:.2f}")
surv = [x for x in res if abs(x[2]["t"]) > bar]
print(f"  survivors after deflation: {len(surv)} of {len(res)}")
print("\n  BREADTH CEILING: this is ONE asset. Even a perfect signal firing")
print("  ~10 independent times a year gives BR=10 -> sqrt(10)=3.2. At IC 0.05")
print("  that is IR 0.16. Market timing cannot carry a strategy on its own.")
