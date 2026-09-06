"""Conditional ICs under a PROPER multi-state HMM, point-in-time.

Compared against the crude median splits, and with episode deflation applied --
because an HMM state is still a MARKET-WIDE label, so the effective sample is
regime episodes, not days, no matter how sophisticated the model is.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from algo import data, features, regimes, research, universe as U

tab = U.collection_table()
tk = [t for t in tab.ticker if t not in U.DIVERSIFIERS]
px = data.load_panel(tk + ["SPY"], start="2016-01-01", end="2026-09-01")
px = px[[c for c in px.columns if px[c].notna().mean() > 0.95]].ffill().dropna()
spy = px["SPY"]; eq = px[[c for c in px.columns if c != "SPY"]]
R = eq.pct_change()

# richer observation vector than SPY returns alone
r = spy.pct_change()
obs = pd.DataFrame({
    "ret": r,
    "vol": r.rolling(21).std(),
    "disp": R.std(axis=1).rolling(21).mean(),
    "trend": spy / spy.rolling(100).mean() - 1.0,
}).dropna()

FEATS = {n: features.REGISTRY[n](eq) for n in
         ["mom_12_1", "mom_21", "reversal_5", "reversal_21"]}
H = 21
fwd = research.forward_returns(eq, H)

def episodes(mask):
    m = mask.fillna(False)
    return int(((m != m.shift()) & m).sum())

rows = []
for K in (2, 3, 4):
    print(f"fitting {K}-state HMM point-in-time...", flush=True)
    st = regimes.filtered_states(obs, n_states=K, min_train=504, refit_every=126)["state"]
    for fn, F in FEATS.items():
        d = research.dynamic_pit(F, method="expanding")
        ic = research.cross_sectional_ic(d, fwd)
        for s in range(K):
            mask = (st == s).reindex(ic.index).fillna(False)
            sub = ic[mask]
            if len(sub) < 100:
                continue
            stt = research.ic_stats(sub, H)
            ep = episodes(mask)
            # deflate for BOTH overlap (already in ic_stats) and episode clustering
            t_ep = stt["ic_t"] / np.sqrt(max(len(sub) / max(ep, 1), 1.0))
            rows.append(dict(K=K, feature=fn, state=s, days=len(sub), episodes=ep,
                             ic=stt["ic_mean"], t=stt["ic_t"], t_ep=t_ep))

df = pd.DataFrame(rows)
print(f"\n{'K':>2} {'feature':<13} {'st':>3} {'days':>6} {'eps':>5} {'IC':>8} "
      f"{'t':>7} {'t_episode':>10}")
for _, x in df.iterrows():
    print(f"{int(x.K):>2} {x.feature:<13} {int(x.state):>3} {int(x.days):>6} "
          f"{int(x.episodes):>5} {x.ic:>8.4f} {x.t:>7.2f} {x.t_ep:>10.2f}")

bar = research.bonferroni_t(len(df))
print(f"\n  {len(df)} tests -> Bonferroni bar |t| > {bar:.2f}")
print(f"  survivors on raw t     : {(df.t.abs() > bar).sum()}")
print(f"  survivors on episode t : {(df.t_ep.abs() > bar).sum()}")
print(f"\n  best raw t: {df.t.abs().max():.2f}   best episode-deflated: {df.t_ep.abs().max():.2f}")
print(f"  median days per episode: {(df.days/df.episodes).median():.0f}")
