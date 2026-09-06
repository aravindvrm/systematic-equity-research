"""The relational space we had NOT tested. One bounded, pre-specified batch.

Six features x three horizons = 18 tests, count declared before running, on the
full 2005-2026 sample. No variants will be added after seeing results; if a cell
looks interesting it gets the sample-split treatment that killed pair_spread_z,
not a search for a better parameterisation.

    net_centrality    position in the correlation graph (weighted by WHOM)
    coint_spread_z    pairs chosen by ADF stationarity, hedge-ratio adjusted
    size_lead_lag     large->small diffusion within sector (Hou 2007), DIRECTIONAL
    down_corr_asym    correlation on down days vs up days (Ang/Chen/Xing)
    beta_instability  volatility of the name's own market beta
    corr_dispersion   spread of the name's correlations across the cross-section

PRIOR, stated up front: five of these six are transformations of the same return
series that we just measured as worth IC ~0.012 against a requirement of ~0.05.
There is no strong reason a further rearrangement escapes that ceiling. This batch
is run to answer "exhausted?" empirically rather than by assertion.
"""
import numpy as np
import pandas as pd
from scipy import stats as sst

from algo import data, network, relational, research

pd.set_option("display.width", 220)

uni = pd.read_parquet("data/collection_universe.parquet")
sec_col = [c for c in uni.columns if "sector" in c.lower()][0]
sym_col = [c for c in uni.columns if c.lower() in ("symbol", "ticker")][0]

c = data.load_panel(uni[sym_col].tolist(), start="2005-01-01", end="2026-09-01",
                    field="close", refresh=False)
c = c.dropna(axis=1, thresh=int(0.9 * len(c))).ffill(limit=5)
v = (data.load_panel(uni[sym_col].tolist(), start="2005-01-01", end="2026-09-01",
                     field="volume", refresh=False)
     .reindex(index=c.index, columns=c.columns).ffill(limit=5))
sectors = uni.set_index(sym_col)[sec_col].reindex(c.columns)
print(f"{c.shape[1]} names, {len(c)} bars, {c.index[0].date()}..{c.index[-1].date()}\n")

print("building features (cointegration screen is the slow one)...", flush=True)
cpeers, cbetas = relational.coint_peer_map(c, sectors, window=504, refit_every=252)
print(f"  coint peers assigned for {cpeers.notna().mean().mean()*100:.0f}% of name-days")
print(f"  median hedge ratio beta {np.nanmedian(cbetas.to_numpy()):.2f} "
      "(1.00 would mean the old unhedged spread was fine)", flush=True)

FEATS = {
    "net_centrality":   network.net_centrality(c),
    "coint_spread_z":   relational.hedged_spread_z(c, cpeers, cbetas),
    "size_lead_lag":    network.size_lead_lag(c, c * v, sectors),
    "down_corr_asym":   network.down_corr_asym(c),
    "beta_instability": network.beta_instability(c),
    "corr_dispersion":  network.corr_dispersion(c),
}

HZ = (1, 5, 21)
K = len(FEATS) * len(HZ)
bar = sst.norm.ppf(1 - 0.05 / (2 * K))
print(f"\n{'='*84}")
print(f"{K} pre-specified tests -> Bonferroni bar |t| > {bar:.2f}")
print(f"{'='*84}\n")
print(f"{'feature':<18} {'h':>3} {'IC':>9} {'t':>8} {'n':>7}")

rows = []
for name, f in FEATS.items():
    for hz in HZ:
        ic = research.cross_sectional_ic(f, research.forward_returns(c, hz)).dropna()
        if len(ic) < 100:
            print(f"{name:<18} {hz:>3} {'--- insufficient overlap ---':>26}")
            continue
        m = float(ic.mean())
        t = (m / (ic.std(ddof=1) / np.sqrt(len(ic)))) / np.sqrt(hz)
        rows.append((name, hz, m, t))
        print(f"{name:<18} {hz:>3} {m:>9.4f} {t:>8.2f} {len(ic):>7}")

df = pd.DataFrame(rows, columns=["feature", "h", "ic", "t"])
surv = df[df.t.abs() > bar]
print(f"\n{'-'*84}")
print(f"survivors: {len(surv)}")
for _, r in surv.iterrows():
    print(f"   {r.feature:<18} h={int(r.h):<3} IC {r.ic:+.4f}  t {r.t:+.2f}")

print(f"\nlargest |t|: {df.t.abs().max():.2f} ({df.loc[df.t.abs().idxmax(),'feature']})")
exp_max = sst.norm.ppf(1 - 1 / (2 * K))
print(f"expected max |t| under the null with {K} tests: ~{exp_max:.2f}")

print(f"\n{'-'*84}")
print("IC needed to break even against the no-signal benchmark: ~0.050")
print(f"largest |IC| in this batch:                            {df.ic.abs().max():.4f}")
df.to_csv("relational_batch.csv", index=False)
