# Methodology Audit — 2026-09-05

Prompted by a simple question: *if this is all so exhaustive, how do any retail
algo traders exist?* The answer turned out to involve an error in how every
result in this project had been judged.

Related: the working state notes (unpublished), [[Strategy Spec v1]], [[Fundamental Law of Active Management]]

---

## 1. Tooling — what was actually running

**No standard library was in use.** Every number came from ~536 lines of
homegrown code: `algo/backtest.py`, `algo/metrics.py`, `algo/diagnostics.py`.
Not vectorbt, not backtrader, not zipline, not pyfolio, not alphalens.

That is defensible for a vectorised cross-sectional book — those frameworks are
mostly event-driven and heavier than needed — but it means nothing was ever
cross-checked. So it was, against `empyrical` (the library under pyfolio):

| metric | verdict |
|---|---|
| Sharpe | agrees to 3e-5 |
| max drawdown | agrees exactly |
| volatility | agrees to 4e-5 |
| CAGR | agrees exactly on a real equity curve |
| **Sortino** | **16% too high — genuine bug** |

### Bug 1 — Sortino (FIXED)
Computed as the std of the *negative subset*. The standard definition is the RMS
of the full series with upside clipped to zero. The subset version both drops
the zeros and centres on the negative mean instead of on zero. Now matches
`empyrical` to 2e-16. Never load-bearing in any conclusion — Sharpe and Calmar
were what got reported — but it was wrong.

### Bug 2 — turnover accounting (FIXED, immaterial)
```python
turnover = (weights.shift(1) - weights.shift(2)).abs().sum(axis=1)   # v1
```
Turnover was charged on changes in the **target**, while `held = weights.shift(1)`
assumed the book always sat exactly on target. Those are inconsistent: holding a
constant target requires trading as prices drift. Consequence — a constant-target
strategy (equal weight, i.e. the **no-signal benchmark**) paid *zero* cost while
being rebalanced for free.

`algo/backtest2.py` now tracks true drifting holdings:
```
w_drift[t]  = w_held[t-1] * (1 + r[t]) / (1 + portfolio_return[t])
turnover[t] = sum |w_target[t-1] - w_drift[t]|
```
**Measured impact: benchmark undercharged by 5.8bp/yr; Sharpe comparisons moved
by 0.003.** Real bug, no conclusion changes. `backtest.run` now delegates to v2;
the old engine survives as `run_legacy` for reproducing pre-audit numbers.

> Note on process: I announced this bug had "tilted every comparison" *before*
> measuring it. It had not. Measure first.

---

## 2. The framing errors — these were the real ones

### Error A — the benchmark bar was set at a historical anomaly
Every result was judged against a vol-targeted equal-weight equity book running
at **Sharpe ~1.05 over 2016-2026**. Long-run US equity Sharpe is ~0.4-0.5. That
demanded every signal beat one of the best equity decades on record, measured
over that same decade.

Extending the sample to 2005-2026 materially changed outcomes — e.g. the filings
sleeve raises both Sharpe (0.938 → 1.070 at 50% weight) and CAGR (9.70% →
10.80%), where the 2016-2026 window had shown it losing.

### Error B — standalone Sharpe was the wrong metric entirely
The stated goal was a **separate sleeve alongside an existing balanced
portfolio**. For that purpose standalone Sharpe is close to irrelevant. What
matters is **marginal contribution**, which is driven by **correlation** — and
correlation to the user's actual portfolio was never measured, not once, in
twelve signal families.

A Sharpe-0.6 sleeve uncorrelated with your core adds more than a Sharpe-0.9
sleeve that is 90% equity beta, because you already own equity beta.

### Error C — de-risking masquerading as skill
The subtlest one. A low-beta sleeve raises portfolio Sharpe simply by lowering
volatility — and **anyone can lower volatility for free by holding less equity**.
So a sleeve with beta *b* must be judged against `b*core + (1-b)*cash`, which
requires no strategy, no data, and no trading.

---

## 3. Re-evaluation under the corrected framework

Core = 60/40 SPY/AGG, monthly rebalanced, 2005-2026. Alpha t-stats use
Newey-West (21 lags) because these return series are autocorrelated by
construction (vol targeting, band rebalancing, overlapping signals).

| sleeve | stand. Sharpe | corr | beta | alpha%/yr | NW t | Δ Sharpe @10% |
|---|---|---|---|---|---|---|
| 30-name book (no signal) | 1.02 | 0.80 | 0.54 | 2.88% | 1.78 | +0.024 |
| large-cap EW (no signal) | 0.87 | 0.83 | 0.74 | 1.90% | 1.02 | +0.015 |
| price composite | 1.03 | 0.80 | 0.73 | 3.96% | **1.98** | +0.033 |
| filings (Lazy Prices) | 0.87 | 0.83 | 0.75 | 1.91% | 1.07 | +0.015 |
| **SPY (control)** | 0.81 | 0.98 | 1.56 | −0.07% | −0.07 | **−0.001** |

The SPY row validates the framework: adding SPY to a portfolio already 60% SPY
contributes exactly nothing, as it must.

### The de-risking control
| sleeve | beta | sleeve @10% | beta-matched @10% | **genuine gain** |
|---|---|---|---|---|
| 30-name book | 0.61 | 0.967 | 0.946 | **+0.022** |
| large-cap EW | 0.84 | 0.962 | 0.941 | +0.021 |
| price composite | 0.71 | 0.980 | 0.944 | **+0.036** |
| filings | 0.85 | 0.970 | 0.941 | +0.028 |
| SPY (control) | 1.61 | 0.928 | 0.927 | +0.001 |

---

## 4. Verdict

**Most of the apparent gain was de-risking**, but not all. A genuine residual of
**+0.02 to +0.036 Sharpe at a 10% allocation** survives the beta-matched control,
positive across all four sleeves.

Two reasons that is not a result:

1. **Correlation 0.80-0.85 — these are not diversifiers.** They are long-only
   equity and inherit equity beta. The uncorrelated sleeve the reframe was
   hoping to find is not here.
2. **Alpha t tops out at 1.98** against a search of ~200 cells across 12
   families. The Bonferroni bar for even 20 independent candidates is 2.87; for
   200 it is 3.48. Noise-consistent.

> The reframe was correct and it moved the answer from "clearly negative" to
> "small positive, not statistically distinguishable from the search." That is a
> real improvement in the conclusion, and it is the difference between "this was
> a waste" and "this produced a modest, defensible de-risking sleeve with a
> possible small alpha on top." It is not a reversal.

---

## 5. What changed in the code

- `algo/backtest2.py` — new engine, true holdings drift. `backtest.run` delegates
  to it; `backtest.run_legacy` preserved.
- `algo/metrics.py` — Sortino corrected.
- `algo/evaluation.py` — NEW. `alpha_beta` (Newey-West), `blend_curve`,
  `optimal_weight`, `evaluate`.
- `algo/diagnostics.py` — **check 6**: pass `core=` and every report now prints
  correlation, beta, alpha with NW t, Sharpe at 10% weight, and the
  beta-matched genuine gain. FAILs when genuine gain ≤ 0.
- `tests/test_evaluation.py` — 7 regression tests, including: a constant target
  must pay turnover; a sleeve that IS the core must add exactly zero; pure
  de-risking must score ~0 genuine gain while still raising naive Sharpe.

## 6. Standing rules from this audit

1. **Always pass `core=`.** Without it the report answers the wrong question.
2. **Match breadth, not just horizon**, when comparing IC to a target. The 0.036
   break-even came from a *daily-regenerating* forecast; a quarterly signal
   needs more.
3. **Beta-match before claiming a Sharpe gain.** De-risking is free.
4. **Measure before announcing a bug's impact.**
5. **A t-stat of 6 is not a business.** The price composite had t=5.98 on IC and
   still lost. Judge on the economic bar, not on significance.


---

# ADDENDUM — the null floor (same day, later)

**Section 4's verdict above is RETRACTED.** The "genuine residual of +0.02 to
+0.036 Sharpe surviving the beta-matched control" was itself an artifact. Code:
`retest_all.py`.

## What the retest did
Re-ran every family under the corrected framework, but ALSO ran **40 random
signals through the identical stack** — same universe, same top-30%, same
inverse-vol weighting, same 10% vol target, same costs, same rebalancing. A
random signal has zero information by construction, so its distribution IS the
noise floor.

```
NULL FLOOR (40 random signals)
  genuine gain @10%:  mean +0.0288   sd 0.0041   p95 +0.0349   max +0.0367
  alpha NW t:         mean  +3.14    sd 0.38     max  +3.95
```

> **A zero-information signal produces alpha with a Newey-West t of 3.14 and a
> "genuine gain" of +0.029.** The residual I reported in section 4 IS the noise
> floor. The |t| > 3 bar I proposed in the standing rules is cleared by pure
> noise, on average.

## Every family against that floor

| strategy | corr | beta | alpha%/yr | NW t | gain@10% | null pctile |
|---|---|---|---|---|---|---|
| 1 price composite | 0.78 | 0.67 | 2.64% | 2.02 | +0.0166 | **0%** |
| 12 PEAD (sue) | 0.81 | 0.72 | 2.98% | 2.35 | +0.0205 | **0%** |
| 7 per-name low-vol state | 0.56 | 0.38 | 0.85% | 0.64 | −0.0050 | 0% |
| 11 filings (jaccard) | 0.81 | 0.72 | 4.07% | 3.34 | +0.0301 | 65% |
| 9 low beta to universe | 0.70 | 0.58 | 4.43% | 3.25 | +0.0311 | 70% |
| **0 large-cap EW (NO SIGNAL)** | 0.83 | 0.74 | 4.25% | 3.59 | +0.0322 | **78%** |
| 0 30-name book (no signal) | 0.80 | 0.57 | 4.26% | 4.08 | +0.0300 | 60% |
| 9 idio vol share | 0.74 | 0.63 | 5.64% | 4.13 | +0.0427 | 100% |

**The price composite and PEAD score at the 0th percentile — WORSE than random
name selection.** The no-signal benchmark beats most of the actual signals. One
family clears p95 out of eight, which is what chance produces (p ≈ 0.34).

## Why the beta-matched control failed
It uses a **static, full-sample beta**. The stack applies **volatility
targeting**, which cuts equity exposure exactly when volatility spikes — exactly
when a 60/40 core crashes. So the sleeve's true beta is time-varying and
conditionally low when it matters most, and no static beta can represent that.
Over 2005-2026 that means partially dodging 2008.

**The vol-targeting overlay was generating the apparent alpha. Not the signals.**

## The corrected control
Hold the ENTIRE construction pipeline constant and vary only the information
content. `algo.evaluation.null_floor(build_returns, n)` takes a callable that
routes a random signal through your stack; `percentile_vs_null` reads the result.
Anything held constant in that callable is controlled for; anything not, is not.

`algo/diagnostics.py` now FAILs when genuine gain is inside the measured floor,
and FAILs on an alpha t below ~4.1 with an explicit note that noise averages
3.14. `tests/test_evaluation.py` pins the artifact directly: a zero-information
sleeve that merely de-risks in high-vol regimes must still beat its own
beta-matched control.

## Revised standing rules
1. **The null floor is the benchmark.** Not a static beta match, not a
   Sharpe-1.05 equity book, not |t| > 3. Run random signals through YOUR stack
   and report a percentile.
2. **Any overlay is part of the strategy.** Vol targeting, inverse-vol
   weighting, top-k truncation and band rebalancing all change the return
   distribution independently of the signal, and all must be inside the control.
3. **Calibrate bars empirically, never by convention.** "t > 3" is meaningless
   without knowing what noise scores in the same pipeline.
4. Rules 1-5 from section 6 above still hold, except the |t| > 3 bar, which is
   replaced by rule 1 here.

## Where this leaves the project
Twelve families, and under the only control that holds the pipeline constant,
**none beats random name selection**. The no-signal book at the 78th percentile
is the honest summary: the construction (diversify, weight by inverse vol,
target volatility, rebalance monthly) is doing all of the work, and every signal
we tested adds nothing to it.

That is a cleaner and more defensible conclusion than section 4's, and it was
reachable only because the null floor was run. It should have been the first
thing built, not the last.
