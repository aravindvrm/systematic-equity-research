# Methodology Audit

How results in this project are validated, which controls proved necessary, and
what each one caught. Several conclusions reached earlier in the work were
overturned by controls added later; those reversals are documented here because
they are the reason the final numbers can be trusted.

Related: [[Evaluation Methodology - Literature]], [[Strategy Spec v1]],
[[Fundamental Law of Active Management]]

---

## 1. The control stack

Every result is evaluated against four controls, applied in this order. Each was
added because the preceding set proved insufficient.

| control | question it answers | implementation |
|---|---|---|
| **Reference metrics** | is the arithmetic correct? | cross-check against `empyrical` |
| **Factor attribution** | is this a known factor tilt? | Fama-French 6, Newey-West |
| **Conditional model** | is the exposure being timed? | Ferson-Schadt (1996) |
| **Null floor** | what does *no information* score here? | 40 random signals, same pipeline |

The fourth is the one that matters most, and it is the one most often missing
from strategy research.

---

## 2. Metric verification

Nothing in this project ran on a standard backtesting library. All performance
figures came from ~536 lines of local code, so the arithmetic was checked against
`empyrical` (the library underneath `pyfolio`).

| metric | result |
|---|---|
| Sharpe | agrees to 3e-5 |
| max drawdown | exact |
| volatility | agrees to 4e-5 |
| CAGR | exact on a real equity curve |
| **Sortino** | **16% high — defect found** |

**Sortino** had been implemented as the standard deviation of the negative return
subset. The standard definition is the RMS of the full series with upside clipped
to zero; the subset version both drops the zeros and centres on the negative mean.
Corrected, it now matches to 2e-16. It was never load-bearing — Sharpe and Calmar
were the reported metrics — but it was wrong.

**Turnover accounting** was charged on changes in the *target* weights while
`held = weights.shift(1)` assumed the book always sat exactly on target. Those
assumptions are mutually inconsistent: holding a constant target requires trading
as prices drift. The effect was that a constant-target strategy — the no-signal
benchmark — paid zero cost while being rebalanced for free.

`algo/backtest2.py` now tracks true drifting holdings:

```
w_drift[t]  = w_held[t-1] * (1 + r[t]) / (1 + portfolio_return[t])
turnover[t] = sum |w_target[t-1] - w_drift[t]|
```

Measured impact: the benchmark was undercharged by 5.8bp/yr, moving Sharpe
comparisons by 0.003. A real defect with an immaterial effect — which is only
knowable by measuring it rather than reasoning about it.

---

## 3. Why standalone Sharpe is the wrong question

The original evaluation judged every strategy on standalone Sharpe against a
vol-targeted equity benchmark. Three problems, in increasing order of severity.

**The bar sat at a historical anomaly.** That benchmark ran at Sharpe ~1.05 over
2016–2026. Long-run US equity Sharpe is 0.4–0.5. The test demanded that every
signal beat one of the best equity decades on record, measured over that same
decade. Extending to 2005–2026 materially changed outcomes.

**The question did not match the use case.** The objective was a sleeve held
*alongside* an existing portfolio. For that purpose what matters is marginal
contribution, which is driven by correlation — a quantity never measured across
twelve signal families. A Sharpe-0.6 sleeve uncorrelated with the core adds more
than a Sharpe-0.9 sleeve that is 90% equity beta.

**De-risking reads as skill.** A low-beta sleeve raises portfolio Sharpe simply by
lowering volatility, and anyone can lower volatility for free by holding less
equity. A sleeve with beta *b* must therefore be judged against
`b*core + (1-b)*cash`, which requires no strategy at all.

---

## 4. Why the beta-matched control is still insufficient

The beta-matched comparison above uses a **static, full-sample beta**. The stack
applies **volatility targeting**, which cuts market exposure when volatility rises
— which is when the benchmark crashes. The book's true beta is therefore
time-varying and conditionally low precisely when it matters, and no constant beta
can represent that. The unmodelled timing surfaces as alpha.

This is the bias Ferson & Schadt described in 1996. Its magnitude here was
established by running signals containing **no information at all** through the
identical pipeline:

```
40 zero-information signals, same universe / selection / weighting / target / costs

  vs a 60/40 core       "genuine gain" +0.029 mean,  alpha NW t = 3.14
  vs Fama-French 6      alpha t = 2.61 mean,  p95 3.29,  max 3.66
  vs Ferson-Schadt      alpha t = 2.58 mean,  p95 3.54
```

**A signal containing nothing clears `|t| > 2` comfortably and reaches
Harvey-Liu-Zhu's `|t| > 3` roughly one time in twenty.**

The conditional model only partially helps, because a single linear interaction
cannot absorb a vol-target rule that is non-linear and capped at 1.0 leverage.

Direct confirmation of the mechanism: moving the sample start from 2005 to 2015
collapses the null floor from t = 2.75 to t = 0.77. Remove the 2008 crash and
there is nothing left for the overlay to dodge.

---

## 5. Results under the full stack

38 strategies spanning every family, one shared pipeline so a single null floor
applies. Ranked by percentile against that floor under the conditional model.
Full table in `results/reassess_all.csv`.

| strategy | Sharpe | alpha%/yr | cond. t | null pctile | R² vs FF6 |
|---|---|---|---|---|---|
| **no-signal book** | 1.02 | 3.31% | 3.59 | **98%** | 0.75 |
| relative strength 126d | 1.02 | 4.27% | 3.41 | 92% | 0.67 |
| filing-text similarity | 0.99 | 3.20% | 3.00 | 62% | 0.71 |
| 7-signal composite | 0.82 | 2.38% | 1.63 | 2% | 0.67 |
| post-earnings drift | 0.88 | 1.95% | 1.42 | 0% | 0.72 |
| insider net buying | 0.38 | −1.31% | −1.18 | 0% | 0.27 |

- **0 of 38 clear the conditional null floor.** Two cleared unconditionally
  against a chance expectation of 1.9; neither survived once beta could vary.
- **The no-signal book ranks first of thirty-eight.**
- R² of 0.66–0.75 for most rows: two thirds to three quarters of what was built
  is known factor exposure purchasable in an ETF.
- Three strategies clear Harvey-Liu-Zhu's `|t| > 3` — **two of them are the
  no-signal books**, which is the clearest available demonstration that a
  conventional bar is meaningless without pipeline-specific calibration.

---

## 6. Conclusions superseded by these controls

Recorded because each was stated confidently before the control that overturned
it existed, and because the pattern — reporting a result when it becomes
interesting rather than when it becomes controlled — is the failure mode these
controls exist to prevent.

| superseded claim | what the control showed |
|---|---|
| A pairwise relative-value signal survives every structural check at t = 2.45 | Extending 7 → 20 years dropped t to 1.28 where a real effect predicts 4.1. Sample-specific; every supporting result reversed. |
| A genuine residual of +0.02 to +0.036 Sharpe survives the beta-matched control | That range *is* the noise floor. Random signals score +0.029 on the same control. |
| Small caps fail because a 12× spread penalty cancels their larger premia | Corwin-Schultz and Abdi-Ranaldo measure the ratio at 2.0×. The 12× figure was assumed, never measured. Small caps still fail — from concentration cost in a high-idiosyncratic-vol universe, and a benchmark at Sharpe 0.11. |
| A turnover defect tilted every comparison in the project | Real defect, measured at 5.8bp/yr and 0.003 Sharpe. Immaterial. The claim preceded the measurement. |

---

## 7. Standing rules

1. **Calibrate every significance bar against the actual pipeline.** Never quote
   `|t| > 2` or `> 3` from convention. Run zero-information signals through the
   identical stack and report a percentile.
2. **Every overlay belongs inside the control.** Volatility targeting, inverse-vol
   weighting, top-k truncation and band rebalancing each change the return
   distribution independently of the signal.
3. **Recalibrate the null floor for every sample window.** It moved from t = 2.75
   to 0.77 purely by changing the start date.
4. **Run factor attribution first.** Alpha that vanishes against known factors is
   a tilt that can be bought in an ETF.
5. **Use a conditional model when exposure is timed**, and know it absorbs only
   the linear component.
6. **Match breadth, not just horizon,** when comparing IC to a target. A
   break-even derived from a daily-regenerating forecast does not apply to a
   quarterly signal.
7. **Beta-match before claiming a Sharpe gain.** De-risking is free.
8. **Measure a defect's impact before characterising it.**
9. **A t-statistic of 6 is not a business.** The signal composite reached exactly
   that and still lost to holding everything.

---

## 8. Implementation

- `algo/backtest2.py` — engine with true holdings drift; `backtest.run` delegates
  to it, `run_legacy` preserved for reproducing pre-audit figures.
- `algo/evaluation.py` — `null_floor`, `percentile_vs_null`, `alpha_beta`
  (Newey-West), `blend_curve`, `optimal_weight`.
- `algo/factors.py` — Fama-French 6 and Ferson-Schadt conditional attribution.
- `algo/diagnostics.py` — check 6 runs the marginal and null-floor tests on every
  result and fails loudly when a figure sits inside the measured floor.
- `tests/test_evaluation.py` — regression tests pinning each defect above,
  including one that constructs a zero-information volatility-timing sleeve and
  asserts it still beats a naive beta-matched control.
