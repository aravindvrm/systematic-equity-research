# Evaluation Methodology — the literature I should have read first

Written 2026-09-05 after being correctly told off: *"framework, methodology is
the kind of thing that would be there, not signals."*

That is exactly right, and I had dismissed it. Signals are the part the
literature **cannot** help with — nobody publishes a live edge. Methodology is
the part it documents thoroughly, for free, and has for decades. I spent two
days rediscovering, badly and late, results that are named and published.

Related: [[Methodology Audit]], the working state notes (unpublished), [[Strategy Spec v1]]

---

## What I built by accident, and what it is actually called

| what I did | what it is | reference |
|---|---|---|
| ran 40 random signals through the stack | **White's Reality Check** / Hansen's SPA test — bootstrap null for data-snooping | White (2000), Hansen (2005) |
| worried about search across 200 cells | **multiple testing in asset pricing**; the |t|>2 convention is far too lenient, |t|>3 argued as a floor | Harvey, Liu & Zhu (2016, RFS) |
| haircut Sharpe for trials | **Deflated Sharpe Ratio** (I did use this) | Bailey & López de Prado |
| never did it | **Probability of Backtest Overfitting** via CSCV | Bailey, Borwein, López de Prado & Zhu |
| never did it | **purged k-fold CV with embargo** | López de Prado, *Advances in Financial ML* |
| never did it until today | **Fama-French factor attribution** — the single most standard control in the field | Fama & French (2015), Carhart (1997) |
| hit it and could not name it | **conditional performance evaluation** — unconditional alpha is BIASED when beta varies with public information | **Ferson & Schadt (1996)** |

The last row is the bug that broke this project's evaluation, published thirty
years ago, with a named fix.

---

## The core problem, stated properly

**Volatility targeting makes beta vary with public information.** It cuts market
exposure when trailing volatility is high, and trailing volatility is public. An
unconditional constant-beta model cannot represent a strategy that times its own
exposure, so the unmodelled timing appears as alpha.

Measured here — 40 **zero-information** signals through a top-30%, inverse-vol,
10%-vol-targeted stack:

```
vs a 60/40 core       : "genuine gain" +0.029 mean,  alpha NW t = 3.14 mean
vs Fama-French 6      : alpha t = 2.61 mean, p95 3.29, max 3.66
vs Ferson-Schadt cond.: alpha t = 2.58 mean, p95 3.54
```

> **A signal with zero information clears |t| > 2 comfortably and reaches
> Harvey-Liu-Zhu's |t| > 3 bar roughly one time in twenty.** Any bar quoted as a
> convention, rather than calibrated against the actual pipeline, is worthless.

Ferson-Schadt only partially helps because its single linear interaction cannot
absorb a vol-target rule that is nonlinear and capped at 1.0 leverage.

---

## Final results, all controls applied

FF6 attribution, 2005-2026, 199 names, IBKR costs:

| strategy | alpha%/yr | uncond t | cond t | timing b | timing t | null pctile |
|---|---|---|---|---|---|---|
| price composite | 2.07% | 1.72 | 1.42 | −0.105 | −12.00 | **0%** |
| PEAD (sue) | 1.95% | 1.72 | 1.42 | −0.117 | −11.12 | **0%** |
| low beta to universe | 2.69% | 2.11 | 1.79 | −0.073 | −14.26 | 8% |
| filings (jaccard) | 3.20% | 2.93 | 3.00 | −0.113 | −10.39 | 82% |
| idio vol share | 4.12% | 3.20 | 2.98 | −0.083 | −10.81 | 82% |
| 30-name book (NO SIGNAL) | 3.36% | 3.14 | 2.81 | −0.054 | −18.01 | 72% |
| **large-cap EW (NO SIGNAL)** | 3.31% | **3.24** | **3.59** | −0.121 | −10.95 | **95%** |

- **3 of 7 clear Harvey-Liu-Zhu |t| > 3 — and two are the NO-SIGNAL books.**
- **0 of 7 clear the unconditional null floor.**
- The only thing clearing the CONDITIONAL null is the no-signal book.
- Timing loadings are −10 to −18 t for everything: the vol-target overlay is the
  dominant term in every strategy.
- R² against FF6 runs 0.55-0.75, so most of what these do is known factor
  exposure buyable through cheap ETFs.

---

## What the alpha actually is

Essentially all of it is the **volatility-targeting overlay**, not stock
selection. That is not nothing — volatility management is itself a documented
effect (Moreira & Muir 2017, *JF*, "Volatility-Managed Portfolios"). But:

1. It is **not our signal.** The no-signal book captures it just as well.
2. It is **free and public** — a published technique, not an edge.
3. It is **contested.** Later work (e.g. Cederburg et al.) finds the effect does
   not survive real-time implementation and out-of-sample testing as cleanly as
   originally claimed. Do not bank on it.

---

## Standing rules (final)

1. **Calibrate every bar empirically against your own pipeline.** Never quote
   |t| > 2 or > 3 from convention. Run zero-information signals through the
   identical stack and report a percentile.
2. **Any overlay is part of the strategy and must be inside the control.** Vol
   targeting, inverse-vol weighting, top-k truncation, band rebalancing.
3. **Run FF6 attribution before anything else.** Alpha that vanishes against
   known factors is a tilt you could have bought.
4. **Use the conditional model when exposure is timed** (Ferson-Schadt), and
   know it only absorbs the linear part.
5. **Match breadth, not just horizon,** when comparing IC to a target.
6. **Measure before announcing a bug's impact.**
7. **Read the methods literature first.** It cannot give you signals. It can
   stop you believing in ones you do not have — which, on this evidence, is the
   more valuable of the two.


---

# FULL REASSESSMENT — all twelve families (2026-09-05)

Code: `reassess_all.py`, results `reassess_all.csv`. 38 comparable strategies
spanning every family, one shared stack (199 large caps, top-30%, inverse-vol,
10% vol target, monthly, IBKR costs) so a single null floor is valid for all.

```
NULL FLOOR (n=40):  uncond alpha t  mean +2.75  sd 0.36  p95 +3.27
                    conditional     mean +2.77  sd 0.49  p95 +3.47

clearing null p95 UNCONDITIONAL: 2
clearing null p95 CONDITIONAL:   0
expected by chance at 5%:        1.9
```

| strategy | Sharpe | alpha%/yr | cond t | cond pctile | R2 |
|---|---|---|---|---|---|
| 0 NO-SIGNAL large-cap EW | 1.02 | 3.31% | 3.59 | 98% | 0.75 |
| 9 rel:rel_strength_126 | 1.02 | 4.27% | 3.41 | 92% | 0.67 |
| 9 net:down_corr_asym | 1.06 | 3.78% | 3.35 | 92% | 0.70 |
| 1 px:vol_trend | 1.01 | 3.53% | 3.19 | 70% | 0.72 |
| 9 net:corr_dispersion | 1.06 | 3.79% | 3.14 | 65% | 0.66 |
| 9 rel:corr_to_univ | 1.11 | 4.12% | 2.98 | 62% | 0.59 |
| 9 net:net_centrality | 1.09 | 3.99% | 3.00 | 62% | 0.60 |
| 9 rel:idio_vol_share | 1.11 | 4.12% | 2.98 | 62% | 0.59 |
| ... | | | | | |
| 12 pead:sue | 0.88 | 1.95% | 1.42 | 0% | 0.72 |
| 9 rel:lead_lag | 0.88 | 2.20% | 1.45 | 0% | 0.67 |
| 1 px:reversal_5 | 0.85 | 2.09% | 1.41 | 0% | 0.66 |
| 7 state:own_trend | 0.82 | 2.71% | 1.39 | 0% | 0.29 |
| 7 state:own_corr | 0.69 | 1.46% | 0.55 | 0% | 0.29 |
| 7 state:own_drawdown | 0.65 | 1.41% | 0.43 | 0% | 0.28 |

**The NO-SIGNAL book ranks #1 of 38 on the conditional percentile.** Two
strategies clear the unconditional null against a chance expectation of 1.9, and
both fail once beta is allowed to vary with public information.

## Three findings worth keeping

1. **The composite is WORSE than its components.** `px:COMPOSITE` sits at the 2nd
   percentile while `px:vol_trend` reaches the 70th. Averaging seven signals of
   which three are 97%-correlated momentum variants averages one signal with
   itself and adds noise from the rest. That is a defect in the CONSTRUCTION, not
   in the signals, and it was invisible until every feature was run individually.
2. **Insider signals are actively negative** — alpha -0.85% to -1.31%, Sharpe
   0.38-0.46, worst of anything tested, consistent across all four variants.
   (2018+ sample; wider error bars.)
3. **R2 vs FF6 runs 0.66-0.75 for most strategies** -- two thirds to three
   quarters of everything built here is known factor exposure purchasable in an
   ETF. The per-name state strategies are the exception at R2 ~0.29: genuinely
   orthogonal to the factors, and among the lowest alphas. Orthogonal AND
   worthless.

## Conclusion

Twelve families, 38 strategies, ~200 signal cells. Under a null calibrated to our
own pipeline, **nothing beats a random signal, and a no-signal portfolio beats
everything.** This is a properly controlled result, not a series of individually
deflated nulls.


---

# TAXONOMY GAP CLOSED — value, profitability, investment (2026-09-05)

Prompted by: *"is the strategy taxonomy you're using also referenced from
literature? Have overlooked approaches?"* — and it was not. My twelve "families"
were improvised as I went, not a principled decomposition.

Checked against the standard taxonomy used across the replication literature
(Hou/Xue/Zhang), which has six categories, our coverage had been:

| category | before | after |
|---|---|---|
| momentum | tested | tested |
| trading frictions | tested | tested |
| **value-vs-growth** | **NOT TESTED** | tested |
| **profitability** | **NOT TESTED** | tested |
| **investment** | **NOT TESTED** | tested |
| intangibles | partial | partial |

**Three of six untested — including three of the five Fama-French factors
(HML, CMA, RMW).** The reason was mundane: they need balance-sheet and
income-statement data that was never ingested, though it is free from the same
DERA source already used twice.

## Data
`algo/fundamentals.py` + `algo/anomalies.py`, `ingest_fundamentals.py`.
SEC Financial Statement Data Sets, 69 of 70 quarters, **15,038 filings, 248
companies, 2009-04-15..2026-06-29**, 19MB after filtering (raw is ~630MB/quarter,
~36GB total, discarded).

2009q1 legitimately returns nothing — it predates the XBRL mandate (fiscal
periods ending after 2009-06-15). The ingest script exited non-zero rather than
report success on a partial run; that guard worked as designed.

## Results — 8 anomalies x 3 horizons, plus portfolio level vs a null floor
recalibrated on this shorter sample (uncond mean +2.15, p95 +2.71).

| anomaly | IC(h=126) | Sharpe | alpha%/yr | cond t | null pctile | R2 |
|---|---|---|---|---|---|---|
| **book_to_market** | **0.0448** | 1.13 | 3.73% | 2.96 | **85%** | 0.65 |
| gross_profit | 0.0188 | 1.16 | 3.19% | 2.66 | 65% | 0.65 |
| earnings_yield | 0.0167 | 1.11 | 3.28% | 2.64 | 65% | 0.64 |
| leverage | 0.0221 | 1.02 | 2.12% | 1.80 | 10% | 0.71 |
| share_issuance | 0.0054 | 1.03 | 2.26% | 1.75 | 8% | 0.64 |
| roe | -0.0190 | 1.04 | 1.83% | 1.55 | 2% | 0.68 |
| roa | -0.0176 | 1.03 | 1.71% | 1.55 | 2% | 0.70 |
| asset_growth | -0.0079 | 0.92 | 1.30% | 0.80 | 0% | 0.61 |

**IC survivors: 0 of 24. Clearing the null p95: 0 of 8.**

## What it shows
1. **Value is the strongest single signal in the entire project.** Book-to-market
   IC 0.0448 at h=126 beats every price feature, the filings signal, and
   everything relational. Value > earnings yield > profitability is the ordering
   the literature predicts, and every sign is correct — reassuring about the
   pipeline.
2. **It still does not clear.** 85th percentile against a 95th-percentile bar.
3. **A clean pass would not have been an edge anyway.** Book-to-market IS value;
   IWD and VTV sell it for 5-20bp. R2 of 0.65 vs FF6 confirms most of the
   strategy is simply HML exposure.
4. Do NOT repeat the filings error of reading IC 0.0448 > the 0.036 target as
   success: at h=126 on a quarterly-updating signal the breadth is low, and the
   portfolio test is what settles it.

## Remaining genuine hole
**Accruals (Sloan 1996)** is untestable here: `cfo` coverage came in at 16%.
`cogs` at 61% also means gross profitability is measured on a biased subset of
filers who tag cost of revenue separately. These are data gaps, not nulls, and
should not be counted as tested.

## Also still untested (named, not silently dropped)
- **Non-linear methods.** Gu, Kelly & Xiu (2020) show trees and neural nets
  roughly double out-of-sample R2 over linear methods on the same characteristic
  set. Everything here was a linear z-score composite — and that composite was
  measurably worse than its own components.
- **Event-driven beyond earnings**: index add/delete, spinoffs, M&A, buyback
  announcements, lockup expirations. Index rebalancing is the cleanest CONSTRAINT
  asymmetry available — index funds must trade regardless of price.
- **The Chen & Zimmermann open-source predictor set** (~200 published predictors
  with code, openassetpricing.com) — the systematic version of this whole
  exercise.


---

# SPREAD MEASUREMENT + NON-LINEAR METHODS (2026-09-05)

## 1. The small-cap cost assumption was invented — and wrong
The conclusion "small caps do not help because wider spreads cancel the larger
premia" rested on spread numbers I ASSUMED and never measured.

Corwin & Schultz (2012) and Abdi & Ranaldo (2017), run on actual OHLC
(`algo/spreads.py`, `measure_spreads.py`):

| | I ASSUMED | MEASURED (CS / AR) |
|---|---|---|
| large cap | 1.0bp | 15.9 / 8.3 -> ~12bp |
| small cap | 12.0bp | 34.2 / 14.7 -> ~24bp |
| **ratio** | **12x** | **2.0x** (1.6x for >$20M/day) |

Absolute levels are NOT trustworthy — both estimators are biased upward and
disagree by ~2x — but they agree on the RATIO, which is the load-bearing number.

### The conclusion survives; the reasoning was wrong
Recomputed break-even with measured cost ratios (`breakeven_v2.py`):

| universe | cost | RT bp | bench Sharpe | break-even IC |
|---|---|---|---|---|
| large cap | IBKR | 4 | 1.07 | 0.0359 |
| small cap (all) | 12bp ASSUMED | 37 | 0.48 | 0.0958 |
| small cap (all) | 2.0x measured | 7 | 0.53 | **>0.20** |
| small cap (>$20M/day) | 1.6x measured | 6 | **0.11** | 0.0245 |

- All small caps with CORRECT (lower) costs: **no IC clears at all.** Not a cost
  effect — costs FELL. It is a CONCENTRATION effect: cutting 345 names to the top
  103 in a high-idiosyncratic-vol universe loses more to diversification than any
  forecast recovers.
- Liquid small caps show a LOW break-even (0.0245) only because the benchmark is
  terrible (Sharpe 0.11). **Break-even IC in isolation is a misleading statistic**
  — it must be read alongside the absolute Sharpe it buys.

> Right answer, wrong reasoning, for two days. The error was self-serving: an
> invented number that happened to support a conclusion already drawn.

**Also**: measured large-cap spread ~12bp vs the 1.0bp in our cost model. Even
discounting the upward bias, 2.2bp one-way is probably optimistic — and every
backtest in this project used it.

## 2. Non-linear methods (Gu, Kelly & Xiu 2020)
Best-motivated remaining test, because we had direct evidence the COMBINATION
method was destroying information: the equal-weighted 7-signal composite ranked
2nd percentile while single features reached the 70th.

Walk-forward (train < Y-1, validate Y-1, test Y, roll annually), 35 features
including fundamentals, monthly non-overlapping observations.

| model | OOS IC | t | portfolio Sharpe | uncond pctile | cond pctile |
|---|---|---|---|---|---|
| equal-wt composite | — | — | 1.00 | 72% | 52% |
| ridge (linear) | 0.0186 | 1.26 | 0.83 | 30% | 18% |
| hist grad boost | 0.0200 | 1.51 | 0.89 | 85% | 72% |
| **random forest** | **0.0240** | 1.69 | 0.92 | **98%** | 88% |

- Trees beat linear by **+29% IC**, not the ~2x GKX report.
- **Random forest is the best result in the project** — 98th percentile
  unconditional, the first clean clear. But it FAILS conditionally (88% vs the
  95% bar), it is 1 of 4 models tried (max-of-4 lands near p88 by chance), and
  the OOS sample is ~11 years.
- **Ridge came LAST, below equal-weighting.** So it is not "fitting beats not
  fitting" — ridge overfits and generalises badly; trees regularise better.

## 3. A confirmation worth noting
The null floor **collapsed from t=2.75 (2005+) to t=0.77 (2015+)**, purely from
changing the start date. That is direct confirmation of the Ferson-Schadt
diagnosis: the fake alpha came from the vol-target overlay dodging 2008, and
removing 2008 removes most of it.

**Consequence**: the null floor must be recalibrated for EVERY sample window. A
floor computed on one period does not transfer to another.


---

# THE RISK PREMIA LITERATURE — the body of work we never consulted (2026-09-05)

Prompted by: *"'Publication doesn't compete it away' — have we looked at
literature already to identify these comprehensively, instead of stumbling upon
them?"* No. We systematically mined the ANOMALY literature (Hou/Xue/Zhang,
Chen/Zimmermann) and never touched the RISK PREMIA literature, which is a
separate body of work and the one that actually addresses persistence.

## The taxonomy (Ilmanen; Roncalli, "Alternative Risk Premia: What Do We Know?")

A pyramid of return sources:
- **Base — traditional market premia**: equity, term (duration), credit.
- **Middle — style premia**: value, momentum/trend, carry, defensive/quality,
  volatility.
- **Top — proprietary alpha**: what twelve families failed to find, which is the
  correct outcome for that layer.

| premium | status here |
|---|---|
| equity / term / credit | user already owns via balanced portfolio |
| **volatility (VRP)** | the wheel — **persistent, 84% positive over 30yr, no decay** |
| value | tested; 85th pctile long-only; needs shorts for the real premium |
| defensive / BAB | tested; INVERTED in the 2009-2026 sample |
| carry | not accessible without futures/FX |
| **trend / time-series momentum** | **was never tested — see below** |

## Honest correction
I stated "publication doesn't compete it away" as a clean rule. Cochrane's
*Discount Rates* shows it is blurrier: behavioural theories ARE discount-rate
theories, since a distorted probability with risk-free discounting is
mathematically equivalent to a different discount rate. The practical test that
survives is: **does someone need to keep taking the other side for a structural
reason?** For the VRP, yes — hedging demand is permanent.

## Volatility risk premium (the wheel), measured
CBOE benchmark indices, 1996-2026 (`wheel_analysis.py`):

| | CAGR | vol | Sharpe | MaxDD | ex kurt |
|---|---|---|---|---|---|
| BXM covered call | 7.27% | 14.01% | 0.57 | -40.1% | 29.3 |
| PUT cash-sec put | 8.48% | 15.24% | 0.61 | -37.1% | **356.2** |
| SPX (price only) | 8.45% | 19.11% | 0.52 | -56.8% | 9.9 |

- **VRP has NOT decayed**: 4.85 / 3.34 / 3.16 / 3.77 pp across four eras, 81-86%
  positive throughout. Unlike every anomaly tested.
- **But the mechanical indices do not beat buy-and-hold.** Add ~1.8%/yr dividends
  to SPX and it returns ~10.25% vs PUT's 8.48%.
- Crisis: much better (GFC -34.8% vs -56.8%); recovery: much worse (+188% vs
  +349%). A risk transformation, not free money.
- As a sleeve beside 60/40 both have NEGATIVE alpha (BXM -1.67%, PUT -0.43%),
  corr 0.77-0.86. Equity exposure with the upside capped.
- Single-name wheeling likely harvests LESS premium than these indices: the index
  VRP includes a correlation risk premium that single names lack.

## Time-series trend — the gap, and the first thing to clear a null floor
`trend_test.py`. 15 multi-asset ETFs, 2007-2026, risk-parity sized, monthly.

| variant | Sharpe | CAGR | MaxDD | corr to core | delta@20% | null pctile |
|---|---|---|---|---|---|---|
| **long/short (margin)** | 0.40 | 1.67% | -13.6% | **-0.15** | **+0.0435** | **100th** |
| long-only (cash) | 0.56 | 3.13% | -21.1% | 0.29 | +0.0367 | 95th (marginal) |
| equal-weight buy&hold | 0.59 | 6.19% | -34.6% | — | — | — |

Crisis: GFC SPY -55.2%, trend L/S **+6.2%**; 2022 SPY -24.5%, trend L/S **+9.3%**.

**Null floor (random signs, identical construction, 40 draws):**
- long/short delta: null mean -0.0056, p95 +0.0213, **max +0.0241** vs real
  **+0.0435** — past the top of 40 draws, nearly double the null max.
- long-only Sharpe: null mean +0.562, real +0.562 — **48th percentile, the null
  MEDIAN.** Clipping shorts turns it into a long-biased multi-asset book that
  random signs replicate exactly.

> **This answers the margin question.** Market-neutral equity was rightly ruled
> out (Sharpe 0.735 vs null max 0.738, and beta 0.53 — not even neutral). But
> trend is a different case: the short leg is where the crisis alpha lives, and
> the cash-account version does not have it.

### Caveats
- Standalone Sharpe 0.40; held for the -0.15 correlation, not for returns.
- ETF proxy for ~50 futures markets — a LOWER BOUND on the premium.
- 2010-2020 was a bad decade for CTAs and is in the sample.
- **FF6 alpha 0.26%, t=0.26** — essentially zero. The value is entirely the
  negative correlation, which is not guaranteed to persist.
- This is family ~13. The null controls for construction, not for having tried
  twelve things first.

## The meta-lesson, again
Two days were spent on the ALPHA layer, which the literature predicts is where
retail loses. The premia layer — which the same literature says persists, and
which is where the user was ALREADY profitable by hand — was never examined until
prompted. **Read the taxonomy before searching the space.**
