# Beta-neutral cross-section — result

Answers `notes/Beta-Neutral Pre-Registration.md`, which is left exactly as
written. Code: `research/beta_neutral.py`, `research/beta_neutral_stability.py`.
Data: `results/beta_neutral_n200.csv`.

## What was run

Three constructions on the random-forest signal, 2015–2026, each with its own
recalibrated null of 200 draws. Betas are daily/1-year/exponentially-weighted
with Vasicek shrinkage, aggregated on held weights. Costs: 2.2bp one-way, 50bp
GC borrow, no short rebate, margin interest on the debit balance, Reg T cap at
2:1 gross (which binds on no days after capping; gross runs 1.998).

| construction | Sharpe | β ex-ante | β realised | β FF6 | alpha t | Sharpe pctile | alpha-t pctile |
|---|---|---|---|---|---|---|---|
| dollar-neutral (reproduces `market_neutral.py`) | 0.735 | 0.255 | 0.528 | 0.426 | 2.191 | 99.5 | 100 |
| beta-neutral (A) | 0.560 | 0.001 | 0.232 | 0.126 | 1.957 | 99.0 | 100 |
| beta-orthogonal (B) | 0.394 | 0.003 | 0.008 | −0.006 | 0.935 | 99.0 | 99.0 |

## Scorecard against the registered predictions

| # | prediction | outcome |
|---|---|---|
| 1 | realised \|β\| < 0.15 but not zero | **wrong both ways.** A realised 0.232, well above the bound; B realised 0.008, effectively zero |
| 2 | Sharpe falls vs the 0.735 dollar-neutral | **held.** 0.735 → 0.560 → 0.394 |
| 3 | lands at or below the recalibrated null's p95 — a fail | **wrong.** All three are above p95; none clears the null max |
| 4 | B weaker than A | **held.** Sharpe 0.394 vs 0.560, alpha t 0.935 vs 1.957 |
| 5 | null floor lower than the long-only t = 2.75 | **held, decisively.** The alpha-t null falls to p95 ≈ −0.6, max ≈ 1.3 |

## The falsifier fired, and the falsifier was badly written

Registered: Sharpe ≥ null p95 **and** realised \|β\| < 0.10 **and** alpha t
clears its own null floor. Variant B meets all three on a literal reading —
Sharpe 0.394 vs p95 0.177, β 0.008, alpha t 0.935 at the 99th percentile of its
own null. Raising the null from 40 to 200 draws firmed this up rather than
dissolving it.

Two objections were considered and **rejected as unsound**:

- *"97.5th percentile is one draw."* Answered by rerunning at n = 200, where it
  became the 99th percentile. Not a resolution in the hoped-for direction.
- *"The null is cost-dragged, so beating it only means covering costs."* Wrong:
  the null books turn over at the same ~23×/yr and pay the same borrow, so the
  drag is common to both and cannot explain the gap.

What the criterion actually omitted was **persistence**. It asked whether the
residual was distinguishable from noise on one sample, which is the exact
question paper 1 shows is insufficient.

## What settles it

Splitting variant B by period:

| window | Sharpe | alpha (ann) | alpha t | realised β |
|---|---|---|---|---|
| 2015–2020 | 0.516 | **+3.16%** | 0.95 | 0.002 |
| 2021–2026 | 0.272 | **−1.23%** | **−0.34** | 0.016 |
| full | 0.394 | +2.34% | 0.94 | 0.008 |

The residual is entirely first-half and **changes sign in the second**. That is
the signature of the pairwise relative-value signal already retracted in the
whitepaper: significant on one window, absent on extension. The cross-sectional
question is closed. Any future criterion of this kind must require the effect to
hold in both halves before the percentile is even consulted.

## Two findings worth keeping

**Ex-ante neutrality is not realised neutrality, and the gap is regime-dependent.**
Variant A targets beta zero and achieves ex-ante 0.001, but realises 0.232 over
the full sample — and drifts from 0.088 in 2015–2020 to **0.415** in 2021–2026.
Scaling legs by estimated beta does not hold a book neutral. Only B, which
removes the tilt from the ranking rather than hedging its symptom, stays neutral
across both halves.

**The null floor collapses when beta is genuinely removed.** From t = +2.75 on
the long-only vol-targeted stack to a p95 near −0.6 and a max near 1.3 here.
This is the first direct test of the whitepaper's causal claim rather than its
diagnosis: the phantom alpha came from vol-targeting a time-varying beta, and a
book with no beta has none to time. It is the most transferable result of the
run, and it is about the method, not the signal.

## Still unsettled

Survivorship. The universe is current index membership carried backward, so the
short leg is measured on survivors and the best short candidates — the companies
that collapsed and left the index — are absent. The direction of that bias is
against the short leg, so it does not rescue the result above, but it means
these figures are a bound rather than a point estimate.
