# What Does Nothing Score?

*Every backtesting library verifies that your code runs. None of them tell you what a worthless strategy scores in your pipeline. That number is often large.*

---

I spent a few days testing whether a retail trader can find tradeable signal in US equities. Twelve families of ideas — momentum, mean reversion, insider filings, institutional ownership, filing-text similarity, post-earnings drift, machine learning on all of it. Thirty-eight strategies in total.

Partway through I built a composite of the seven signals that had survived every check. It produced an information coefficient with a **t-statistic of 5.98**.

For anyone who hasn't spent time with these: a t of 6 is not borderline. In most of empirical finance it is a career-making number. Published factors clear 2. Harvey, Liu and Zhu argued in 2016 that given how many factors have been mined, new ones should be held to 3.

Mine was 6. And it lost to holding everything.

That gap — overwhelming statistical significance, negative economic value — is where this post lives.

## The question I hadn't asked

I had been careful in all the conventional ways. Deflated Sharpe ratios for multiple testing. Out-of-sample splits with signs fixed on the training half. Newey-West standard errors for autocorrelation. Bonferroni bars declared before running. Every deflation the literature recommends, applied honestly.

What I never did, for about eleven of those twelve families, was ask the most basic question available:

**What does a signal containing no information score on these same tests?**

Not a shuffled version of my signal. Not a placebo drawn from a different distribution. Literally random numbers, routed through the *identical* pipeline — same universe, same selection rule, same weighting, same volatility target, same transaction costs, same rebalance schedule. Vary only the information content and hold everything else fixed.

When I finally ran it, forty times, the answer was:

```
Zero-information signals, identical pipeline, n = 40

  Fama-French 6 alpha, Newey-West t     mean +2.75   p95 +3.27
  Ferson-Schadt conditional t           mean +2.77   p95 +3.47
  "genuine gain" vs a beta-matched core mean +0.029
```

A signal containing **nothing** clears the conventional `|t| > 2` bar comfortably, and reaches the Harvey-Liu-Zhu `|t| > 3` threshold roughly one time in twenty.

Every significance test I had run was uncalibrated. Not wrong in its arithmetic — uncalibrated. I was comparing my results against a threshold from a textbook rather than against what my own machinery produced from noise.

## Where I thought the phantom alpha came from

The cause looked like a single line in my portfolio construction that I had never thought of as a modelling choice: **volatility targeting.**

The rule is unremarkable. Estimate the portfolio's trailing volatility, and if it exceeds your target, scale the whole book down. Almost every systematic strategy does something like it. It is risk management, not forecasting.

But it means market exposure falls when volatility rises — which is precisely when the market crashes. The strategy's beta is therefore *time-varying, and conditionally low exactly when being low pays most.* A factor model with constant betas cannot represent that. The unmodelled timing has to go somewhere, and where it goes is the intercept.

The intercept is alpha.

Here is the same portfolio — a randomly chosen 30% of the universe, no forecasting whatsoever — with and without the overlay:

```
Maximum drawdown, identical zero-information signal

  without volatility targeting    -37.9%
  with volatility targeting       -20.0%
```

Half the 2008 loss avoided, by a strategy that knows nothing. Run that through a constant-beta model and it reads as skill.

None of this is new. Ferson and Schadt described the bias in 1996 and proposed a conditional model to correct it. I rediscovered it by accident, thirty years late, and only because I had run the null.

It is also not the answer.

## Where it actually comes from

Someone reviewing this work asked the obvious question I had never asked: what does the floor look like with the overlay *switched off*? Same universe, same ranking, same weighting, same schedule, same costs — two hundred null draws in each arm.

```
Alpha-t null floor, identical stack, 200 draws per arm

  with 10% volatility target      +2.66
  no overlay at all               +2.94
```

Taking the overlay away makes the floor *worse*. Whatever was manufacturing the phantom alpha, it was not the thing I had spent the project blaming — and the thing I was blaming had been quietly reducing it.

So I stripped everything out. No selection, no overlay, no forecast: hold every name in the universe and see what a factor model says.

```
FF6 alpha, holding everything, predicting nothing

  whole universe, equal weight     +4.53%   t = +5.79
  SPY, same window                 -0.40%   t = -1.77
```

There it is. My universe is the index membership of *today*, carried back to 2005. The companies that went bankrupt and dropped out are not in it. A portfolio holding all of them and forecasting nothing beats a six-factor model by four and a half points a year, while the actual market over the same window returns a negative alpha. That gap is survivorship bias.

It had been sitting in my caveats section the whole time — listed as a limitation, the way everyone lists it, rather than measured as an effect. Any long book drawn from this universe inherits it in proportion to how much market exposure it carries. Volatility targeting cuts that exposure, taking beta from 0.90 to 0.48, which is exactly why removing the overlay raises the floor.

I had the right mechanism and the wrong cause. Ferson-Schadt is real, it is present in this data, and it is second-order.

## The part I got right for the wrong reason

I had already re-run the null floor on 2015–2026 instead of 2005–2026 — same universe, same everything, just a later start.

```
  2005-2026 window    null floor  t = 2.75
  2015-2026 window    null floor  t = 0.77
```

I read that as confirmation: remove the 2008 crash and the overlay has no crisis left to dodge. The simpler reading, which I did not consider, is that one decade of survivor drift compounds into less than two decades of it.

The rule it yields survives either way: **the null floor must be recalibrated for every sample window.** A floor computed on one period does not transfer to another. I just had the wrong reason for believing it.

## What it cost me

Being specific about this, because the general claim is easy and the particulars are the evidence.

The null floor invalidated **four conclusions I had already reached and believed**. None had been published — catching them is what the floor is for — but I had stopped questioning all four:

A pairwise relative-value signal that survived every structural check — monotone response to a mechanism-motivated filter, sector-neutral, coherent horizon profile, t = 2.45. Extending the sample from seven years to twenty dropped it to 1.28, where a real effect predicts 4.1.

A "genuine residual" of +0.02 to +0.036 Sharpe that I reported as surviving a beta-matched control. That range *is* the noise floor. Random signals score +0.029 on the same control.

A claim that small caps fail because a 12× spread penalty cancels their larger premia. I had invented the 12×. Measured with two independent estimators, the ratio is 2.0×. (Small caps still fail — from concentration cost in a high-idiosyncratic-volatility universe, and a benchmark not worth beating. Right answer, wrong reasoning, repeated for two days.)

And a turnover accounting bug I announced had "tilted every comparison in the project" — before measuring it. Measured impact: 5.8 basis points a year, 0.003 Sharpe. Immaterial.

The pattern in all four is the same: **I reported a result at the point it became interesting rather than the point it became controlled.**

There is a fifth, and it is the one that should worry you most, because it did not stay in the notebook. The volatility-targeting explanation above — the mechanism section of this essay, the centrepiece of the companion paper — was published, and stood, and was wrong. It survived every control I had. What caught it was a control I did not own: someone asked what the floor did with the overlay removed, and the answer went the wrong way.

The lesson is not that I should have been more careful. I was careful, in every direction I knew to be careful in. It is that **a calibration procedure only covers the choices you thought to vary.** Mine varied the signal and held the universe fixed, so it could never have found a defect in the universe. That is not a flaw you patch; it is a permanent property of controls, and the only remedy is other people.

## The final tally

With the floor in place, all thirty-eight strategies re-evaluated under Fama-French attribution, a conditional model, and a percentile against forty random signals:

```
  signal strategies tested                    38
  clearing the null floor's 95th percentile    0
  no-signal portfolio's percentile          97.5   (best signal: 92.5)
```

The no-signal book is scored against its own null, since it holds the whole
universe rather than a selection. It outscored every one of the thirty-eight.

The construction — diversify, weight by inverse volatility, target volatility, rebalance monthly — was doing all of the work. Every signal I tested added nothing to it.

## Tests that check financial logic, not execution

The other half of this is a test suite, and it is deliberately not a code-coverage exercise. Thirty-five tests, each pinning a specific defect that actually occurred. The interesting ones assert things about *finance*, not about Python:

**Lookahead, with its own control.** One test sets portfolio weights from the return *into* bar *t* — information you only possess once *t* has closed. A naive engine captures every up-move and posts an absurd Sharpe, so the test fails loudly if the one-bar lag is ever removed. It is paired with a second test using a genuine one-bar-ahead forecast, which must still win big. Together they prove the first test fails for the right reason: the engine rejects future information, not signal.

**Phantom alpha.** A test builds a synthetic market with volatility clustering, runs a zero-information sleeve that merely de-risks when volatility rises, and asserts it *still* beats its own beta-matched control. That test pins the exact illusion behind two of the four reversals above.

**De-risking is not skill.** A "strategy" that is simply 60% of the benchmark plus cash must score approximately zero genuine gain — *while still raising naive standalone Sharpe*. The second clause is the whole point.

**The framework cannot manufacture alpha.** A sleeve that *is* the benchmark must contribute exactly zero. If that ever returns a positive number, the evaluation is producing alpha from nothing and every result downstream is void.

## Why this matters more with AI assistance

This codebase was written with heavy LLM assistance, and I think that makes the test harness more important rather than less.

The failure mode of AI-accelerated development is code that reads plausibly and computes the wrong thing. The failure mode of backtesting is a result that looks plausible and means nothing. **These are the same failure mode**, and a test suite that verifies financial logic rather than execution answers both at once.

Every defect the suite pins was found by a test, not by reading. The Sortino implementation that ran 16% high, the turnover charged on target changes rather than realised drift, a cache refresh that silently overwrote years of history with one month, a `pd.NA` that promoted frames to object dtype and surfaced as an unrelated error in scipy. All plausible-looking. None caught by inspection.

## What to take from this

If you run backtests, or A/B tests, or any experiment where you built the measurement apparatus yourself:

**Run your pipeline on noise.** Not on a shuffled label, not on a placebo — on random input through the identical machinery. Report a percentile against that distribution rather than a t-statistic against convention.

**Every overlay is part of the strategy.** Volatility targeting, position weighting, top-k truncation, rebalancing bands. Each changes the return distribution independently of your signal, and any one left outside the control will be attributed to the signal.

**A large t-statistic is not evidence of a business.** Mine was 6 and it lost money.

**Recalibrate per window.** The floor moved from 2.75 to 0.77 by changing a start date.

The uncomfortable part is that the null floor takes about ten minutes to run. I built it twelfth instead of first, and every conclusion before it was uncalibrated. It should be the first thing in the pipeline, not the last thing you think of.

---

*The framework, the tests, and the full negative result are [on GitHub](https://github.com/aravindvrm/systematic-equity-research). The whitepaper is [here](https://aravindvrm.github.io/systematic-equity-research/). Analysis code and drafting were produced in collaboration with Claude (Anthropic); conclusions and errors are mine.*
