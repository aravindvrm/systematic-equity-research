# Beta-neutral cross-section — pre-registration

Written **before** implementation. Predictions below are committed in advance so the
result can falsify them, which is the control the four overturned conclusions in the
whitepaper did not have.

## The open question

`market_neutral.py` tested a dollar-neutral long/short book on the random forest and
found Sharpe 0.735 against a null maximum of 0.738 — a fail. But the book carried a
**market beta of 0.53**, so it was never beta-neutral, and the test does not actually
answer whether the signal has anything beyond a beta tilt. Equal dollars is not equal
risk when the model's edge is a volatility tilt: its predictions correlate −0.472 with
low-volatility, so the long leg holds higher-beta names than the short leg.

The question this run settles: **with beta actually removed, is anything left?**

## Design, and where each choice comes from

**Beta estimation — daily returns, one-year window, exponentially weighted, shrunk.**
Hollstein, Prokopczuk & Simen and the market-neutrality forecasting literature find the
smallest ex-ante/realized gap from daily data over a trailing year, and that exponential
weighting brings the realized beta of ex-ante-neutral portfolios measurably closer to
zero. Shrinkage follows Vasicek (1973) toward a prior of 1.0; Frazzini & Pedersen report
a mean shrinkage weight near 0.51 on US equities, which is the default here. Estimated
point-in-time on a rolling basis, so no beta uses data from after the date it is applied.

**Both legs scaled by their own portfolio beta**, so ex-ante portfolio beta is zero —
the Frazzini & Pedersen construction, which on US equities averages roughly $1.52 long
against $0.71 short.

**Betas computed on actual holdings, not on rank weights.** Novy-Marx & Velikov show
BAB's rank-weighting is a backdoor to equal-weighting that concentrates the book in
microcaps and **overstates** the profitability of the beta-neutral portfolio. Two
guards: betas are aggregated using the weights actually held, and the universe here is
211 large caps, which closes the microcap channel their critique identifies. Realized
size and gross exposure are reported so the tilt is visible if it appears.

**Ex-ante neutrality is not assumed to be realized neutrality.** BAB's own realized
market loading is not zero, and a portfolio targeting zero beta off five-year monthly
Fama-MacBeth betas has been shown to realize an ex-post beta above one. Both numbers are
reported. Claiming neutrality without measuring it is exactly the error that produced
the 0.53.

**A factor-residual variant as well as a market-beta variant.** Blitz, Huij & Martens
show momentum carries large time-varying exposures to the Fama-French factors and that
ranking on residuals removes them without harming returns. With FF6 R² running 0.66–0.75
across this project, hedging market beta alone will leave the value, size and volatility
exposure intact. Two constructions are therefore run: (A) market-beta-neutral, and
(B) trading the residual after projecting out FF6.

**Costs.** Borrow at 50bp general collateral with no rebate on short proceeds — the
conservative retail case, as in `market_neutral.py` — plus margin interest on the debit
balance, which no test in this project has yet charged, and an explicit Reg T check
(50% initial, 25% maintenance) since beta-scaling makes gross exposure drift away from
200%.

**A fresh null floor for each construction.** The project's standing rule: a floor
computed for one construction does not transfer to another. Forty random signals through
each identical stack.

## Predictions

Committed in advance. Each is falsifiable.

1. **Realized beta falls to |β| < 0.15** from 0.53, but not to zero — estimation noise
   leaves a residual, and the literature is explicit that it does.
2. **Sharpe falls** relative to the 0.735 dollar-neutral figure. The 0.53 beta was
   contributing return in a sample (2015–2026) that was strongly directional; removing
   it removes that contribution.
3. **The result lands at or below the recalibrated null's 95th percentile** — a fail,
   closing the cross-sectional question.
4. **The FF-residual variant (B) is weaker than the market-neutral variant (A)**,
   because the model's edge is a volatility tilt and B removes more of it.
5. **The null floor for a genuinely beta-neutral book is lower than the long-only
   t = 2.75**, because the phantom alpha in the whitepaper came from vol-targeting a
   time-varying beta, and a book with no beta has none to time. This is the one
   prediction that tests the whitepaper's causal claim rather than the signal.

## What would falsify the conclusion

A Sharpe at or above the recalibrated null's 95th percentile **while** realized
|β| < 0.10 and the FF6 alpha t-statistic clears its own null floor. That would be
genuine residual alpha, would contradict predictions 2–4, and would be worth its own
writeup. Anything less closes the question.

## What this cannot settle

Survivorship. The universe is current index membership carried backward, and the best
short candidates are the companies that collapsed and left it. The short leg is measured
on survivors only, so its contribution is understated by an unmeasured amount. Fixing
this needs delisted-securities data, which is where free data runs out. The result is
therefore a bound, not a point estimate, and prediction 3 is the direction that bound
can support.

## References

- Frazzini, A. & Pedersen, L. (2014). Betting against beta. *JFE*.
- Novy-Marx, R. & Velikov, M. (2022). Betting against betting against beta. *JFE*.
- Vasicek, O. (1973). A note on using cross-sectional information in Bayesian estimation
  of security betas. *Journal of Finance*.
- Blitz, D., Huij, J. & Martens, M. (2011). Residual momentum. *Journal of Empirical Finance*.
- Hollstein, F., Prokopczuk, M. & Simen, C. (2019). Estimating beta: forecast adjustments
  and the impact of stock characteristics. *Journal of Financial Markets*.
- Clarke, R., de Silva, H. & Thorley, S. (2002). Portfolio constraints and the fundamental
  law. *FAJ*.
