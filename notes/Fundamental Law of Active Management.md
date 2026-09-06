---
tags: [quant, trading, finance-concepts, primer]
created: 2026-09-04
---

# Fundamental Law of Active Management

> [!abstract] The one-line version
> Your realized performance = your per-prediction skill × the square root of how many independent bets you make. Skill is nearly impossible to improve. Number of bets is a data problem.

All three terms below are **statistics concepts wearing finance costumes.** If you know correlation and standard error, you already know the math.

---

## The three terms

### IC — Information Coefficient

Just a **correlation coefficient**. You predict something, it happens or it doesn't:

$$IC = \mathrm{corr}(\text{prediction},\ \text{outcome})$$

Range −1 to +1. Zero = worthless forecast.

> [!warning] The number that shocks everyone
> Good professional quant signals have **IC of 0.02–0.05**.
>
> An IC of 0.03 means you get the direction right about **51%** of the time. Barely distinguishable from a coin flip. This is not a sign of incompetence — it is what the ceiling actually looks like in a competitive market.

### Breadth (BR)

How many **independent** bets you make per year. This is just your **sample size, `n`**.

The word *independent* is doing enormous work — see [[#Effective bets]] below.

### IR — Information Ratio

Your payoff: return per unit of risk. Essentially the [[Sharpe ratio]] of your active decisions.

| IR | Verdict |
|----|---------|
| 0.2 | negligible |
| 0.5 | decent |
| 1.0 | very good |
| 2.0+ | rare / elite |

---

## The law

$$IR \approx IC \times \sqrt{BR}$$

In plain statistics: **realized signal-to-noise = per-prediction skill × √(sample size).**

That square root is the same one from the standard error of a mean, $\sigma/\sqrt{n}$. Random errors cancel at rate $\sqrt{n}$, not $n$. The whole formula is just the central limit theorem applied to bets.

### The casino intuition

Roulette has a house edge of ~5.26% — the casino wins roughly **52.6%** of even-money bets. That is an IC-like edge of almost nothing. No single spin is meaningfully profitable, and the casino loses constantly.

Run millions of independent spins and the outcome becomes essentially deterministic.

**Quant funds are casinos.** Tiny edge, enormous number of trials.

### Why the square root matters so much

Because it means breadth has *diminishing* returns:

- Double your bets → IR × 1.41
- **Quadruple** your bets → IR × 2.0

To double performance you need 4× the independent bets. Painful — but still far easier than doubling skill, which is roughly impossible.

---

## Verified by simulation

A simulated forecaster with skill fixed at exactly IC = 0.03, given varying numbers of bets:

| bets/year | √bets | predicted IR | measured IR |
|-----------|-------|--------------|-------------|
| 32 | 5.7 | 0.17 | 0.18 |
| 300 | 17.3 | 0.52 | 0.39 |
| 720 | 26.8 | 0.80 | 0.64 |
| 37,800 | 194.4 | 5.83 | 4.63 |

**Same forecaster throughout.** Identical, barely-better-than-random skill. Only the number of chances changes — and it goes from useless to world-class.

> [!note] Why measured < predicted
> The simulation bets only on the **sign** of the forecast, discarding magnitude. The law assumes bets sized in proportion to conviction. That ~20% gap is a real lesson: implementation quality costs you actual IR.

*Code: `algo/explain_law.py`*

---

## Effective bets

**N assets ≠ N independent bets**, because assets move together.

When the market drops, SPY / QQQ / IWM / EFA / EEM / XLF all drop at once. You don't have ten opinions — you have roughly one about "stocks," one about "bonds," one about "gold," and noise.

**Effective bets** = how many genuinely distinct things you're betting on. Computed from the eigenvalues of the correlation matrix (a participation ratio) — the same idea as asking how many real dimensions a dataset has:

$$N_{\text{eff}} = \frac{\left(\sum \lambda_i\right)^2}{\sum \lambda_i^2}$$

Perfectly correlated assets → 1. Perfectly independent → N.

> Measured for our 10-ETF universe: **2.65 out of 10.**

### The arithmetic that follows

```
2.65 effective assets × 12 rebalances/year  =  ~32 independent bets/year
√32                                         =  5.6
IR = IC × 5.6
     at realistic IC = 0.03  →  IR = 0.17     ← negligible
```

Running it backwards: reaching IR 1.0 with this universe needs **IC ≈ 0.177** — about **5× better forecasting than professional quant funds achieve.** Not a modeling challenge; a fantasy.

This is a **structural** cap. It has nothing to do with how clever the model is. A perfect model still only gets 32 chances a year.

### Why more names fixes it

```
500 stocks → ~60 effective bets × 12/yr = 720 bets/yr
√720 = 26.8
IR = 0.03 × 26.8 = 0.80
```

**Same mediocre skill. IR goes from 0.17 → 0.80.**

> [!tip] The counterintuitive part for an ML background
> Your instinct is to improve the model. The law says improving the model 4.8× is impossible, while getting 4.8× more independent bets is *just a data problem*.

---

## Point-in-time data

A separate idea — about not fooling yourself.

Grab today's S&P 500 list, backtest to 2010, and you have quietly guaranteed that **every company in your test survived.** Lehman, Bear Stearns, Enron, Circuit City, Blockbuster — gone, not on today's list, so the backtest never buys them.

You built a strategy that only ever picked winners, because you selected the universe using information from the future.

> [!danger] This is label leakage
> Exactly the kind you'd catch instantly in an ML pipeline. Known as **survivorship bias**. Can inflate returns by several % per year and is completely invisible in the output.

**Point-in-time data** answers: *"what was actually in the S&P 500 on 2013-03-03?"* — the list as known **then**, delisted companies included.

This is why the working universe is 10 long-lived ETFs. Small, but honest: none were chosen because they did well.

---

## Why it's a data-engineering problem

Getting to a few hundred real names requires:

- [ ] Historical index membership, as-of each date
- [ ] Delisted tickers with price history intact
- [ ] Corporate actions — splits, dividends, spinoffs
- [ ] Ticker symbol reuse handled

All plumbing. No modeling, no cleverness. Which is exactly why it's the next *build* rather than the next *idea*.

---

## Takeaway

A 10-ETF monthly-rebalanced universe **cannot produce a good result regardless of what goes into it** — not because the strategies are bad, but because 32 bets a year is too few trials for a small edge to show through noise.

More names is the lever with real leverage on it.

## Related
- [[Sharpe ratio]]
- [[Survivorship bias]]
- [[Transaction costs and position sizing]]
- [[Deflated Sharpe ratio]]
