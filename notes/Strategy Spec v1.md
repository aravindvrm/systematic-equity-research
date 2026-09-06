---
tags: [quant, trading, spec, algo-project]
created: 2026-09-04
status: agreed baseline before non-price signal work
---

# Strategy Spec v1

> [!abstract] One line
> ~24-30 names, weekly-to-biweekly rebalance, band-triggered, vol-targeted, in an
> Alpaca cash account — needing a **time-varying non-price signal with IC ≥ 0.032**
> to be worth running at all.

Every number below is measured in this repo, not assumed. Where a figure is an
estimate or is contaminated, it says so.

---

## 1. Universe — size and shape

> [!abstract] Two universes, and conflating them was the original mistake
> **COLLECTION (262)** — what we snapshot options for daily. Wide on purpose:
> collection is *irreversible*. A name not captured today can never be backfilled.
> Narrowing later is free; widening retroactively is impossible.
>
> **TRADING (30)** — selected *from* that pool by rule. Re-derivable any time.

### The rule (reproducible, not hand-picked)

```
S&P 500 members (Wikipedia, cached with GICS sector + SEC CIK)
  -> liquidity screen: >= $50M FULL-WINDOW median daily dollar volume,
     >= 1000 bars of history                                  [503 -> 484]
  -> top 250 by dollar volume, capped at 30 per GICS sector    [-> 250]
  -> plus 12 diversifiers                                      [-> 262 COLLECTION]
  -> greedy maximize effective bets, n=30                      [-> 30 TRADING]
```

Use the **full-window** median for liquidity, not a trailing 120 days. A recent-volume
screen ranked LITE, PLTR and MRVL above most mega-caps — smuggling a momentum tilt
into what is supposed to be a neutral filter.

The **sector cap matters**: ranking on dollar volume alone hands the universe to
Information Technology, where turnover concentrates.

### Core variants — the diversifier constraint

| | div | eq | eff.bets | vol% | corr | IC@wk | IC@2wk | IC@mo |
|---|---|---|---|---|---|---|---|---|
| none | 4 | 26 | **16.49** | 37.4 | 0.14 | 0.020 | 0.026 | 0.037 |
| core3 | 4 | 26 | **16.50** | 36.9 | 0.13 | 0.020 | 0.026 | 0.037 |
| **core6** ← default | 7 | 23 | **14.50** | 33.7 | 0.15 | **0.021** | 0.028 | 0.039 |
| core12 | 12 | 18 | **11.85** | 28.4 | 0.16 | 0.023 | 0.031 | 0.043 |

`IC@` = information coefficient required for **IR 0.5** at that rebalance frequency.

**core3 is free** — forcing TLT/GLD/DBC costs nothing in effective bets; the optimizer
picks SHY unprompted even unconstrained. Past that you trade breadth for lower vol.
core6 is the default: 14.50 bets at 33.7% vol.

### Composition (core6)

```
diversifiers (7)   DBC EFA GLD HYG SHY TIP TLT
equities (23)
  Communication Services   CHTR PSKY TTWO VZ
  Consumer Staples         CLX DG DLTR GIS KDP KR
  Consumer Discretionary   CVNA TSLA
  Health Care              BIIB DXCM GILD HUM LLY
  Materials                CF NEM
  Energy EQT | Industrials NOC | Info Tech ORCL | Utilities PCG
```

> [!note] What the composition reveals
> **Zero Financials**, in any variant — banks are too correlated with the market and
> with each other to earn a diversification slot. **Consumer Staples takes 6 slots in
> every variant** — defensives are the genuine diversifier among equities. Only 1-2
> Info Tech names despite tech being 30 of the 262. The optimizer is systematically
> avoiding the market factor, which is the point.

> [!warning] Maximizing effective bets selects for IDIOSYNCRATIC VOLATILITY
> Unconstrained, it picked MRNA, CVNA, WBD, ALB, PCG — no mega-caps, no diversifiers,
> **43.7% average volatility**. Mathematically correct: lowest-correlation names are
> the high-idio-vol ones (story stocks, biotech binaries, distressed). The objective
> was underspecified — asking for diversification in a single-factor market gets you
> volatility. Hence the forced diversifier core and the vol column above.

### Two hard constraints on honesty

> [!danger] Survivorship: still ~+0.23 Sharpe
> This is TODAY's S&P 500 membership. Making the rule reproducible did NOT make it
> point-in-time. Delisted names are absent from yfinance entirely (8 of 9 tested
> returned zero rows) and `FRCB` returned a *different company's* prices — ticker
> reuse silently splicing two firms into one series.

> [!warning] Pick luck was larger than any strategy effect found
> 200 random 11-name draws, identical strategy: Sharpe **0.75 to 1.41**. 30 names
> halves that dispersion vs 11 (sd 0.125 → 0.059). That, not breadth, is the argument
> for 30 over 15.

## 2. Time horizon

**Rebalance weekly to biweekly (5–10 trading days), band-triggered.**

Why not the alternatives:

| Horizon | Verdict | Reason |
|---|---|---|
| Intraday / minutes | **Closed** | Cost scales linearly with trades, IR only as √BR. At 5-min holds: 1,966%/yr in costs at 5bp. Doesn't close at *any* IC — profitability requires *earning* the spread, which is a different business |
| Daily | Marginal | BR is good but IC would need to be high; whipsaw hurt in testing |
| **Weekly–biweekly** | **Target** | BR 164–328 with 6.5 effective bets. Cost drag only 0.04–0.08 Sharpe units on Alpaca |
| Monthly | Hard | Needs IC 0.059 — above the professional range |
| Quarterly+ | Closed | Needs IC ≥ 0.099 |

Counterintuitive but it follows: **breadth comes from frequency**, and with only 6.5 effective bets you need the frequency to accumulate decisions. Zero commissions make that affordable.

**Band rebalancing, ~10%**: check daily, trade only past the band. Measured to beat both daily (8.25% vs 3.68% CAGR at $1k) *and* monthly calendar rebalancing — and it still won at **zero** commissions, so it filters noise, not just cost. Cold-start rule: open a position regardless of band, apply the band only once held.

---

## 3. Signal type

**Fast-moving, time-varying, non-price.** Candidates: estimate revisions, sentiment, flows/positioning, news, alternative data.

### Why not price
32 features across 5 categories (momentum, reversal, volatility, trend quality, volume, intraday range, relational), 2 universes, both IC formulations, plus the actual binary decision rule tested directly. After removing static tilts, **nothing clears the corrected significance bar**. Time-series IC on the momentum family is zero to negative; the rule the old portfolio used had a *negative* return spread (8.59% on vs 13.24% off).

### Why not slow fundamentals — at this size
Value/quality/profitability want monthly–quarterly horizons, which need IC 0.06–0.10. Real factor funds solve this with **breadth from names** (500+, quarterly), requiring point-in-time fundamentals. That's a data purchase before it can even be tested. See [[Fundamental Law of Active Management]].

> [!note] A refinement to the static/dynamic test
> `research.static_vs_dynamic()` correctly kills fake signals — `dollar_volume`
> scored t=5.92 and was purely "hold SPY/QQQ" (rank stability 0.91). But it cannot
> distinguish a *spurious* tilt from a **compensated risk premium**. Value and size
> are slow-moving by nature; for those, persistence IS the strategy. Validate a
> factor tilt with an economic story plus out-of-sample across time AND universes —
> not with the demeaning test.

---

### Prefer idiosyncratic triggers over market-wide ones

> [!important] Added after the setup tangent
> A conditional rule fired by a **market-wide** move gives far less evidence than
> its event count suggests. Testing "buy after a 21d −2sd move" produced 2,090
> events — but only **80 independent episodes**, because dozens of names trigger on
> the same macro day. Naive t = 9.39; after deflating for overlap *and* clustering,
> **t = 0.40**.
>
> It also failed for a second reason: ~90% of the apparent edge was market beta.
> Names returned 3.09% after the trigger while the index returned 2.77% over the
> same dates. Excess: 0.32%.
>
> **Selection criterion for any event-driven signal:** does it fire independently
> across names (single-company earnings, a filing, a name-specific shock), or all at
> once on a macro day (index put/call ratio, VIX spikes, market drawdowns)?
> Idiosyncratic triggers generate real independent episodes. Market-wide triggers
> generate one bet wearing many hats.
>
> `research.event_study()` now reports `n_episodes` and a deflated `t` alongside
> `t_naive`. Never quote `t_naive`.

## 4. IC required

> [!important] The bar: **IC ≥ 0.021**, weekly, time-varying, out of sample.
> Lowered from 0.032 by the universe rebuild — 6.33 → 14.50 effective bets. This is
> the single largest feasibility gain in the project, and it came from portfolio
> construction rather than from any signal.

At **14.50 effective bets** (core6, 30 names):

```
horizon        BR    IR=0.3   IR=0.5   IR=0.8
1 week        754     0.014    0.021    0.032
2 weeks       377     0.018    0.028    0.042
1 month       174     0.025    0.039    0.061
1 quarter      58     0.042    0.066    0.102
```

Monthly now needs IC 0.039 — borderline plausible where it was fantasy at 0.059.
Weekly at 0.021 sits comfortably inside the professional 0.02-0.05 band.

Calibration: professional signals run **IC 0.02–0.05**. IC 0.10 is exceptional. IC 0.20+ is a bug.

**IR is the ALPHA on top of the base portfolio, not total Sharpe.** With a ~1.0 baseline and an uncorrelated IR of 0.5, total ≈ √(1.0² + 0.5²) ≈ **1.12**. So a successful signal moves total Sharpe by roughly +0.12, not to 0.5.

Breadth moves the bar: 100 stocks long-only → IC 0.053; 50 stocks **market-neutral** → **IC 0.029**. Shorting roughly halves the requirement — the clearest statement of what a margin account would buy. Gated at $2,000 equity, and Alpaca fractional shares are long-only so shorts need whole shares (realistically $25k+ to size sensibly).

---

## 5. Execution

- **Alpaca cash account.** $0 commissions, fractional shares, free data adequate for daily bars. IBKR needs Pro for API market data, and Pro Tiered's $0.35/order minimum is the binding constraint at small size
- **Vol target 10%, max leverage 1.0** — only ever cuts exposure; cash-account legal
- **Position sizing**: inverse volatility within the selected set
- Risk limits as **fractions of equity**, never absolute dollars
- Live capital: **$1,000** as validation tuition, not investment. Income at that size is arithmetically negligible ($5–42/month across any plausible return)

---

## 6. The baseline to beat

> [!warning] This is the demanding part
> A **vol-targeted, equal-weight, NO-SIGNAL** book on this universe scored **1.12**
> (SPY 0.83 over the same window). Of 16 signal configurations tested at various
> horizons, **11 underperformed it**.
>
> Benchmark against *that*, not against SPY and not against plain equal-weight.
> I flattered a result earlier by choosing the wrong benchmark; the diagnostics
> reported "all checks passed" on what was actually a null.
>
> Discount the 1.12 for survivorship (~0.23) → honest baseline **~0.9**.

---

## 7. Validation gates

Any candidate signal must clear all of these before it goes near live capital:

- [ ] **Time-varying** — survives `research.static_vs_dynamic()` (unless it's an explicit factor tilt, which needs the economic-story test instead)
- [ ] **Significant after correction** — Bonferroni for the true number of tests run, counted honestly
- [ ] **Beats its shuffled control** — same values, cross-section permuted
- [ ] **Stable across halves** — `split_stability` retention > 70%
- [ ] **Survives walk-forward**, not just one split (n=1 misled me twice this session)
- [ ] **Deflated Sharpe** with the real trial count
- [ ] **Beats the vol-targeted no-signal baseline**, with costs, at realistic order sizes
- [ ] **Ensembled, not tuned** — parameter selection lost to naive averaging twice, out of sample

---

## Open decisions
- Exact universe rule (which diversification objective, sector caps, refit cadence)
- Which non-price data source, and whether it's worth paying for
- Whether to revisit shorting at $25k+
- Whether to keep any signal at all, or run the no-signal book as the honest default

## Related
- [[Fundamental Law of Active Management]]


---

## Appendix: bugs this spec exists to prevent

Each was found only because something was questioned, and each would have silently
corrupted results rather than failing loudly.

1. **Cache truncation.** The daily options collector called `latest_prices(refresh=True)`,
   which refetched a 30-day window and **overwrote** the cached parquet — replacing
   years of bars with 23 rows, for all 262 tickers, every day. Every subsequent
   backtest would have run on one month of data. *Fix: cache merges, never replaces.*

2. **Static tilts read as signal.** `dollar_volume` scored t=5.92 — and was purely
   "hold SPY/QQQ" (rank stability 0.91). *Fix: `research.static_vs_dynamic()`.*

3. **Event clustering.** "Buy after a 21d -2sd move" showed 2,090 events and t=9.39;
   it was **80 independent episodes** and t=0.40. *Fix: `event_study()` reports
   `n_episodes` and a deflated t. Never quote `t_naive`.*

4. **Object dtype poisoning.** `pd.NA` promoted every frame to object; arithmetic
   still "worked" until scipy raised somewhere unrelated. *Fix: float coercion + test.*

5. **Unpopulated final row.** `panel.iloc[-1]` during market hours is today's
   unfinished bar — all NaN. Produced NaN spot prices, so every moneyness-filtered
   options metric returned empty while unfiltered ones looked fine. *Fix:
   `data.latest_prices()` + a hard raise.*

6. **Flattering benchmark.** Diagnostics reported "all checks passed" against plain
   equal-weight (1.05) instead of the vol-targeted no-signal book (1.12). Against the
   right baseline, 11 of 16 signal cells *underperformed*. *Fix: always benchmark
   against the no-signal version of the same construction.*


---

## SUCCESS CRITERIA — SUPERSEDED 2026-09-05

The original criteria in this spec asked for standalone Sharpe above a
vol-targeted equity benchmark. That is the wrong test for a sleeve held
alongside an existing portfolio. See [[Methodology Audit]].

**Current criteria — all four must hold:**

1. **Genuine gain > 0** against a beta-matched `b*core + (1-b)*cash` mix at a
   realistic (~10%) allocation. De-risking is free; only what beats it counts.
2. **Correlation to core < ~0.6**, or the sleeve is just more of what is already
   owned regardless of its Sharpe.
3. **Newey-West alpha |t| > 3**, not 2 — the honest multiple-testing bar after
   ~200 cells across twelve families.
4. **Evaluated on the longest available sample**, not 2016-2026, which flatters
   equity and penalises everything measured against it.

Run every backtest as `result.report(core=<your portfolio returns>, n_trials=N)`.
Without `core`, the report answers the wrong question.
