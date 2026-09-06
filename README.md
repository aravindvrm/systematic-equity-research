# Systematic Equity Research

A signal-research and evaluation framework for systematic equity strategies, and
the negative result it produced.

Twelve signal families were tested across 38 strategies on 211 US large caps over
2005–2026. **None beat a random signal routed through the same pipeline, and a
no-signal portfolio ranked first of the thirty-eight.** The useful output is not
the strategies — it is the calibration procedure that made that conclusion
trustworthy, and the arithmetic that explains it.

📄 **[Read the writeup →](https://aravindvrm.github.io/systematic-equity-research/)**

---

## The finding in three numbers

| | |
|---|---|
| **0.036** | information coefficient required to beat holding everything, long-only |
| **0.012** | best stable IC achievable from free price data on this universe |
| **0.0223** | asymptotic ceiling from the correlation structure of those features |

Because the features correlate at ρ = 0.234, combining arbitrarily many converges
to 0.0223 — roughly 60% of the requirement, approached and never crossed. The
nulls were not a search failure; they were the arithmetic asserting itself.

## The control that matters

The central methodological point, and the one that invalidated four earlier
conclusions in this project:

> Forty **zero-information** signals routed through the identical stack — same
> universe, same top-30% selection, same inverse-volatility weighting, same 10%
> volatility target, same costs — produced a Fama-French 6-factor alpha with a
> Newey-West **t of 2.75**.

A signal containing nothing clears the conventional `|t| > 2` bar comfortably and
reaches the Harvey-Liu-Zhu `|t| > 3` threshold about one time in twenty. The
cause is the volatility-targeting overlay giving the book a time-varying beta that
a constant-beta factor model cannot represent — the bias Ferson & Schadt described
in 1996.

**Calibrate every significance bar against your own pipeline. Never quote one from
convention.** `algo/evaluation.py` implements this; `algo/diagnostics.py` fails
loudly when a result sits inside the measured floor.

## Layout

```
algo/        the library — engine, metrics, evaluation, data ingestion
research/    ~90 one-off analysis scripts, one per question asked
results/     CSV output from the main test batteries
notes/       methodology audit, literature review, state of play
tests/       35 regression tests, each pinning a specific bug found
```

Reading order for `algo/`: `backtest2.py` (engine), `evaluation.py` (the null
floor), `factors.py` (Fama-French and Ferson-Schadt attribution),
`diagnostics.py` (what runs automatically on every result).

## Data sources, all free

| source | what | volume |
|---|---|---|
| SEC DERA Form 4 | insider transactions | 523k rows |
| SEC DERA Form 13F | institutional holdings, 43 quarters | 370k positions/qtr |
| SEC Financial Statement Data Sets | XBRL fundamentals, 69 quarters | 15k filings |
| SEC EDGAR | 10-K/10-Q full text | 19k documents |
| Ken French Data Library | FF5 + momentum, daily | 1963–2026 |
| yfinance | prices, options chains | 250 names |

Ingested data is gitignored (~3.4 GB) and reproducible from the scripts in
`research/`.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest tests/          # 35 tests
.venv/bin/python research/reassess_all.py  # every family vs the null floor
```

Broker integration is dry-run by default; `submit_orders()` refuses to send
anything without an explicit `live=True`. Copy `.env.example` to `.env` for
Alpaca paper credentials — `.env` is gitignored and should never be committed.

## What it does not claim

Every result is survivorship-biased: the universe is current index membership
carried backward. Fundamentals begin at full coverage in 2012 due to the XBRL
phase-in. Accruals were untestable at 16% cash-flow tag coverage. Spread
estimators are biased upward in level, so only their ratio is used.

Backtested research output on historical data. Not investment advice, and not
achieved returns.
