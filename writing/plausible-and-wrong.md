# Plausible and Wrong

*I wrote a quantitative research project with heavy LLM assistance, kept a record of every defect, and found that reading the output carefully caught none of them.*

---

The companion paper to this one has a colophon saying its analysis and drafting were produced in collaboration with Claude. That sentence is true and almost useless. It tells you nothing about what the collaboration consisted of, what went wrong, or what caught it.

This is that detail. It is worth writing down because the project kept a record — every defect, when it appeared, and what found it — and the record contradicts the advice usually given about working this way.

The headline: **zero defects in this project were caught by reading the output carefully.** Every single one was caught by a test, by recomputing a number, or by another person.

## The failure mode is not what you are told to watch for

Hallucinated APIs and syntax errors are the standard warning, and they are not the problem. They fail immediately and loudly. An import that does not exist raises on the first run.

The actual failure mode is **confident, well-formed, and wrong**, and it clusters in three places.

**Numbers asserted rather than derived.** The paper once claimed small caps fail because a 12× bid-ask spread penalty cancels their larger premia. The 12× was invented. Measured with two independent estimators, the ratio is 2.0×. The conclusion survived; the reasoning was fiction, and it was repeated for two days.

**Mechanisms inferred from a plausible story.** A coherent causal account, consistent with the literature and with three separate results, and backwards. More on this below — it is the case that matters.

**Claims about work not examined.** A résumé bullet described a segmentation pipeline feeding an image-editing model. Searching the repository's full history for that work returned nothing: no Python at all, no segmentation code, in any commit.

All three are the same defect. An output shaped exactly like a verified statement, produced without the verification. It reads like a conclusion because it is written like one.

## What actually caught things

Here is the record. Every defect I logged, what it actually was, and what found it.

| Defect | What it was | Caught by |
|---|---|---|
| Null-floor mechanism | Volatility targeting blamed; ablation shows the opposite sign | A reader's question |
| Universe size | "211 names" against a filter that yields 199 | Recomputing during unrelated work |
| Small-cap cost penalty | 12× asserted; 2.0× measured | Two independent estimators |
| Turnover defect severity | "Tilted every comparison"; 5.8bp/yr | Measuring before publishing |
| The drift guard itself | Passed while its own target bug was present | Breaking it on purpose |
| Orphaned CSS, three times | Regex deletion left stranded declarations | Counting brace depth |
| Chart coordinates | Series plotted against the wrong scale | Recomputing against the axis |
| Two bibliography entries | Filed under "Anomalies tested", never tested | Checking each citation against the code |
| A résumé bullet | Described work absent from that repo's entire history | Searching the history |
| Three line and module counts | Stale or inflated by duplicate directories | Counting |
| An equicorrelation caveat | Claimed a shortcut understated; it agrees to 3% | Computing the full spectrum |
| — | — | **Reading it carefully: nothing** |

Ordered by hit rate, not by cleverness.

**Recomputing from source.** The highest-yield control by a wide margin. Three published figures had silently drifted from the code that produced them — a universe of "211 names" when the coverage filter yields 199, a line count, a module count. In all three the code was right and the prose was stale. Prose does not raise exceptions.

That gap is now closed mechanically. A script derives every published number from its source and asserts the document still states it, in context, exiting non-zero if not:

```
CLAIM                                        VALUE   STATUS
null floor mean                              +2.75   ok
IC asymptote                                0.0209   ok
beta-orthogonal: realised beta                0.01   ok
universe size                                  199   ok
...
All 30 derived claims appear in the documents, in context.
```

**Structural invariants, never greps.** Cleaning up CSS with a regex left orphaned declarations three separate times. Grepping for the deleted selector found nothing each time — *because the surviving damage does not contain the thing you would search for.* Counting brace depth caught it immediately. The generalisation is worth stating plainly: **to verify a removal, check a property that holds over the whole structure. Never search for the token you removed.**

**Testing the control by breaking it.** The drift-checking script above passed its first run while the exact bug it was written for was present in the document. It had searched for the value as a bare substring, and "199" occurs incidentally all over a long page. I only discovered this by deliberately reintroducing the bug to watch it fail — and it did not. A control that has never been observed to fail has not been tested.

**Measuring before characterising.** A turnover accounting defect was announced as having "tilted every comparison in the project." Measured: 5.8 basis points a year, 0.003 Sharpe. Immaterial. The claim preceded the measurement by a full day.

**Registering predictions before implementing.** For one follow-up study the design, five numbered predictions and an explicit falsifier were written and committed before any code existed. Two predictions were wrong and the falsifier fired. Pre-registration did not prevent a bad criterion — the criterion omitted persistence, which is the very thing the project exists to check. It did make the badness visible instead of negotiable.

## The one that got through

The paper's central finding is that a strategy containing no information scores a Fama-French alpha t-statistic of 2.75 on this pipeline, which clears every significance bar in the literature. The paper explained that floor as an artifact of volatility targeting: the overlay cuts market exposure when volatility rises, giving the book a time-varying beta that a constant-beta factor model cannot represent, so the unmodelled timing surfaces as alpha.

That explanation was excellent. It was consistent with a real, published effect. It was consistent with the drawdown figure, since the overlay does halve the 2008 loss. It was consistent with a robustness check on a later sample window. It was consistent with a market-neutral result showing the floor collapse when beta was removed.

It was also wrong.

It survived every control the project had. What refuted it was a question from the first person to read the thing critically: what does the floor look like with the overlay simply switched off? Same universe, same ranking, same weighting, same costs, two hundred null draws in each arm.

```
Alpha-t null floor, identical stack

  with 10% volatility target      +2.66
  no overlay at all               +2.94
```

Removing the overlay makes the floor *worse*. The thing I had spent the project blaming was quietly reducing the problem.

The real cause is survivorship in the universe — index membership as it stands today, carried backward, so the companies that failed and left are absent while the companies added are present for the run-up that earned them their place. A book holding all of them and forecasting nothing beats a six-factor model by 4.53% a year at t = 5.79, while the market itself over the same window returns a negative alpha. Every long book inherits that in proportion to its market exposure, and volatility targeting cuts exposure — which is exactly why removing it raises the floor.

Brown, Goetzmann, Ibbotson and Ross published this in 1992: a sample truncated by survivorship produces the appearance of predictability, strongly enough to account for the evidence then being offered for return predictability. I had it in a caveats section the whole time, listed as a limitation the way everyone lists it, rather than measured as an effect.

## Why that one got through

Not carelessness. The floor had been calibrated forty different ways, and every calibration varied the signal while holding the universe fixed.

**A calibration procedure only covers the choices you thought to vary.** Mine could not have found a defect in the universe, because the universe was never a variable. That is not an oversight to be fixed by more diligence; it is a permanent property of controls. Whatever you hold constant is invisible to your own checks, and you cannot enumerate what you are holding constant, because the whole problem is that you are not thinking about it.

The only remedy is someone outside the loop. That is a strange thing to conclude about a workflow built for working alone faster, and I think it is the most important thing in this piece.

## What did not work

**Reading carefully.** Zero for the project. Every defect was caught by a test, a recomputation, or another person.

**Confidence.** There is no relationship between how settled a claim felt and whether it held. The three most confidently stated claims in the project — the 12× penalty, the turnover characterisation, the volatility-targeting mechanism — are all in the list of things that turned out to be wrong. If anything, the correlation runs the wrong way: a claim gets stated confidently when it fits a story, and fitting a story is exactly what a plausible-and-wrong output does.

**Citing without checking.** Two references sat in a bibliography section titled "Anomalies tested" having never been tested. Two more were listed and used nowhere. Checking each citation against the code took twenty minutes and should have happened before publishing, not after.

## The rules I would keep

1. Derive every published number from something the code can recompute, then assert the document contains it. Automate the assertion.
2. Verify a removal with a structural invariant, never by searching for the removed token.
3. Break the control on purpose. If it does not fail, it does not work.
4. Measure a defect's impact before describing its severity.
5. Register predictions and a falsifier before implementing. Score them afterwards, in public, including the misses.
6. Treat "this reads plausibly" as carrying no evidential weight whatsoever.
7. Do not describe work you have not opened.

None of these are about prompting. They are ordinary engineering controls, and the reason they matter more here is throughput: the volume of plausible output goes up, the cost of checking each piece does not, and the gap between those two is where the defects live.

The failure mode of AI-accelerated development is code that reads well and computes the wrong thing. The failure mode of backtesting is a result that looks credible and means nothing. **They are the same failure mode**, and one discipline answers both.

---

*The project, its tests, and the drift-checking script are [on GitHub](https://github.com/aravindvrm/systematic-equity-research). The research this describes is [here](https://aravindvrm.github.io/systematic-equity-research/), and the companion essay on calibration is [here](https://aravindvrm.github.io/systematic-equity-research/null-floor.html). Written in collaboration with Claude (Anthropic), which is the point; conclusions and errors are mine.*
