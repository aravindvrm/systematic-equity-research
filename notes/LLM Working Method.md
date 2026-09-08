# The LLM working method, and what it actually caught

Internal reference. The whitepaper's colophon says the analysis and drafting were
produced in collaboration with Claude; this is the mechanical detail behind that
sentence, written down because the interesting part is not the collaboration but
the **control structure around it**, and because that structure has a measured
record rather than an assumed one.

The claim this document is organised around: *LLM-assisted work fails in one
specific way — output that reads plausibly and is wrong — and the only things
that catch it are mechanical.* Everything below is evidence for or against that.

---

## 1. The setup

- **Claude Code** in a terminal, working directly in the repo, with shell, file
  edits, and web search. No copy-paste loop.
- **One repository, everything committed.** Analysis scripts, notes, results
  CSVs, the published HTML. Nothing lives only in a chat log.
- **`research/` is one script per question.** ~90 of them. Each is disposable
  and self-documenting; the reusable parts get promoted to `algo/`.
- **`algo/` is the library and carries the tests.** 35 of them, each pinning a
  specific defect that actually occurred.
- **The published documents are generated or guarded.** `build_essay.py` renders
  the essay from markdown; `make_figures.py` emits both the figure data and the
  SVG; `check_figures.py` asserts every published number still matches its
  source.

---

## 2. The controls, ordered by what they actually caught

Not by how clever they are. By hit rate.

**1. Recomputation from source.** The single highest-yield control. Most errors
were found by deriving a number again rather than reading it. Three published
figures had drifted from their sources with nobody noticing (universe size,
Bender's runtime count, Swapaday's module count) and in every case the code was
right and the prose was stale.

**2. Structural invariants over greps.** Counting braces caught CSS damage that
grepping for the selector could not, three separate times — because a deletion
leaves behind lines that *do not contain the thing you would grep for*. The
generalisation: to verify a removal, check a property that holds over the whole
structure, never search for the removed token.

**3. Negative tests on the controls themselves.** `check_figures.py` v1 passed
while the exact bug it was written for was present, because it searched for a
bare number that occurred incidentally elsewhere. It was only discovered by
deliberately reintroducing the bug. **A control that has never been observed to
fail has not been tested.**

**4. Measure before characterising.** A turnover defect was announced as having
"tilted every comparison in the project"; measured, it was 5.8bp/yr and 0.003
Sharpe. A small-cap cost penalty was asserted at 12×; measured with two
independent estimators, 2.0×. Both claims were confident, plausible, and
invented.

**5. Pre-registration.** Design, numbered predictions and a falsifier committed
before the code existed. Two of five predictions were wrong and the falsifier
fired. It did not prevent a bad criterion — the criterion omitted persistence —
but it made the badness visible instead of negotiable.

**6. External review.** The only control that caught the error which had been
*published*. See §4.

---

## 3. What the failure mode actually looks like

Not hallucinated APIs or syntax errors — those fail loudly and immediately. The
real mode is **confident, well-formed, wrong**, and it clusters in three places:

- **Numbers asserted rather than derived.** The 12×. The "ranks first of
  thirty-eight" when the no-signal book was not among the thirty-eight. A `0.45`
  point estimate invented from a documented `0.4–0.5` range.
- **Mechanisms inferred from a plausible story.** Volatility targeting as the
  source of the null floor: coherent, literature-backed, consistent with two
  other results, and wrong.
- **Claims about work not examined.** A resume bullet describing a DeepLabV3+
  segmentation pipeline that appears nowhere in that repository's history.

Each is the same defect: an output shaped like a verified statement, produced
without the verification.

---

## 4. The case that matters most

The paper attributed its central finding — a null floor of *t* = 2.75 — to
volatility targeting inducing a time-varying beta. That explanation was:

- consistent with a real, published effect (Ferson & Schadt, 1996)
- consistent with the drawdown figure (the overlay does halve the 2008 loss)
- consistent with the 2015-window result (floor falls to 0.77)
- consistent with the beta-neutral result (floor falls below zero)
- **wrong**

It survived every internal control, was published, and stood for weeks. It was
refuted by an ablation a reviewer suggested: run the identical stack with the
overlay switched off. The floor went *up*, from +2.66 to +2.94. The actual cause
is survivorship in the universe, which every long book inherits in proportion to
its market exposure — and which Brown, Goetzmann, Ibbotson & Ross had described
in 1992.

The transferable lesson is not "be more careful." It is that **a calibration
procedure only covers the choices you thought to vary.** This one varied the
signal and held the universe fixed, so it could not have found a defect in the
universe. That is a permanent property of controls, not a fixable oversight, and
the only remedy is someone outside the loop.

---

## 5. What did not work

- **Reading carefully.** Zero defects in this project were caught by inspection.
  Every one was caught by a test, a recomputation, or an outside reader.
- **Grepping to confirm a deletion.** Failed three times for the same structural
  reason.
- **Confidence as a signal.** The strongest-held claims — the 12×, the turnover
  characterisation, the vol-targeting mechanism — were among the wrong ones.
  There is no correlation between how settled a claim felt and whether it held.
- **Citing without checking.** Two references sat under "Anomalies tested"
  having never been tested; two more were listed and never used anywhere.

---

## 6. Standing rules

1. Derive every published number from a source the code can recompute, then
   assert the document contains it. `check_figures.py` does this for 30 claims.
2. Verify a removal with a structural invariant, never with a search for the
   removed token.
3. Test the control by breaking the thing it watches. If it does not fail, it
   does not work.
4. Measure a defect's impact before characterising its severity.
5. Register predictions and a falsifier before implementing. Record the score
   afterwards, including the misses.
6. Treat "this reads plausibly" as carrying no evidential weight at all.
7. Do not describe work you have not opened. Repository history is checkable;
   recollection is not.

---

## 7. If this is ever published

The genre is saturated with process posts that assert a method and show no
record. The only thing here worth publishing is the part almost none of them
have: **a catalogue of specific failures, what caught each one, and one case
where the method failed in public and an outsider caught it.**

Written without that, it would be the exact failure mode it describes.
