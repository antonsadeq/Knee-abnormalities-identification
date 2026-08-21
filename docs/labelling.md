# Verifying the report labeller

The labeller manufactures 4,349 of the 4,407 training targets. If it is systematically wrong, the
imaging model learns the error faithfully and no amount of architecture recovers it. So the
verification protocol has to be designed before the labeller, not after.

## Why the gold set cannot carry this alone

Fifty-eight studies have image-derived labels. Broken down by report language:

| lang | reports | gold studies |
|---|---:|---:|
| en | 1,736 | 28 |
| es | 682 | 10 |
| tr | 546 | 6 |
| hr | 406 | 4 |
| el | 321 | 3 |
| de | 262 | **2** |
| bg | 220 | 3 |
| nl | 153 | **2** |
| fr | 81 | **0** |

**French has no gold labels at all.** German and Dutch have two each. And the dangerous failure mode
is exactly language-specific: miss one negation idiom and every report in that language flips
polarity for that finding. Averaged over the corpus it looks like mild label noise; within that
language it is an inverted label, and the imaging model dutifully learns to predict the opposite.

A per-finding rate measured on n = 58 has a Wilson interval roughly ±0.13 wide even at its most
favourable. Per (language × finding) it is not measurable at all. So gold agreement is one check
among four, and for six of the nine languages it is not available.

## The four checks

Implemented in [`rsna_knee.verify`](../src/rsna_knee/verify.py).

### 1. Gold agreement — `gold_agreement`

Per-finding precision/recall against the 58, every rate reported with its Wilson interval and its
own `n` (the number of gold studies the labeller actually asserted on, usually well under 58).

Read it for **gross failure only**: a finding whose recall interval excludes 0.5, or whose polarity
is plainly inverted. It cannot distinguish a good labeller from a slightly better one, and treating
it as if it can is how you overfit 58 studies.

### 2. Coverage — `coverage`

Fraction of reports yielding *any* assertion, per language × finding. **Needs no ground truth**,
which is why it carries the load for the six languages the gold set cannot reach.

Read by comparing a cell against the same finding in other languages. A finding asserted in 70 % of
English reports and 8 % of Turkish ones means missing Turkish vocabulary — not an absence of Turkish
knees with that finding. This is the check that finds *silence*.

### 3. Polarity — `polarity`

Positive share among assertions, per language × finding, with binomial outliers flagged. **This is
the inverted-negation detector**, and it needs no ground truth either.

If a negation idiom is missed, every negative statement of that finding is read as a positive, so
the cell's positive share jumps toward 1 while other languages sit near the true prevalence.

One implementation detail is load-bearing: the reference rate is the **median across languages, not
the pooled rate**. Pooling is the obvious first attempt and it fails — the corrupted language
contributes to the reference it is judged against, dragging it far enough that the clean languages
are flagged too, at which point the output no longer says which language is wrong. This is not
hypothetical; the pooled version flagged English at z = −7.1 in a test where English was correct and
French was inverted. The median over nine languages tolerates up to four bad ones.

A flag is a prompt to read, not a verdict — case mix genuinely differs across sites. But a real
case-mix difference rarely reaches |z| > 3, and a polarity inversion always does.

### 4. Cross-labeller disagreement — `disagreement`, `adjudication_queue`

Where the rule lexicon and the LLM differ is where the labels are wrong. Needs no ground truth, and
concentrates manual reading where it pays: studies where both agree almost never need a human,
studies where they differ almost always do.

`adjudication_queue` samples those disagreements **stratified by language**. Unstratified, English
would dominate the queue and French — the language with no gold labels at all — would rarely appear.
Reading ~20 reports per language, machine-translated where needed, is a few hours of work and is the
highest-information-per-hour activity available on this project.

### Later: the imaging model as a fifth opinion

Once any model exists, compute per (language × finding) AUC of the report label against the model's
prediction. A language whose labels are inverted shows AUC well below 0.5 for that finding — the
model, trained mostly on other languages, has learned the true signal and disagrees. This is the
strongest available check and it costs nothing once training is running.

## Should unclear reports be dropped?

Your instinct is right about *what* to avoid — guessing a label is worse than having none — but
dropping the study is the wrong lever. Three reasons.

**The decision is per-finding, not per-study.** A report can be explicit about the ACL and silent
about synovitis. Dropping the study to avoid one bad label discards eleven good ones. Since the loss
takes a per-finding sample weight, "drop" is available as weight = 0 on a single (study, finding)
cell — strictly more granular, with nothing given up.

**Silence is not random, so dropping introduces bias.** Radiology reports mention what is abnormal
and what was specifically looked for. Drop every study where a finding is not explicitly asserted
and you preferentially remove the *normal* knees for that finding — the model then trains on a
population enriched for positives and never learns what a normal one looks like. Noise averages out
as data grows; selection bias does not, and it hurts the rare findings worst, which is precisely
where the metric is most exposed (a finding left at chance costs ≈ 0.029 of the final score).

**4,407 studies is not enough to spend.** With 12 heads and a weak-supervision setup, discarding
whole studies is expensive in a way that down-weighting is not.

So: **grade and weight per finding; never drop the study.** Three assertion states, mapping to a
target and a weight:

| Report says | target | weight |
|---|---|---|
| Asserts the finding | 1.0 | high |
| Explicitly negates it | 0.0 | high |
| Hedges ("possible", "cannot exclude") | 0.5 | **~0 — this is where your instinct applies** |
| Silent | low prior | low |

The hedged case is exactly where dropping is correct, and it is available as a weight of zero on
that one cell. Silence gets a weak negative rather than a confident one, because a loss that cannot
express the difference trains every silence as a confident negative.

## Severity calibration — fitted, and what it is worth

`python scripts/calibrate_targets.py`. Implemented in
[`rsna_knee.calibrate`](../src/rsna_knee/calibrate.py): a single logistic maps severity onto
P(gold = 1), sharing one slope across all twelve findings and giving each its own ridge-shrunk
intercept, fitted on 384 usable cells from the 58 gold studies. Cross-validation groups by **study**,
because one report drives up to twelve correlated cells and splitting them at random would put the
same report on both sides of the divide.

| model | grouped-CV log loss | Brier |
|---|---:|---:|
| raw severity | 0.593 | 0.166 |
| per-finding base rate | 0.665 | 0.237 |
| **calibrated** | **0.503** | 0.167 |

Fitted P(gold = 1), the readable form of the map:

| finding | sev 0.0 | sev 0.35 | sev 0.65 | sev 0.95 |
|---|---:|---:|---:|---:|
| Synovitis | 0.17 | 0.44 | 0.71 | 0.88 |
| Effusion | 0.15 | 0.41 | 0.68 | 0.87 |
| Lateral Meniscus | 0.14 | 0.38 | 0.65 | 0.85 |
| ACL | 0.11 | 0.33 | 0.60 | 0.82 |
| Medial OA | 0.09 | 0.27 | 0.53 | 0.78 |
| MCL | 0.06 | 0.18 | 0.41 | 0.69 |
| **Contusion** | **0.05** | **0.16** | **0.38** | **0.65** |

Contusion sits lowest at every severity, which is the over-calling seen from the other side: even a
report describing marked marrow oedema only carries the gold label about two thirds of the time. MCL
is the next lowest, consistent with the generic collateral cue added for Greek and Bulgarian
harvesting some LCL statements.

### What this does and does not buy

Log loss improves 15 %; Brier is flat. That difference says the gain is concentrated in the tails —
confidently wrong targets — which is exactly what a BCE training loss is most damaged by and what
Brier is least sensitive to.

But the map is **monotone within each finding** (unit-tested), so it cannot change the ranking of
studies for that finding. The competition metric is per-class ROC-AUC, which reads only rank.
**Calibration therefore leaves the labeller's own AUC untouched**, and the binary agreement table at
a 0.5 cut barely moves — down 0.04 on Medial OA, 0.03 on Contusion, unchanged elsewhere.

That is not a disappointing result, it is the correct scope: the value is entirely in the *training
targets*. A loss fed systematically over-confident targets learns the over-confidence, and Contusion
was being handed 0.8 for cells that are positive half the time. Whether it pays is settled by
training a model on calibrated versus raw targets and comparing CV — not by this table.

## The severity threshold nobody mentions

The gold labels were produced by radiologists reading images against **explicitly
severity-thresholded criteria**. The reports do not use those thresholds. A French report says
`déchirure partielle de bas grade` of the ACL; a Spanish one says `Condropatía con lesiones grado 2`
at the medial femoral condyle. Whether either crosses the threshold for a positive label is a
calibration question, not an extraction question.

This is the **highest-value use of the 58 gold studies** — better than scoring the labeller. Emit a
graded severity from the report, then fit one scalar cut per finding against the gold labels.
Estimating one threshold per finding from ~30 labelled examples is statistically reasonable;
estimating a precision to ±0.02 from the same 58 is not. Spend the gold set on the thing it can
actually determine.

## v3 — the LLM pass, and the blend that ships

`qwen3:14b` at 4-bit via Ollama, `think=false`, the whole 4,407-report corpus in one overnight pass
at **18.3 reports/min** on the RTX 5090 (4,406 written; one report is not a string). Then
`scripts/build_targets.py` blends, calibrates, overlays gold, and writes what training reads.

| variant | macro label AUC | cells asserted | vs rules (95 % CI) |
|---|---:|---:|---|
| rules only | 0.801 | 0.490 | — |
| LLM only | 0.841 | 0.643 | +0.040 [+0.010, +0.074] |
| rules first, LLM fills silence | 0.830 | 0.686 | +0.030 [+0.010, +0.053] |
| LLM first, rules fill silence | 0.838 | 0.686 | +0.038 [+0.008, +0.069] |
| **mean where both assert** | **0.853** | **0.686** | **+0.053 [+0.027, +0.082]** |

**0.801 → 0.853**, and unlike every rule edit in v2 this one clears the measurement floor
comfortably. The ordering is the ordinary ensembling result: two labellers reading the same reports
by different mechanisms have only partly shared errors, so the mean cancels some of both.

Coverage is where the LLM earns it, and it closed exactly the gaps the v2 audit named:

| gap the rules had | rules | LLM |
|---|---:|---:|
| Spanish MCL | 0.09 | 0.63 |
| Turkish effusion | 0.08 | 0.96 |
| Turkish medial meniscus | 0.52 | 1.00 |
| MCL overall (cells asserted) | 0.55 | 0.91 |
| Baker's overall | 0.40 | 0.68 |

### Where the LLM is worse, and why it matters

**Baker's: 0.841 against the rules' 0.918.** A cyst is a keyword — the rules are near-perfect on it
and the LLM adds nothing but noise. The blend keeps most of the rule's edge (0.912).

**Effusion recall 0.32 against the rules' 0.72; Synovitis 0.38 against 0.73.** Not a bug — the
prompt's "label what the report states" makes the LLM turn silence into a confident `absent`, and
these are precisely the findings reports under-report relative to the images the gold labels were
read from. For them the rules' treatment of silence — a weak 0.15 prior at low weight, never a
confident zero — is the better model, which is why the blend beats the LLM on both.

The corollary is the important one: **Synovitis tops out at 0.688 and that is not recoverable from
text.** Gold prevalence is 0.466 and the reports mention it in 16 % of studies at best. Its
remaining ≈0.03 of the score has to come from the imaging model, not from better labelling.

### Two notes on the numbers

- **Calibration costs a little AUC here (0.853 → 0.844), and `docs/labelling.md` used to claim it
  could not.** The old claim holds only if silent cells are excluded. Training fills silence with a
  fixed 0.15 prior, and calibration moves asserted cells across that constant, so it does reorder
  asserted against silent. The effect is small and within noise; calibration is kept because its
  purpose is to stop handing the loss over-confident targets, not to raise this number.
- **Disagreement is not rare** — 8 % of cells on MCL up to 34 % on Lateral OA. Hence the weight rule
  in `blend_labellers`: weight falls with disagreement and reaches zero on a flat contradiction,
  the same treatment hedging already gets.

### Still owed: the adjudication queue

`artifacts/adjudication_queue.csv`, 483 rows sampled from rule-vs-LLM disagreements and stratified
by language. This is the only scrutiny French (81 reports, **0 gold studies**), German (262, 2) and
Dutch (153, 2) can get, and it is a human task — a few hours of reading, machine-translated where
needed, and the highest information per hour left on the labelling side.

## Rule labeller v2 — measured on the real corpus

`python scripts/run_labeller.py`. The headline number is now **label AUC against the 58 gold
studies** (`verify.label_auc`), not the precision/recall table below it. Precision/recall reads the
graded target through an arbitrary 0.5 cut and counts hedged cells — target exactly 0.5, weight
zero, never trained on — as positive predictions, so it scores something training never sees.

| finding | label AUC | asserted | silent |
|---|---:|---:|---:|
| MCL | 0.947 | 0.55 | 0.45 |
| Baker's | 0.918 | 0.40 | 0.60 |
| ACL | 0.916 | 0.68 | 0.32 |
| Lateral Meniscus | 0.872 | 0.69 | 0.31 |
| Medial Meniscus | 0.828 | 0.72 | 0.28 |
| Lateral OA | 0.794 | 0.38 | 0.62 |
| Medial OA | 0.785 | 0.44 | 0.56 |
| Fracture | 0.772 | 0.20 | **0.80** |
| PF OA | 0.741 | 0.50 | 0.50 |
| Effusion | 0.717 | 0.74 | 0.26 |
| Contusion | 0.673 | 0.38 | 0.62 |
| Synovitis | 0.643 | 0.12 | **0.88** |
| **MACRO** | **0.801** | | |

### What v2 changed, and what it bought

Four structural defects were found by reading the code against the corpus, each reproducible on a
single sentence and each locked into `tests/test_labeller.py`:

1. **Severity matched by raw substring** while every other cue set is word-anchored. `sever` fired
   on *several*, `extens` on *extension*, `complet` on *incomplete*; the strongest grade wins, so
   any one of them promoted a mild finding to 0.95. `incomplete tear` graded 0.95 inverts the exact
   distinction the severity channel exists to carry.
2. **Laterality unbound from its noun.** Requiring only co-presence in a clause meant
   "degenerative changes in the medial compartment with a tear of the lateral meniscus" asserted all
   four laterality-paired findings. Fixed by a per-finding proximity window plus an explicit
   medial-vs-lateral contest — the window alone cannot do it, because OA is phrased far more
   diffusely than a meniscus and needs a window three times as wide.
3. **Croatian `medijaln` in the MCL cue list.** It means *medial*, not *medial collateral*, so every
   Croatian medial meniscus tear also became an MCL tear: 0.96 coverage at a 0.57 positive share
   against a gold prevalence of 0.16, and a polarity z of +29. **This is the one unambiguous win —
   MCL went 0.918 → 0.947.**
4. **Negation spanning the whole clause.** "Tear of the lateral meniscus, medial meniscus intact"
   marked the real tear negative. Stopping at commas is the obvious repair and is wrong twice over,
   so scope is now decided by whether the negation has a subject: claimed by the nearest preceding
   finding and its coordinated peers, or unclaimed and distributing forward over its list.

**Net effect on the macro: 0.804 → 0.801, paired-bootstrap 95 % CI [−0.010, +0.003].** In other
words, statistically indistinguishable. The changes were kept anyway, and the reasoning should be
explicit because it is a judgement call rather than a measurement:

- Every changed behaviour is verifiably right on a concrete sentence, and the behaviour it replaced
  was verifiably wrong.
- The errors removed are **systematic** — bound to a language or a report template, not to a study.
  Independent label noise averages out over 4,407 studies and an imaging model can score above its
  labels; noise correlated with the report's language does not average out, and the imaging model
  learns it faithfully. The gold AUC cannot see this distinction at all.
- Two of the losses came with a mechanism worth naming: tightening precision on a low-coverage
  finding *reduces* its AUC, because AUC over the whole set rewards ranking asserted studies above
  silent ones, and a smaller asserted set discriminates over fewer pairs. Where a finding is 60 %
  silent, coverage buys more than precision does.

### The measurement floor, and what it implies

A paired bootstrap over 58 studies resolves about ±0.017 on the macro. Every remaining rule edit is
smaller than that. **Rule tuning against the gold set is finished** — not because the rules are
perfect but because the instrument cannot read the difference, and continuing means fitting 58
studies.

The headroom is legible in the table above, and it is coverage, not precision:

- **Synovitis — 88 % silent, AUC 0.643.** Gold prevalence is 0.466, so nearly half these knees have
  it and the reports almost never say so. This one finding forfeits roughly 0.03 of the final score.
- **Fracture — 80 % silent, AUC 0.772.**
- **Contusion — 62 % silent, AUC 0.673**, still the severity-threshold problem.
- **Effusion — only 26 % silent yet AUC 0.717.** The exception that proves the rule: high coverage,
  poor discrimination. This one is a *quality* problem and worth reading by hand.

That list is the brief for the LLM salvage pass, which targets exactly the silent and hedged cells.

## Rule labeller v1 — results (superseded, kept for the audit trail)

`python scripts/run_labeller.py`. Agreement against the 58 gold studies, `n` = gold studies the
labeller actually asserted on:

| finding | n | precision | recall | accuracy |
|---|---:|---:|---:|---:|
| ACL | 49 | 0.83 | 0.91 | 0.88 |
| MCL | 33 | 0.70 | 1.00 | 0.91 |
| Medial Meniscus | 42 | 0.74 | 0.94 | 0.83 |
| Lateral Meniscus | 39 | 0.87 | 0.76 | 0.85 |
| PF OA | 26 | 0.75 | 0.75 | 0.77 |
| Effusion | 49 | 0.81 | 0.72 | 0.73 |
| Synovitis | 15 | 0.80 | 0.73 | 0.67 |
| Baker's | 23 | 0.77 | 0.91 | 0.83 |
| Fracture | 24 | 0.77 | 0.83 | 0.79 |
| Medial OA | 25 | 0.60 | 0.75 | 0.64 |
| Lateral OA | 17 | 0.56 | 0.62 | 0.59 |
| Contusion | 36 | 0.50 | 0.88 | 0.56 |

Read against Wilson intervals these are wide — at n = 25 a precision of 0.60 spans roughly 0.41–0.77.
The ranking is still informative: **the three weakest are the two tibiofemoral OA classes and
Contusion**, and all three are exactly the findings whose report language is graded rather than
binary. Contusion recalls 0.88 at precision 0.50 — it fires on every marrow signal change, while the
gold label counts only those above a severity threshold. This is the severity-calibration problem
above, showing up precisely where it was predicted, and it is the first thing to fix.

### What the audits caught

The label-free checks earned their place immediately. Four bugs, none of which the gold set could
have localised:

1. **Forward-only negation scope.** Spanish sat at 0.95 positive share on nearly every finding
   because `Menisco lateral … sin signos de rotura` puts the negation *after* the structure. Turkish
   is verb-final, so its negation is always last. Scope is now bidirectional within a clause,
   terminated by a contrast conjunction.
2. **Substring matching.** `normal` occurs inside the Spanish `anormal` — *abnormal* — so the
   strongest positive in the language read as a negation. Cues are now anchored at word boundaries,
   with an explicit opt-out.
3. **Anchored stems miss German compounds.** `Innenmeniskus` appears in 70 % of German reports and a
   boundary-anchored `menisk` reaches none of them. German menisci scored 0.08 coverage; unanchoring
   the stem took them to 0.64.
4. **Fixed multi-word cues miss Slavic declension.** Croatian writes `koštani edem`, `koštanog
   edema`, `kostanim edemom`. Splitting Contusion into an oedema cue and a bone cue took Croatian
   from 0.00 to 0.45 — and has the side benefit of excluding Hoffa fat-pad oedema, which shares the
   first cue and is not a target.

### Residual coverage gaps

Real, and the queue for LLM salvage rather than more vocabulary:

- **Turkish effusion, 0.08** against 0.93 in English. Verified against the raw text: `mayi` appears
  in 1 % of Turkish reports and `efüzyon` in 7 %. The vocabulary is not missing — these reports
  mostly do not comment on effusion at all. Silence, correctly detected.
- **Turkish medial meniscus, 0.12** against lateral at 0.63. An asymmetry this large within one
  language is not case mix; something in the medial cue is failing there.
- **Synovitis everywhere, 0.01–0.19.** The rarest finding in the reports by a wide margin, and the
  one where silence is most likely to be uninformative rather than negative.
- **MCL in el/hr/nl/bg, ≤ 0.09.** Almost certainly missing vocabulary.

## Order of work

1. Build the rule lexicon and run the three label-free checks. They will find most of the errors.
2. Fix coverage and polarity outliers by adding vocabulary, re-run, repeat until no cell is flagged.
3. Add the LLM salvage pass on unmatched and hedged clauses; run `disagreement`.
4. Adjudicate the stratified queue by hand — this is where French and German get their only real
   scrutiny.
5. Only then fit the severity thresholds against the 58, and report gold agreement once, as a sanity
   check rather than a target.
