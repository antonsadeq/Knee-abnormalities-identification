# Architecture: options evaluated and the design chosen

Written after profiling the data and reading the top public notebooks. Evidence for each claim is
cited so the reasoning can be re-checked when the rules pages are finally read by a human.

## Two constraints that eliminate most of the design space

### 1. There is no report at inference time

`train.csv` has a `Report` column. **`test.csv` does not** — it is `StudyInstanceUID` and nothing
else, and `test_series.csv` carries only plane and sequence flags. Verified directly against the
downloaded files, and stated independently in the most-read public notebook on the competition
([prvsiyan, "read the report, then the knee"](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee),
218 votes): *"Text is available when fitting and absent when predicting. That rules out a fusion
model with a text branch — at inference it would have nothing to read."*

This kills any architecture with a text encoder in the inference graph. There is no late-fusion
decision head over (LLM output, detector output), because one of those two inputs does not exist at
prediction time.

The LLM does not disappear — it moves. It becomes an **offline label generator** that runs once over
the 4,349 unlabelled training reports and is then discarded. The pipeline is staged in *time*, not
in the forward pass:

```
Stage A  (offline, once)     reports ──LLM/rules──▶ 12 soft targets + per-finding confidence
Stage B  (training)          DICOM + Stage A targets ──▶ image-only model
Stage C  (inference, Kaggle) DICOM ──▶ 12 probabilities        [no text anywhere]
```

RSNA's press language — "the first to use both images and the text of radiology reports to train and
test AI models" — describes the *test labels* being derived by radiologists, not reports being handed
to your model at test time. Do not design against the press release.

### 2. There are no segmentation masks

The supervision is twelve binary flags attached to a **study**. Not to a slice, not to a pixel.
Nothing in the dataset says *where* a finding is.

That rules out UNet, DeepLabv3+ and DDRNet-23-slim **as segmentation networks**. A segmentation
decoder produces a per-pixel map for which there is no target and no use — you would train it
through a pooling layer, paying its parameters and FLOPs for a spatial output you immediately
discard. It is strictly dominated by using the encoder alone.

DDRNet-23-slim deserves a specific note because the efficiency track makes it tempting: it is a
real-time semantic-segmentation network tuned for driving scenes at high resolution, and its dual-
resolution design exists to keep boundary detail for dense prediction. Strip the decoder and what
remains is a modest ImageNet-class backbone with no medical pretraining — weaker than an
EfficientNet of the same cost and far weaker than a DINOv2/RadImageNet initialisation. The
efficiency prize is not won by a cheap backbone (see below); it is won on I/O.

The task is **multi-label study classification under weak supervision**, and the shape that fits is
encoder → slice aggregation → series aggregation → 12 heads.

## The design

### Stage A — report labeller

Input: 4,407 reports, nine languages across three scripts (Latin, Greek, Cyrillic), ~1,100
characters each. Output: a 4,407 × 12 matrix of **graded** targets plus a per-finding confidence.

Two properties matter more than raw accuracy:

- **Negation and hedging.** For several findings, *most* mentions are negative — "No hay
  alteraciones de señal", "kein Hinweis auf", "Aucune". A matcher that misses these produces
  inverted labels, which is worse than no label. Hedges ("possible", "cannot exclude") are a third
  state, not a positive.
- **Silence ≠ negative.** A report that never mentions synovitis should pull weakly on the synovitis
  head. Emitting a confident 0 trains every silence as a confident negative. This is what the
  confidence channel is for, and it is why the targets are graded rather than binary.

Known corpus traps: reports are hard-wrapped mid-sentence at a fixed column, so splitting on
newlines severs findings from their anatomy; headings sit on their own line above their value
(`Fracturas :` / `Aucune.`); Greek reports spell mu as `U+00B5` rather than `U+03BC`, so normalise
NFKD before matching; post-operative descriptions and comparisons to priors mention structures
without asserting current pathology.

#### LLM options

Your instinct to use an LLM here is right — this is the single highest-leverage component. On the
model choice, the literature is clear that you do not need a frontier API model:

| Option | Assessment |
|---|---|
| **Open-weight multilingual LLM run locally** (Qwen3-class, Gemma-3-class, ~14–32B, 4-bit) | **Recommended.** 4,407 reports is a trivial corpus — one overnight pass on the 16 GB RTX 5000. Fully reproducible, no API dependency, no external-data ambiguity, and the labelling script ships with the winning code. Peer-reviewed work finds open-weight models [competitive with GPT-4o at extracting findings from free-text reports](https://pubs.rsna.org/doi/10.1148/radiol.240895) and [top open-weight models exceeding rule-based extractors and matching pooled radiologist performance](https://link.springer.com/article/10.1186/s41747-026-00738-7). |
| **Frontier API model (GPT-class)** | Viable and cheap — ~400k input tokens for the whole corpus. But it introduces a rules question (is an external paid service "reasonably available to all participants"?), a reproducibility liability if you win and must open-source, and a dependency you cannot re-run deterministically. Use it as a *cross-check* on a sample, not as the production labeller. |
| **Rule lexicon alone** | Fast, auditable, and the public notebooks show it works — but coverage is the failure mode: a phrasing no language contributes stays unmatched. |
| **Hybrid: rules first, LLM salvage** | The best-performing published pattern for this exact task ([F1 0.99 on PE CT reports](https://link.springer.com/article/10.1186/s41747-026-00738-7)). Rules handle the high-confidence majority; the LLM adjudicates only the clauses rules cannot resolve. Cheapest to run, easiest to audit. |

**Recommendation: hybrid.** Build the rule lexicon first because it is auditable and forces you to
read the corpus, then route unmatched or hedged clauses to a local open-weight LLM. Validate both
against the 58 gold studies. Keep an API model as an occasional referee, never in the pipeline.

Do not skip the rule pass and go straight to the LLM — you will have no way to tell when the LLM is
confidently wrong in Turkish.

### Stage B — imaging model

**Slots.** Each series has exactly two degrees of freedom in the metadata: plane (Sagittal /
Coronal / Axial) and fluid-sensitivity (`Fluid_Sensitive`, which is bit-identical to
`Fat_Suppression` in all 24,371 rows). That is **6 slots** per study. Every study has all three
planes, so slot presence is high but not guaranteed at the sequence level.

**Physical scale, not pixel count.** DICOM pixel spacing varies across the 16 sites, so a fixed
resize hands the encoder anatomy at wildly different physical scales. Crop to a constant physical
extent (~130 mm covers the joint and sits inside almost every acquired FOV) and then resize.

The resolution choice follows from anatomy, not convention. A feature of width *d* mm cannot survive
resampling to a pitch coarser than *d*/2. **A meniscal tear is 1–3 mm**, so the pitch must be
≤ 0.5 mm/px. At a 130 mm crop:

| Input size | Pitch | 1 mm tear survives? |
|---|---|---|
| 224 px | 0.580 mm/px | **No** |
| 336 px | 0.387 mm/px | Yes |
| 384 px | 0.339 mm/px | Yes |

This is a property of the resize, not of the network — capacity downstream cannot recover signal
destroyed before the first convolution. It also explains why 224-px baselines plateau on the
meniscus classes specifically. Treat 336 as the floor for the accuracy model; it is the single
highest-value hyperparameter here.

**Laterality normalisation.** Four targets are medial/lateral pairs and MCL is named for its side.
Medial and lateral are defined relative to the body midline, so which side of the *image* they fall
on depends on which knee was scanned — unnormalised, those five labels learn from an axis the model
cannot observe. Correction differs by plane: coronal and axial flip the last axis; **sagittal
reverses slice order** (mirroring a sagittal slice does nothing). The DICOM `Laterality` tag is
Type 2C and is absent on roughly half the studies, by whole vendors — so recover the side from the
sign of the image-centre *x* coordinate in the patient coordinate system, and leave studies near the
midline unresolved rather than guessing.

This is also the fix for the flip-augmentation trap already recorded in `config.LATERALITY_PAIRS`.

**Encoder.** Public solutions converge on self-supervised ViTs — DINOv2-small/base and DINOv3 — with
RadImageNet-pretrained ResNet-50 and EfficientNet-B3 as strong diverse ensemble members. Pick by
track: a fine-tuned ViT for accuracy, a smaller CNN for efficiency, and use their disagreement as
ensemble diversity rather than picking one winner.

**Aggregation.** Per-slice embeddings pool within a slot (MIL / attention), then **each of the 12
diagnoses gets its own learned query attending over the 6 slot embeddings**, with absent slots
masked out of the softmax. This matters because each finding is read on particular sequences — an
effusion on fluid-sensitive images, cartilage on others — and a plain mean over slots dilutes the one
slot carrying the evidence with five that do not. Masked softmax also renormalises over whatever the
study actually contains instead of feeding a zero vector.

Keep this head **small**. The label is attached to the study, so nothing in the supervision says
which part of the study carries the finding; a richer aggregator has no signal to learn from. Where
supervision is coarse, so is the head.

**Loss.** Weighted BCE against Stage A's graded targets, with the per-finding confidence as the
sample weight, and the 58 gold studies upweighted heavily.

### Stage C — ensembling

The metric is the unweighted mean of 12 ROC-AUCs, which is invariant under any strictly increasing
map of each label's scores. Two consequences:

- **Calibration is worth nothing**, and no threshold matters.
- **Combine by rank, not by probability.** Averaging raw probabilities lets whichever member happens
  to be most confident dominate; averaging ranks combines the only information the metric reads.

And the sharpest consequence: **every label costs the same.** If a good model averages AUC *M*, a
single label left at chance forfeits (*M* − 0.5)/12 — at *M* = 0.85 that is **0.029**, larger than
the gap between many neighbouring leaderboard places. A rare finding therefore deserves *more*
attention than a common one, because a rare finding is where a model most easily lands at chance.
Spend your effort on the findings the reports mention least.

## The efficiency track

The critical fact, from the official [Efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)
(1,395 teams, downloaded and inspected):

**Efficiency rank #1 scores 0.936.** Rank #2 scores 0.943. The main leaderboard leader is at 0.949.

So the efficiency prize is not won by being fast and mediocre — it is won by being *nearly as
accurate as the leaders* and much faster. A weak fast model does not place; the median efficiency-LB
score is 0.836 and the bottom is 0.468.

**Where the time actually goes:** a study holds ~150 DICOM files across its series. The pipeline
cost is dominated by **reading and decoding, not arithmetic**. This inverts the usual intuition:
choosing a cheaper backbone buys far less than reading fewer slices. The efficiency levers, in
order of payoff:

1. **Decode fewer slices per slot.** Linear, immediate, and the accuracy cost is sublinear.
2. **Fewer slots.** Drop the slots that the per-diagnosis attention shows it ignores.
3. **Resolution.** Cache size grows with the *square* of resolution and only linearly with slice
   count — so coverage is the cheap axis and resolution the expensive one. This is exactly the
   tension with the Nyquist bound above: the accuracy model wants 336+, the efficiency model wants
   224. Run them as two separate submissions.
4. **Then** backbone size, half precision, ONNX/TensorRT.

Distil the accuracy ensemble into a single small model rather than training the small model from
scratch — the ensemble's soft outputs are a much richer target than the report labels.

The exact efficiency formula is on the competition's Efficiency Prize Evaluation page and has not
been read yet — it is JS-rendered. Kaggle's standard form trades a normalised score penalty against
normalised runtime; read it before optimising, since the runtime normaliser determines whether the
last 20 % of speed is worth any accuracy at all.

## Validation traps specific to this dataset

- **Duplicate reports.** Some reports are byte-identical across studies — a template read for an
  unremarkable knee — so those studies share a derived target vector. Splitting such a group across
  the train/holdout boundary scores the model on a target whose source it trained on. **Group the
  split by a hash of the report text.**
- **The 58 gold studies are an audit set, not a validation set.** Per-class AUC on n = 58 has a
  standard error around ±0.07. Keep them *in* training at elevated weight — they are the only labels
  read from images rather than text — and measure agreement only on the held-out portion. Scoring a
  model on upweighted training examples measures memorisation and reports it as skill.
- **Prevalence drift.** The data description warns prevalence need not match across train, public
  and private sets. AUC is in expectation invariant to the positive rate, so this is survivable —
  but any threshold you introduce for validation convenience must never reach a submitted score.

## Public baselines to build from

| Notebook | Score | What to take |
|---|---|---|
| [pilkwang/rsna-knee-baseline-v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | 0.891 | The reference 20-checkpoint DINOv2-small ensemble and its weights package |
| [prvsiyan/rsna-knee-read-the-report-then-the-knee](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) | 0.906 | The nine-language report lexicon, laterality recovery, slot attention head |
| [ryanholbrook/rsna-knee-abnormalities-efficiency-lb](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb) | — | Official efficiency standings |
| Main LB leader | 0.949 | Target |

Starting from a 0.891 public baseline and improving it is a far better use of the remaining nine
weeks than reimplementing the decode pipeline. Reproduce first, then attack the report labeller,
then the resolution bound.
