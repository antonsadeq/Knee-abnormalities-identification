# Plan

Target: a competitive macro-ROC-AUC on the 12 findings, with an inference path cheap enough to also
contest the efficiency award. Deadline **2026-10-22 23:59** — 65 days from 2026-08-18. Public LB
leader is at 0.949 with 1,572 teams entered. **Stated goal: 0.97.**

Design rationale, model comparisons and the traps behind each step are in
[docs/architecture.md](docs/architecture.md). This file is the order of work.

---

## Where this stands (2026-08-18)

| | |
|---|---|
| Submissions | 1 — **0.518**, a series-metadata baseline that reads no pixels. See [docs/experiments.md](docs/experiments.md) |
| Report labelling | **Done.** Rule + LLM blend, macro label AUC **0.853** against the 58 gold studies |
| Training targets | **Built** — `artifacts/train_targets.parquet` (4,407 x 12 graded + weights) |
| Decode path | **Built and tested** — `src/rsna_knee/dicom.py`, 128 tests passing |
| Cache | Kernel written (`notebooks/cache_336/`), **not yet run on Kaggle** |
| Model | **Nothing trained.** This is the whole remaining project |

Reference points: public baselines at **0.891** (20x frozen DINOv2-small, 224 px) and **0.906**
(rules labeller); efficiency-LB rank 1 at **0.936**.

---

## What 0.97 actually requires

Worth doing the arithmetic before committing to the number, because it is unforgiving. A macro mean
of 0.97 over twelve columns needs the AUCs to sum to **11.64**. The leader's 0.949 sums to 11.388, so
0.97 is **+0.021 per class averaged across all twelve**, against a leader who has two more months to
improve as well.

The metric's structure says where that can and cannot come from. **Every column costs the same**, so
one weak class dominates:

| if eleven classes average… | the twelfth must reach… |
|---|---|
| 0.975 | 0.915 |
| 0.980 | 0.860 |
| 0.985 | 0.805 |

So 0.97 is not reached by lifting everything a little. It is reached by **being ordinary on the
detectable findings and decisively better on the two or three that are hard for everyone**. Inverting
the leader's score the same way: if their Synovitis sits near 0.60, their other eleven average 0.981.

### Score budget, honestly labelled

Starting from the 0.906 public rules-labeller notebook as the thing to beat:

| lever | expected | confidence | why |
|---|---|---|---|
| Laterality from DICOM geometry | **+0.010 – 0.030** | high | The `Laterality` tag is Type 2C and absent on ~half the studies by whole vendors. Five of twelve targets are side-named. If the public baselines trust the tag, those five are learning from an axis they cannot observe. Biggest single differentiator available. |
| 336 px vs 224 px | +0.005 – 0.010 | high | 0.387 vs 0.580 mm/px against the ≤0.5 a 1–3 mm meniscal tear needs. Concentrated in the 4 meniscus/cartilage classes, so ~1/3 of the metric. |
| Better labels (0.801 → 0.853) | +0.010 – 0.020 | medium | Downstream effect is not 1:1; unbiased label noise partly averages out either way. |
| Per-diagnosis slot attention, masked | +0.005 – 0.015 | medium | Mean-pooling 6 slots dilutes the one carrying the evidence. Only 566 of 4,407 studies have all six slots, so the mask is load-bearing. |
| Ensemble, rank-averaged | +0.010 – 0.020 | high | Multi-seed / multi-architecture. AUC reads rank only, so average ranks not probabilities. |
| Fixing the weakest 2–3 classes | **+0.020 – 0.040** | low | The only lever big enough to clear the leader. Also the least certain. |

Optimistic sum lands near 0.97; the pessimistic sum lands near 0.93. **Treat 0.97 as the stretch
target and ~0.94–0.95 as the realistic landing zone**, and note that the efficiency track (rank 1 at
0.936, median 0.836, a first-time award, less crowded) is the better-odds prize for the same model.

### The one class that decides it

**Synovitis.** Gold prevalence 0.466 — nearly half these knees have it — and the reports mention it
in **at most 16 % of studies in any language**. Label AUC tops out at 0.688 and the LLM pass barely
moved it, because the information is not in the text at all. Every team using report-derived labels
faces the same wall.

That makes it the highest-leverage target on the board: it is worth ~0.03 of the macro on its own,
and it cannot be won by better labelling. It has to be won from images — pretraining, semi-supervised
learning on the unlabelled signal, or accepting the 58 gold studies as the only real supervision for
that head and training it accordingly. Fracture (80 % silent, AUC 0.772) and Contusion (62 % silent,
0.673) are the same problem one step less severe.

---

## Phase 0 — Access and ground truth *(mostly done)*

- [x] Kaggle token installed at `~/.kaggle/access_token`; API access confirmed.
- [x] Rules accepted (`userHasEntered: True`).
- [x] Metadata CSVs downloaded to `data/` and profiled.
- [x] Real label column names recorded in `rsna_knee.config.LABELS`.
- [x] Metric confirmed as the unweighted mean of the twelve ROC-AUCs.
- [ ] Read the Overview / Rules / Efficiency-Prize-Evaluation tabs and clear the remaining boxes in
      [docs/competition.md](docs/competition.md) — the external-data and pretrained-weights policy,
      the exact efficiency formula, and **whether an external LLM API may be used to label the
      reports** (an open-weight local model sidesteps this entirely — see Phase 2).

---

## Phase 0.5 — Take the baseline's code, not its ceiling

A 0.891 public baseline exists (`pilkwang/rsna-knee-baseline-v1`) and a 0.906 notebook builds on it
(`prvsiyan/rsna-knee-read-the-report-then-the-knee`). Both are worth mining. Neither is worth
inheriting as the thing you improve, because both are **20 frozen DINOv2-small checkpoints at
224 px** and that is a dead end in three separate ways:

- 224 px at a 130 mm crop is 0.580 mm/px, coarser than the 0.5 mm/px a meniscal tear needs. The
  meniscus classes are capped by the resize before any weight is trained.
- The weights are frozen and the kernel is inference-only, so "improving" it means tuning blend
  weights over checkpoints someone else trained. That is the definition of getting stuck early.
- A 20-member ensemble is disqualifying for the efficiency track, where rank #1 scores 0.936.

A strong **single** model serves both tracks; a 20-member ensemble serves only one. Build the single
model, and keep ensembling as the last step rather than the first.

- [ ] Run the public baseline once, unmodified, to confirm the environment and get a CV↔LB anchor.
      Record it in `docs/experiments.md` as a reference row, not as the thing to beat incrementally.
- [ ] Lift the *code*: the DICOM decode path, laterality recovery, the six-slot layout and the
      per-diagnosis attention head. Attribute both notebooks.
- [ ] Train one model at **336 px** on your own labels as the real starting point. Expect it to
      score below 0.891 at first — a single fine-tuned model against a 20-member ensemble — while
      having headroom in three independent directions (labels, resolution, architecture) that the
      ensemble does not.

**Exit:** a reference submission on the board, and a single-model training run that is yours end to
end.

---

## Phase 1 — EDA

The metadata is small; this should take hours, not days.

- [ ] Report corpus: language detection over all 4,407, length distribution, section structure per
      language, how often each finding's vocabulary appears.
- [ ] Site / scanner inference. There is **no site column** in the CSVs, but the challenge spans 16
      sites. Extract `InstitutionName`, `Manufacturer`, `ManufacturerModelName`,
      `MagneticFieldStrength`, `StudyDate` from a single DICOM header per study — one file read per
      study, not per slice. This is the basis for grouped validation.
- [ ] Series geometry from headers: slice count, in-plane resolution, slice thickness, per plane and
      per site.
- [ ] Confirm the 58 labelled studies are not clustered in one site or language — if they are, they
      are an even weaker calibration set than their size suggests.

**Exit:** `notebooks/01_eda.ipynb`, a written summary in `docs/eda.md`, and
`artifacts/study_meta.parquet` with one row per study carrying site/scanner fields.

---

## Phase 2 — Report labelling *(done)*

Rules first, LLM salvage, then blended. Full results in [docs/labelling.md](docs/labelling.md).

| labeller | macro label AUC vs the 58 gold | cells asserted |
|---|---:|---:|
| rules v1 | 0.804 | 0.49 |
| rules v2 (four structural bugs fixed) | 0.801 | 0.49 |
| qwen3:14b over the full corpus | 0.841 | 0.64 |
| **mean where both assert** | **0.853** | **0.69** |

- [x] Normalisation, negation and hedging scope, per-finding graded targets and weights.
- [x] Four structural defects found and fixed, each pinned by a test: unanchored severity cues,
      laterality unbound from its noun, Croatian `medijaln` in the MCL list, clause-wide negation.
      Net effect on the gold macro was **nil** (CI [-0.010, +0.003]) — they were kept because the
      errors were systematic rather than random, and systematic label error does not average out.
- [x] LLM pass: 4,406 reports at 18.3/min on the RTX 5090, one overnight run.
- [x] Blend, calibrate, gold overlay → `artifacts/train_targets.parquet`.
- [ ] **Adjudicate `artifacts/adjudication_queue.csv`** (483 rows, language-stratified). Still the
      only scrutiny French (0 gold studies), German (2) and Dutch (2) can get. A human task.

**Lesson worth carrying forward:** a paired bootstrap over 58 studies resolves about ±0.017, so any
change smaller than that is unmeasurable here. Rule tuning hit that floor; the LLM pass cleared it.
Do not tune against the gold set below its resolution.

## Phase 3 — Validation protocol *(done)*

Implemented in [`src/rsna_knee/folds.py`](src/rsna_knee/folds.py) (16 tests), frozen to
`artifacts/folds.csv` by [`scripts/build_folds.py`](scripts/build_folds.py).

- [x] **Site metadata recovered** via a header-only Kaggle kernel
      ([`notebooks/study_meta/`](notebooks/study_meta/)), one header per series, pixels never touched.
- [x] **Grouped by report-text hash.** 204 studies in 54 clusters share a report byte-for-byte after
      whitespace normalisation, hence share a derived target vector exactly.
- [x] **Two fold columns, and the reason is a measurement, not a preference** — see below.
- [x] **Verified: 0 groups span folds** on both columns. Fold sizes 882/882/881/881/881; worst
      per-label prevalence spread across folds **0.006**.
- [x] Gold studies distributed 13/10/14/5/16 — kept *in* training, so the annotation check is
      restricted to the held-out portion.
- [ ] Log every run in [docs/experiments.md](docs/experiments.md) with CV and, once submitted, LB.

### There is no site column, and there is no recovering one

`DeviceSerialNumber`, `InstitutionName` and `StationName` are **0 % present** across all 4,407
studies — anonymised out. The sharpest available identifier is vendor + model + field strength, which
yields **51 scanner keys** after normalising the vendor spellings ("Siemens Healthineers", "SIEMENS"
and "Siemens" are one vendor; "3" and "3.0" one field strength). Two sites running the same model
cannot be separated; two sites sharing a model get grouped together, which errs conservative.

### Why two fold columns

The obvious design — group by report hash *and* scanner as connected components — collapses. **30 of
the 54 duplicate-report clusters span more than one scanner**, so templates chain scanners
transitively: 51 scanners fuse into 38 components and the largest holds **53 % of the corpus**. No
5-fold split of that is balanced, and the first attempt produced a fold with 2,324 studies against
~520 for the rest.

The two constraints are not equally hard, so they are separated:

| column | construction | use |
|---|---|---|
| `fold` | grouped by report text; **stratified** on the 12 labels *and* scanner | model selection — balanced (882/882/881/881/881) and free of the hard leak |
| `fold_site` | grouped by scanner, each report cluster assigned whole to its majority scanner | cross-site generalisation — 50 groups, 10–16 scanners per fold |

`fold_site` carries a residual leak of **149 studies (3.4 %)** whose report cluster spans scanners.
Read it as the pessimistic number and `fold` as the selection number. **If they diverge badly, the
model is riding site signal** — which is precisely the 0.072 shortcut the 0.518 submission exposed.

### Laterality: the geometry method, validated at scale

The header pass made this measurable, and it is the strongest single result so far. On the 2,170
studies where both a `Laterality` tag and a geometry call exist:

| | |
|---|---|
| Agreement | **97.65 %** (51 disagreements) |
| Tag coverage | 49.6 % of studies — the docs' "absent on about half" was right |
| Geometry coverage | **99.0 %** |
| Either | 99.3 % |

So geometry **doubles** the usable laterality coverage at 97.7 % accuracy. Five of the twelve targets
depend on it, and any competitor trusting the tag alone is normalising half the corpus.

One improvement left on the table: where the tag exists it is probably the better authority, so a
future cache rebuild should prefer tag-then-geometry rather than geometry throughout. Worth at most
~2 % of studies being un-mirrored, so it is not worth rebuilding 45 GB for now — revisit if the
medial/lateral classes underperform.

---

## Phase 4 — Preprocessing and cache *(module done; cache not yet built)*

Implemented in [`src/rsna_knee/dicom.py`](src/rsna_knee/dicom.py), 34 tests in
`tests/test_dicom.py`, validated against a real Siemens axial series.

- [x] **Crop to a constant physical extent (130 mm), then resize.** `crop_to_extent` returns whether
      it actually cropped, because a crop wider than the field of view is a silent no-op that leaves
      that study at a different scale from every other. Verified: 150 mm FOV → exactly 130.0 mm.
- [x] **336 px** → 0.387 mm/px, under the ≤0.5 a 1 mm meniscal tear needs.
- [x] **Laterality from geometry**, not the Type 2C tag: sign of the mean image-centre *x* in the
      patient frame, `unknown` within 15 mm of the midline. On the validation study the geometry says
      `L` and the DICOM tag independently agrees. Note the centre is **not** `ImagePositionPatient`,
      which is the first *voxel* — they differ by 63 mm in x on that series.
      Resolved **once per study**, not per series: a study cannot be half-normalised.
- [x] **Slice order and in-plane axes normalised from the direction cosines**, not assumed. Sorting
      by `InstanceNumber` would inherit the vendor's acquisition direction. Tested by feeding slices
      in reverse and requiring identical output.
- [x] Per-series percentile windowing over the **stack**, not per slice.
- [x] Slot selection, one series per (plane × fluid-sensitive) cell. Measured occupancy across all
      4,407 studies: mean **4.84 of 6**, distribution `{3: 4, 4: 1259, 5: 2578, 6: 566}` — so the
      presence mask is load-bearing, not defensive.
- [x] Kaggle kernel written: [`notebooks/cache_336/build_cache.py`](notebooks/cache_336/build_cache.py).
      Dry-run against a synthetic mount (which caught a real `np.save` extension bug).
- [x] **Ran on Kaggle, 4 shards, 2026-08-18.** Driven from the CLI by
      [`scripts/run_cache_kernel.py`](scripts/run_cache_kernel.py) — no UI. 45 GB in
      `artifacts/kaggle_cache/`, combined manifest at `manifest_all.parquet`.
- [x] **Verified, not assumed:** 4,407 arrays, 4,407 manifest rows, **zero cross-shard overlap, zero
      missing, zero unexpected, zero misassigned**, and `status == ok` for all 4,407.

| | |
|---|---|
| Laterality | R 2,310 / L 2,052 / **unknown 45 (1.02 %)** — against the ~50 % the Type 2C tag would give |
| Slots per study | mean **4.84**, distribution `{3: 4, 4: 1259, 5: 2578, 6: 566}` — *identical* to the prediction from `train_series.csv`, so series discovery missed nothing |
| Uncropped (FOV < 130 mm) | 160 studies (3.6 %) — these sit at a different physical scale; revisit if the OA classes underperform |
| Slot with < 16 slices | 32 studies (0.7 %), so slice repetition is negligible |
| Decode cost | 3.0 h CPU total, median 2.19 s/study, ~15 min wall per shard on 4 vCPUs |

Slot presence, which is what the masked softmax has to cope with:
`Axial_1` 100 %, `Sagittal_0` 96.8 %, `Coronal_1` 96.4 %, `Sagittal_1` 94.2 %, `Coronal_0` 77.3 %,
**`Axial_0` 19.5 %**. That last one is the reason absent slots must be masked rather than zero-filled.

Two bugs the run exposed, both fixed and worth remembering:

1. **`sorted(INPUT.rglob(name))` cost 451 of 464 seconds** on the validation run — `sorted`
   materialises every match, forcing a walk of an 819k-file mount. Real decode was 8 seconds. The
   mount is `competitions/<comp>/` at depth 2 but `datasets/<user>/<name>/` at depth 3, so both
   depths are now probed explicitly before any walk.
2. **The driver read a `ConnectionError` traceback as the kernel failing**, because it substring-matched
   `"error"` against the whole command output, and abandoned a job that was running fine. Terminal
   state is now parsed from the `KernelWorkerStatus` enum, unreachable-API is treated as transient,
   and `--attach` recovers a running version without re-pushing.

**Why on Kaggle rather than locally:** the raw archive is ~570 GB and Kaggle serves it as one zip,
needing ~1.1 TB to download and extract against 673 GB free. Kaggle already has it mounted. The
decisive reason is different though — this is the same decode path the submission kernel must run, so
debugging it there means not rediscovering every environment difference on submission day.

**Exit:** four cache shards downloaded, plus a manifest that says how many studies failed and why.

---

## Phase 5 — Single-plane baseline

- [ ] 2.5D: sample N slices per series, feed a 2D CNN (ConvNeXt-T / EfficientNet-B0 / ResNet-50)
      with a per-slice encoder plus attention pooling.
- [ ] Multi-label BCE against the Phase 2 soft targets; upweight the 58 gold studies heavily.
- [ ] Mixed precision. **32 GB VRAM on the RTX 5090**, so 336 px at a real batch size is
      straightforward and gradient accumulation is mostly unnecessary — the docs' 16 GB
      Quadro RTX 5000 was a different machine.
- [ ] Augmentation: rotation, intensity jitter, random slice subsets. **Horizontal flips need
      care** — flipping a coronal or axial image swaps medial and lateral anatomy, so the
      `Medial/Lateral Meniscus` and `Medial/Lateral OA` label pairs must swap with it. See
      `config.LATERALITY_PAIRS`. Silently corrupting four of twelve targets is an easy way to lose
      the competition without noticing.

**Exit:** a CV number beating the prior baseline, and one submission to calibrate CV↔LB.

---

## Phase 6 — Multi-plane study-level model

- [ ] **Give each of the 12 diagnoses its own learned query** attending over the 6 slot embeddings,
      with absent slots masked out of the softmax. Each finding is read on particular sequences, so a
      plain mean over slots dilutes the one carrying the evidence with five that do not; the masked
      softmax renormalises over whatever the study actually contains instead of feeding a zero
      vector.
- [ ] Keep the head **small**. The label is attached to the study, so nothing in the supervision says
      which part of the study carries the finding — a richer aggregator has no signal to learn from.
- [ ] Encoder diversity for the ensemble: fine-tuned DINOv2/DINOv3 ViT alongside RadImageNet-
      pretrained ResNet-50 and EfficientNet-B3. Use their disagreement rather than picking a winner.
- [ ] Findings are anatomically localised — test whether coarse region cropping (medial vs. lateral
      compartment, patellofemoral) helps the OA classes specifically.
- [ ] **Spend effort on the rare findings.** A label left at chance forfeits (M − 0.5)/12 ≈ 0.029 at
      M = 0.85, larger than the gap between many neighbouring leaderboard places. A rare finding is
      where a model most easily lands at chance, so it deserves more attention than a common one.

**Exit:** best single model, plus a per-class AUC table identifying which findings remain weak.

---

## Phase 7 — Efficiency track

Run alongside Phase 6, not after — efficiency constraints affect architecture choice.

**This prize is not won by being fast and mediocre.** On the official efficiency leaderboard,
rank #1 scores 0.936 against a main-LB leader of 0.949, and the median is 0.836. The winner is
nearly as accurate as the leaders *and* much faster.

**The cost is I/O, not arithmetic.** A study is ~150 DICOM files, and decode dominates. A cheaper
backbone buys far less than reading fewer slices. Levers in order of payoff:

- [ ] Read the Efficiency Prize Evaluation page first — the runtime normaliser decides whether the
      last 20 % of speed is worth any accuracy at all.
- [ ] Measure inference wall-clock per study on Kaggle hardware, split into decode vs. forward pass.
- [ ] **Decode fewer slices per slot** — linear saving, sublinear accuracy cost.
- [ ] **Drop slots** the per-diagnosis attention shows it ignores.
- [ ] Lower resolution — the direct tension with the Nyquist bound in Phase 4, which is why this is
      a separate submission rather than a tuning of the accuracy model.
- [ ] **Then** backbone size, half precision, ONNX/TensorRT.
- [ ] Distil the accuracy ensemble into one small model rather than training it from scratch — the
      ensemble's soft outputs are a far richer target than the report labels.
- [ ] Plot the accuracy/latency curve and pick the point that maximises the efficiency score.

**Exit:** a separate lightweight submission kernel.

---

## Phase 8 — Ensemble and submission

- [ ] Multi-seed, multi-fold, multi-architecture ensemble; per-class blend weights fitted on OOF.
- [ ] **Blend by rank, not by probability.** AUC is invariant under any strictly increasing map of
      each label's scores, so averaging raw probabilities lets whichever member is most confident
      dominate, while averaging ranks combines the only information the metric reads.
- [ ] TTA within the runtime budget.
- [ ] Submission kernel: internet off, end-to-end within the cap.
- [ ] Dry-run against the full hidden test size with headroom. Submit early; do not discover a
      timeout on 22 October.

**Exit:** two selected final submissions — best-CV and most-robust.

---

## Risks

| Risk | Mitigation |
|---|---|
| **Label quality is the ceiling** — 58 gold labels, 4,349 report-derived | Phase 2 gets the most effort; graded targets; audit against the 58 |
| Negation handling fails in one language → inverted labels for that site | Per-language spot-checks; LLM salvage on unmatched clauses |
| Duplicate template reports leak across the validation split | Group the split by report-text hash (Phase 3) |
| Resolution below the Nyquist bound silently caps the meniscus classes | ≥ 336 px at a 130 mm crop for the accuracy model (Phase 4) |
| Unrecovered laterality makes 5 of 12 targets unlearnable | Recover side from DICOM geometry, not the Type 2C tag (Phase 4) |
| CV does not track LB (site shift across 16 sites) | Site-grouped folds from DICOM headers; submit early to calibrate |
| Overfitting to the 58 gold studies by using them for model selection | They are an audit set only; never a selection signal |
| Runtime timeout in the submission kernel | Measure per-study inference from Phase 5; keep headroom |
| Raw archive does not fit extracted (247 GB zip + 570 GB extracted vs 668 GB free) | Decode from inside the zip, or per-study extract-and-delete; never a full extraction |
| Rare classes drag the macro mean | Per-class monitoring; targeted sampling and loss weighting |
| Horizontal flips silently corrupt medial/lateral labels | Explicit label swap on flip, unit-tested |
