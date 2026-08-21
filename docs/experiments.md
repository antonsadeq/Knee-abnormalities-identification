# Experiment log

One row per scored run. Fill in the LB column only after an actual submission — the CV↔LB
relationship is the single most valuable thing this table records.

| Date | ID | Description | CV (macro AUC) | LB | Notes |
|------|----|-------------|----------------|----|-------|
| 2026-08-16 | B0 | Series-metadata baseline. Features are per-study counts over the 6 (plane × fluid-sensitive) slots; **no pixel is read**. Targets are calibrated report labels + the 58 gold at weight 8. | 0.590 derived / **0.521 gold** | **0.518** | Deliberate floor, not a failed model — the kernel header predicted ~0.52. See below. |
| 2026-08-18 | **M1** | **First model that reads pixels.** convnext_tiny (`in_chans=1`) → attention-pool over slices → 12 per-diagnosis queries over 6 slots, absent slots masked out of the softmax. 336 px, 8 slices/slot, fold 0, 12 epochs, bs 4 × accum 2. | **0.783 derived / 0.846 gold** (ep 8) | — | Not submitted. Overfits from epoch 8; gold rests on 13 held-out studies, so read the derived column. |

## Per-class tracking

Keep a per-finding AUC breakdown for the best run of each phase. The macro mean hides which
findings are near-random, and the rare classes (fracture, synovitis) are where the headroom is.

| Run | ACL | MCL | Med Men | Lat Men | Med OA | Lat OA | PF OA | Effusion | Synovitis | Baker's | Contusion | Fracture |
|-----|-----|-----|---------|---------|--------|--------|-------|----------|-----------|---------|-----------|----------|
| B0 | — | — | — | — | — | — | — | — | — | — | — | — |
| M1 derived (n=882) | 0.835 | 0.742 | 0.821 | 0.795 | 0.744 | 0.725 | 0.761 | 0.795 | 0.792 | 0.787 | 0.768 | 0.820 |
| M1 gold (n=13) | 0.725 | 1.000 | 0.889 | 0.905 | 0.955 | 0.750 | 0.905 | 0.933 | 0.619 | 1.000 | 0.583 | 0.889 |

The M1 gold row is shown for completeness and should not be acted on: 13 studies put two or three positives behind each column, which is how MCL and Baker's reach exactly 1.000. The derived row is the usable per-class signal.

B0's per-class breakdown was not captured. Capture it from now on: the macro mean cannot tell you
which findings are at chance, and a single column left there costs ≈ 0.029.

## What M1 established

**Derived and gold climbed together.** That is the single most important thing in the table. B0's
warning was that a model can ride the site→report-style shortcut and inflate the derived number while
learning nothing about the knee. Here both metrics rose for eight epochs and both fell afterwards, so
the model is learning image signal, not the labeller.

| epoch | 1 | 4 | 6 | **8** | 10 | 12 |
|---|---|---|---|---|---|---|
| derived | 0.633 | 0.742 | 0.783 | **0.782** | 0.768 | 0.762 |
| gold | 0.638 | 0.732 | 0.811 | **0.846** | 0.821 | 0.797 |
| train loss | 0.607 | 0.549 | 0.526 | 0.499 | 0.472 | 0.458 |

**Gold exceeds derived, and that is the expected direction.** The labeller's own macro AUC against the
58 is 0.853; the model reaches 0.846 against the same gold while being trained only on the labeller's
noisy output. Label noise that is independent of the images partly averages out over 3,525 studies, so
a model can approach — and in principle pass — the quality of its own labels. It has essentially
matched them on the first attempt.

**It overfits from epoch 8.** Train loss keeps falling while both validation curves turn. Fixes in
order of expected value: more slices per slot (8 of the cached 16 are used), stronger augmentation,
layer-wise LR decay on the encoder, and early stopping at ~8 epochs.

**Read the derived column, not the gold one, for per-class detail.** Fold 0 holds 13 gold studies, so a
per-class gold AUC there rests on two or three positives — MCL and Baker's both read exactly 1.000,
which means "not wrong yet", not "solved". The 882-study derived numbers are the usable per-class
signal: strongest on ACL 0.835, Medial Meniscus 0.821, Fracture 0.820; weakest on Lateral OA 0.725,
MCL 0.742, Medial OA 0.744.

**Cost:** ~7.7 min/epoch, 1.5 h for 12 epochs, 7.16 GiB peak inside a self-imposed 34 % cap on a
shared GPU.

## What B0 established

B0 is worth more than its score. Three things came out of it, and only the first is about B0.

### 1. The submission path works end to end

Data mount, code-dataset reconstruction, label order, and the CSV format are all confirmed against
the live grader. Nothing structural is wrong — no column shuffle, no id mismatch, no constant
column. That is a real result and it means the next submission's score is about the model.

### 2. Gold CV tracks LB; derived-target CV does not

| measured against | value | vs LB |
|---|---:|---:|
| report-derived targets | 0.590 | **+0.072** |
| the 58 image-annotated studies | 0.521 | +0.003 |
| leaderboard | 0.518 | — |

The derived-target CV overstated the true score by 0.072 while the gold CV landed within 0.003.

The mechanism matters more than the numbers, because it will not go away. **Site and protocol
determine both the series metadata and the language of the report**, and the report is what produced
the derived targets. So a model reading protocol metadata can score 0.59 against derived targets
while knowing nothing whatsoever about the knee. Any future CV run against derived labels inherits
the same shortcut.

Two consequences, both already in the plan and now backed by evidence rather than argument:

- **Site-grouped folds are mandatory** (PLAN Phase 3), not a refinement. Without them this 0.072 is
  invisible and every CV number is quietly inflated.
- **Report gold-CV alongside derived CV on every run.** The gap between them *is* the shortcut
  estimate. Watch the gap, not just the headline. Read the gold number with its ±0.07 standard error
  on n = 58 — it is a bias check, never a model-selection signal.

Caveat on the ±0.003 agreement: both numbers sit at the floor, and everything agrees near 0.5. This
is one anchor point at chance, not a demonstration that gold CV tracks LB at 0.9. Re-anchor with the
first submission that carries real signal.

### 3. The remaining gap is imaging, and only imaging

B0 extracts what series metadata contains, which is roughly nothing: 0.518 against a public baseline
at 0.891 and a leader at 0.949. There is no tuning path from here — the input carries no anatomy.
The next run has to read pixels.

## Standing rules for this table

- Log **every** submission, including deliberate floors like B0. An unlogged submission is an
  uninterpretable leaderboard number three weeks later.
- Record derived CV, gold CV, and LB on the same row. The first two disagree, and the disagreement
  is the finding.
- Run `python scripts/diagnose_submission.py submission.csv --reference data/train.csv` before every
  upload. A misaligned submission passes visual inspection and scores at chance; the kernel builds
  its output with a positional column rename (`submission.columns = [STUDY_ID] + LABELS`), which is
  one refactor away from silently shuffling all twelve.
