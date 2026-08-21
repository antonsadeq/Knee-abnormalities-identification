# RSNA Knee Abnormality Detection (Kaggle, 2026)

Multi-label detection of twelve clinically important abnormalities from multi-planar knee MRI
examinations, with multilingual radiology reports available as weak supervision.

Competition: <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection>

## The task

Each sample is a **complete knee MRI study** (`StudyInstanceUID`) containing several DICOM series
acquired in different planes (sagittal / coronal / axial, various sequences). For each study we
predict **12 independent binary probabilities**:

| # | Finding | Group |
|---|---------|-------|
| 1 | ACL tear | Ligament |
| 2 | MCL tear | Ligament |
| 3 | Medial meniscus tear | Meniscus |
| 4 | Lateral meniscus tear | Meniscus |
| 5 | Medial compartment OA | Cartilage / OA |
| 6 | Lateral compartment OA | Cartilage / OA |
| 7 | Patellofemoral OA | Cartilage / OA |
| 8 | Joint effusion | Inflammatory |
| 9 | Synovitis | Inflammatory |
| 10 | Baker's cyst | Inflammatory |
| 11 | Bone contusion | Bone |
| 12 | Fracture | Bone |

Labels are **not mutually exclusive** — one study commonly carries several positive findings.
Column names above are the literal `sample_submission.csv` header, in submission order.

## What makes this competition unusual

1. **Only 58 of 4,407 training studies carry ground-truth labels.** The other 4,349 have their
   twelve label columns entirely blank — all you get is a free-text radiology report. This is the
   defining constraint of the competition. Turning multilingual narrative reports into reliable
   per-finding labels *is* the problem; the vision model is downstream of it.
2. **Study-level, multi-series input.** The model consumes a whole exam — 3 to 14 series across
   sagittal, coronal and axial planes, heterogeneous protocols and vendors — not a single image.
   Series selection and cross-plane aggregation matter.
3. **Efficiency is scored.** The prize pool includes first-time awards for the *most efficient*
   models, so inference cost is an explicit objective alongside accuracy.
4. **Domain shift is built in.** 16 sites, five continents, mixed scanners and protocols. Robust
   normalization and site-aware validation are essential.

## Key facts

| | |
|---|---|
| Host | RSNA + Kaggle (Research category) |
| Deadline | **2026-10-22 23:59 UTC** (winners announced November 2026) |
| Prize pool | **$77,000**, including first-time efficiency awards |
| Teams entered | 1,572 — top public leaderboard score **0.949** (as of 2026-08-15) |
| Training data | 4,407 studies / 24,371 series / ~819k DICOM files (**247 GB zipped**, ~570 GB extracted) |
| Labelled studies | **58** of 4,407. The rest are report-only. |
| Metric | Macro (column-wise mean) ROC-AUC over the 12 findings *(unverified)* |
| Format | Code competition — notebook submission, internet off, ≤ 9 h runtime *(unverified)* |
| Recognition | AI Theater, RSNA 2026 Annual Meeting, Chicago (29 Nov – 3 Dec) |

> Rows marked *unverified* come from public participant repositories, not the official pages — the
> Kaggle Overview/Rules tabs are JavaScript-rendered and are not machine-readable. Everything else
> is confirmed from the downloaded data or the Kaggle API. See
> [docs/competition.md](docs/competition.md).

## Data files

Downloaded and verified (see `data/`, git-ignored):

| File | Shape | Contents |
|---|---|---|
| `train.csv` | 4,407 × 14 | `StudyInstanceUID`, `Report` (free text), 12 label columns — **mostly NaN** |
| `train_series.csv` | 24,371 × 5 | `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` |
| `test.csv` | 3 × 1 | Study ids (public placeholder; the real test set is hidden) |
| `test_series.csv` | 15 × 5 | Same schema as `train_series.csv` |
| `sample_submission.csv` | 3 × 13 | `StudyInstanceUID` + the 12 probability columns, all 0.5 |
| `train_series/…`, `test_series/…` | 247 GB zipped | DICOM, nested `StudyInstanceUID/SeriesInstanceUID/*.dcm` |

Note there is no separate reports file — the report text is the `Report` column of `train.csv`, and
it is present for all 4,407 studies. Reports are Spanish, Dutch, German, Greek, Turkish, English and
more, roughly 1,100 characters on average.

Series structure is unusually clean: every study has all three planes, 3–14 series each (mean 5.5).
`Fluid_Sensitive` and `Fat_Suppression` are **perfectly collinear** — identical in all 24,371 rows —
so they carry one bit, not two.

Submission is a CSV of `StudyInstanceUID` plus one probability column per finding, matching
`sample_submission.csv` exactly.

## Repository layout

```
src/rsna_knee/     library code (data, reports, models, training, inference)
notebooks/         Kaggle kernels — one folder per submitted kernel
scripts/           local CLI entry points
docs/              competition notes, strategy, experiment log
data/              raw competition files (git-ignored)
artifacts/         caches, folds, checkpoints, predictions (git-ignored)
tests/             unit tests
```

## Setup

```powershell
conda env create -f environment.yml
conda activate rsna-knee
```

Data download requires a valid Kaggle API token — see [docs/setup.md](docs/setup.md).

## Status

Metadata CSVs downloaded and profiled; label sparsity confirmed. No DICOM pulled, nothing trained
yet. Next: EDA and the report labeller. See [PLAN.md](PLAN.md).
