# Competition notes

Status of every fact: **verified** (from the downloaded data or the Kaggle API) or **unverified**
(from press material / public participant repos). The Kaggle Overview / Data / Rules / Evaluation
tabs are JavaScript-rendered and are not machine-readable, so a handful of rules-level facts still
need a human to read them.

## Verified — Kaggle API (2026-08-15)

| Fact | Value |
|---|---|
| Deadline | `2026-10-22 23:59` |
| Category | Research |
| Reward | 77,000 USD |
| Teams entered | 1,572 |
| Account status | `userHasEntered: True` (rules already accepted for `antonsadeq`) |
| Top public LB score | 0.949 |

A top score of 0.949 on a 12-class macro AUC is high, and it arrives from teams working with the
same 58 gold labels everyone else has. That number is the target to reason about — it implies the
report-derived supervision is capable of producing a very strong signal.

## Verified — from the downloaded files

### Label columns

Literal `sample_submission.csv` header, in order:

```
StudyInstanceUID, ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture
```

Mirrored exactly in `rsna_knee.config.LABELS`. Note the apostrophe in `Baker's` and the spaces in
the multi-word names — quote them.

### The label sparsity problem

`train.csv` is 4,407 rows × 14 columns. **58 rows have labels. 4,349 do not** — their twelve label
columns are `NaN`, not zero. Every row has a `Report`.

Prevalence among the 58 labelled studies:

| Finding | Positives / 58 | Rate |
|---|---|---|
| Effusion | 35 | 0.60 |
| Synovitis | 27 | 0.47 |
| Medial Meniscus | 26 | 0.45 |
| ACL | 24 | 0.41 |
| Lateral Meniscus | 23 | 0.40 |
| PF OA | 21 | 0.36 |
| Contusion | 19 | 0.33 |
| Fracture | 18 | 0.31 |
| Medial OA | 15 | 0.26 |
| Baker's | 12 | 0.21 |
| Lateral OA | 11 | 0.19 |
| MCL | 9 | 0.16 |

Read these with care. 58 studies is far too small to be a validation set on its own — a per-class
AUC computed on 58 samples has a standard error of roughly ±0.07, so it cannot distinguish a 0.80
model from a 0.90 one. The set also looks enriched for pathology relative to any plausible clinical
population, so these rates are **not** trustworthy priors for the hidden test set. Use the 58 to
*calibrate the report labeller*, not to select models.

### Reports

- Present for all 4,407 studies, in the `Report` column of `train.csv` (there is no separate reports
  file).
- Mean ~1,098 characters, median 977, max 4,743.
- Observed languages in a small sample: Spanish, Dutch, German, Greek, Turkish, English. Reports are
  semi-structured — most carry `Technique` / `Findings` / `Impression`-style sections in their own
  language.
- Negation is everywhere and idiomatic: "No evidence of knee effusion", "No hay alteraciones de
  señal", "kein Hinweis auf". A keyword matcher without negation handling will invert these.
- Some reports describe post-operative anatomy or prior comparisons, which confounds naive
  finding-mention extraction.

### Series metadata

`train_series.csv`: 24,371 rows × 5 columns —
`StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane`.

- 3–14 series per study, mean 5.5.
- Plane counts: Sagittal 9,864 / Coronal 8,609 / Axial 5,898.
- **Every study has all three planes** — no missing-plane handling is needed for training.
- `Fluid_Sensitive` and `Fat_Suppression` are identical in all 24,371 rows (14,010 ones, 10,361
  zeros). They encode one bit between them; treating them as two features is redundant.

So each series is characterised by exactly 2 degrees of freedom: plane (3 levels) and
fluid-sensitive/fat-suppressed (2 levels) — 6 possible series types. Any finer protocol
discrimination has to come from the DICOM headers.

### Imaging

`train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`, ~819k files.

**Measured 2026-08-18: the download is 247 GB**, not the ~570 GB quoted elsewhere in these notes.
570 GB is the *uncompressed* footprint; Kaggle serves it as a single DEFLATE zip at roughly 2.3:1,
and `kaggle competitions download` reports `247G`. Sustained throughput measured at **~20 MB/s**,
so the full archive is a **~3.5 hour** download.
Sampled test DICOMs are ~1.8 MB each.

## Unverified — a human needs to read the Kaggle pages

- [ ] **Metric**: reported as macro (column-wise mean) ROC-AUC. Confirm it is unweighted and how
      degenerate columns are handled.
- [ ] **Code competition**: notebook-only, internet off, ~9 h runtime. Confirm the exact CPU/GPU
      limits.
- [ ] **Efficiency award**: how efficiency is scored (wall-clock? FLOPs? a trade-off curve?) and
      whether it is a separate leaderboard.
- [ ] **External data / pretrained weights** policy — decisive, since both ImageNet-pretrained
      vision backbones and multilingual text encoders are the obvious starting points.
- [ ] Team size cap, daily submission cap, number of final submissions.
- [ ] Winner licensing obligation (RSNA challenges usually require open-sourcing winning code).
- [ ] Whether an LLM may be used to label the reports, and whether an API call at *training* time is
      permitted (inference is internet-off regardless).

That last one is worth checking early — it changes the shape of Phase 3 entirely.

## Sources

- <https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge>
- <https://www.rsna.org/news/2026/august/ai-challenge-knee-mri>
- <https://www.auntminnie.com/imaging-informatics/artificial-intelligence/news/15831808/rsna-launches-2026-knee-abnormality-detection-ai-challenge>
- <https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection>
- <https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection>

## Local hardware (updated 2026-08-18)

- GPU: **NVIDIA RTX 5090, 32 GB VRAM**, CUDA 13.0, Linux.
- Disk: 673 GB free on `/`, 493 GB on `/mnt/samsung990`.
- Local LLM: Ollama 0.32.13 in `~/.local` (no sudo), `qwen3:14b`. Measured **18.3 reports/min** on
  the labelling pass with `think=false`.

The earlier notes described a Windows machine with a 16 GB Quadro RTX 5000 Max-Q and C:/S:/Z: drives,
including a runbook step about thermal throttling to 300 MHz. **None of that applies here.** Two
consequences: 32 GB VRAM means 336 px training at a real batch size without gradient accumulation,
and the full-corpus LLM labelling pass is one overnight run rather than the six days that machine
implied.

The 247 GB zip plus a 48 GB cache is 295 GB, which fits `/` with room to spare. Fully *extracting*
it does not — 247 + 570 = 817 GB against 668 GB free — so a local build reads members straight out
of the zip, or extracts one study at a time and deletes it. See PLAN Phase 4.
