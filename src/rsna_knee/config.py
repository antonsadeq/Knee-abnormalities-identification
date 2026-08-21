"""Paths and label definitions.

``LABELS`` is the literal ``sample_submission.csv`` header, in submission order. Verified against
the downloaded file — do not reorder.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("RSNA_DATA_ROOT", REPO_ROOT / "data"))
CACHE_ROOT = Path(os.environ.get("RSNA_CACHE_ROOT", REPO_ROOT / "artifacts" / "cache"))
ARTIFACT_ROOT = Path(os.environ.get("RSNA_ARTIFACT_ROOT", REPO_ROOT / "artifacts"))

STUDY_ID = "StudyInstanceUID"
SERIES_ID = "SeriesInstanceUID"
REPORT = "Report"

#: The twelve target findings, in submission column order.
LABELS: list[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

#: Labels that refer to a specific side of the joint. A horizontal flip of a coronal or axial
#: image swaps medial and lateral anatomy, so flipping without swapping these pairs silently
#: corrupts the target. Either swap on flip, or disable horizontal flips entirely.
LATERALITY_PAIRS: list[tuple[str, str]] = [
    ("Medial Meniscus", "Lateral Meniscus"),
    ("Medial OA", "Lateral OA"),
]

#: Values of the `Anatomical_Plane` column in {train,test}_series.csv. Every study has all three.
PLANES = ("Sagittal", "Coronal", "Axial")

#: `Fluid_Sensitive` and `Fat_Suppression` are identical in all 24,371 training rows, so they encode
#: a single bit. Prefer this column and ignore the other.
SEQUENCE_FLAG = "Fluid_Sensitive"
