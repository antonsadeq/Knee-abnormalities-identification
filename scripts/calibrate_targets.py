"""Fit the severity → probability map against the 58 gold studies and write calibrated targets.

    python scripts/calibrate_targets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee.calibrate import curve, fit  # noqa: E402
from rsna_knee.config import ARTIFACT_ROOT, DATA_ROOT, LABELS, STUDY_ID  # noqa: E402
from rsna_knee.labeller import label_corpus  # noqa: E402
from rsna_knee.verify import gold_agreement  # noqa: E402


def main() -> None:
    pd.set_option("display.width", 200)
    train = pd.read_csv(DATA_ROOT / "train.csv")
    targets, weights, _ = label_corpus(train)

    gold = train.set_index(STUDY_ID)[LABELS]
    gold = gold[gold.notna().all(axis=1)]

    cal = fit(targets, gold, weights)
    print(f"fitted on {cal.n_cells} cells from {cal.n_studies} studies, C={cal.C}\n")

    print("=== grouped-CV, split by study (lower is better) ===")
    print(cal.cv.round(4).to_string())

    print("\n=== P(gold = 1) by severity ===")
    print(curve(cal).round(2).to_string())

    calibrated = cal.apply(targets)
    calibrated.to_parquet(ARTIFACT_ROOT / "calibrated_targets.parquet")

    print("\n=== gold agreement at a 0.5 cut, before vs after ===")
    cols = ["precision", "recall", "accuracy"]
    merged = pd.concat(
        {
            "raw": gold_agreement(targets, gold)[cols],
            "calibrated": gold_agreement(calibrated, gold)[cols],
        },
        axis=1,
    )
    merged[("delta", "accuracy")] = (
        merged[("calibrated", "accuracy")] - merged[("raw", "accuracy")]
    )
    print(merged.round(2).to_string())
    print(f"\ncalibrated targets -> {ARTIFACT_ROOT / 'calibrated_targets.parquet'}")
    print(
        "\nNote: these agreement numbers are in-sample — the same 58 studies fitted the map.\n"
        "The grouped-CV table above is the honest estimate; this one shows direction, not size."
    )


if __name__ == "__main__":
    main()
