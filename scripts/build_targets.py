"""Produce the final training targets the imaging model consumes.

    python scripts/build_targets.py

Reads the rule labeller's output and the LLM checkpoint, blends them, applies the fitted severity
calibration, overlays the 58 image-derived gold labels at elevated weight, and writes

    artifacts/train_targets.parquet     4,407 x 12 graded targets in [0, 1]
    artifacts/train_weights.parquet     4,407 x 12 per-cell sample weights

This is the only file the training code should read for supervision. Everything upstream of it —
lexicon edits, LLM re-runs, calibration refits — is an input to this script, not to training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsna_knee.calibrate import fit as fit_calibration  # noqa: E402
from rsna_knee.config import ARTIFACT_ROOT, DATA_ROOT, LABELS, STUDY_ID  # noqa: E402
from rsna_knee.labeller import blend_labellers, label_corpus, training_targets  # noqa: E402
from rsna_knee.verify import label_auc  # noqa: E402

GOLD_WEIGHT = 8.0


def main() -> None:
    pd.set_option("display.width", 200)
    train = pd.read_csv(DATA_ROOT / "train.csv")
    gold = train.set_index(STUDY_ID)[LABELS]
    gold = gold[gold.notna().all(axis=1)]

    rule_t, rule_w, _ = label_corpus(train)

    llm_path = ARTIFACT_ROOT / "llm_targets.parquet"
    if llm_path.exists():
        llm_t = pd.read_parquet(llm_path)
        llm_w = pd.read_parquet(ARTIFACT_ROOT / "llm_weights.parquet")
        targets, weights = blend_labellers(rule_t, rule_w, llm_t, llm_w)
        source = "rule + LLM blend"
    else:
        # Still produces a usable file, so training is never blocked on the LLM pass.
        print(f"note: {llm_path.name} absent — falling back to rules only")
        targets, weights, source = rule_t, rule_w, "rules only"

    # Calibration maps severity onto P(gold = 1). It is monotone per finding, so it cannot change
    # the labeller's own AUC — the value is entirely in not handing the loss over-confident targets.
    calibration = fit_calibration(targets, gold, weights)
    calibrated = calibration.apply(targets)

    print(f"=== {source} ===")
    print(f"cells asserted: {float(targets.notna().mean().mean()):.3f}")
    print("\n=== label AUC against the 58 gold studies ===")
    print(label_auc(calibrated.loc[gold.index], gold).round(3).to_string())

    final_t, final_w = training_targets(calibrated, weights, gold, gold_weight=GOLD_WEIGHT)

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    final_t.to_parquet(ARTIFACT_ROOT / "train_targets.parquet")
    final_w.to_parquet(ARTIFACT_ROOT / "train_weights.parquet")

    print("\n=== written ===")
    print(f"  {ARTIFACT_ROOT / 'train_targets.parquet'}  {final_t.shape}")
    print(f"  {ARTIFACT_ROOT / 'train_weights.parquet'}  {final_w.shape}")
    print("\n=== weight distribution (per finding) ===")
    desc = pd.DataFrame(
        {
            "mean_w": final_w.mean(),
            "zero_w": (final_w <= 1e-9).mean(),
            "mean_target": final_t.mean(),
        }
    )
    print(desc.round(3).to_string())


if __name__ == "__main__":
    main()
