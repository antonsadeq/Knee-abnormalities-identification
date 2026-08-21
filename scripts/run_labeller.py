"""Label the report corpus and print the three audits.

Usage:  python scripts/run_labeller.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee.config import (  # noqa: E402
    ARTIFACT_ROOT,
    DATA_ROOT,
    LABELS,
    LATERALITY_PAIRS,
    STUDY_ID,
)
from rsna_knee.labeller import label_corpus  # noqa: E402
from rsna_knee.reports import detect_language  # noqa: E402
from rsna_knee.verify import coverage, gold_agreement, label_auc, polarity  # noqa: E402


def main() -> None:
    pd.set_option("display.width", 220)
    train = pd.read_csv(DATA_ROOT / "train.csv")
    # Before the first write, not before the second — the language cache below lands here too.
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    lang_path = ARTIFACT_ROOT / "lang.csv"
    if lang_path.exists():
        lang = pd.read_csv(lang_path).set_index(STUDY_ID)["lang"]
    else:
        lang = train.set_index(STUDY_ID)["Report"].fillna("").map(detect_language)
        lang.rename("lang").to_csv(lang_path)

    targets, weights, states = label_corpus(train)
    targets.to_parquet(ARTIFACT_ROOT / "rule_targets.parquet")
    weights.to_parquet(ARTIFACT_ROOT / "rule_weights.parquet")

    gold = train.set_index(STUDY_ID)[LABELS]
    gold = gold[gold.notna().all(axis=1)]

    print("=== assertion states (share of 4,407 reports) ===")
    share = pd.DataFrame(
        {s: (states == s).mean() for s in ["positive", "negative", "hedged", "silent"]}
    )
    print(share.round(3).to_string())

    print("\n=== coverage: share of reports with any assertion, by language ===")
    print(coverage(targets, lang).round(2).to_string())

    print("\n=== polarity: positive share among assertions ===")
    share_p, z = polarity(targets, lang)
    print(share_p.round(2).to_string())

    print("\n=== polarity outliers (|z| > 3) ===")
    flags = z.stack().rename("z")
    flags = flags[flags.abs() > 3].sort_values(key=abs, ascending=False)
    print(flags.round(1).to_string() if len(flags) else "  none")

    print("\n=== label AUC against the 58 gold studies (the headline quality number) ===")
    auc = label_auc(targets, gold)
    print(auc.round(3).to_string())

    # Medial and lateral genuinely co-occur, but only about as often as the gold labels say. A
    # labeller that fires both whenever one sentence names both compartments shows up here as a
    # both-positive rate well above gold, and that inflation lands on four of the twelve columns.
    print("\n=== laterality coupling: both-positive rate, labeller vs gold ===")
    for a, b in LATERALITY_PAIRS:
        pair = targets[[a, b]].dropna()
        lab = float((pair[a].ge(0.5) & pair[b].ge(0.5)).mean()) if len(pair) else float("nan")
        gld = float(((gold[a] == 1) & (gold[b] == 1)).mean())
        print(f"  {a:16} + {b:16}  labeller {lab:.2f}  gold {gld:.2f}  (n={len(pair)})")

    print("\n=== gold agreement (n = studies where the labeller asserted) ===")
    print(
        gold_agreement(targets, gold)[
            ["n", "tp", "fp", "fn", "tn", "precision", "recall", "accuracy"]
        ]
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
