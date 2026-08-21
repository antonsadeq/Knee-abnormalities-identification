"""Convert the LLM JSONL into the labeller schema and audit it against the rule labeller.

    python scripts/compare_labellers.py

Produces `artifacts/llm_targets.parquet` / `llm_weights.parquet`, prints coverage, polarity, gold
agreement and rule-vs-LLM disagreement, and writes a language-stratified reading queue to
`artifacts/adjudication_queue.csv`.

The queue is the deliverable. Neither labeller can be trusted on French or German from the gold set
alone — 0 and 2 gold studies respectively — so the disagreements between two independent labellers,
sampled evenly across languages, are where a human should spend their reading time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsna_knee.config import ARTIFACT_ROOT, DATA_ROOT, LABELS, REPORT, STUDY_ID  # noqa: E402
from rsna_knee.labeller import SILENT_TARGET, label_corpus  # noqa: E402
from rsna_knee.verify import (  # noqa: E402
    adjudication_queue,
    coverage,
    disagreement,
    gold_agreement,
    polarity,
)
from llm_label import KEY_TO_LABEL  # noqa: E402


def load_llm(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the JSONL checkpoint into (targets, weights, evidence).

    Mirrors the rule labeller's conventions exactly so the two are comparable: NaN marks silence,
    a hedge carries zero weight, and severity becomes the graded target.
    """
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    ids = [r["id"] for r in rows]
    targets = pd.DataFrame(np.nan, index=ids, columns=LABELS, dtype=float)
    weights = pd.DataFrame(np.nan, index=ids, columns=LABELS, dtype=float)
    evidence = pd.DataFrame("", index=ids, columns=LABELS, dtype=object)

    for row in rows:
        for key, verdict in row["findings"].items():
            label = KEY_TO_LABEL.get(key)
            if label is None:
                continue
            state = verdict["state"]
            if state == "not_mentioned":
                continue
            if state == "present":
                # Severity 0 on a "present" verdict is a model slip, not a negative; floor it so the
                # finding is not silently inverted.
                targets.at[row["id"], label] = max(float(verdict["severity"]), SILENT_TARGET)
                weights.at[row["id"], label] = 1.0
            elif state == "absent":
                targets.at[row["id"], label] = 0.0
                weights.at[row["id"], label] = 1.0
            else:  # uncertain
                targets.at[row["id"], label] = 0.5
                weights.at[row["id"], label] = 0.0
            evidence.at[row["id"], label] = verdict.get("evidence", "")

    return targets, weights, evidence


def main() -> None:
    pd.set_option("display.width", 220)
    # Prefer a full-corpus run; fall back to the cheap gold-only experiment.
    path = ARTIFACT_ROOT / "llm_labels.jsonl"
    if not path.exists():
        path = ARTIFACT_ROOT / "llm_labels_gold.jsonl"
    if not path.exists():
        sys.exit(
            f"no LLM labels in {ARTIFACT_ROOT} — run one of:\n"
            "  python scripts/llm_label.py --gold-only    # the cheap decisive experiment\n"
            "  python scripts/llm_label.py                # full corpus"
        )
    print(f"reading {path.name}")

    train = pd.read_csv(DATA_ROOT / "train.csv")
    lang = pd.read_csv(ARTIFACT_ROOT / "lang.csv").set_index(STUDY_ID)["lang"]
    reports = train.set_index(STUDY_ID)[REPORT]

    llm_t, llm_w, evidence = load_llm(path)
    llm_t.to_parquet(ARTIFACT_ROOT / "llm_targets.parquet")
    llm_w.to_parquet(ARTIFACT_ROOT / "llm_weights.parquet")
    evidence.to_parquet(ARTIFACT_ROOT / "llm_evidence.parquet")

    # Score the rule labeller on exactly the studies the LLM covered, or the two are not comparable.
    rule_t, _, _ = label_corpus(train[train[STUDY_ID].isin(llm_t.index)])

    gold = train.set_index(STUDY_ID)[LABELS]
    gold = gold[gold.notna().all(axis=1)]

    print(f"=== {len(llm_t)} reports labelled by the LLM ===")

    # Coverage first, because it is the larger prize. Half of all rule-labeller cells are silent or
    # hedged, so converting silence into correct assertions matters more than sharpening the cells
    # that already carry one.
    print("\n=== cells asserted (share of studies x findings) ===")
    asserted = pd.DataFrame(
        {"rule": rule_t.notna().mean(), "llm": llm_t.notna().mean()}
    )
    asserted["delta"] = asserted["llm"] - asserted["rule"]
    print(asserted.round(2).to_string())

    print("\n=== LLM coverage by language ===")
    print(coverage(llm_t, lang).round(2).to_string())

    print("\n=== LLM polarity outliers (|z| > 3) ===")
    _, z = polarity(llm_t, lang)
    flags = z.stack().rename("z")
    flags = flags[flags.abs() > 3].sort_values(key=abs, ascending=False)
    print(flags.round(1).to_string() if len(flags) else "  none")

    common = gold.index.intersection(llm_t.index)
    if len(common):
        cols = ["n", "asserted", "precision", "recall", "accuracy"]
        print(f"\n=== gold agreement on {len(common)} annotated studies ===")
        merged = pd.concat(
            {
                "rule": gold_agreement(rule_t, gold)[cols],
                "llm": gold_agreement(llm_t, gold)[cols],
            },
            axis=1,
        )
        print(merged.round(2).to_string())

    print("\n=== rule vs LLM ===")
    print(disagreement(rule_t, llm_t).round(2).to_string())

    queue = adjudication_queue(rule_t, llm_t, reports, lang, per_cell=5)
    queue.to_csv(ARTIFACT_ROOT / "adjudication_queue.csv", index=False)
    print(f"\nadjudication queue -> {ARTIFACT_ROOT / 'adjudication_queue.csv'} ({len(queue)} rows)")
    if len(queue):
        print(queue.groupby("lang").size().to_string())


if __name__ == "__main__":
    main()
