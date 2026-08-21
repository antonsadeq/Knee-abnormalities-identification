"""Out-of-fold evaluation — the closest estimate of a leaderboard score we can make offline.

    python scripts/evaluate_oof.py --tag-pattern "M3_f{fold}_convnext_tiny_16sl"

Why this exists rather than just reading the per-epoch numbers. Training prints two validation
metrics and **neither one predicts the leaderboard**:

``derived`` — 882 studies, but scored against *report-derived* targets. Large n, so it is stable
enough to shape a training curve, and biased, because it measures agreement with the labeller rather
than with the knee. B0 quantified the bias at **+0.072**: site and protocol predict how a report is
written, so a model can score well here while knowing nothing.

``gold`` — image-derived labels, the same kind the hidden test set uses, so unbiased. But a single
fold holds only ~12 of the 58, which puts two or three positives behind each column. M1 read exactly
1.000 on MCL and Baker's, meaning "not wrong yet".

Pooling **out-of-fold predictions across every fold** fixes the second problem without reintroducing
the first: each of the 58 gold studies is predicted by the one model that never saw it, giving one
honest 58-study estimate with test-set-like labels. That is the number to read before submitting, and
it comes with a bootstrap interval because a macro AUC over 58 studies is not precise to a hundredth.

Also reported:

* **per-class AUC**, because the macro hides a column left at chance and each one costs ≈ 0.029;
* **the derived-vs-gold gap**, which is the shortcut estimate;
* **rank correlation between folds**, which says whether an ensemble will add anything — models that
  already agree have nothing to average away.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee.config import ARTIFACT_ROOT, DATA_ROOT, LABELS, STUDY_ID  # noqa: E402
from rsna_knee.data import KneeStudies, build_index  # noqa: E402
from rsna_knee.metric import per_class_auc  # noqa: E402
from rsna_knee.model import KneeModel, ModelConfig  # noqa: E402

CACHE_GLOB = "artifacts/kaggle_cache/cache_336_16sl_train_shard*"


def collate(batch):
    pixels, targets, weights, present = zip(*batch)
    return (
        torch.from_numpy(np.stack(pixels)),
        torch.from_numpy(np.stack(targets)),
        torch.from_numpy(np.stack(weights)),
        torch.from_numpy(np.stack(present)),
    )


def load_model(path: Path, device: str):
    """Rebuild the exact architecture the checkpoint was trained with."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stored = checkpoint.get("model_config")
    state = checkpoint["model"]
    if stored:
        config = ModelConfig(**{**stored, "pretrained": False})
    else:
        # Older checkpoints predate the stored config; infer from the tensors themselves.
        config = ModelConfig(
            backbone=checkpoint.get("config", {}).get("backbone", "convnext_tiny"),
            pretrained=False,
            slot_embedding="slot_embedding" in state,
            slice_position="slice_position" in state,
            dim=state["diagnosis.query"].shape[1],
            max_slices=state["slice_position"].shape[0] if "slice_position" in state else 32,
        )
    model = KneeModel(config).to(device).eval()
    model.load_state_dict(state, strict=True)
    return model, checkpoint


def bootstrap_macro(truth: np.ndarray, pred: np.ndarray, rounds: int, seed: int = 0):
    """Percentile interval for the macro AUC, resampling studies.

    A macro AUC over 58 studies is not a precise number, and quoting it to three decimals invites
    over-reading. The interval is the honest form.
    """
    rng = np.random.default_rng(seed)
    n = len(truth)
    draws = []
    for _ in range(rounds):
        idx = rng.choice(n, n, replace=True)
        scores = []
        for j in range(truth.shape[1]):
            column = truth[idx, j]
            if len(np.unique(column)) == 2:
                from sklearn.metrics import roc_auc_score

                scores.append(roc_auc_score(column, pred[idx, j]))
        if scores:
            draws.append(float(np.mean(scores)))
    if not draws:
        return float("nan"), float("nan")
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag-pattern", default="M3_f{fold}_convnext_tiny_16sl",
                    help="checkpoint stem with {fold} substituted")
    ap.add_argument("--fold-column", default="fold", choices=["fold", "fold_site"])
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--slices", type=int, default=0, help="0 = whatever the checkpoint trained with")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--memory-fraction", type=float, default=0.25)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--runs", type=Path, default=ARTIFACT_ROOT / "runs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and 0 < args.memory_fraction < 1:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)

    index = build_index(sorted(glob.glob(str(ROOT / CACHE_GLOB))))
    targets = pd.read_parquet(ARTIFACT_ROOT / "train_targets.parquet")
    weights = pd.read_parquet(ARTIFACT_ROOT / "train_weights.parquet")
    manifest = pd.read_parquet(
        ROOT / "artifacts/kaggle_cache/manifest_all.parquet"
    ).set_index(STUDY_ID)
    folds = pd.read_csv(ARTIFACT_ROOT / "folds.csv").set_index(STUDY_ID)
    index = index[index[STUDY_ID].isin(targets.index)].reset_index(drop=True)

    train = pd.read_csv(DATA_ROOT / "train.csv")
    gold = train.set_index(STUDY_ID)[LABELS]
    gold = gold[gold.notna().all(axis=1)]

    wanted = [int(f) for f in args.folds.split(",")]
    oof: dict[str, np.ndarray] = {}
    per_fold: dict[int, pd.Series] = {}

    for fold in wanted:
        path = args.runs / f"{args.tag_pattern.format(fold=fold)}.pt"
        if not path.exists():
            print(f"fold {fold}: {path.name} missing — skipped")
            continue
        model, checkpoint = load_model(path, device)
        n_slices = args.slices or int(checkpoint.get("config", {}).get("slices", 8))

        assignment = folds[args.fold_column].reindex(index[STUDY_ID]).to_numpy()
        held = index[assignment == fold].reset_index(drop=True)
        data = KneeStudies(held, targets, weights, present=manifest["present"],
                           n_slices=n_slices, train=False)
        loader = DataLoader(data, batch_size=args.batch, shuffle=False,
                           num_workers=args.workers, collate_fn=collate)

        logits = []
        with torch.no_grad():
            for pixels, _y, _w, present in loader:
                with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    out = model(pixels.to(device), present.to(device))
                logits.append(out.float().cpu().numpy())
        pred = np.concatenate(logits)
        for study, row in zip(held[STUDY_ID], pred):
            oof[study] = row

        truth = (targets.loc[held[STUDY_ID], LABELS].to_numpy() >= 0.5).astype(int)
        per_fold[fold] = per_class_auc(truth, pred)
        print(f"fold {fold}: {len(held)} studies, {n_slices} slices, "
              f"derived macro {per_fold[fold].mean(skipna=True):.4f} "
              f"(checkpoint epoch {checkpoint.get('epoch')})", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not oof:
        raise SystemExit("no checkpoints found for that tag pattern")

    ids = list(oof)
    pred = np.stack([oof[s] for s in ids])
    frame = pd.DataFrame(pred, index=pd.Index(ids, name=STUDY_ID), columns=LABELS)
    frame.to_parquet(args.runs / "oof_predictions.parquet")

    print(f"\n=== out-of-fold coverage: {len(ids)} of {len(index)} studies ===")

    derived_truth = (targets.loc[ids, LABELS].to_numpy() >= 0.5).astype(int)
    derived = per_class_auc(derived_truth, pred)

    gold_ids = [s for s in ids if s in gold.index]
    table = derived.to_frame("derived (report labels)")
    summary = {"derived": float(derived.mean(skipna=True))}

    if gold_ids:
        gold_truth = gold.loc[gold_ids, LABELS].to_numpy().astype(int)
        gold_pred = np.stack([oof[s] for s in gold_ids])
        gold_scores = per_class_auc(gold_truth, gold_pred)
        table["gold (image labels)"] = gold_scores
        summary["gold"] = float(gold_scores.mean(skipna=True))
        lo, hi = bootstrap_macro(gold_truth, gold_pred, args.bootstrap)
        summary["gold_ci"] = (lo, hi)
        summary["gold_n"] = len(gold_ids)

    print(table.round(3).to_string())
    print(f"\nMACRO derived : {summary['derived']:.4f}   (n={len(ids)}, biased — agreement with the labeller)")
    if "gold" in summary:
        lo, hi = summary["gold_ci"]
        print(f"MACRO gold    : {summary['gold']:.4f}   95% CI [{lo:.3f}, {hi:.3f}]   "
              f"(n={summary['gold_n']}, image labels — the leaderboard proxy)")
        print(f"gap (derived - gold): {summary['derived'] - summary['gold']:+.4f}   "
              "— B0's site shortcut measured +0.072")

    if len(per_fold) > 1:
        print("\n=== per-fold derived macro (consistency across sites) ===")
        spread = pd.Series({f: s.mean(skipna=True) for f, s in per_fold.items()})
        print(spread.round(4).to_string())
        print(f"spread {spread.max() - spread.min():.4f} — a large spread means fold-dependent "
              "behaviour, and an ensemble that will disagree usefully")

    print(f"\nwritten -> {args.runs / 'oof_predictions.parquet'}")
    print("Record the gold macro and its interval in docs/experiments.md before submitting.")


if __name__ == "__main__":
    main()
