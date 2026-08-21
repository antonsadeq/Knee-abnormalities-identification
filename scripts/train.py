"""Train one fold of the study-level model.

    python scripts/train.py --fold 0 --epochs 12
    python scripts/train.py --fold 0 --backbone efficientnet_b0 --slices 8 --batch 8
    python scripts/train.py --fold 0 --fold-column fold_site      # the pessimistic cross-site number

**Two validation numbers are printed every epoch, and the gap between them is the point.** `derived`
scores against the report-derived targets on ~880 held-out studies; `gold` scores against the 58
image-derived labels, restricted to the ones held out of this fold. The 0.518 submission established
why both are needed: its derived CV read 0.590 against a leaderboard of 0.518, because site and
protocol predict *how a report is written*. A model whose derived score climbs while its gold score
does not is learning the labeller, not the knee.

Selection uses **gold** where the held-out count allows it and derived otherwise, but neither is
trusted to a hundredth: a per-class AUC on ~12 gold studies is nearly meaningless, and the paired
bootstrap over all 58 resolves about ±0.017. Read the per-class table, not the third decimal.
"""

from __future__ import annotations

import os

# Fragmentation matters more than usual here: a step pushes batch x 6 slots x slices images through
# one encoder, so the allocator sees a few very large transient blocks per step.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

# Persist Inductor's compiled kernels across processes. torch.compile with max-autotune costs ~9
# minutes of warmup on this model; caching means only the first fold pays it, not all five.
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    str(Path(__file__).resolve().parents[1] / "artifacts" / "inductor_cache"),
)

import argparse
import glob
import json
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee.config import ARTIFACT_ROOT, DATA_ROOT, LABELS, STUDY_ID  # noqa: E402
from rsna_knee.data import KneeStudies, build_index  # noqa: E402
from rsna_knee.metric import per_class_auc  # noqa: E402
from rsna_knee.data import label_swap_index  # noqa: E402
from rsna_knee.model import (  # noqa: E402
    KneeModel,
    ModelConfig,
    WeightEma,
    class_balance,
    predict_with_mirror_tta,
    weighted_bce,
)

CACHE_GLOB = "artifacts/kaggle_cache/cache_336_16sl_train_shard*"


def collate(batch):
    pixels, targets, weights, present = zip(*batch)
    return (
        torch.from_numpy(np.stack(pixels)),
        torch.from_numpy(np.stack(targets)),
        torch.from_numpy(np.stack(weights)),
        torch.from_numpy(np.stack(present)),
    )


def evaluate(model, loader, device, gold: pd.DataFrame, ids: np.ndarray, tta: bool = False):
    """Per-class AUC against derived targets and against whatever gold studies are held out."""
    model.eval()
    swap = torch.as_tensor(label_swap_index(), device=device)
    logits, targets = [], []
    with torch.no_grad():
        for pixels, y, _w, present in loader:
            pixels, present = pixels.to(device, non_blocking=True), present.to(device)
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                out = (
                    predict_with_mirror_tta(model, pixels, present, swap)
                    if tta else model(pixels, present)
                )
            logits.append(out.float().cpu().numpy())
            targets.append(y.numpy())
    pred = np.concatenate(logits)
    derived = np.concatenate(targets)

    scores = {"derived": per_class_auc((derived >= 0.5).astype(int), pred)}

    held = [i for i, s in enumerate(ids) if s in gold.index]
    if held:
        truth = gold.loc[[ids[i] for i in held], LABELS].to_numpy()
        # Columns where the held-out gold has only one class present are undefined, not zero.
        scores["gold"] = per_class_auc(truth.astype(int), pred[held])
        scores["gold_n"] = len(held)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--fold-column", default="fold", choices=["fold", "fold_site"])
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--slices", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-checkpoint", dest="checkpoint", action="store_false",
                    help="disable gradient checkpointing (needs a GPU to yourself)")
    ap.set_defaults(checkpoint=True)
    ap.add_argument("--memory-fraction", type=float, default=0.34,
                    help="cap this process's share of the GPU. This is a shared workstation and a "
                         "step's activation memory is large enough to starve other users; a cap "
                         "makes us fail fast instead of taking the card.")
    ap.add_argument("--drop-path", type=float, default=0.0,
                    help="stochastic depth in the encoder; M1 overfitted from epoch 8 with none")
    ap.add_argument("--slot-dropout", type=float, default=0.0,
                    help="hide a present slot with this probability during training")
    ap.add_argument("--no-slot-embedding", dest="slot_embedding", action="store_false")
    ap.add_argument("--no-slice-position", dest="slice_position", action="store_false")
    ap.set_defaults(slot_embedding=True, slice_position=True)
    ap.add_argument("--patience", type=int, default=4,
                    help="stop after this many epochs without improvement (0 disables)")
    ap.add_argument("--no-class-balance", dest="class_balance", action="store_false",
                    help="disable per-class gradient equalisation (the loss then optimises a "
                         "different objective from the unweighted-mean-of-12 metric)")
    ap.set_defaults(class_balance=True)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the encoder. Measured 1.45x on this model (271 -> 187 ms/step, "
                         "8.0 -> 5.5 min/epoch) for ~9 min of one-off warmup, cached across runs. Only "
                         "the encoder is compiled: the head's slot dropout and variable slice count "
                         "would otherwise trigger recompiles.")
    ap.add_argument("--ema", type=float, default=0.999,
                    help="weight-EMA decay; 0 disables. Evaluated and saved instead of the live model")
    ap.add_argument("--tta", action="store_true",
                    help="mirror TTA at validation time, so the reported number matches submission")
    ap.add_argument("--log-every", type=int, default=50,
                    help="print progress every N optimiser steps. An epoch here is ~18 minutes on a "
                         "contended GPU, and a run that prints only at epoch boundaries is opaque "
                         "for exactly as long as you most want to see it.")
    ap.add_argument("--tag", default=None,
                    help="output name; defaults to one derived from every result-changing flag")
    ap.add_argument("--limit", type=int, default=0, help="smoke test on this many train studies")
    ap.add_argument("--out", type=Path, default=ARTIFACT_ROOT / "runs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.fold)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if device == "cuda" and 0 < args.memory_fraction < 1:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        free = torch.cuda.mem_get_info()[0] / 2**30
        print(f"GPU: {total:.1f} GiB total, {free:.1f} GiB free now, "
              f"self-capped at {args.memory_fraction:.0%} = {args.memory_fraction * total:.1f} GiB")

    index = build_index(sorted(glob.glob(str(ROOT / CACHE_GLOB))))
    targets = pd.read_parquet(ARTIFACT_ROOT / "train_targets.parquet")
    weights = pd.read_parquet(ARTIFACT_ROOT / "train_weights.parquet")
    manifest = pd.read_parquet(ROOT / "artifacts/kaggle_cache/manifest_all.parquet").set_index(
        STUDY_ID
    )
    folds = pd.read_csv(ARTIFACT_ROOT / "folds.csv").set_index(STUDY_ID)

    index = index[index[STUDY_ID].isin(targets.index)].reset_index(drop=True)
    assignment = folds[args.fold_column].reindex(index[STUDY_ID]).to_numpy()
    train_index = index[assignment != args.fold].reset_index(drop=True)
    val_index = index[assignment == args.fold].reset_index(drop=True)
    if args.limit:
        train_index = train_index.head(args.limit)

    train_data = KneeStudies(
        train_index, targets, weights, present=manifest["present"],
        n_slices=args.slices, train=True, seed=args.fold,
    )
    val_data = KneeStudies(
        val_index, targets, weights, present=manifest["present"],
        n_slices=args.slices, train=False,
    )
    common = dict(num_workers=args.workers, collate_fn=collate, pin_memory=True,
                  persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, drop_last=True,
                              **common)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, **common)

    train = pd.read_csv(DATA_ROOT / "train.csv")
    gold = train.set_index(STUDY_ID)[LABELS]
    gold = gold[gold.notna().all(axis=1)]

    model_config = ModelConfig(
        backbone=args.backbone,
        checkpoint=args.checkpoint,
        drop_path=args.drop_path,
        slot_dropout=args.slot_dropout,
        slot_embedding=args.slot_embedding,
        slice_position=args.slice_position,
        max_slices=max(32, args.slices),
    )
    model = KneeModel(model_config).to(device)
    # The encoder is pretrained and the head is not, so they do not want the same learning rate.
    encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    opt = torch.optim.AdamW(
        [{"params": encoder_params, "lr": args.encoder_lr},
         {"params": head_params, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )
    balance = None
    if args.class_balance:
        scale = class_balance(weights.loc[train_index[STUDY_ID]])
        balance = torch.as_tensor(scale, dtype=torch.float32, device=device)
        print("class balance multipliers (metric weights all 12 equally; the loss did not):")
        print("  " + "  ".join(f"{k} {v:.2f}" for k, v in zip(LABELS, scale)))

    if args.compile and device == "cuda":
        # Only the encoder: `state_dict` keys are unchanged because compile wraps the forward, so EMA
        # and checkpointing are unaffected.
        model.encoder = torch.compile(model.encoder, mode="max-autotune-no-cudagraphs")
        print("encoder compiled — the first epoch includes one-off autotune warmup")

    ema = WeightEma(model, decay=args.ema) if args.ema else None

    steps = max(1, len(train_loader) // args.accum) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.encoder_lr, args.lr], total_steps=steps, pct_start=0.15
    )

    print(f"fold {args.fold} ({args.fold_column})  train {len(train_index)}  val {len(val_index)}  "
          f"held-out gold {int(gold.index.isin(val_index[STUDY_ID]).sum())}")
    print(f"{args.backbone}  {sum(p.numel() for p in model.parameters())/1e6:.1f}M params  "
          f"batch {args.batch} x accum {args.accum}  {args.slices} slices/slot  device {device}")
    print(f"regularisation: drop_path {args.drop_path}  slot_dropout {args.slot_dropout}  "
          f"mirror 0.5  |  slot_embedding {args.slot_embedding}  "
          f"slice_position {args.slice_position}\n")

    args.out.mkdir(parents=True, exist_ok=True)
    # The tag must encode everything that changes the result, or two runs silently share a
    # checkpoint path and the earlier one is destroyed the first time the later one improves. This
    # already cost one baseline overwrite; `--tag` overrides it entirely.
    tag = args.tag or "_".join(
        [
            f"f{args.fold}",
            args.fold_column,
            args.backbone,
            f"{args.slices}sl",
            f"dp{args.drop_path:g}",
            f"sd{args.slot_dropout:g}",
            "se" if args.slot_embedding else "nose",
            "sp" if args.slice_position else "nosp",
        ]
    )
    print(f"run tag: {tag}")
    history, best, stale = [], -1.0, 0

    # Written after **every** epoch, not at the end. A run is 1-2 hours and the whole point of a log
    # is to be readable while it is still running — `tail -f` on the .log for progress, and this
    # JSONL for the per-class detail that decides whether to stop early.
    log_path = args.out / f"{tag}_log.jsonl"
    log_path.write_text("")

    total_batches = len(train_loader)
    for epoch in range(args.epochs):
        model.train()
        started, running, seen = time.time(), 0.0, 0
        window = 0.0
        for step, (pixels, y, w, present) in enumerate(train_loader):
            pixels = pixels.to(device, non_blocking=True)
            y, w, present = y.to(device), w.to(device), present.to(device)
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                loss = weighted_bce(model(pixels, present), y, w, balance) / args.accum
            loss.backward()
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                if sched.last_epoch < steps - 1:
                    sched.step()
                if ema is not None:
                    ema.update(model)
            running += loss.item() * args.accum
            window += loss.item() * args.accum
            seen += 1

            if args.log_every and (step + 1) % args.log_every == 0:
                done = step + 1
                per_batch = (time.time() - started) / done
                # Recent-window loss as well as the epoch mean: the mean moves too slowly to show
                # whether anything is happening in the last minute.
                print(f"    [{epoch + 1:2d}] {done:5d}/{total_batches}  "
                      f"loss {running / seen:.4f} (last {args.log_every}: {window / args.log_every:.4f})  "
                      f"{args.batch / per_batch:.1f} studies/s  "
                      f"eta {(total_batches - done) * per_batch / 60:.1f} min", flush=True)
                window = 0.0

        # Validate (and checkpoint) the EMA weights: they are what a submission would use.
        if ema is not None:
            live = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state_dict(model))
        scores = evaluate(model, val_loader, device, gold, val_index[STUDY_ID].to_numpy(),
                          tta=args.tta)
        derived = float(scores["derived"].mean(skipna=True))
        gold_macro = float(scores["gold"].mean(skipna=True)) if "gold" in scores else float("nan")
        selection = gold_macro if np.isfinite(gold_macro) else derived

        weakest = scores["derived"].nsmallest(3)
        print(f"epoch {epoch + 1:2d}/{args.epochs}  loss {running / max(seen, 1):.4f}  "
              f"derived {derived:.4f}  gold {gold_macro:.4f}  "
              f"({time.time() - started:.0f}s)  weakest: "
              + ", ".join(f"{k} {v:.3f}" for k, v in weakest.items()), flush=True)

        record = {
            "epoch": epoch + 1,
            "loss": round(running / max(seen, 1), 5),
            "derived": round(derived, 5),
            "gold": round(gold_macro, 5) if np.isfinite(gold_macro) else None,
            "lr_head": round(opt.param_groups[1]["lr"], 8),
            "seconds": round(time.time() - started, 1),
            "per_class_derived": {k: round(float(v), 4) for k, v in scores["derived"].items()},
        }
        history.append(record)
        with log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        pd.DataFrame(history).drop(columns=["per_class_derived"]).to_csv(
            args.out / f"{tag}_history.csv", index=False
        )

        if selection > best:
            best, stale = selection, 0
            # `model_config` as well as the CLI args, so the submission kernel can rebuild the exact
            # architecture instead of inferring it from tensor names.
            torch.save({"model": model.state_dict(), "config": vars(args),
                        "model_config": asdict(model_config), "epoch": epoch + 1,
                        "derived": derived, "gold": gold_macro}, args.out / f"{tag}.pt")
            scores["derived"].to_frame("derived").assign(
                gold=scores.get("gold")
            ).to_csv(args.out / f"{tag}_per_class.csv")
        else:
            stale += 1
            # M1 turned at epoch 8 and spent four more epochs getting worse. Stopping early costs
            # nothing, since the best checkpoint is already on disk.
            if args.patience and stale >= args.patience:
                if ema is not None:
                    model.load_state_dict(live)
                print(f"no improvement for {stale} epochs — stopping early")
                break

        if ema is not None:
            model.load_state_dict(live)   # resume training from the live weights, not the average

    print(f"\nbest {best:.4f} -> {args.out / (tag + '.pt')}")
    print("per-class table:")
    print(pd.read_csv(args.out / f"{tag}_per_class.csv", index_col=0).round(3).to_string())
    print("\nRecord this run in docs/experiments.md with BOTH validation numbers.")


if __name__ == "__main__":
    main()
