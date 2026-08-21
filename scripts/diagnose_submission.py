"""Check a submission before uploading it, and explain a chance-level score after.

Usage:
    python scripts/diagnose_submission.py submission.csv
    python scripts/diagnose_submission.py submission.csv --observed 0.518
    python scripts/diagnose_submission.py submission.csv --reference data/train.csv

`--reference` supplies a frame carrying the twelve label columns — `data/train.csv` (the 58 gold
studies) or an `artifacts/*_targets.parquet` — which switches on the column-alignment check. That
check is the reason this script exists: a shuffled submission looks completely normal on inspection
and scores at chance, and the kernel builds its output with a positional column rename.

Exit status is 1 if any check is fatal, so it can gate an upload.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee.config import DATA_ROOT, LABELS, STUDY_ID  # noqa: E402
from rsna_knee.diagnose import (  # noqa: E402
    FATAL,
    degeneracy,
    degeneracy_checks,
    hypotheses,
    permutation_checks,
    recover_permutation,
    structural,
)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submission", type=Path)
    ap.add_argument("--sample", type=Path, default=DATA_ROOT / "sample_submission.csv")
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--observed", type=float, default=None, help="leaderboard score, if known")
    ap.add_argument("--working", type=float, default=0.89, help="AUC a healthy model would reach")
    args = ap.parse_args()

    pd.set_option("display.width", 200)
    sub = _read(args.submission)
    sample = _read(args.sample) if args.sample and args.sample.exists() else None
    if sample is None:
        print(f"note: {args.sample} not found — id and order checks are skipped\n")

    checks = structural(sub, sample)

    have_all = all(lab in sub.columns for lab in LABELS)
    if have_all:
        stats = degeneracy(sub)
        checks += degeneracy_checks(sub, stats)

        if args.reference and args.reference.exists():
            ref = _read(args.reference)
            if STUDY_ID in ref.columns:
                ref = ref.set_index(STUDY_ID)
            ref = ref[LABELS].dropna(how="any")
            if len(ref) >= 20:
                checks += permutation_checks(recover_permutation(sub, ref))
            else:
                print(f"note: only {len(ref)} complete reference rows — alignment check skipped\n")

    print("=== checks ===")
    for c in checks:
        print(f"  {c}")

    if have_all:
        print("\n=== per-column spread (a constant column scores exactly 0.500) ===")
        print(stats.round(4).to_string())

    if args.observed is not None:
        print(f"\n=== hypotheses against the observed {args.observed:.3f} ===")
        print(hypotheses(args.observed, args.working).round(3).to_string(index=False))

    fatal = [c for c in checks if not c.ok and c.severity == FATAL]
    print(f"\n{len(fatal)} fatal, {sum(1 for c in checks if not c.ok) - len(fatal)} warnings")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
