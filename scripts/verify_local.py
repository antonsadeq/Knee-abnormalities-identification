"""Report what this checkout already has, what is missing, and exactly how to get each missing piece.

Run this before re-downloading anything. The pixel cache is 45 GB and the report labels cost an
overnight GPU pass, so the expensive mistake is fetching something that is already on disk — and the
*dangerous* mistake is assuming a partial cache is complete, since a half-populated shard trains
without complaining and quietly scores lower.

    python scripts/verify_local.py

Exit status is 1 if anything required is missing or inconsistent.
"""

from __future__ import annotations

import glob
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts"
CACHE = ART / "kaggle_cache"

EXPECTED_STUDIES = 4407
EXPECTED_SHARDS = 4

#: Almost every artifact here is parquet, and pandas needs an engine for it. Checked once, up front,
#: because without it a dozen separate reads each raise the same ImportError and the report never
#: reaches the findings that matter.
try:
    import pyarrow  # noqa: F401

    PARQUET = True
except ImportError:
    try:
        import fastparquet  # noqa: F401

        PARQUET = True
    except ImportError:
        PARQUET = False
OK, MISSING, BAD = "  ok  ", "MISSING", " BAD  "
problems: list[str] = []


def say(status: str, what: str, detail: str = "", fix: str = "") -> None:
    print(f"[{status}] {what}" + (f" — {detail}" if detail else ""))
    if status != OK:
        problems.append(what)
        if fix:
            print(f"         fix: {fix}")


# --- competition CSVs ---------------------------------------------------------------------------
if not PARQUET:
    say(BAD, "parquet engine", "neither pyarrow nor fastparquet is importable — every artifact here "
                               "is parquet, so those checks cannot run",
        fix="conda activate rsna-knee && conda env update -f environment.yml --prune   "
            "(or: pip install pyarrow)")
    print()

print("=== competition metadata ===")
for name, rows, cols in [
    ("train.csv", 4407, 14),
    ("train_series.csv", 24371, 5),
    ("test.csv", 3, 1),
    ("test_series.csv", 15, 5),
    ("sample_submission.csv", 3, 13),
]:
    path = DATA / name
    if not path.exists():
        say(MISSING, f"data/{name}", fix=f"kaggle competitions download -c "
            f"rsna-knee-abnormality-detection -f {name} -p ./data")
        continue
    frame = pd.read_csv(path)
    good = frame.shape == (rows, cols)
    say(OK if good else BAD, f"data/{name}", f"{frame.shape}, expected ({rows}, {cols})")

# --- the pixel cache ----------------------------------------------------------------------------
print("\n=== pixel cache (45 GB — the expensive one) ===")
shards = sorted(glob.glob(str(CACHE / "cache_336_16sl_train_shard*")))
if not shards:
    say(MISSING, "artifacts/kaggle_cache", "no shard directories",
        fix="kaggle kernels output antonsadeq/rsna-knee-cache-336 -p artifacts/kaggle_cache  "
            "(latest version only — rerun the other shards with scripts/run_cache_kernel.py)")
else:
    owner: dict[str, str] = {}
    per_shard = {}
    for directory in shards:
        names = [Path(p).stem for p in glob.glob(str(Path(directory) / "*.npy"))]
        per_shard[Path(directory).name] = len(names)
        for study in names:
            owner.setdefault(study, Path(directory).name)
    total = len(owner)
    for name, count in per_shard.items():
        print(f"         {name}: {count} arrays")
    say(OK if len(shards) == EXPECTED_SHARDS else BAD,
        "shard count", f"{len(shards)} of {EXPECTED_SHARDS}")
    say(OK if total == EXPECTED_STUDIES else BAD,
        "cached studies", f"{total} of {EXPECTED_STUDIES}"
        + ("" if total == EXPECTED_STUDIES else f" — {EXPECTED_STUDIES - total} short"))

    # A study in the wrong shard means a rebuild wrote into a stale directory.
    stray = [
        s for s, d in owner.items()
        if f"shard{hashlib.sha1(s.encode()).digest()[0] % EXPECTED_SHARDS}of" not in d
    ]
    say(OK if not stray else BAD, "shard assignment", f"{len(stray)} studies in the wrong shard")

    if owner:
        import numpy as np

        sample = next(iter(owner))
        path = next(p for p in glob.glob(str(CACHE / "*" / f"{sample}.npy")))
        array = np.load(path, mmap_mode="r")
        good = array.shape == (6, 16, 336, 336) and array.dtype == np.uint8
        say(OK if good else BAD, "array shape", f"{array.shape} {array.dtype}, "
            "expected (6, 16, 336, 336) uint8")

manifest = CACHE / "manifest_all.parquet"
if manifest.exists() and PARQUET:
    try:
        man = pd.read_parquet(manifest)
        say(OK if len(man) == EXPECTED_STUDIES else BAD, "cache manifest",
            f"{len(man)} rows, {int((man['status'] == 'ok').sum())} ok")
    except Exception as exc:  # noqa: BLE001
        say(BAD, "cache manifest", f"unreadable: {type(exc).__name__}: {exc}"[:140],
            fix="if this is a Parquet version error, take the compat bundle "
                "(rsna_transfer_compat.tar.gz), which is written at format version 1.0")
elif manifest.exists():
    say(BAD, "cache manifest", "present but no parquet engine installed")
else:
    say(MISSING, "artifacts/kaggle_cache/manifest_all.parquet",
        "carries the slot-presence mask the masked attention needs",
        fix="copy from the transfer bundle (artifacts/cache_manifests/manifest_all.parquet)")

# --- irreplaceable artifacts --------------------------------------------------------------------
print("\n=== artifacts that must not be regenerated ===")
#: Columns `build_folds.py` writes. A file missing any of them is not the frozen split, whatever it
#: is called — most likely a leftover from an earlier stage of the project.
FOLD_COLUMNS = {"StudyInstanceUID", "group", "fold", "site_group", "fold_site", "scanner"}

folds_path = ART / "folds.csv"
if not folds_path.exists():
    say(MISSING, "artifacts/folds.csv", "FROZEN — rebuilding invalidates every recorded CV number",
        fix="copy from the transfer bundle; do NOT run build_folds.py")
else:
    folds = pd.read_csv(folds_path)
    absent = FOLD_COLUMNS - set(folds.columns)
    if absent:
        # Reported rather than raised: a stale file with the right name is the failure this check
        # exists to catch, so crashing on it would defeat the purpose.
        say(BAD, "artifacts/folds.csv",
            f"not the frozen split — missing {sorted(absent)}; found {list(folds.columns)}",
            fix="replace it with the one from the transfer bundle. Do not merge: a foreign fold "
                "assignment makes every number in docs/experiments.md incomparable.")
    else:
        spanning = int((folds.groupby("group")["fold"].nunique() > 1).sum())
        detail = (f"{len(folds)} rows, {folds['fold'].nunique()} folds, "
                  f"{spanning} groups spanning folds")
        say(OK if len(folds) == EXPECTED_STUDIES and spanning == 0 else BAD,
            "artifacts/folds.csv", detail)

for name, note in [
    ("llm_labels.jsonl", "one overnight qwen3:14b pass, not reproducible"),
    ("train_targets.parquet", "what training reads"),
    ("train_weights.parquet", "per-cell sample weights"),
]:
    path = ART / name
    if not path.exists():
        say(MISSING, f"artifacts/{name}", note, fix="copy from the transfer bundle")
    elif name.endswith(".parquet") and not PARQUET:
        say(BAD, f"artifacts/{name}", "cannot be checked without a parquet engine")
    else:
        try:
            if name.endswith(".parquet"):
                frame = pd.read_parquet(path)
                say(OK if frame.shape == (EXPECTED_STUDIES, 12) else BAD,
                    f"artifacts/{name}", f"{frame.shape}, expected ({EXPECTED_STUDIES}, 12)")
            else:
                rows = sum(1 for _ in path.open(encoding="utf-8"))
                say(OK if rows > 4000 else BAD, f"artifacts/{name}", f"{rows} rows, {note}")
        except Exception as exc:  # noqa: BLE001 — unreadable is a finding, not a crash
            say(BAD, f"artifacts/{name}", f"unreadable: {type(exc).__name__}: {exc}"[:120],
                fix="replace it from the transfer bundle")

# --- checkpoints (optional) ---------------------------------------------------------------------
print("\n=== checkpoints (optional — training can start without them) ===")
found = sorted(glob.glob(str(ART / "runs" / "*.pt"))) + sorted(glob.glob(str(ROOT / "checkpoints" / "*.pt")))
print(f"         {len(found)} found" + (f": {[Path(p).name for p in found]}" if found else ""))

print("\n=== staleness ===")
stale = []
for name in ("rule_targets.parquet", "rule_weights.parquet"):
    path = ART / name
    if path.exists():
        stale.append(name)
if stale:
    print(f"         {stale} present. These come from the rule labeller, which was changed after the")
    print("         earlier runs (severity anchoring, laterality proximity, the Croatian MCL cue).")
    print("         Regenerate with scripts/run_labeller.py rather than trusting an old copy, or take")
    print("         the bundle's versions — do not mix the two.")
else:
    print("         no known-stale artifacts found")

print()
if problems:
    print(f"{len(problems)} problem(s): {', '.join(problems)}")
    sys.exit(1)
print("Everything required is present and consistent — safe to train.")
