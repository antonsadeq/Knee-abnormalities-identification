"""Build the decode cache locally from the downloaded competition zip.

    python scripts/build_cache_local.py --zip /path/to/rsna-knee-abnormality-detection.zip

Local rather than on Kaggle because the archive turns out to be **247 GB zipped**, not the ~570 GB
these notes long assumed — that figure is the extracted size. 247 GB downloads in about 3.5 hours at
the measured 20 MB/s and sits comfortably on a 668 GB disk alongside a 48 GB cache. What does *not*
fit is a full extraction (247 + 570 = 817 GB), so studies are extracted one at a time and deleted.

Two things make this worth more than the Kaggle route despite the download:

* **24 cores against Kaggle's 4**, and no 9-hour session limit or 20 GB output cap, so no sharding.
* **The raw data stays.** Re-caching at another resolution or slice count then costs ~15 minutes
  instead of four more notebook runs and a 48 GB re-download. Over a two-month competition where the
  accuracy model wants 336 px and the efficiency model wants 224, that is the deciding argument.

The one thing this cannot do is prove the decode path works *on Kaggle*, which the submission kernel
needs. That is what `notebooks/cache_336/` is for — run it once over the three public test studies.

Resumable: a study whose ``.npy`` already exists is skipped, so interrupting is free.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee.config import ARTIFACT_ROOT, SEQUENCE_FLAG, SERIES_ID, STUDY_ID  # noqa: E402
from rsna_knee.dicom import CROP_MM, N_SLICES, SIZE, SLOTS, decode_study  # noqa: E402

#: Set once in each worker so the zip's central directory is parsed per process rather than per study.
_STATE: dict = {}


def _init(
    zip_path: str, scratch: str, series_csv: str, size: int, n_slices: int, mm: float
) -> None:
    _STATE.update(
        zip_path=zip_path,
        handle=zipfile.ZipFile(zip_path),
        scratch=Path(scratch),
        size=size,
        n_slices=n_slices,
        mm=mm,
        meta=pd.read_csv(series_csv),
    )


def _members_by_study(zip_path: Path, split: str) -> dict[str, list[str]]:
    """Group the archive's DICOM member names by study id, read once up front.

    Cheaper and far more reliable than trusting the CSV to describe what is actually in the archive:
    a series listed in `train_series.csv` but absent from the zip would otherwise surface as a
    mysterious per-study failure deep into the run.
    """
    prefix = f"{split}_series/"
    grouped: dict[str, list[str]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.startswith(prefix) or not name.endswith(".dcm"):
                continue
            parts = name.split("/")
            if len(parts) >= 4:
                grouped.setdefault(parts[1], []).append(name)
    return grouped


def build_one(task: tuple[str, list[str], str]) -> dict:
    """Extract one study out of the zip, decode it, delete the raw files."""
    study_id, members, out_dir = task
    dest = Path(out_dir) / f"{study_id}.npy"
    if dest.exists():
        return {STUDY_ID: study_id, "status": "cached"}

    work = _STATE["scratch"] / study_id
    started = time.time()
    try:
        zf: zipfile.ZipFile = _STATE["handle"]
        for name in members:
            target = work / "/".join(name.split("/")[2:])
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)

        rows = []
        for series_dir in sorted(p for p in work.iterdir() if p.is_dir()):
            n = sum(1 for _ in series_dir.glob("*.dcm"))
            if n:
                rows.append({SERIES_ID: series_dir.name, "n_files": n})
        if not rows:
            return {STUDY_ID: study_id, "status": "no_series"}

        series = pd.DataFrame(rows).merge(
            _STATE["meta"][[SERIES_ID, "Anatomical_Plane", SEQUENCE_FLAG]], on=SERIES_ID, how="inner"
        )
        if series.empty:
            return {STUDY_ID: study_id, "status": "no_metadata"}

        pixels, present, info = decode_study(
            work, series, n_slices=_STATE["n_slices"], size=_STATE["size"], mm=_STATE["mm"]
        )
        if not present.any():
            return {STUDY_ID: study_id, "status": "no_slots"}

        tmp = dest.with_suffix(".npy.part")
        # Explicit handle: np.save appends ".npy" to any path not already ending in it, which would
        # write "<id>.npy.part.npy" and leave the rename with nothing to move.
        with open(tmp, "wb") as fh:
            np.save(fh, pixels)
        tmp.rename(dest)

        slots = info.get("slots", {})
        return {
            STUDY_ID: study_id,
            "status": "ok",
            "side": info.get("side", "unknown"),
            "n_slots": int(present.sum()),
            "present": "".join("1" if p else "0" for p in present),
            "any_uncropped": any(not s.get("cropped", True) for s in slots.values()),
            "min_slices_available": min((s["n_available"] for s in slots.values()), default=0),
            "seconds": round(time.time() - started, 2),
            "detail": json.dumps(slots),
        }
    except Exception as exc:  # noqa: BLE001 — one bad study must not end the run
        return {STUDY_ID: study_id, "status": f"error: {type(exc).__name__}: {exc}"[:300]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", required=True, type=Path, help="the downloaded competition zip")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/rsna_extract"))
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--slices", type=int, default=N_SLICES)
    ap.add_argument("--mm", type=float, default=CROP_MM)
    ap.add_argument("--series-csv", type=Path, default=None,
                    help="defaults to data/<split>_series.csv")
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count")
    ap.add_argument("--limit", type=int, default=0, help="smoke test on this many studies")
    args = ap.parse_args()

    import os

    out = args.out or ARTIFACT_ROOT / f"cache_{args.size}_{args.slices}sl_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    series_csv = args.series_csv or ROOT / "data" / f"{args.split}_series.csv"
    print(f"indexing {args.zip.name} ...", flush=True)
    grouped = _members_by_study(args.zip, args.split)
    print(f"  {len(grouped)} studies, {sum(len(v) for v in grouped.values())} DICOM members")

    studies = sorted(grouped)
    if args.limit:
        studies = studies[: args.limit]
    todo = [s for s in studies if not (out / f"{s}.npy").exists()]

    per_study_mb = len(SLOTS) * args.slices * args.size * args.size / 1e6
    print(f"{len(todo)} to decode  |  {per_study_mb:.1f} MB/study  "
          f"|  projected {per_study_mb * len(studies) / 1000:.1f} GB")

    workers = args.workers or (os.cpu_count() or 4)
    tasks = [(s, grouped[s], str(out)) for s in todo]

    results: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init,
        initargs=(str(args.zip), str(args.scratch), str(series_csv),
                  args.size, args.slices, args.mm),
    ) as pool:
        for i, row in enumerate(pool.map(build_one, tasks, chunksize=1), 1):
            results.append(row)
            if i % 50 == 0 or i == len(tasks):
                rate = i / (time.time() - t0)
                gb = sum(f.stat().st_size for f in out.glob("*.npy")) / 1e9
                print(f"  {i}/{len(tasks)}  {rate * 60:.0f}/min  "
                      f"~{(len(tasks) - i) / rate / 60:.0f} min left  {gb:.1f} GB", flush=True)

    manifest = pd.DataFrame(results)
    if len(manifest):
        manifest.to_parquet(out / "manifest.parquet", index=False)
        print("\n=== status ===")
        print(manifest["status"].value_counts().to_string())
        ok = manifest[manifest["status"].isin(["ok", "cached"])]
        if "side" in ok and len(ok):
            print("\n=== laterality ===")
            print(ok["side"].value_counts(dropna=False).to_string())
        if "n_slots" in ok and len(ok):
            print("\n=== slots per study ===")
            print(ok["n_slots"].value_counts().sort_index().to_string())
        if "any_uncropped" in ok and ok["any_uncropped"].any():
            print(f"\nWARNING: {int(ok['any_uncropped'].sum())} studies had a slot with FOV < {args.mm} mm")

    gb = sum(f.stat().st_size for f in out.glob("*.npy")) / 1e9
    print(f"\n{out}  {len(list(out.glob('*.npy')))} arrays  {gb:.1f} GB")


if __name__ == "__main__":
    main()
