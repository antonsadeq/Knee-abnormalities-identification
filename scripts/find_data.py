"""Search this machine for anything that saves a download, before starting one.

Three different assets can each spare you hours, and they are worth very different amounts:

* **A decoded cache** (`cache_336_*/**.npy`) — nothing left to do; point training at it.
* **Raw DICOM**, extracted or still zipped — build the cache locally with `build_cache_local.py`
  instead of transferring 45 GB. Decode is CPU-bound and took 3.0 h of CPU for the whole corpus, so
  on a desktop that is one to two hours of wall clock and no network at all.
* **A partial competition zip** (`*.kaggle-partial`) — resume rather than restart.

    python scripts/find_data.py                     # sensible defaults
    python scripts/find_data.py C:\\ D:\\           # or scan whole drives

Directory trees that cannot plausibly hold 45 GB of medical imaging are pruned, so a drive scan is
minutes rather than hours.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Pruned during the walk. Scanning these is pure cost — they cannot hold the competition data.
SKIP = {
    "$recycle.bin", "system volume information", "windows", "program files",
    "program files (x86)", "programdata", "appdata", "node_modules", ".git",
    "__pycache__", "site-packages", "envs", "anaconda3", "miniconda3", ".cache",
    "proc", "sys", "dev", "run", "snap", "lost+found",
}

MAX_DEPTH = 7


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} TB"


def tree_size(path: Path, cap: int = 20000) -> tuple[int, int]:
    """(bytes, files) under `path`, stopping after `cap` files so a 45 GB tree does not stall this."""
    total = files = 0
    for current, dirs, names in os.walk(path):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP]
        for name in names:
            try:
                total += os.path.getsize(os.path.join(current, name))
                files += 1
            except OSError:
                pass
            if files >= cap:
                return total, files
    return total, files


def scan(roots: list[Path]) -> dict[str, list[tuple[Path, int, int]]]:
    found: dict[str, list] = {"cache": [], "dicom_dir": [], "zip": [], "partial": []}
    seen: set[str] = set()

    for root in roots:
        if not root.exists():
            print(f"  (skipping {root} — not present)")
            continue
        print(f"  scanning {root} ...", flush=True)
        base_depth = len(root.parts)
        for current, dirs, names in os.walk(root, topdown=True):
            here = Path(current)
            if len(here.parts) - base_depth >= MAX_DEPTH:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d.lower() not in SKIP and not d.startswith("$")]

            key = str(here).lower()
            name = here.name.lower()

            # Name plus content: `notebooks/cache_336` is the Kaggle kernel folder, not a cache.
            if name.startswith("cache_336") and key not in seen:
                seen.add(key)
                arrays = [n for n in names if n.endswith(".npy")]
                if arrays:
                    size, count = tree_size(here)
                    found["cache"].append((here, size, len(arrays)))
                    dirs[:] = []
                    continue

            # A DICOM tree is <split>_series/<study>/<series>/*.dcm — recognise the top of it.
            if name.endswith("_series") and key not in seen:
                if any((here / d).is_dir() for d in dirs[:3]):
                    seen.add(key)
                    size, count = tree_size(here)
                    if count:
                        found["dicom_dir"].append((here, size, count))
                        dirs[:] = []
                        continue

            for entry in names:
                low = entry.lower()
                path = here / entry
                if low.endswith(".kaggle-partial") or (
                    low.endswith(".zip.part") and "rsna" in low
                ):
                    found["partial"].append((path, path.stat().st_size, 1))
                elif "rsna-knee" in low and low.endswith(".zip"):
                    found["zip"].append((path, path.stat().st_size, 1))
    return found


def main() -> None:
    roots = [Path(a) for a in sys.argv[1:]]
    if not roots:
        roots = [ROOT, ROOT.parent]
        home = Path.home()
        roots += [home / "Downloads", home / "Documents", home / "Desktop"]
        if os.name == "nt":
            roots += [Path("C:/"), Path("D:/")]
        roots = [r for r in dict.fromkeys(roots) if r.exists()]

    print("=== searching ===")
    found = scan(roots)

    print("\n=== decoded cache (best case — nothing left to do) ===")
    if found["cache"]:
        for path, size, count in found["cache"]:
            complete = "COMPLETE" if count >= 1000 else f"partial ({count} arrays)"
            print(f"  {path}\n      {human(size)}, {count} files — {complete}")
        print("\n  If these total 4,407 arrays across 4 shards, run verify_local.py and start training.")
    else:
        print("  none")

    print("\n=== raw DICOM (saves the 45 GB transfer) ===")
    if found["dicom_dir"] or found["zip"] or found["partial"]:
        for path, size, count in found["dicom_dir"]:
            note = " (count capped)" if count >= 20000 else ""
            print(f"  extracted: {path}\n      {human(size)}, {count} files{note}")
        for path, size, _ in found["zip"]:
            print(f"  archive:   {path}\n      {human(size)} "
                  f"({'looks complete' if size > 200e9 else 'INCOMPLETE — the full zip is ~247 GB'})")
        for path, size, _ in found["partial"]:
            print(f"  partial:   {path}\n      {human(size)} — resume with the same "
                  "`kaggle competitions download` command")
        print("\n  With the raw data present, build the cache locally instead of transferring it:")
        print("     python scripts/build_cache_local.py --zip <path-to-zip>        # from the archive")
        print("  Decode cost the whole corpus 3.0 h of CPU, so expect 1-2 h wall clock on a desktop.")
    else:
        print("  none — the 45 GB cache transfer is the fastest route")

    print("\n=== verdict ===")
    cached = sum(c for _, _, c in found["cache"])
    if cached >= 4407:
        print("  A complete cache is already here. Do not download anything.")
    elif found["dicom_dir"] or found["zip"]:
        print("  Raw DICOM is here. Build the cache locally — no network needed.")
    elif cached:
        print(f"  Only {cached} of 4,407 arrays cached, and no raw DICOM. Transfer the cache.")
    else:
        print("  Nothing reusable found. Transfer the 45 GB cache from the workstation.")


if __name__ == "__main__":
    main()
