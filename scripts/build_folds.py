"""Freeze the cross-validation folds. Run once; commit the result; never regenerate.

    python scripts/build_folds.py

Writes ``artifacts/folds.csv`` with one row per study: group, scanner, ``fold`` and ``fold_site``.

**Two fold columns, because one cannot serve both purposes.** The first attempt grouped by report
hash *and* scanner as connected components, and it collapsed: 30 of the 54 duplicate-report clusters
span more than one scanner, so the templates chain the scanners transitively and 51 scanners fuse into
38 components, the largest holding **53 % of the corpus**. No 5-fold split of that is balanced.

The two constraints are not equally hard, so they get separated:

``fold`` — **grouped by report text, stratified by labels and scanner.** The hard leak is closed
absolutely: studies sharing a report text share a derived target vector exactly, and never straddle
the split. Scanner enters the stratification objective instead of the grouping, so every fold sees a
similar scanner mix. This is the column to select models with — balanced, and leak-free in the sense
that matters.

``fold_site`` — **grouped by scanner**, with each duplicate-report cluster assigned whole to its
majority scanner. This is the cross-site generalisation estimate, and it is the one that speaks to the
0.072 shortcut the 0.518 submission exposed. It carries a small, quantified leak: the studies whose
report cluster spans scanners, printed below, can appear on both sides. Read it as the pessimistic
number and ``fold`` as the selection number; if they diverge badly, the model is riding site signal.

The scanner key itself is a proxy, and weaker than intended: ``DeviceSerialNumber``,
``InstitutionName`` and ``StationName`` are anonymised out of this dataset — 0 % present across all
4,407 studies — so vendor, model and field strength is the sharpest identifier available. Vendor
strings need normalising first ("Siemens Healthineers", "SIEMENS", "Siemens" are one vendor; "3" and
"3.0" one field strength).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee.config import ARTIFACT_ROOT, DATA_ROOT, LABELS, REPORT, STUDY_ID  # noqa: E402
from rsna_knee.folds import assign_folds, audit, build_groups, report_hash  # noqa: E402

K = 5

#: Vendor spellings that mean the same manufacturer.
VENDORS = {
    "SIEMENS": "siemens", "SIEMENS HEALTHINEERS": "siemens",
    "PHILIPS": "philips", "PHILIPS MEDICAL SYSTEMS": "philips", "PHILIPS HEALTHCARE": "philips",
    "GE MEDICAL SYSTEMS": "ge", "GE HEALTHCARE": "ge",
    "TOSHIBA": "toshiba", "CANON MEDICAL SYSTEMS": "canon",
    "HITACHI": "hitachi", "HITACHI MEDICAL CORPORATION": "hitachi",
}


def vendor(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        return "unknown"
    return VENDORS.get(name.strip().upper(), name.strip().upper().split()[0].lower())


def field_strength(value: object) -> str:
    try:
        return f"{round(float(value), 1):.1f}"
    except (TypeError, ValueError):
        return "unknown"


def main() -> None:
    pd.set_option("display.width", 200)
    train = pd.read_csv(DATA_ROOT / "train.csv")
    meta = pd.read_parquet(ARTIFACT_ROOT / "meta" / "study_meta.parquet")

    scanner = (
        meta[STUDY_ID].to_frame()
        .assign(
            scanner=(
                meta["Manufacturer"].map(vendor)
                + "|" + meta["ManufacturerModelName"].fillna("unknown").str.strip().str.lower()
                + "|" + meta["MagneticFieldStrength"].map(field_strength) + "T"
            )
        )
        .set_index(STUDY_ID)["scanner"]
    )
    print(f"{scanner.nunique()} distinct scanner keys over {len(scanner)} studies")
    print(scanner.value_counts().head(10).to_string())

    studies = train[STUDY_ID]
    hashes = train[REPORT].map(report_hash)
    duplicated = hashes[hashes.duplicated(keep=False) & (hashes != "missing")]
    print(f"\nstudies sharing a report text with another: {len(duplicated)} "
          f"in {duplicated.nunique()} clusters")

    scanner_by_study = scanner.reindex(studies).to_numpy()
    targets = pd.read_parquet(ARTIFACT_ROOT / "train_targets.parquet")

    # --- fold: report-grouped, stratified on labels AND scanner ---------------------------------
    groups = build_groups(studies, hashes)
    print(f"report-text groups: {groups.nunique()} (largest "
          f"{groups.value_counts().iloc[0]})")

    targets = targets.reindex(groups.index)
    strat = pd.concat(
        [(targets >= 0.5).astype(float),
         pd.get_dummies(pd.Series(scanner_by_study, index=groups.index)).astype(float)],
        axis=1,
    )
    folds = assign_folds(strat, groups, k=K, labels=list(strat.columns))

    checks = audit(folds, groups, targets, sites=scanner.reindex(groups.index))
    print("\n=== fold: leak check ===")
    print(checks["leak"].to_string(index=False))
    print("\n=== fold sizes ===")
    print(checks["size"].to_string())
    print("\n=== positive rate per fold (target >= 0.5) ===")
    print(checks["prevalence"].round(3).to_string())
    spread = checks["prevalence"].max() - checks["prevalence"].min()
    print(f"worst per-label spread across folds: {spread.max():.3f} ({spread.idxmax()})")

    # --- fold_site: scanner-grouped, bridges broken by majority vote ----------------------------
    resolved = pd.Series(scanner_by_study, index=groups.index)
    bridging = 0
    for digest, cluster in hashes.groupby(hashes.to_numpy()):
        if digest == "missing" or len(cluster) < 2:
            continue
        ids = studies.iloc[cluster.index]
        keys = resolved.reindex(ids)
        if keys.nunique() > 1:
            bridging += len(ids)
            resolved.loc[ids] = keys.mode().iloc[0]

    site_groups = build_groups(studies, pd.Series(["missing"] * len(studies)),
                              resolved.reset_index(drop=True))
    fold_site = assign_folds(strat, site_groups, k=K, labels=list(strat.columns))
    site_checks = audit(fold_site, site_groups, targets, sites=scanner.reindex(groups.index))
    print(f"\n=== fold_site: {site_groups.nunique()} scanner groups ===")
    print(site_checks["size"].to_string())
    print(f"studies whose report cluster spans scanners (the residual leak): {bridging} "
          f"({bridging / len(studies):.1%})")
    print("\nscanners per fold_site:")
    print(site_checks["sites"].astype(bool).sum().rename("scanners").to_frame().T.to_string())

    gold = train.set_index(STUDY_ID)[LABELS]
    gold = gold[gold.notna().all(axis=1)]
    print("\n=== the 58 gold studies per fold ===")
    print(folds.reindex(gold.index).value_counts().sort_index().to_string())

    out = pd.DataFrame(
        {
            STUDY_ID: groups.index,
            "group": groups.to_numpy(),
            "fold": folds.to_numpy(),
            "site_group": site_groups.to_numpy(),
            "fold_site": fold_site.to_numpy(),
        }
    ).assign(scanner=lambda d: scanner.reindex(d[STUDY_ID]).to_numpy())

    path = ARTIFACT_ROOT / "folds.csv"
    if path.exists():
        raise SystemExit(f"{path} already exists — folds are frozen. Delete it deliberately.")
    out.to_csv(path, index=False)
    print(f"\nwritten -> {path}  ({len(out)} rows)")
    print("Commit this file. Regenerating it invalidates every CV number recorded against it.")


if __name__ == "__main__":
    main()
