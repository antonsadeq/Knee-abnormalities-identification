"""Fold assignment must not leak, and the leak it prevents is silent.

A group split across folds does not raise; it just makes every CV number optimistic for the rest of
the competition. So the no-span property is asserted directly rather than inferred from a score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsna_knee.config import LABELS
from rsna_knee.folds import assign_folds, audit, build_groups, report_hash

RNG = np.random.default_rng(0)


def _labels(n, index):
    return pd.DataFrame(
        (RNG.random((n, len(LABELS))) < 0.3).astype(float), columns=LABELS, index=index
    )


# --- report hashing -----------------------------------------------------------------------------

def test_hash_ignores_whitespace_and_case():
    a = "Rotura del menisco medial.\nSin derrame."
    b = "  rotura del  menisco medial.   Sin   derrame.  "
    assert report_hash(a) == report_hash(b)


def test_hash_separates_different_reports():
    assert report_hash("ACL tear") != report_hash("MCL tear")


@pytest.mark.parametrize("value", [None, np.nan, 3, float("nan")])
def test_non_text_reports_hash_to_missing(value):
    assert report_hash(value) == "missing"


# --- grouping -----------------------------------------------------------------------------------

def test_duplicate_reports_land_in_one_group():
    studies = pd.Series(["a", "b", "c"])
    hashes = pd.Series(["same", "same", "other"])
    groups = build_groups(studies, hashes)
    assert groups["a"] == groups["b"]
    assert groups["c"] != groups["a"]


def test_missing_reports_are_not_pooled_together():
    # "missing" must not fuse every report-less study into one component.
    studies = pd.Series(["a", "b"])
    groups = build_groups(studies, pd.Series(["missing", "missing"]))
    assert groups["a"] != groups["b"]


def test_same_site_lands_in_one_group():
    studies = pd.Series(["a", "b", "c"])
    hashes = pd.Series(["h1", "h2", "h3"])
    sites = pd.Series(["siteA", "siteA", "siteB"])
    groups = build_groups(studies, hashes, sites)
    assert groups["a"] == groups["b"] != groups["c"]


def test_blank_sites_are_singletons_not_one_giant_group():
    studies = pd.Series(["a", "b", "c"])
    hashes = pd.Series(["h1", "h2", "h3"])
    sites = pd.Series([None, "", np.nan])
    groups = build_groups(studies, hashes, sites)
    assert groups.nunique() == 3


def test_a_template_shared_across_sites_merges_both_sites():
    # This is why the two constraints cannot be applied independently: honouring the report
    # constraint and the site constraint separately would force one of them to be violated.
    studies = pd.Series(["a", "b", "c", "d"])
    hashes = pd.Series(["tpl", "h2", "tpl", "h4"])
    sites = pd.Series(["A", "A", "B", "B"])
    groups = build_groups(studies, hashes, sites)
    assert groups.nunique() == 1


# --- fold assignment ----------------------------------------------------------------------------

def test_no_group_is_split_across_folds():
    n = 300
    ids = [f"s{i:03d}" for i in range(n)]
    studies = pd.Series(ids)
    hashes = pd.Series([f"h{i % 40}" for i in range(n)])       # 40 duplicate clusters
    sites = pd.Series([f"site{i % 6}" for i in range(n)])
    groups = build_groups(studies, hashes, sites)
    y = _labels(n, pd.Index(ids))

    folds = assign_folds(y, groups, k=5)
    checks = audit(folds, groups, y, sites=sites.set_axis(pd.Index(ids)))
    assert int(checks["leak"]["groups_spanning_folds"].iloc[0]) == 0


def test_every_fold_is_used_and_sizes_are_not_wild():
    n = 400
    ids = [f"s{i:03d}" for i in range(n)]
    groups = build_groups(
        pd.Series(ids), pd.Series([f"h{i}" for i in range(n)])  # all singletons
    )
    y = _labels(n, pd.Index(ids))
    folds = assign_folds(y, groups, k=5)

    sizes = folds.value_counts()
    assert set(sizes.index) == {0, 1, 2, 3, 4}
    # Singleton groups admit a near-perfect split, so tolerate only a small imbalance.
    assert sizes.max() - sizes.min() <= max(2, n // 50)


def test_prevalence_is_balanced_across_folds():
    n = 600
    ids = [f"s{i:03d}" for i in range(n)]
    groups = build_groups(pd.Series(ids), pd.Series([f"h{i}" for i in range(n)]))
    y = _labels(n, pd.Index(ids))
    folds = assign_folds(y, groups, k=5)

    prevalence = audit(folds, groups, y)["prevalence"]
    # Each label's per-fold rate should sit close to its overall rate.
    overall = (y >= 0.5).mean()
    assert (prevalence - overall).abs().to_numpy().max() < 0.12


def test_folds_are_deterministic():
    n = 200
    ids = [f"s{i:03d}" for i in range(n)]
    groups = build_groups(pd.Series(ids), pd.Series([f"h{i % 25}" for i in range(n)]))
    y = _labels(n, pd.Index(ids))
    a = assign_folds(y, groups, k=5)
    b = assign_folds(y, groups, k=5)
    assert a.equals(b)


def test_audit_detects_a_deliberately_broken_split():
    ids = ["a", "b", "c", "d"]
    groups = pd.Series([0, 0, 1, 1], index=ids)
    y = _labels(4, pd.Index(ids))
    broken = pd.Series([0, 1, 0, 1], index=ids)   # splits both groups
    checks = audit(broken, groups, y)
    assert int(checks["leak"]["groups_spanning_folds"].iloc[0]) == 2
