"""The alignment check has to catch a shuffle that looks entirely normal on inspection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsna_knee.config import LABELS, STUDY_ID
from rsna_knee.diagnose import (
    FATAL,
    alphabetical_permutation,
    degeneracy,
    degeneracy_checks,
    expected_macro_auc,
    hypotheses,
    permutation_checks,
    recover_permutation,
    structural,
)

RNG = np.random.default_rng(0)


def _truth(n: int = 800) -> pd.DataFrame:
    """Labels with the real correlation structure: OA classes together, effusion with synovitis."""
    oa = RNG.normal(size=n)
    inflam = RNG.normal(size=n)
    tear = RNG.normal(size=n)
    drivers = {
        "ACL": tear, "MCL": tear, "Medial Meniscus": tear, "Lateral Meniscus": tear,
        "Medial OA": oa, "Lateral OA": oa, "PF OA": oa,
        "Effusion": inflam, "Synovitis": inflam, "Baker's": inflam,
        "Contusion": RNG.normal(size=n), "Fracture": RNG.normal(size=n),
    }
    # Distinct base rates, so prevalence order is informative on top of the correlation structure.
    cuts = np.linspace(-1.0, 1.0, len(LABELS))
    cols = {
        lab: (drivers[lab] + 0.6 * RNG.normal(size=n) > cut).astype(float)
        for lab, cut in zip(LABELS, cuts)
    }
    return pd.DataFrame(cols, columns=LABELS)


def _submission(values: pd.DataFrame) -> pd.DataFrame:
    ids = [f"study-{i:04d}" for i in range(len(values))]
    return pd.DataFrame({STUDY_ID: ids, **{c: values[c].to_numpy() for c in LABELS}})


def _predictions(truth: pd.DataFrame) -> pd.DataFrame:
    """A believable model: correlated with truth, continuous, in range."""
    noise = RNG.normal(scale=0.8, size=truth.shape)
    raw = truth.to_numpy() + noise
    return pd.DataFrame(1 / (1 + np.exp(-raw)), columns=LABELS)


def test_alphabetical_permutation_moves_eleven_of_twelve():
    perm = alphabetical_permutation()
    fixed = [i for i, j in enumerate(perm) if i == j]
    assert [LABELS[i] for i in fixed] == ["ACL"]


def test_clean_submission_passes_every_check():
    truth = _truth()
    sub = _submission(_predictions(truth))
    sample = pd.DataFrame({STUDY_ID: sub[STUDY_ID]})

    checks = structural(sub, sample) + degeneracy_checks(sub, degeneracy(sub))
    assert not [c for c in checks if not c.ok and c.severity == FATAL]

    assert permutation_checks(recover_permutation(sub, truth))[0].ok


def test_shuffled_columns_are_detected_and_named():
    truth = _truth()
    pred = _predictions(truth)

    # Exactly the bug the kernel's positional rename would produce: values written in sorted()
    # order, headers left in submission order.
    perm = alphabetical_permutation()
    shuffled = pd.DataFrame(pred.to_numpy()[:, perm], columns=LABELS)
    sub = _submission(shuffled)

    # It passes every check that reads the file as a file — that is what makes it dangerous.
    assert not [c for c in structural(sub) if not c.ok and c.severity == FATAL]
    assert not [c for c in degeneracy_checks(sub, degeneracy(sub)) if not c.ok]

    recovered = recover_permutation(sub, truth)
    assert not recovered.is_identity
    assert recovered.best_cost < recovered.identity_cost

    check = permutation_checks(recovered)[0]
    assert not check.ok and check.severity == FATAL


def test_constant_column_is_fatal_and_named():
    truth = _truth()
    pred = _predictions(truth)
    pred["Fracture"] = 0.5
    checks = degeneracy_checks(_submission(pred), degeneracy(_submission(pred)))
    dead = [c for c in checks if c.name == "constant columns"]
    assert dead and not dead[0].ok and "Fracture" in dead[0].detail


def test_all_constant_submission_reports_row_variation():
    flat = pd.DataFrame({lab: [0.5] * 50 for lab in LABELS})
    checks = degeneracy_checks(_submission(flat), degeneracy(_submission(flat)))
    assert any(c.name == "row variation" and c.severity == FATAL for c in checks)


def test_structural_catches_missing_and_extra_headers():
    truth = _truth(30)
    sub = _submission(_predictions(truth)).drop(columns=["Baker's"]).assign(Bakers=0.3)
    header = [c for c in structural(sub) if c.name == "header names"][0]
    assert not header.ok and "Baker's" in header.detail


def test_structural_catches_id_mismatch_and_duplicates():
    truth = _truth(30)
    sub = _submission(_predictions(truth))
    sample = pd.DataFrame({STUDY_ID: list(sub[STUDY_ID]) + ["study-9999"]})
    cov = [c for c in structural(sub, sample) if c.name == "study coverage"][0]
    assert not cov.ok and "1 missing" in cov.detail

    dup = pd.concat([sub, sub.iloc[[0]]], ignore_index=True)
    assert not [c for c in structural(dup) if c.name == "duplicate ids"][0].ok


def test_expected_macro_auc_spans_the_two_endpoints():
    assert expected_macro_auc(0, working=0.89) == pytest.approx(0.89)
    assert expected_macro_auc(12, working=0.89, dead=0.5) == pytest.approx(0.5)
    # One surviving column out of twelve is barely above chance — the alphabetical-shuffle case.
    assert expected_macro_auc(11, working=0.89, dead=0.5) == pytest.approx(0.5325)


def test_hypotheses_ranks_the_closest_explanation_first():
    ranked = hypotheses(0.518, working=0.89)
    assert ranked.loc[0, "|observed − expected|"] <= ranked.loc[1, "|observed − expected|"]
    assert set(ranked["fix"]) >= {"file", "model"}
