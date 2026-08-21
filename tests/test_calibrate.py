import numpy as np
import pandas as pd
import pytest

from rsna_knee.calibrate import build_dataset, curve, fit
from rsna_knee.config import LABELS


def _synthetic(n=120, over_caller="Contusion", seed=0):
    """A corpus where one finding is systematically over-called.

    Every finding's severity predicts the gold label except ``over_caller``, which fires at high
    severity on labels that are usually negative — the Contusion pattern in the real data.
    """
    rng = np.random.default_rng(seed)
    ids = [f"s{i}" for i in range(n)]
    targets = pd.DataFrame(np.nan, index=ids, columns=LABELS, dtype=float)
    gold = pd.DataFrame(np.nan, index=ids, columns=LABELS, dtype=float)

    for label in LABELS:
        sev = rng.choice([0.0, 0.35, 0.65, 0.95], size=n)
        targets[label] = sev
        p = sev if label != over_caller else sev * 0.25
        gold[label] = (rng.random(n) < p).astype(float)
    return targets, gold


def test_build_dataset_uses_only_asserted_cells():
    targets, gold = _synthetic(n=20)
    targets.iloc[:5, 0] = np.nan  # silence
    X, y, groups, finding = build_dataset(targets, gold)
    assert len(X) == len(y) == len(groups) == len(finding)
    assert len(X) == 20 * len(LABELS) - 5


def test_build_dataset_drops_hedged_cells():
    targets, gold = _synthetic(n=20)
    weights = pd.DataFrame(1.0, index=targets.index, columns=LABELS)
    weights.iloc[:4, 1] = 0.0  # hedged: asserted nothing, so it must not teach the map
    X, _, _, _ = build_dataset(targets, gold, weights)
    assert len(X) == 20 * len(LABELS) - 4


def test_build_dataset_rejects_an_empty_gold_frame():
    targets, _ = _synthetic(n=10)
    with pytest.raises(ValueError, match="no usable"):
        build_dataset(targets, pd.DataFrame(columns=LABELS, dtype=float))


def test_groups_are_studies_so_a_report_cannot_span_the_split():
    targets, gold = _synthetic(n=30)
    _, _, groups, _ = build_dataset(targets, gold)
    # Each study contributes twelve correlated cells; they must share one group id.
    assert len(np.unique(groups)) == 30


def test_calibration_beats_the_uncalibrated_severity():
    targets, gold = _synthetic(n=200)
    cal = fit(targets, gold)
    assert cal.cv.loc[f"calibrated (C={cal.C})", "log_loss"] < cal.cv.loc["raw severity", "log_loss"]


def test_the_over_caller_is_mapped_down():
    targets, gold = _synthetic(n=200, over_caller="Contusion")
    c = curve(fit(targets, gold))
    # At the top of the severity scale, the over-called finding must sit below the others.
    top = c["sev=0.95"]
    assert top["Contusion"] < top.drop("Contusion").median()


def test_calibration_is_monotone_in_severity():
    """The property that bounds what calibration can buy.

    The map is monotone within each finding, so it cannot change the ranking of studies for that
    finding — and the competition metric is per-class ROC-AUC, which reads only rank. Calibration
    therefore leaves the labeller's own AUC untouched. Its value is entirely in the *training*
    targets: a BCE loss fed systematically over-confident targets learns that over-confidence.
    """
    targets, gold = _synthetic(n=200)
    c = curve(fit(targets, gold), severities=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    for label in LABELS:
        row = c.loc[label].to_numpy()
        assert np.all(np.diff(row) > 0), f"{label} is not monotone"


def test_apply_preserves_silence():
    targets, gold = _synthetic(n=60)
    targets.iloc[:10, 3] = np.nan
    out = fit(targets, gold).apply(targets)
    assert out.iloc[:10, 3].isna().all()
    assert out.iloc[10:, 3].notna().all()


def test_apply_returns_probabilities():
    targets, gold = _synthetic(n=60)
    out = fit(targets, gold).apply(targets)
    values = out.to_numpy()
    values = values[~np.isnan(values)]
    assert values.min() >= 0.0
    assert values.max() <= 1.0
