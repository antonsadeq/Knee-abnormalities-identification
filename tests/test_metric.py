import numpy as np
import pandas as pd
import pytest

from rsna_knee.config import LABELS, STUDY_ID
from rsna_knee.metric import macro_auc, per_class_auc, score_frames


def _frame(ids, values):
    return pd.DataFrame(values, columns=LABELS).assign(**{STUDY_ID: ids})


def test_perfect_ranking_scores_one():
    y_true = np.tile([[0], [1]], (1, len(LABELS)))
    y_pred = np.tile([[0.1], [0.9]], (1, len(LABELS)))
    assert macro_auc(y_true, y_pred) == pytest.approx(1.0)


def test_inverted_ranking_scores_zero():
    y_true = np.tile([[0], [1]], (1, len(LABELS)))
    y_pred = np.tile([[0.9], [0.1]], (1, len(LABELS)))
    assert macro_auc(y_true, y_pred) == pytest.approx(0.0)


def test_degenerate_column_is_nan_and_skipped():
    y_true = np.zeros((4, len(LABELS)))
    y_true[2:, 1:] = 1  # column 0 stays all-negative
    y_pred = np.random.default_rng(0).random((4, len(LABELS)))
    y_pred[2:, 1:] += 1  # make the defined columns perfectly ranked

    aucs = per_class_auc(y_true, y_pred)
    assert np.isnan(aucs.iloc[0])
    assert macro_auc(y_true, y_pred) == pytest.approx(1.0)


def test_all_degenerate_raises():
    y_true = np.zeros((4, len(LABELS)))
    y_pred = np.random.default_rng(0).random((4, len(LABELS)))
    with pytest.raises(ValueError, match="undefined"):
        macro_auc(y_true, y_pred)


def test_score_frames_realigns_shuffled_rows():
    ids = ["a", "b", "c", "d"]
    truth_vals = np.tile([[0], [0], [1], [1]], (1, len(LABELS)))
    pred_vals = np.tile([[0.1], [0.2], [0.8], [0.9]], (1, len(LABELS)))

    truth = _frame(ids, truth_vals)
    pred = _frame(ids, pred_vals).iloc[::-1].reset_index(drop=True)

    score, aucs = score_frames(truth, pred)
    assert score == pytest.approx(1.0)
    assert list(aucs.index) == LABELS


def test_score_frames_rejects_missing_studies():
    truth = _frame(["a", "b"], np.zeros((2, len(LABELS))))
    pred = _frame(["a"], np.zeros((1, len(LABELS))))
    with pytest.raises(ValueError, match="study mismatch"):
        score_frames(truth, pred)
