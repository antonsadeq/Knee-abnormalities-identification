"""Competition metric: macro (column-wise mean) ROC-AUC over the twelve findings.

The exact definition is unverified — confirm against the Evaluation tab, in particular whether the
mean is unweighted and how a column with a single class present is handled. Here such columns are
skipped, which is the usual Kaggle convention, and reported so the omission is never silent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import LABELS, STUDY_ID


def per_class_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> pd.Series:
    """Per-finding ROC-AUC. Columns with only one class present yield NaN."""
    labels = labels or LABELS
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.shape[1] != len(labels):
        raise ValueError(f"expected {len(labels)} columns, got {y_true.shape[1]}")

    scores = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        # A degenerate column (all-positive or all-negative) has no defined ROC-AUC.
        scores.append(np.nan if len(np.unique(col)) < 2 else roc_auc_score(col, y_pred[:, j]))
    return pd.Series(scores, index=labels, name="auc")


def macro_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> float:
    """Mean of the per-finding AUCs, ignoring undefined columns."""
    aucs = per_class_auc(y_true, y_pred, labels)
    if aucs.isna().all():
        raise ValueError("no column had both classes present; AUC is undefined")
    return float(aucs.mean(skipna=True))


def score_frames(
    truth: pd.DataFrame,
    pred: pd.DataFrame,
    labels: list[str] | None = None,
) -> tuple[float, pd.Series]:
    """Score two dataframes keyed by study id, aligning rows and columns explicitly.

    Returns ``(macro_auc, per_class_auc)``. Raises if the two frames do not cover exactly the
    same studies — a silent inner join is how a submission ends up scored on a subset.
    """
    labels = labels or LABELS
    truth = truth.set_index(STUDY_ID)
    pred = pred.set_index(STUDY_ID)
    if set(truth.index) != set(pred.index):
        missing = len(set(truth.index) - set(pred.index))
        extra = len(set(pred.index) - set(truth.index))
        raise ValueError(f"study mismatch: {missing} missing, {extra} unexpected")

    pred = pred.loc[truth.index]
    y_true = truth[labels].to_numpy()
    y_pred = pred[labels].to_numpy()
    aucs = per_class_auc(y_true, y_pred, labels)
    return float(aucs.mean(skipna=True)), aucs
