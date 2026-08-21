"""Mapping the labeller's severity onto the probability the gold standard calls it positive.

The reference labels were produced by radiologists applying **severity-thresholded** criteria. The
reports do not share those thresholds: a Turkish report says ``grade II dejenerasyon``, a French one
``déchirure partielle de bas grade``, a Spanish one ``Condropatía grado 2``. Whether any of those
crosses the bar is a calibration question, and it is why the two osteoarthritis classes and
Contusion are the labeller's weakest — Contusion recalls 0.88 at precision 0.50 because it fires on
every marrow signal change while the gold label counts only what clears the threshold.

**Why one pooled model rather than twelve thresholds.** The obvious approach is a cut per finding
fitted against the gold labels. With 17–49 usable examples per finding that overfits badly, and a
hard cut also throws away the graded information the labeller worked to produce. Instead a single
logistic maps severity onto P(gold = 1), sharing one slope across all twelve findings and giving
each its own intercept. The slope is estimated from every usable cell at once — several hundred
rather than a few dozen — and the intercept is the part that genuinely differs per finding, which is
exactly where Contusion's systematic over-calling lives.

Ridge shrinks those intercepts toward the global rate, so a finding with few examples borrows
strength from the rest instead of trusting its own handful.

**Cross-validation groups by study, not by cell.** Each study contributes up to twelve cells, and
they are not independent — one report drives all of them. Splitting cells at random puts the same
report on both sides of the divide and reports a number that cannot be reproduced on new studies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

from .config import LABELS

#: Regularisation grid. Small C means heavy shrinkage of the per-finding intercepts toward the
#: pooled rate, which is the right default when a finding has only a handful of gold examples.
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


@dataclass
class Calibration:
    """A fitted severity → probability map, plus the evidence that it beats not calibrating."""

    model: LogisticRegression
    labels: list[str]
    C: float
    n_cells: int
    n_studies: int
    cv: pd.DataFrame

    def apply(self, targets: pd.DataFrame) -> pd.DataFrame:
        """Map graded targets onto calibrated probabilities, preserving NaN for silence."""
        out = pd.DataFrame(np.nan, index=targets.index, columns=self.labels, dtype=float)
        for label in self.labels:
            col = targets[label]
            asserted = col.notna()
            if not asserted.any():
                continue
            X = _design(col[asserted].to_numpy(), label, self.labels)
            out.loc[asserted, label] = self.model.predict_proba(X)[:, 1]
        return out


def _design(severity: np.ndarray, label: str, labels: list[str]) -> np.ndarray:
    """One severity column followed by a one-hot over findings."""
    onehot = np.zeros((len(severity), len(labels)))
    onehot[:, labels.index(label)] = 1.0
    return np.column_stack([severity, onehot])


def build_dataset(
    targets: pd.DataFrame,
    gold: pd.DataFrame,
    weights: pd.DataFrame | None = None,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Assemble every (study, finding) cell where the labeller asserted and gold is known.

    Cells the labeller hedged carry zero weight and are dropped — a hedge is a refusal to assert, so
    including it would teach the map from a value that was never a claim.

    Returns ``(X, y, groups, finding)``.
    """
    labels = labels or LABELS
    common = targets.index.intersection(gold.index)
    rows, ys, groups, findings = [], [], [], []

    for label in labels:
        t = targets.loc[common, label]
        g = gold.loc[common, label]
        usable = t.notna() & g.notna()
        if weights is not None:
            usable &= weights.loc[common, label].fillna(0) > 0
        if not usable.any():
            continue
        sev = t[usable].to_numpy()
        rows.append(_design(sev, label, labels))
        ys.append(g[usable].to_numpy().astype(int))
        groups.extend(t[usable].index.tolist())
        findings.extend([label] * int(usable.sum()))

    if not rows:
        raise ValueError("no usable (study, finding) cells — is the gold frame empty?")
    return (
        np.vstack(rows),
        np.concatenate(ys),
        np.asarray(groups),
        pd.Series(findings, name="label"),
    )


def _cv_scores(X, y, groups, C: float, n_splits: int = 5) -> tuple[float, float]:
    """Grouped-CV log loss and Brier score for one regularisation strength."""
    n_splits = min(n_splits, len(np.unique(groups)))
    oof = np.full(len(y), np.nan)
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        # A fold whose training half is single-class cannot fit; fall back to its base rate.
        if len(np.unique(y[train])) < 2:
            oof[test] = y[train].mean()
            continue
        model = LogisticRegression(C=C, max_iter=2000)
        model.fit(X[train], y[train])
        oof[test] = model.predict_proba(X[test])[:, 1]
    return log_loss(y, oof, labels=[0, 1]), brier_score_loss(y, oof)


def fit(
    targets: pd.DataFrame,
    gold: pd.DataFrame,
    weights: pd.DataFrame | None = None,
    labels: list[str] | None = None,
) -> Calibration:
    """Fit the severity → probability map and score it against the alternatives.

    The comparison table is the point. Calibration is only worth adopting if it beats *both* using
    the raw severity as a probability and ignoring severity entirely in favour of a per-finding base
    rate — and on this many examples that is not a foregone conclusion.
    """
    labels = labels or LABELS
    X, y, groups, finding = build_dataset(targets, gold, weights, labels)

    scored = [(C, *_cv_scores(X, y, groups, C)) for C in C_GRID]
    best_C = min(scored, key=lambda r: r[1])[0]
    best_ll, best_brier = next((ll, b) for C, ll, b in scored if C == best_C)

    raw = np.clip(X[:, 0], 1e-6, 1 - 1e-6)
    prior = finding.map(pd.Series(y, index=finding.index).groupby(finding).mean()).to_numpy()
    prior = np.clip(prior, 1e-6, 1 - 1e-6)

    cv = pd.DataFrame(
        [
            {"model": "raw severity", "log_loss": log_loss(y, raw, labels=[0, 1]),
             "brier": brier_score_loss(y, raw)},
            {"model": "per-finding base rate", "log_loss": log_loss(y, prior, labels=[0, 1]),
             "brier": brier_score_loss(y, prior)},
            {"model": f"calibrated (C={best_C})", "log_loss": best_ll, "brier": best_brier},
        ]
    ).set_index("model")

    model = LogisticRegression(C=best_C, max_iter=2000).fit(X, y)
    return Calibration(
        model=model,
        labels=labels,
        C=best_C,
        n_cells=len(y),
        n_studies=len(np.unique(groups)),
        cv=cv,
    )


def curve(calibration: Calibration, severities=(0.0, 0.35, 0.65, 0.8, 0.95)) -> pd.DataFrame:
    """P(gold = 1) at representative severities, per finding.

    The readable form of the fit. A finding whose row stays low even at severity 0.95 is one the
    reports over-call relative to the reference standard; a row that is already high at severity 0
    is one where an explicit negation in the report still often accompanies a positive gold label.
    """
    rows = {}
    for label in calibration.labels:
        X = _design(np.asarray(severities, dtype=float), label, calibration.labels)
        rows[label] = calibration.model.predict_proba(X)[:, 1]
    return pd.DataFrame(rows, index=[f"sev={s}" for s in severities]).T
