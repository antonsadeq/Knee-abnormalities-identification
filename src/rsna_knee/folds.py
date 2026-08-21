"""Assigning studies to cross-validation folds without leaking anything.

Two leaks are specific to this dataset, and the 0.518 submission demonstrated the cost of the second
one before any model existed: its CV against report-derived targets read **0.590** while the
leaderboard said **0.518**. That 0.072 was not noise. Site and protocol jointly determine the series
metadata *and* the language and phrasing of the report, and the report is what manufactured the
targets — so a model can score well against derived labels by recognising the site and knowing
nothing about the knee.

So the fold assignment has to defeat both:

**Duplicate reports.** 131 of the 4,407 report texts are byte-identical to another — a template read
for an unremarkable knee. Studies sharing a report share a derived target vector exactly, so putting
two of them on opposite sides of the split scores the model on a target whose source it trained on.

**Site.** Grouping by site is what makes the CV measure cross-site generalisation, which is the thing
the hidden test set actually asks for across 16 sites and five continents.

Both are handled by one mechanism: build groups as **connected components** over "shares a report
hash" and "shares a scanner", then never split a component. Doing them separately does not work,
because a duplicated template can appear at more than one site, and the union-find merges exactly
those cases instead of silently choosing which constraint to violate.

On top of that the folds are balanced for the twelve labels at once. There is no standard multilabel
stratified *grouped* K-fold — sklearn's `StratifiedGroupKFold` is single-label — so
:func:`assign_folds` does the usual greedy thing: walk groups from largest to smallest and put each in
whichever fold currently leaves the per-label totals most even.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .config import LABELS, STUDY_ID


def report_hash(text: object) -> str:
    """Stable hash of a report's text, whitespace-normalised.

    Normalised before hashing because the duplicates are templates re-emitted by a reporting system,
    and those differ in trailing whitespace and line wrapping often enough that a raw hash misses
    some of them — which would leave exactly the leak this is meant to close.
    """
    if not isinstance(text, str):
        return "missing"
    normalised = " ".join(text.split()).casefold()
    return hashlib.sha1(normalised.encode()).hexdigest()[:16]


class _Union:
    """Minimal union-find over arbitrary hashable keys."""

    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(
    studies: pd.Series,
    report_hashes: pd.Series,
    sites: pd.Series | None = None,
) -> pd.Series:
    """Group id per study: connected components over shared report text and shared site.

    ``sites`` may be None, in which case only the duplicate-report constraint applies — useful before
    the header metadata exists, but it leaves the CV measuring within-site performance and therefore
    optimistic. A site value that is null or blank is treated as its own singleton rather than being
    pooled with every other unknown, since pooling them would fuse unrelated studies into one giant
    component and collapse the fold structure.
    """
    union = _Union()
    for study in studies:
        union.find(("study", study))

    for study, digest in zip(studies, report_hashes):
        if digest and digest != "missing":
            union.union(("report", digest), ("study", study))

    if sites is not None:
        for study, site in zip(studies, sites):
            token = None if site is None or pd.isna(site) or str(site).strip() == "" else str(site)
            if token is not None:
                union.union(("site", token), ("study", study))

    roots = [union.find(("study", s)) for s in studies]
    codes = {root: i for i, root in enumerate(dict.fromkeys(roots))}
    return pd.Series([codes[r] for r in roots], index=studies.to_numpy(), name="group")


def assign_folds(
    y: pd.DataFrame,
    groups: pd.Series,
    k: int = 5,
    labels: list[str] | None = None,
) -> pd.Series:
    """Greedy multilabel stratified group K-fold. Returns a fold index per study.

    Groups are placed largest first, each into the fold whose per-label totals are furthest from
    their target share. Largest-first matters: a big group placed late has nowhere balanced to go, and
    with site-sized groups the largest few dominate the outcome.

    Balance is scored on the label totals *and* the group sizes, because a fold can be perfectly
    balanced on positives while holding half the studies.
    """
    labels = labels or LABELS
    y = y.reindex(columns=labels).fillna(0.0)
    positives = (y >= 0.5).astype(float)

    order = (
        pd.DataFrame({"group": groups.to_numpy(), "n": 1}, index=groups.index)
        .groupby("group")["n"]
        .sum()
        .sort_values(ascending=False)
    )

    per_fold = np.zeros((k, len(labels)))
    counts = np.zeros(k)
    assignment: dict[int, int] = {}

    for group in order.index:
        members = groups.index[groups.to_numpy() == group]
        block = positives.loc[positives.index.intersection(members)].to_numpy()
        block_sum = block.sum(axis=0) if len(block) else np.zeros(len(labels))
        size = float(len(members))

        # Cost of adding this group to each fold: how uneven the totals would become. Label counts
        # and study counts are both normalised so neither term drowns the other.
        best, best_cost = 0, None
        for fold in range(k):
            trial_labels = per_fold.copy()
            trial_labels[fold] += block_sum
            trial_counts = counts.copy()
            trial_counts[fold] += size
            cost = (
                trial_labels.std(axis=0).sum() / max(trial_labels.sum(), 1.0)
                + trial_counts.std() / max(trial_counts.sum(), 1.0)
            )
            if best_cost is None or cost < best_cost:
                best, best_cost = fold, cost

        per_fold[best] += block_sum
        counts[best] += size
        assignment[group] = best

    return pd.Series(
        [assignment[g] for g in groups.to_numpy()], index=groups.index, name="fold"
    )


def audit(
    folds: pd.Series,
    groups: pd.Series,
    y: pd.DataFrame,
    sites: pd.Series | None = None,
    labels: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Checks that the split is what it claims to be.

    The first one is the only one that can fail catastrophically and silently: if any group spans two
    folds the leak is back and every subsequent number is optimistic.
    """
    labels = labels or LABELS
    frame = pd.DataFrame({"group": groups, "fold": folds})

    spanning = frame.groupby("group")["fold"].nunique()
    out = {
        "leak": pd.DataFrame(
            {
                "groups": [len(spanning)],
                "groups_spanning_folds": [int((spanning > 1).sum())],
            }
        ),
        "size": frame.groupby("fold").size().rename("studies").to_frame(),
    }

    positives = (y.reindex(columns=labels).fillna(0.0) >= 0.5).astype(float)
    out["prevalence"] = positives.groupby(folds.reindex(positives.index)).mean()

    if sites is not None:
        aligned = sites.reindex(frame.index)
        out["sites"] = (
            pd.crosstab(aligned.fillna("unknown"), frame["fold"]).pipe(
                lambda t: t.loc[t.sum(axis=1).sort_values(ascending=False).index]
            )
        )
    return out
