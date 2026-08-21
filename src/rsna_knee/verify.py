"""Checking a report labeller when there is almost nothing to check it against.

Fifty-eight studies carry image-derived labels. Broken down by language they are: en 28, es 10,
tr 6, hr 4, bg 3, el 3, de 2, nl 2, **fr 0** — against corpora of 1,736 / 682 / 546 / 406 / 220 /
321 / 262 / 153 / 81 reports respectively. So the gold set can say whether the labeller works *on
average*, and can say almost nothing about whether it works *in French*.

That gap matters because the dangerous failure is language-specific and systematic: miss one
negation idiom and every report in that language flips polarity for that finding. Averaged over the
corpus it looks like mild noise; within that language it is an inverted label.

The three label-free checks below are therefore not a supplement to the gold check — for six of the
nine languages they are the only check there is.

- :func:`gold_agreement` — agreement with truth, with intervals wide enough to be honest about n.
- :func:`coverage` — what fraction of reports yield any assertion. Catches *missing* vocabulary.
- :func:`polarity` — the positive share among assertions, per language. Catches *inverted* polarity.
- :func:`disagreement` — where two labellers differ, which is where to spend manual reading.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import LABELS


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used in preference to the normal approximation because the counts here are small enough that
    the naive interval leaves the unit range and reports impossible bounds.
    """
    if trials == 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def gold_agreement(
    targets: pd.DataFrame,
    gold: pd.DataFrame,
    threshold: float = 0.5,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Per-finding agreement between derived targets and the image-derived gold labels.

    ``targets`` holds graded values in [0, 1] with NaN where the labeller made no assertion; ``gold``
    holds the 58 studies' binary labels. Both are indexed by study id.

    The returned ``n`` column is the number of gold studies on which the labeller actually asserted
    something, which is usually well below 58 — read every rate on this table against its own ``n``
    and its interval, never against the headline count.
    """
    labels = labels or LABELS
    common = targets.index.intersection(gold.index)
    rows = []

    for label in labels:
        t = targets.loc[common, label]
        g = gold.loc[common, label]
        both = t.notna() & g.notna()
        pred = (t[both] >= threshold).astype(int)
        true = g[both].astype(int)

        tp = int(((pred == 1) & (true == 1)).sum())
        fp = int(((pred == 1) & (true == 0)).sum())
        fn = int(((pred == 0) & (true == 1)).sum())
        tn = int(((pred == 0) & (true == 0)).sum())

        prec_lo, prec_hi = wilson(tp, tp + fp)
        rec_lo, rec_hi = wilson(tp, tp + fn)
        rows.append(
            {
                "label": label,
                "n": int(both.sum()),
                "asserted": float(t.notna().mean()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": tp / (tp + fp) if tp + fp else float("nan"),
                "precision_lo": prec_lo,
                "precision_hi": prec_hi,
                "recall": tp / (tp + fn) if tp + fn else float("nan"),
                "recall_lo": rec_lo,
                "recall_hi": rec_hi,
                "accuracy": (tp + tn) / both.sum() if both.sum() else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("label")


def label_auc(
    targets: pd.DataFrame,
    gold: pd.DataFrame,
    silent_target: float = 0.15,
    labels: list[str] | None = None,
) -> pd.Series:
    """Per-finding ROC-AUC of the derived targets against the gold labels, plus the macro mean.

    This is the labeller's score **in the units the competition uses**, and it is the single most
    useful summary of label quality — more so than the precision/recall table, which reads the
    graded target through an arbitrary 0.5 cut and counts hedged cells (target exactly 0.5, weight
    zero, never trained on) as positive predictions.

    Silent cells are filled with the prior rather than dropped, because that is what training sees.
    Dropping them would score the labeller only where it chose to speak, which flatters a labeller
    that stays silent whenever it is unsure.

    Read it as a **per-class diagnostic, not a ceiling**. Label noise that is independent of the
    images partly averages out over 4,407 studies, so a model can score above its labels; noise that
    is systematic — a laterality cue bound to the wrong noun, a negation idiom missed in one
    language — does not average out, and there the label AUC is the real bound. The macro mean is
    returned under the ``MACRO`` key.

    With n = 58 each per-class figure carries roughly ±0.07, so compare rankings and large moves,
    never two runs that differ by 0.01.
    """
    labels = labels or LABELS
    from sklearn.metrics import roc_auc_score

    common = targets.index.intersection(gold.index)
    scores = {}
    for label in labels:
        t = targets.loc[common, label].fillna(silent_target)
        g = gold.loc[common, label]
        ok = g.notna()
        scores[label] = (
            float(roc_auc_score(g[ok].astype(int), t[ok]))
            if ok.sum() and g[ok].nunique() == 2
            else float("nan")
        )
    out = pd.Series(scores, name="label_auc")
    out["MACRO"] = float(np.nanmean(list(scores.values())))
    return out


def coverage(
    targets: pd.DataFrame,
    lang: pd.Series,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Fraction of reports yielding any assertion, per language × finding.

    This needs no ground truth at all, which is why it carries the load for the six languages the
    gold set cannot reach. Read it by comparing a cell against the same finding in other languages:
    a finding asserted in 70 % of English reports and 8 % of Turkish ones indicates missing Turkish
    vocabulary, not an absence of Turkish knees with that finding.
    """
    labels = labels or LABELS
    idx = targets.index.intersection(lang.index)
    return (
        targets.loc[idx, labels]
        .notna()
        .groupby(lang.loc[idx])
        .mean()
        .rename_axis(index="lang", columns="label")
    )


def polarity(
    targets: pd.DataFrame,
    lang: pd.Series,
    threshold: float = 0.5,
    labels: list[str] | None = None,
    z_flag: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Positive share among assertions, per language × finding, with outliers flagged.

    **This is the inverted-negation detector.** If a negation idiom is missed in one language, every
    negative statement of that finding is read as a positive, so the positive share for that cell
    jumps toward 1 while the other languages sit near the true prevalence. Nothing else available
    without labels localises that failure so directly.

    The reference rate is the **median across languages**, not the pooled rate. Pooling is what a
    first attempt reaches for and it does not work: the corrupted language contributes to the
    reference it is being judged against, dragging it far enough that the *clean* languages are
    flagged too, and the output no longer says which one is wrong. The median across nine languages
    tolerates up to four bad ones and leaves the inverted cell as the sole outlier.

    Returns ``(share, z)``. ``z`` is the binomial deviation of each cell from that reference; cells
    with ``|z| > z_flag`` are the ones to read by hand. Case mix does differ across sites, so a flag
    is a prompt to look, not a verdict — but a genuine case-mix difference rarely reaches |z| > 3
    while a polarity inversion always does.
    """
    labels = labels or LABELS
    idx = targets.index.intersection(lang.index)
    t = targets.loc[idx, labels]
    lg = lang.loc[idx]

    asserted = t.notna()
    positive = t.ge(threshold) & asserted

    n = asserted.groupby(lg).sum()
    k = positive.groupby(lg).sum()
    share = (k / n.replace(0, np.nan)).rename_axis(index="lang", columns="label")

    # Clipped away from the boundaries so a reference of exactly 0 or 1 gives a finite variance
    # rather than dividing by zero and silently reporting no outlier at all.
    ref = share.median(axis=0, skipna=True).clip(1e-3, 1 - 1e-3)
    expected = n.mul(ref, axis=1)
    sd = n.mul(ref * (1 - ref), axis=1) ** 0.5
    z = ((k - expected) / sd.replace(0, np.nan)).rename_axis(index="lang", columns="label")
    return share, z


def disagreement(
    a: pd.DataFrame,
    b: pd.DataFrame,
    threshold: float = 0.5,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Per-finding disagreement rate between two labellers.

    Disagreement needs no ground truth and concentrates manual reading where it pays: a study where
    the rules and the LLM agree is very unlikely to need a human, and a study where they differ
    almost always does. Use this to build the adjudication queue.
    """
    labels = labels or LABELS
    common = a.index.intersection(b.index)
    rows = []
    for label in labels:
        x, y = a.loc[common, label], b.loc[common, label]
        both = x.notna() & y.notna()
        differ = (x[both] >= threshold) != (y[both] >= threshold)
        only_a = int((x.notna() & y.isna()).sum())
        only_b = int((y.notna() & x.isna()).sum())
        rows.append(
            {
                "label": label,
                "both_asserted": int(both.sum()),
                "disagree": int(differ.sum()),
                "disagree_rate": float(differ.mean()) if both.sum() else float("nan"),
                "only_a": only_a,
                "only_b": only_b,
            }
        )
    return pd.DataFrame(rows).set_index("label")


def adjudication_queue(
    a: pd.DataFrame,
    b: pd.DataFrame,
    reports: pd.Series,
    lang: pd.Series,
    per_cell: int = 5,
    threshold: float = 0.5,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Sample the disagreements into a bounded, language-stratified reading list.

    Stratifying by language rather than sampling at random is the point: unstratified, English would
    dominate the queue and French — the language with no gold labels at all — would rarely appear.
    """
    labels = labels or LABELS
    common = a.index.intersection(b.index).intersection(reports.index)
    rows = []
    for label in labels:
        x, y = a.loc[common, label], b.loc[common, label]
        both = x.notna() & y.notna()
        differ = both & ((x >= threshold) != (y >= threshold))
        picks = differ[differ].index
        for lg, group in pd.Series(lang.loc[picks]).groupby(lang.loc[picks]):
            for sid in group.index[:per_cell]:
                rows.append(
                    {
                        "label": label,
                        "lang": lg,
                        "StudyInstanceUID": sid,
                        "a": float(x[sid]),
                        "b": float(y[sid]),
                        "report": reports[sid],
                    }
                )
    return pd.DataFrame(rows)
