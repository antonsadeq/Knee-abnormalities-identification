"""Why did a submission score at chance?

A macro ROC-AUC of 0.5 is not a weak model. It is a submission carrying no per-study signal in any
column, and the causes divide cleanly into three families that need completely different fixes:

**Structural** — the file is malformed, so the score is chance no matter how good the model is.
Wrong column order, missing or duplicated study ids, values outside [0, 1].

**Degenerate** — the model emitted no per-study variation. A column with a single distinct value has
ROC-AUC of exactly 0.5 by construction, because every pair of studies is a tie.

**Misaligned** — the signal exists but is attached to the wrong column or the wrong row. This is the
dangerous family: the file passes every eyeball check, the probabilities look like probabilities,
and the score is chance. The commonest instance is writing columns in ``sorted()`` order while the
header is submission order, which for these twelve labels misplaces **eleven of twelve**.

Only the first family is visible by reading the file. The other two need the checks below.

The metric is the unweighted mean of twelve per-column AUCs, which makes the arithmetic of a partial
failure simple and worth internalising: if a working model averages *M* and *k* of the twelve
columns are dead, the score is ``((12 - k) * M + k * c) / 12`` where *c* is what a misplaced column
scores — 0.5 for a constant, and typically 0.50–0.60 for a permuted one, since the findings
co-occur. See :func:`expected_macro_auc`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import LABELS, STUDY_ID

FATAL, WARN, INFO = "fatal", "warn", "info"


@dataclass(frozen=True)
class Check:
    """One diagnostic result. ``ok=False`` at ``FATAL`` means this alone explains a chance score."""

    name: str
    ok: bool
    severity: str
    detail: str

    def __str__(self) -> str:
        mark = "PASS" if self.ok else self.severity.upper()
        return f"[{mark:5}] {self.name}: {self.detail}"


def alphabetical_permutation(labels: list[str] | None = None) -> list[int]:
    """Where each submission-order column lands if the writer sorted the label names.

    Returned as ``perm`` such that ``perm[i]`` is the submission-order index of the label that
    would occupy position *i* under ``sorted()``. Only ``ACL`` survives in place, so this bug costs
    eleven of the twelve columns — which is why it is worth testing for by name rather than hoping
    a generic permutation search finds it.
    """
    labels = labels or LABELS
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    return order


def structural(
    sub: pd.DataFrame,
    sample: pd.DataFrame | None = None,
    labels: list[str] | None = None,
) -> list[Check]:
    """Checks that read the file as a file: header, ids, value range.

    ``sample`` is ``sample_submission.csv`` when available. It is the only authority on column order
    and on the exact study set, so the checks that need it degrade to warnings without it.
    """
    labels = labels or LABELS
    checks: list[Check] = []

    cols = list(sub.columns)
    if STUDY_ID not in cols:
        checks.append(Check("id column", False, FATAL, f"{STUDY_ID!r} absent; columns are {cols}"))
        return checks

    value_cols = [c for c in cols if c != STUDY_ID]
    missing = [lab for lab in labels if lab not in value_cols]
    extra = [c for c in value_cols if c not in labels]
    if missing or extra:
        checks.append(
            Check(
                "header names",
                False,
                FATAL,
                f"{len(missing)} missing {missing}, {len(extra)} unexpected {extra}",
            )
        )
    else:
        checks.append(Check("header names", True, INFO, "all twelve findings present"))
        ordered = value_cols == labels
        checks.append(
            Check(
                "header order",
                ordered,
                INFO if ordered else WARN,
                "matches submission order"
                if ordered
                else f"differs from submission order; got {value_cols}",
            )
        )

    ids = sub[STUDY_ID]
    dupes = int(ids.duplicated().sum())
    checks.append(
        Check("duplicate ids", dupes == 0, FATAL if dupes else INFO, f"{dupes} duplicated")
    )

    if sample is not None and STUDY_ID in sample.columns:
        want, have = set(sample[STUDY_ID]), set(ids)
        gone, spare = want - have, have - want
        ok = not gone and not spare
        checks.append(
            Check(
                "study coverage",
                ok,
                INFO if ok else FATAL,
                f"exactly the {len(want)} expected studies"
                if ok
                else f"{len(gone)} missing, {len(spare)} unexpected",
            )
        )
        if ok:
            same_order = list(ids) == list(sample[STUDY_ID])
            checks.append(
                Check(
                    "row order",
                    True,
                    INFO,
                    "matches sample_submission"
                    if same_order
                    else "differs from sample_submission (harmless if the grader joins on id, "
                    "fatal if any upstream step wrote predictions positionally)",
                )
            )

    present = [lab for lab in labels if lab in value_cols]
    if present:
        vals = sub[present].to_numpy(dtype=float)
        n_nan = int(np.isnan(vals).sum())
        checks.append(
            Check("missing values", n_nan == 0, FATAL if n_nan else INFO, f"{n_nan} NaN/blank")
        )
        finite = vals[np.isfinite(vals)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            in_range = lo >= 0.0 and hi <= 1.0
            checks.append(
                Check(
                    "value range",
                    in_range,
                    INFO if in_range else WARN,
                    f"[{lo:.4g}, {hi:.4g}]"
                    + ("" if in_range else " — outside [0,1]; harmless for AUC but a smell"),
                )
            )
    return checks


def degeneracy(sub: pd.DataFrame, labels: list[str] | None = None) -> pd.DataFrame:
    """Per-column spread. A column with one distinct value scores exactly 0.5, whatever the truth."""
    labels = labels or LABELS
    present = [lab for lab in labels if lab in sub.columns]
    rows = []
    for lab in present:
        col = pd.to_numeric(sub[lab], errors="coerce")
        counts = col.value_counts(dropna=True)
        rows.append(
            {
                "label": lab,
                "n_unique": int(col.nunique(dropna=True)),
                "std": float(col.std(ddof=0)) if len(col) else float("nan"),
                "min": float(col.min()) if len(col) else float("nan"),
                "mean": float(col.mean()) if len(col) else float("nan"),
                "max": float(col.max()) if len(col) else float("nan"),
                "mode_share": float(counts.iloc[0] / len(col)) if len(counts) else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("label")


def degeneracy_checks(
    sub: pd.DataFrame,
    stats: pd.DataFrame,
    tol: float = 1e-9,
) -> list[Check]:
    """Turn the spread table into verdicts, including the ones that span columns."""
    checks: list[Check] = []

    dead = stats.index[(stats["n_unique"] <= 1) | (stats["std"] <= tol)].tolist()
    checks.append(
        Check(
            "constant columns",
            not dead,
            FATAL if dead else INFO,
            f"{len(dead)} column(s) carry a single value and score exactly 0.500: {dead}"
            if dead
            else "every column varies across studies",
        )
    )

    coarse = stats.index[(stats["n_unique"] > 1) & (stats["n_unique"] <= 3)].tolist()
    if coarse:
        checks.append(
            Check(
                "near-constant columns",
                False,
                WARN,
                f"{coarse} take ≤3 distinct values; mostly ties, so AUC is pinned near 0.5",
            )
        )

    present = [lab for lab in stats.index]
    if present:
        vals = sub[present].to_numpy(dtype=float)
        identical_rows = len(np.unique(vals, axis=0)) == 1 if len(vals) > 1 else False
        if identical_rows:
            checks.append(
                Check(
                    "row variation",
                    False,
                    FATAL,
                    "every study received an identical vector — this is a per-class prior, "
                    "not a model, and scores 0.500 by construction",
                )
            )

        twins = []
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                if np.allclose(sub[a].to_numpy(float), sub[b].to_numpy(float), equal_nan=True):
                    twins.append((a, b))
        if twins:
            checks.append(
                Check(
                    "duplicate columns",
                    False,
                    WARN,
                    f"{len(twins)} identical column pair(s), e.g. {twins[:3]} — the twelve heads "
                    "collapsed to fewer than twelve outputs",
                )
            )
    return checks


@dataclass
class Permutation:
    """Evidence about whether the value columns sit under the right headers."""

    best: list[int]
    best_cost: float
    identity_cost: float
    alphabetical_cost: float
    labels: list[str] = field(default_factory=lambda: list(LABELS))

    @property
    def is_identity(self) -> bool:
        return self.best == list(range(len(self.labels)))

    @property
    def matches_alphabetical(self) -> bool:
        return self.best == alphabetical_permutation(self.labels)

    def mapping(self) -> pd.DataFrame:
        """Which finding each header appears to actually contain."""
        return pd.DataFrame(
            {
                "header": self.labels,
                "looks_like": [self.labels[j] for j in self.best],
                "moved": [self.labels[i] != self.labels[j] for i, j in enumerate(self.best)],
            }
        )


def _corr(frame: np.ndarray) -> np.ndarray:
    """Correlation matrix with degenerate columns zeroed rather than NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(frame, rowvar=False)
    return np.nan_to_num(np.atleast_2d(c), nan=0.0)


def recover_permutation(
    sub: pd.DataFrame,
    reference: pd.DataFrame,
    labels: list[str] | None = None,
    rounds: int = 400,
) -> Permutation:
    """Infer which finding each submitted column actually holds.

    Two properties survive a column shuffle and can therefore expose one.

    **Prevalence order.** The twelve findings have very different base rates, so the ranking of a
    submission's column means should track the ranking of the reference prevalences. This gives the
    starting assignment. On its own it is weak — a submission's probabilities need not be calibrated,
    and the reference prevalence here comes from an enriched sample.

    **Co-occurrence structure.** The three osteoarthritis classes move together, effusion and
    synovitis move together, and a permutation carries that whole pattern with it. Matching the
    submitted correlation matrix against the reference one is far more robust to prevalence drift
    than matching means, so it does the real work: the search below minimises
    ``‖corr(sub) − corr(ref)[perm, perm]‖`` by pairwise swaps from the prevalence-based start.

    ``reference`` is any frame carrying the twelve label columns — the 58 gold studies, or the
    report-derived targets over the full corpus. Returns the best assignment together with the cost
    of the identity, so a caller can see whether "nothing moved" is in fact the better explanation.
    """
    labels = labels or LABELS
    n = len(labels)

    s_vals = sub[labels].to_numpy(dtype=float)
    r_vals = reference[labels].to_numpy(dtype=float)

    s_corr, r_corr = _corr(s_vals), _corr(r_vals)

    def cost(perm: list[int]) -> float:
        idx = np.ix_(perm, perm)
        return float(np.linalg.norm(s_corr - r_corr[idx]))

    # Start from prevalence-rank agreement, solved exactly; refine against the correlation structure.
    s_rank = pd.Series(np.nanmean(s_vals, axis=0)).rank().to_numpy()
    r_rank = pd.Series(np.nanmean(r_vals, axis=0)).rank().to_numpy()
    try:
        from scipy.optimize import linear_sum_assignment

        _, perm = linear_sum_assignment(np.abs(s_rank[:, None] - r_rank[None, :]))
        current = list(int(j) for j in perm)
    except Exception:
        current = list(range(n))

    best_cost = cost(current)
    for _ in range(rounds):
        improved = False
        for a in range(n):
            for b in range(a + 1, n):
                trial = list(current)
                trial[a], trial[b] = trial[b], trial[a]
                c = cost(trial)
                if c < best_cost - 1e-12:
                    current, best_cost, improved = trial, c, True
        if not improved:
            break

    return Permutation(
        best=current,
        best_cost=best_cost,
        identity_cost=cost(list(range(n))),
        alphabetical_cost=cost(alphabetical_permutation(labels)),
        labels=list(labels),
    )


def permutation_checks(perm: Permutation, margin: float = 0.15) -> list[Check]:
    """Verdict on the recovered assignment.

    ``margin`` is the fractional improvement over the identity below which the search is treated as
    noise. Correlation matching on twelve columns will always find *some* rearrangement that fits
    marginally better; only a decisive gap is evidence of a real shuffle.
    """
    ident, best = perm.identity_cost, perm.best_cost
    decisive = ident > 0 and (ident - best) / ident > margin

    if perm.is_identity or not decisive:
        return [
            Check(
                "column alignment",
                True,
                INFO,
                f"co-occurrence structure is consistent with the headers "
                f"(identity cost {ident:.3f} vs best {best:.3f}) — no shuffle detected",
            )
        ]

    moved = int(sum(perm.mapping()["moved"]))
    detail = (
        f"{moved}/{len(perm.labels)} columns appear shuffled "
        f"(identity cost {ident:.3f} → {best:.3f})"
    )
    if perm.matches_alphabetical:
        detail += " — and the recovered order is exactly sorted(LABELS), i.e. the columns were "
        detail += "written alphabetically while the header is submission order"
    return [Check("column alignment", False, FATAL, detail)]


def expected_macro_auc(n_dead: int, working: float = 0.89, dead: float = 0.5, n: int = 12) -> float:
    """Macro AUC when ``n_dead`` of ``n`` columns carry no usable signal.

    ``dead`` is what a broken column scores: 0.5 for a constant one, and rather more for a permuted
    one, because a prediction for medial OA scored against lateral OA still beats chance.
    """
    if not 0 <= n_dead <= n:
        raise ValueError(f"n_dead must be in [0, {n}], got {n_dead}")
    return ((n - n_dead) * working + n_dead * dead) / n


def hypotheses(observed: float, working: float = 0.89) -> pd.DataFrame:
    """Score each failure mode against the observed leaderboard number.

    The point of the table is to separate "the file is broken" from "the model learned nothing".
    Both land near 0.5, they are fixed by completely different work, and the gap between them is
    the whole question.
    """
    rows = [
        ("all columns constant / per-class prior", expected_macro_auc(12, working, 0.50), "file"),
        ("alphabetical column shuffle", expected_macro_auc(11, working, 0.52), "file"),
        ("row order scrambled", expected_macro_auc(12, working, 0.50), "file"),
        ("model learned nothing (columns fine)", 0.50, "model"),
        ("half the columns dead", expected_macro_auc(6, working, 0.50), "mixed"),
        ("working model, correctly written", expected_macro_auc(0, working), "—"),
    ]
    out = pd.DataFrame(rows, columns=["hypothesis", "expected", "fix"])
    out["|observed − expected|"] = (out["expected"] - observed).abs()
    return out.sort_values("|observed − expected|").reset_index(drop=True)
