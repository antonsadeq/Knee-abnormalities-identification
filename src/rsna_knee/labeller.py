"""Rule labeller: reports in, graded targets and per-finding weights out.

The output is deliberately *not* a binary matrix. Each (study, finding) cell carries a target in
[0, 1] and a weight, so the training loss can distinguish "the report says there is no ACL tear"
from "the report never mentions the ACL". Collapsing those two into a 0 trains every silence as a
confident negative, which is the single most damaging thing a labeller can do here.

Assertion states and what they emit:

============  ========  ========  ==========================================================
state         target    weight    when
============  ========  ========  ==========================================================
positive      severity  1.0       finding asserted; target graded by severity modifier
negative      0.0       1.0       finding explicitly negated in scope
hedged        0.5       0.0       "possible", "cannot exclude" — recorded, never trained on
silent        0.15      0.15      no clause mentioned it; a weak prior, not a confident zero
============  ========  ========  ==========================================================

The hedged row is where dropping is the right call, and it is expressed as a zero weight on a single
cell rather than by discarding the study — silence in a radiology report is not random, so dropping
studies preferentially removes the normal knees for a finding. See ``docs/labelling.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import LABELS, LATERALITY_PAIRS, REPORT, STUDY_ID
from .lexicon import (
    CONTRAST,
    FINDINGS,
    FINDINGS_BY_LABEL,
    HEDGE,
    NEGATION,
    PATHOLOGY,
    SEVERITY,
    Finding,
)
from .reports import fold, segment

POSITIVE, NEGATIVE, HEDGED, SILENT = "positive", "negative", "hedged", "silent"

#: Target and weight for a finding no clause mentioned. A weak negative rather than a confident one.
SILENT_TARGET = 0.15
SILENT_WEIGHT = 0.15

#: How much further from the shared anatomy one side of a medial/lateral pair may sit before it is
#: treated as losing the contest rather than co-asserting. Wide enough for the words that separate
#: the two in a genuine "medial and lateral ..." coordination. See :func:`_resolve_laterality`.
LATERALITY_SLACK = 20

#: How far an unclaimed negation distributes forward over the list it introduces, in characters.
NEG_SCOPE_FORWARD = 120

#: How far back a negation may look for the subject that claims it. Generous enough for the
#: verb-final languages, where the negation trails the whole predicate. See :func:`_negated`.
NEG_CLAIM_WINDOW = 80

@lru_cache(maxsize=None)
def _compile(cues: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a cue set into one alternation, anchored at a word boundary.

    Plain substring matching is not safe here. ``"normal"`` occurs inside the Spanish ``anormal``,
    which means *abnormal* — a substring match reads the strongest possible positive as a negation.
    Anchoring the start at ``\\b`` fixes that while still allowing the agglutinative suffixes these
    languages attach freely: Turkish ``normaldir`` and Croatian ``meniskusa`` must still match.

    Two markers in the lexicon adjust the boundary rule:

    - a **trailing space** demands a closing boundary too, which keeps the short function words
      honest — bare ``no`` would otherwise match ``nodular``;
    - a **leading ``*``** drops the opening boundary, for stems that appear inside compounds.
      German fuses its anatomy into single words — ``Innenmeniskus``, ``Knochenmarksödem``,
      ``Innenmeniskusriss`` — and 70 % of the German reports write the meniscus that way, so an
      anchored stem reaches none of them.
    """
    parts = []
    for cue in cues:
        inner = cue.startswith("*")
        exact = cue.endswith(" ")
        body = re.escape(cue.strip().lstrip("*"))
        if exact:
            body += r"\b"
        parts.append(body if inner else r"\b" + body)
    return re.compile("(?:" + "|".join(parts) + ")")


def _first_index(text: str, cues: tuple[str, ...]) -> int | None:
    """Position of the earliest matching cue, or None."""
    m = _compile(cues).search(text)
    return m.start() if m else None


def _all_indices(text: str, cues: tuple[str, ...]) -> list[int]:
    """Positions of every matching cue, in order."""
    return [m.start() for m in _compile(cues).finditer(text)]


def _severity(text: str) -> float:
    """Strongest severity modifier present, defaulting to a plain positive.

    Matched through :func:`_compile` rather than by plain substring, for the same reason every other
    cue set is. Unanchored, the 0.95 tier fires on ``several`` (via ``sever``), ``incomplete`` (via
    ``complet``) and ``extension`` (via ``extens``) — and since the strongest grade wins, one such
    hit promotes a mild finding to severe. ``incomplete tear`` graded 0.95 inverts the very
    distinction the severity channel exists to carry.
    """
    grades = [g for g, cues in SEVERITY.items() if _first_index(text, cues) is not None]
    return max(grades) if grades else 0.8


def _negated(text: str, hits: dict[str, tuple[int, int]]) -> set[str]:
    """The labels a negation governs in this clause.

    Scope runs in **both directions** within the clause. English and German put the negation first
    ("no joint effusion", "kein Gelenkserguss"), but Spanish and French routinely place it after the
    structure — "Menisco lateral de morfología conservada, sin signos de rotura" — and Turkish is
    verb-final, so its negation is essentially always last: "Ön çapraz bağ ... izlenmemektedir".

    A forward-only scope silently reads all three of those as positives. That failure is invisible in
    aggregate and glaring in ``verify.polarity``, which is how it was found: Spanish sat at 0.95
    positive on nearly every finding.

    A contrast conjunction between the negation and the finding terminates the scope, so "no medial
    meniscus tear but the lateral meniscus is torn" negates only the first half.

    **A negation belongs to a subject, and only distributes when it has none.** Unbounded
    bidirectional scope makes every negation clause-wide, and clauses here routinely assert several
    structures:

        "Tear of the lateral meniscus, medial meniscus intact."

    read *intact* backwards over the whole clause and marked a real tear negative. Stopping the
    scope at commas is the obvious repair and it is wrong, because the commonest negation in the
    corpus is a list — "No evidence of meniscal tear, ligament injury, or fracture" — where the
    negation has to reach every item, and because Spanish and French write "Menisco lateral, sin
    signos de rotura" with the comma *inside* a single assertion.

    What separates those is not punctuation but whether the negation already has a subject. A
    negation preceded by a finding cue is that finding's predicate — *intact* in "medial meniscus
    intact" says something about the medial meniscus and nothing about anything else, so it is
    claimed by the nearest preceding finding and negates only that one. A negation with nothing
    before it — "No evidence of…", "Kein Nachweis…" — is unclaimed, and distributes forward over
    every finding within reach until a contrast conjunction stops it.

    This also keeps the verb-final languages working: Turkish "Ön çapraz bağ … izlenmemektedir" puts
    the negation last, where it is claimed by the ACL cue that precedes it.
    """
    positions = sorted((pos, label) for label, (pos, _gap) in hits.items())
    negated: set[str] = set()

    # Comma-delimited spans. A claimed negation governs its own span and sweeps back over earlier
    # ones until it meets a span that already states a pathology, which is the signal that the
    # subject there has its own predicate and is not part of this negation's list.
    edges = [0] + [m.start() for m in re.finditer(",", text)] + [len(text)]
    spans = list(zip(edges, edges[1:]))

    def span_of(i: int) -> int:
        return next((k for k, (lo, hi) in enumerate(spans) if lo <= i < hi), len(spans) - 1)

    for m in _compile(NEGATION).finditer(text):
        start = m.start()
        claimed = [
            (pos, label)
            for pos, label in positions
            if pos <= start and start - pos <= NEG_CLAIM_WINDOW
        ]

        if claimed:
            here = span_of(start)
            reachable = {here}
            for k in range(here - 1, -1, -1):
                lo, hi = spans[k]
                if _first_index(text[lo:hi], PATHOLOGY) is not None:
                    break
                reachable.add(k)
            negated.update(label for pos, label in claimed if span_of(pos) in reachable)
            continue

        # Unclaimed, so it distributes forward over the list it introduces.
        for pos, label in positions:
            if pos < start or pos - start > NEG_SCOPE_FORWARD:
                continue
            if _first_index(text[start:pos], CONTRAST) is None:
                negated.add(label)
    return negated


def _resolve_laterality(hits: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """Drop the losing side when medial and lateral compete for the same anatomy.

    An absolute window alone cannot separate "degenerative changes in the **medial** compartment
    with a tear of the **lateral** meniscus" from a genuine bilateral statement, because the OA
    window has to stay wide enough for the diffuse phrasing the corpus actually uses. What does
    separate them is that the two sides are *competing*: here medial sits 27 characters from the
    degenerative term and lateral 62, so medial plainly owns it.

    The slack is what keeps coordination working. "medial and lateral femorotibial articular
    cartilage" is a real bilateral assertion, and the two gaps differ only by the width of the words
    between them — well inside the slack, so both survive.
    """
    for a, b in LATERALITY_PAIRS:
        if a not in hits or b not in hits:
            continue
        gap_a, gap_b = hits[a][1], hits[b][1]
        if gap_a > gap_b + LATERALITY_SLACK:
            hits.pop(a)
        elif gap_b > gap_a + LATERALITY_SLACK:
            hits.pop(b)
    return hits


def _match(finding: Finding, text: str) -> tuple[int, int] | None:
    """``(position, gap)`` at which ``finding`` is cued in ``text``, or None.

    A qualified finding needs both cue sets present **and close together**. Requiring mere
    co-presence anywhere in the clause is what made a single sentence naming two compartments assert
    all four of the laterality-paired findings:

        "Degenerative changes in the medial compartment with a tear of the lateral meniscus"

    contains *medial*, *lateral*, *menisc* and a degenerative term, so co-presence fires Medial
    Meniscus and Lateral OA as well as the two that are actually stated. Measured over the corpus,
    Medial OA and Lateral OA both fired on 44 % of the studies asserting either, against 10 % in the
    gold labels — a 4.4× over-coupling landing on four of the twelve columns.

    Proximity separates them because a genuine binding is short in every language in the corpus —
    "medial meniscus", "menisco interno", "Innenmeniskus", "medijalnog meniskusa", "corne
    postérieure du ménisque médial" all sit well under the gap — while the spurious pairing above
    spans 43 characters. The nearest cue/qualifier pair is the one that counts, and the returned
    position is the later of that pair, so negation scope is judged where the claim completes.
    """
    primary = _all_indices(text, finding.cues)
    if not primary:
        return None
    if not finding.qualifiers:
        return primary[0], 0
    secondary = _all_indices(text, finding.qualifiers)
    if not secondary:
        return None
    gap, at = min((abs(p - s), max(p, s)) for p in primary for s in secondary)
    return (at, gap) if gap <= finding.max_gap else None


@dataclass(frozen=True)
class Assertion:
    label: str
    state: str
    target: float
    weight: float
    clause: str


def label_report(text: str) -> dict[str, Assertion]:
    """Extract one assertion per finding from a single report.

    Clauses are read in order and the strongest assertion wins: an explicit positive anywhere beats
    a negative elsewhere, because a report that negates a finding in the Findings section and then
    asserts it in the Impression is asserting it. Hedges never override a definite statement.
    """
    best: dict[str, Assertion] = {}
    rank = {POSITIVE: 3, NEGATIVE: 2, HEDGED: 1}

    for clause in segment(text):
        folded = fold(clause.subject)

        hits: dict[str, tuple[int, int]] = {}
        for finding in FINDINGS:
            m = _match(finding, folded)
            if m is not None:
                hits[finding.label] = m

        hits = _resolve_laterality(hits)
        negated = _negated(folded, hits)
        for label, (at, _gap) in hits.items():
            finding = FINDINGS_BY_LABEL[label]

            if label in negated:
                state, target, weight = NEGATIVE, 0.0, 1.0
            elif _first_index(folded, HEDGE) is not None:
                state, target, weight = HEDGED, 0.5, 0.0
            elif finding.needs_pathology and _first_index(folded, PATHOLOGY) is None:
                # The structure was named but nothing was claimed about it — no information.
                continue
            else:
                state, target, weight = POSITIVE, _severity(folded), 1.0

            current = best.get(label)
            if current is None or rank[state] > rank[current.state]:
                best[label] = Assertion(label, state, target, weight, clause.text[:160])
    return best


def label_corpus(
    reports: pd.DataFrame,
    labels: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Label every report.

    Returns ``(targets, weights, states)``, all indexed by study id. ``targets`` carries NaN wherever
    the labeller made no assertion, which is what :mod:`rsna_knee.verify` reads as silence — the
    training-ready target with the silent prior filled in is produced by :func:`training_targets`.
    """
    labels = labels or LABELS
    ids = reports[STUDY_ID].to_numpy()
    targets = pd.DataFrame(np.nan, index=ids, columns=labels, dtype=float)
    weights = pd.DataFrame(np.nan, index=ids, columns=labels, dtype=float)
    states = pd.DataFrame("", index=ids, columns=labels, dtype=object)

    for sid, text in zip(ids, reports[REPORT].fillna("")):
        for label, a in label_report(text).items():
            targets.at[sid, label] = a.target
            weights.at[sid, label] = a.weight
            states.at[sid, label] = a.state

    states = states.replace("", SILENT)
    return targets, weights, states


def blend_labellers(
    rule_targets: pd.DataFrame,
    rule_weights: pd.DataFrame,
    llm_targets: pd.DataFrame,
    llm_weights: pd.DataFrame,
    labels: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine the rule and LLM labellers into one target/weight matrix.

    Measured against the 58 gold studies, averaging the two wherever both spoke beats either alone
    and beats both precedence orders:

    ==================  ======  ======  ==========================
    variant             macro   cells   vs rules (95 % CI)
    ==================  ======  ======  ==========================
    rules only          0.801   0.490   —
    LLM only            0.841   0.643   +0.040 [+0.010, +0.074]
    rules, LLM fills     0.830   0.686   +0.030 [+0.010, +0.053]
    LLM, rules fill     0.838   0.686   +0.038 [+0.008, +0.069]
    **mean where both** 0.853   0.686   **+0.053 [+0.027, +0.082]**
    ==================  ======  ======  ==========================

    That ordering is the ordinary ensembling result — the two labellers read the same reports by
    different mechanisms, so their errors are only partly shared and the mean cancels some of both.
    It is also the only variant whose interval clears zero by a comfortable margin.

    **Weight falls with disagreement.** Where the two agree, the cell keeps the average of their
    weights; where they disagree the weight is scaled down in proportion, reaching zero when one says
    present and the other absent. This is the same judgement the module already applies to hedging —
    a cell nobody can settle is recorded and not trained on — and it reuses the premise behind
    ``verify.disagreement``: disagreement between two independent labellers is where the labels are
    wrong. Disagreement is not rare, so this matters: it runs from 8 % of cells on MCL to 34 % on
    Lateral OA.

    Cells neither labeller asserted stay NaN, for :func:`training_targets` to fill with the prior.
    """
    labels = labels or LABELS
    idx = rule_targets.index.union(llm_targets.index)
    rt = rule_targets.reindex(index=idx, columns=labels)
    rw = rule_weights.reindex(index=idx, columns=labels)
    lt = llm_targets.reindex(index=idx, columns=labels)
    lw = llm_weights.reindex(index=idx, columns=labels)

    both = rt.notna() & lt.notna()
    agreement = 1.0 - (rt - lt).abs()

    targets = ((rt + lt) / 2.0).where(both, rt.where(rt.notna(), lt))
    weights = (((rw + lw) / 2.0) * agreement).where(both, rw.where(rw.notna(), lw))
    return targets, weights


def training_targets(
    targets: pd.DataFrame,
    weights: pd.DataFrame,
    gold: pd.DataFrame | None = None,
    gold_weight: float = 8.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill silent cells with the weak prior and overlay the gold labels at elevated weight.

    Gold labels are read from the images rather than from text, so where they exist they replace the
    derived target outright. They stay *in* training — they are the only image-derived supervision
    available — which is why the annotation check must be restricted to held-out studies.
    """
    t = targets.fillna(SILENT_TARGET)
    w = weights.fillna(SILENT_WEIGHT)
    if gold is not None:
        common = t.index.intersection(gold.index)
        known = gold.loc[common].notna()
        t.loc[common] = gold.loc[common].where(known, t.loc[common])
        w.loc[common] = w.loc[common].where(~known, gold_weight)
    return t, w
