import numpy as np
import pandas as pd
import pytest

from rsna_knee.config import LABELS
from rsna_knee.verify import coverage, disagreement, gold_agreement, polarity, wilson


def _frame(n, value=np.nan, prefix="s"):
    return pd.DataFrame(
        value, index=[f"{prefix}{i}" for i in range(n)], columns=LABELS, dtype=float
    )


def test_wilson_stays_inside_the_unit_range_at_tiny_n():
    lo, hi = wilson(2, 2)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_is_wide_at_n_58_and_narrow_at_n_5000():
    lo58, hi58 = wilson(29, 58)
    lo5k, hi5k = wilson(2500, 5000)
    # The whole argument for the label-free checks rests on this gap being large.
    assert (hi58 - lo58) > 4 * (hi5k - lo5k)


def test_wilson_undefined_without_trials():
    assert all(np.isnan(v) for v in wilson(0, 0))


def test_gold_agreement_scores_a_perfect_labeller():
    gold = _frame(10, 0.0)
    gold.iloc[:4] = 1.0
    out = gold_agreement(gold.copy(), gold)
    assert (out["precision"] == 1.0).all()
    assert (out["recall"] == 1.0).all()
    assert (out["n"] == 10).all()


def test_gold_agreement_counts_only_asserted_studies():
    gold = _frame(10, 0.0)
    targets = gold.copy()
    targets.iloc[5:] = np.nan  # labeller stayed silent on half
    out = gold_agreement(targets, gold)
    assert (out["n"] == 5).all()
    assert out["asserted"].iloc[0] == pytest.approx(0.5)


def test_coverage_reports_a_language_with_missing_vocabulary():
    targets = _frame(200, 1.0)
    lang = pd.Series(["en"] * 100 + ["tr"] * 100, index=targets.index)
    targets.loc[lang == "tr", "Effusion"] = np.nan  # no Turkish cue for effusion
    cov = coverage(targets, lang)
    assert cov.loc["en", "Effusion"] == pytest.approx(1.0)
    assert cov.loc["tr", "Effusion"] == pytest.approx(0.0)


#: The nine languages actually present in the corpus — the polarity check is only meaningful with
#: enough of them for a median to be robust.
_LANGS = ["en", "es", "tr", "hr", "el", "de", "bg", "nl", "fr"]


def _corpus(per_lang=100, positive_rate=0.2, corrupt=None):
    """Nine languages at a shared positive rate, optionally with one reading everything positive."""
    index, langs, acl = [], [], []
    for lg in _LANGS:
        for i in range(per_lang):
            index.append(f"{lg}{i}")
            langs.append(lg)
            acl.append(1.0 if lg == corrupt else float(i < per_lang * positive_rate))
    targets = pd.DataFrame(np.nan, index=index, columns=LABELS, dtype=float)
    targets["ACL"] = acl
    return targets, pd.Series(langs, index=index)


def test_polarity_flags_an_inverted_negation_in_one_language():
    # French misses its negation idiom and reads every report as positive. It has zero gold studies,
    # so this failure is invisible to gold_agreement — and glaring here.
    targets, lang = _corpus(corrupt="fr")
    share, z = polarity(targets, lang)

    assert share.loc["fr", "ACL"] == pytest.approx(1.0)
    assert share.loc["en", "ACL"] == pytest.approx(0.2)
    assert z.loc["fr", "ACL"] > 3.0
    # And the eight clean languages must stay unflagged, or the check cannot say which one is wrong.
    assert (z.drop(index="fr")["ACL"].abs() < 3.0).all()


def test_polarity_does_not_flag_a_consistent_corpus():
    targets, lang = _corpus()
    _, z = polarity(targets, lang)
    assert (z["ACL"].abs() < 3.0).all()


def test_polarity_survives_a_reference_pinned_at_zero():
    # No language ever asserts a positive: the reference is 0, and a naive variance divides by zero.
    targets, lang = _corpus(positive_rate=0.0)
    _, z = polarity(targets, lang)
    assert z["ACL"].notna().all()


def test_disagreement_separates_conflict_from_one_sided_silence():
    a = _frame(10, 1.0)
    b = _frame(10, 1.0)
    b.iloc[:3, b.columns.get_loc("ACL")] = 0.0  # genuine conflict
    b.iloc[5:, b.columns.get_loc("ACL")] = np.nan  # b simply silent

    out = disagreement(a, b)
    assert out.loc["ACL", "disagree"] == 3
    assert out.loc["ACL", "both_asserted"] == 5
    assert out.loc["ACL", "only_a"] == 5
    assert out.loc["MCL", "disagree"] == 0
