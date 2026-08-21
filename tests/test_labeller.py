"""Regression tests for the rule labeller.

Every case here is a real failure the corpus audits surfaced, kept so the fix cannot be undone
silently by a later vocabulary edit.
"""

import numpy as np
import pandas as pd
import pytest

from rsna_knee.config import LABELS, REPORT, STUDY_ID
from rsna_knee.labeller import (
    HEDGED,
    NEGATIVE,
    POSITIVE,
    SILENT_TARGET,
    SILENT_WEIGHT,
    label_corpus,
    label_report,
    training_targets,
)


def state(text, label):
    a = label_report(text).get(label)
    return a.state if a else None


# --- negation scope ---------------------------------------------------------------------------

def test_negation_before_the_finding():
    assert state("No joint effusion.", "Effusion") == NEGATIVE
    assert state("Kein Gelenkserguss.", "Effusion") == NEGATIVE


def test_negation_after_the_finding():
    # Spanish routinely places the negation after the structure. A forward-only scope read the whole
    # Spanish corpus as ~0.95 positive.
    assert state("Menisco lateral de morfologia conservada, sin signos de rotura.",
                 "Lateral Meniscus") == NEGATIVE


def test_negation_at_the_end_of_a_verb_final_clause():
    # Turkish is verb-final, so its negation is essentially always last.
    assert state("On capraz bag normaldir.", "ACL") == NEGATIVE
    assert state("Medial menisküste yirtik izlenmemektedir.", "Medial Meniscus") == NEGATIVE


def test_contrast_conjunction_terminates_negation_scope():
    text = "No medial meniscus tear but the lateral meniscus is torn."
    assert state(text, "Medial Meniscus") == NEGATIVE
    assert state(text, "Lateral Meniscus") == POSITIVE


# --- word boundaries --------------------------------------------------------------------------

def test_abnormal_is_not_read_as_a_negation():
    # "normal" is a substring of the Spanish "anormal", which means the opposite.
    assert state("Senal anormal del menisco medial con rotura.", "Medial Meniscus") == POSITIVE


def test_short_negation_cue_does_not_match_inside_a_word():
    # Bare "no" must not fire on "nodular".
    assert state("Nodular synovial proliferation.", "Synovitis") == POSITIVE


def test_german_compound_words_are_reached():
    # 70% of the German reports fuse the meniscus into a single word; anchored stems miss them all.
    assert state("Innenmeniskusriss.", "Medial Meniscus") == POSITIVE
    assert state("Unauffalliger Aussenmeniskus.", "Lateral Meniscus") == NEGATIVE
    assert state("Knochenmarksodem im medialen Femurkondylus.", "Contusion") == POSITIVE


def test_croatian_declension_is_reached():
    assert state("Degenerativne promjene medijalnog meniskusa s rupturom.", "Medial Meniscus") == POSITIVE
    assert state("Kostani edem tibije.", "Contusion") == POSITIVE


# --- what counts as an assertion ---------------------------------------------------------------

def test_naming_a_structure_asserts_nothing_on_its_own():
    # An intact ACL is present in every knee; the mention alone carries no label.
    assert state("The anterior cruciate ligament is visualised on sagittal images.", "ACL") is None


def test_an_entity_mention_is_itself_the_assertion():
    assert state("Moderate joint effusion.", "Effusion") == POSITIVE


def test_osteoarthritis_needs_both_a_compartment_and_a_degenerative_term():
    assert state("Medial femoral condyle is unremarkable in outline.", "Medial OA") is None
    assert state("Medial compartment osteoarthritis.", "Medial OA") == POSITIVE


def test_bone_oedema_outside_bone_is_not_a_contusion():
    # Hoffa fat-pad oedema shares the first cue and is not a target.
    assert state("Edem hoffinog masnog jastucica.", "Contusion") is None


def test_popliteal_fossa_without_a_cyst_is_not_bakers():
    assert state("Poplitealna jama bez osobitosti.", "Baker's") is None
    assert state("Poplitealna cista.", "Baker's") == POSITIVE


# --- grading and weighting ----------------------------------------------------------------------

def test_severity_grades_the_target():
    mild = label_report("Minimal joint effusion.")["Effusion"]
    severe = label_report("Large joint effusion.")["Effusion"]
    assert mild.target < severe.target
    assert mild.weight == severe.weight == 1.0


def test_a_hedge_is_recorded_at_zero_weight():
    a = label_report("Possible tear of the medial meniscus.")["Medial Meniscus"]
    assert a.state == HEDGED
    assert a.weight == 0.0


def test_a_definite_statement_outranks_a_hedge_elsewhere():
    text = "Possible medial meniscus tear.\nImpression: medial meniscus tear."
    assert state(text, "Medial Meniscus") == POSITIVE


def test_a_positive_in_the_impression_outranks_an_earlier_negative():
    text = "Findings: no ACL tear.\nImpression: ACL rupture."
    assert state(text, "ACL") == POSITIVE


# --- corpus level ---------------------------------------------------------------------------------

def test_label_corpus_marks_silence_as_nan_not_zero():
    reports = pd.DataFrame({STUDY_ID: ["a"], REPORT: ["Moderate joint effusion."]})
    targets, weights, states = label_corpus(reports)
    assert targets.loc["a", "Effusion"] > 0
    # Collapsing an unmentioned finding to 0.0 would train every silence as a confident negative.
    assert pd.isna(targets.loc["a", "Fracture"])
    assert states.loc["a", "Fracture"] == "silent"


def test_training_targets_fills_silence_with_a_weak_prior():
    reports = pd.DataFrame({STUDY_ID: ["a"], REPORT: ["Moderate joint effusion."]})
    targets, weights, _ = label_corpus(reports)
    t, w = training_targets(targets, weights)
    assert t.loc["a", "Fracture"] == SILENT_TARGET
    assert w.loc["a", "Fracture"] == SILENT_WEIGHT
    assert w.loc["a", "Effusion"] == 1.0


def test_training_targets_overlays_gold_at_elevated_weight():
    reports = pd.DataFrame({STUDY_ID: ["a"], REPORT: ["No joint effusion."]})
    targets, weights, _ = label_corpus(reports)
    gold = pd.DataFrame(0.0, index=["a"], columns=LABELS)
    gold.loc["a", "Effusion"] = 1.0  # the images disagree with the report

    t, w = training_targets(targets, weights, gold, gold_weight=8.0)
    assert t.loc["a", "Effusion"] == pytest.approx(1.0)
    assert w.loc["a", "Effusion"] == pytest.approx(8.0)


# --- severity cues must be anchored, like every other cue set ----------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        # "sever"/"severa" are prefixes of "several", "extens" of "extension" (a movement, not a
        # severity), "complet" of nothing once anchored — but "incomplete" was reaching it.
        ("several small foci of marrow oedema", 0.35),
        ("limited extension of the knee", 0.8),
        ("incomplete tear of the anterior cruciate ligament", 0.8),
        ("signal level within normal limits", 0.8),
        # ...while the grades they exist to catch still land.
        ("severe chondropathy", 0.95),
        ("condropatia severa", 0.95),
        ("lesion extensa", 0.95),
        ("complete tear", 0.95),
        ("moderate effusion", 0.65),
        ("condropatia leve", 0.35),
    ],
)
def test_severity_is_word_anchored(text, expected):
    from rsna_knee.labeller import _severity
    from rsna_knee.reports import fold

    assert _severity(fold(text)) == expected


# --- laterality binds to its noun ---------------------------------------------------------------

def positives(text):
    return sorted(k for k, v in label_report(text).items() if v.state == POSITIVE)


def test_naming_two_compartments_does_not_assert_all_four():
    # Co-presence anywhere in the clause used to fire Medial Meniscus and Lateral OA as well.
    assert positives(
        "Degenerative changes in the medial compartment with a tear of the lateral meniscus."
    ) == ["Lateral Meniscus", "Medial OA"]


def test_genuine_coordination_still_asserts_both_sides():
    assert positives("Medial and lateral meniscal tears.") == [
        "Lateral Meniscus",
        "Medial Meniscus",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Rotura del menisco interno.",          # es, noun-first
        "Innenmeniskusriss.",                   # de, fused compound
        "Ruptura medijalnog meniskusa.",        # hr, declined
    ],
)
def test_short_bindings_survive_in_every_language(text):
    assert "Medial Meniscus" in positives(text)


def test_diffuse_oa_phrasing_is_not_dropped_by_the_window():
    # "lateral ... cartilage" spans 31 characters; the meniscus window would have discarded it.
    assert positives("Thinning of lateral femorotibial articular cartilage.") == ["Lateral OA"]


def test_croatian_medial_no_longer_means_medial_collateral():
    # "medijaln" is *medial*, so it matched "medijalnog meniskusa" and made every Croatian medial
    # meniscus tear an MCL tear too (polarity z = +29).
    assert "MCL" not in positives("Ruptura medijalnog meniskusa.")


# --- a negation belongs to a subject ------------------------------------------------------------

@pytest.mark.parametrize(
    "text,negated,asserted",
    [
        # A trailing negation belongs to the nearest subject, not to the whole clause.
        ("Tear of the lateral meniscus, medial meniscus intact.",
         "Medial Meniscus", "Lateral Meniscus"),
        ("Medial meniscus intact, lateral meniscus torn.",
         "Medial Meniscus", "Lateral Meniscus"),
        # ...but a contrast conjunction still terminates scope in the other direction.
        ("No medial meniscus tear but the lateral meniscus is torn.",
         "Medial Meniscus", "Lateral Meniscus"),
    ],
)
def test_trailing_negation_does_not_span_the_clause(text, negated, asserted):
    assert state(text, negated) == NEGATIVE
    assert state(text, asserted) == POSITIVE


@pytest.mark.parametrize(
    "text,labels",
    [
        # Coordination: one predicate, several subjects. Sweeping only the nearest lost the rest.
        ("The ACL and MCL are intact.", ["ACL", "MCL"]),
        ("Medial and lateral menisci are intact.", ["Lateral Meniscus", "Medial Meniscus"]),
    ],
)
def test_negation_sweeps_coordinated_subjects(text, labels):
    for label in labels:
        assert state(text, label) == NEGATIVE, label


@pytest.mark.parametrize(
    "text",
    [
        "Menisco lateral, sin signos de rotura.",                     # comma inside one assertion
        "Menisco lateral de morfologia conservada, sin signos de rotura.",
    ],
)
def test_a_comma_alone_does_not_stop_a_negation(text):
    assert state(text, "Lateral Meniscus") == NEGATIVE


def test_unclaimed_negation_distributes_over_its_list():
    assert state("No evidence of meniscal tear, ligament injury, or fracture.", "Fracture") == (
        NEGATIVE
    )


# --- blending the two labellers -----------------------------------------------------------------

def test_blend_averages_where_both_assert():
    from rsna_knee.labeller import blend_labellers

    rt = pd.DataFrame([[0.8] + [np.nan] * 11], index=["s1"], columns=LABELS)
    lt = pd.DataFrame([[0.6] + [np.nan] * 11], index=["s1"], columns=LABELS)
    rw = pd.DataFrame([[1.0] + [np.nan] * 11], index=["s1"], columns=LABELS)
    lw = pd.DataFrame([[1.0] + [np.nan] * 11], index=["s1"], columns=LABELS)

    t, w = blend_labellers(rt, rw, lt, lw)
    assert t.at["s1", "ACL"] == pytest.approx(0.7)
    # Weight drops in proportion to the disagreement: |0.8 - 0.6| = 0.2.
    assert w.at["s1", "ACL"] == pytest.approx(0.8)


def test_blend_zeroes_weight_on_flat_contradiction():
    from rsna_knee.labeller import blend_labellers

    rt = pd.DataFrame([[1.0] + [np.nan] * 11], index=["s1"], columns=LABELS)
    lt = pd.DataFrame([[0.0] + [np.nan] * 11], index=["s1"], columns=LABELS)
    ones = pd.DataFrame([[1.0] + [np.nan] * 11], index=["s1"], columns=LABELS)

    t, w = blend_labellers(rt, ones, lt, ones)
    assert t.at["s1", "ACL"] == pytest.approx(0.5)
    assert w.at["s1", "ACL"] == pytest.approx(0.0)


def test_blend_takes_whichever_labeller_spoke():
    from rsna_knee.labeller import blend_labellers

    nan = [np.nan] * 12
    rt = pd.DataFrame([[0.9] + nan[1:]], index=["s1"], columns=LABELS)
    lt = pd.DataFrame([nan[:1] + [0.4] + nan[2:]], index=["s1"], columns=LABELS)
    rw = pd.DataFrame([[1.0] + nan[1:]], index=["s1"], columns=LABELS)
    lw = pd.DataFrame([nan[:1] + [0.7] + nan[2:]], index=["s1"], columns=LABELS)

    t, w = blend_labellers(rt, rw, lt, lw)
    assert t.at["s1", "ACL"] == pytest.approx(0.9)
    assert w.at["s1", "ACL"] == pytest.approx(1.0)
    assert t.at["s1", "MCL"] == pytest.approx(0.4)
    assert w.at["s1", "MCL"] == pytest.approx(0.7)
    # Neither spoke about the rest, which stays NaN for training_targets to fill.
    assert t.at["s1", "Fracture"] != t.at["s1", "Fracture"]


def test_blend_unions_the_study_index():
    from rsna_knee.labeller import blend_labellers

    nan = [np.nan] * 12
    rt = pd.DataFrame([[0.9] + nan[1:]], index=["s1"], columns=LABELS)
    lt = pd.DataFrame([[0.5] + nan[1:]], index=["s2"], columns=LABELS)
    rw = pd.DataFrame([[1.0] + nan[1:]], index=["s1"], columns=LABELS)
    lw = pd.DataFrame([[1.0] + nan[1:]], index=["s2"], columns=LABELS)

    t, _ = blend_labellers(rt, rw, lt, lw)
    assert sorted(t.index) == ["s1", "s2"]
