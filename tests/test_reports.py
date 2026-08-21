from rsna_knee.reports import Clause, fold, segment, unwrap


def test_fold_maps_micro_sign_onto_mu():
    # Greek reports spell mu both ways; NFKD is what makes them compare equal.
    assert fold("µm") == fold("μm")


def test_fold_strips_diacritics_and_case():
    assert fold("Œdème OSSEUX") == fold("œdeme osseux")
    assert fold("MENISCO") == "menisco"


def test_unwrap_rejoins_a_hard_wrapped_sentence():
    text = "Rotura compleja degenerativa del cuerno anterior, cuerpo, cuerno\nposterior y parte"
    assert unwrap(text) == "Rotura compleja degenerativa del cuerno anterior, cuerpo, cuerno posterior y parte"


def test_unwrap_keeps_genuine_breaks():
    text = "Kein Gelenkserguss.\nKein freier Gelenkkorper."
    assert unwrap(text).count("\n") == 1


def test_unwrap_does_not_swallow_the_line_after_a_heading():
    assert unwrap("Fractures :\nAucune.") == "Fractures :\nAucune."


def test_unwrap_keeps_bulleted_items_separate():
    # The bullet marks a new item even though the previous line lacks closing punctuation.
    text = "> First finding without a full stop\n> Second finding"
    assert unwrap(text).count("\n") == 1


def test_segment_attaches_a_block_heading_to_its_clauses():
    clauses = segment("Fractures :\nAucune.\nFracture du plateau tibial.")
    assert [c.heading for c in clauses] == ["Fractures", "Fractures"]
    assert clauses[0].subject == "Fractures. Aucune"


def test_segment_handles_an_inline_heading():
    clauses = segment("Ligament croise posterieur : Normal.")
    assert clauses == [Clause(text="Normal", heading="Ligament croise posterieur", line=0)]


def test_inline_heading_does_not_leak_onto_later_lines():
    # "Normal" belongs to the PCL; the effusion clause that follows must not inherit that subject.
    clauses = segment("Ligament croise posterieur : Normal.\nEpanchement articulaire modere.")
    assert clauses[1].heading is None


def test_segment_splits_on_terminators_and_strips_bullets():
    clauses = segment("> No definite tear of medial meniscus. Minimal joint effusion")
    assert [c.text for c in clauses] == [
        "No definite tear of medial meniscus",
        "Minimal joint effusion",
    ]


def test_segment_drops_empty_fragments():
    assert all(len(c.text) > 1 for c in segment("A.. ; . Real clause here."))
