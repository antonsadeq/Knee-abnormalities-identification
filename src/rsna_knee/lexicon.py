"""Multilingual cue vocabulary for the twelve findings.

**No language is identified before matching.** Every cue list below carries all nine languages at
once and each clause is tested against the union. Routing on a detected language would mean
committing to a guess before any evidence is read, and a misrouted report loses every cue at once;
pooling costs little because Greek and Cyrillic cues cannot collide with Latin-script ones at all,
and among the Latin-script languages the vocabulary of interest is close enough that a shared cue is
usually right. The price is paid in coverage instead, which is the thing ``verify.coverage``
measures.

All cues are written naturally here and folded at import, so they compare equal against text passed
through :func:`rsna_knee.reports.fold` regardless of case or diacritics.

Each finding declares two independent properties, because conflating them gets one of the twelve
wrong every time:

``qualifiers``
    When present, a second cue set that must **also** match. Osteoarthritis needs a compartment
    *and* a degenerative term — "medial femoral condyle" is not a statement about arthritis, and
    "arthrosis" alone does not say where. The menisci need a laterality cue and the meniscus stem,
    which is what reaches declined forms like Croatian "medijalnog meniskusa".
``needs_pathology``
    Whether naming the thing is enough. "Effusion" is itself the finding, so a mention asserts it.
    "Medial meniscus" is an anatomical structure that is present in every knee, so a mention asserts
    nothing until a pathology term or a negation says what is true of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .reports import fold


def _f(*terms: str) -> tuple[str, ...]:
    return tuple(fold(t) for t in terms)


# --------------------------------------------------------------------------------------------
# Scope cues
# --------------------------------------------------------------------------------------------

#: Pooled negation. Ordered longest-first at use so "no evidence of" wins over bare "no".
NEGATION = _f(
    # en
    "no evidence of", "no definite", "no significant", "without", "no ", "not ", "absent",
    "unremarkable", "normal", "intact", "preserved", "within normal limits", "free of", "negative for",
    # es
    "sin signos de", "sin evidencia", "no hay", "ausencia de", "ausente", "sin ", "conservad",
    "integr", "dentro de limites normales", "no se observa", "no se identifica",
    # fr
    "pas de", "absence de", "aucun", "aucune", "sans particularite", "sans ", "respecte", "libre de",
    # de
    "kein nachweis", "keine", "kein ", "ohne ", "unauffallig", "regelrecht", "intakt", "nicht ",
    # nl
    "geen ", "zonder ", "onopvallend", "normaal", "afwezig",
    # tr
    "izlenmemektedir", "saptanmamis", "gorulmedi", "mevcut degil", "yoktur", " yok", "dogal",
    "olagan",
    # hr — stems, not full words: "urednog signala" is the standard normal phrasing and matches
    # none of "uredan"/"uredna"/"uredni".
    "bez ", "nema ", "uredn", "intaktan", "ocuvan", "prikazuje se normaln",
    # el
    "χωρις", "δεν ", "ουδεν", "φυσιολογικ", "ακεραι", "ελευθερ",
    # bg
    "без ", "не ", "липсва", "нормал", "интакт", "запазен",
)

#: Pooled hedging. A hedged assertion becomes weight ~0 rather than a guessed label.
HEDGE = _f(
    "possible", "probable", "suspect", "suspicious", "cannot exclude", "questionable", "equivocal",
    "likely", "may represent", "might", "suggest", "consistent with", "concerning for",
    "posible", "sospecha", "sugiere", "no se puede excluir", "dudoso", "impresiona",
    "evoquant", "compatible avec", "ne peut etre exclu", "suspicion",
    "moglich", "verdacht", "fraglich", "vereinbar mit", "nicht auszuschliessen", "am ehesten",
    "mogelijk", "verdenking", "verdacht", "suspect",
    "supheli", "olasi", "muhtemel", "ekarte edilemez", "izlenimi",
    "mogu", "sumnja", "vjerojatn", "suspektn",
    "πιθαν", "υποψια", "συμβατ",
    "възможно", "вероятно", "съмнение", "подозр",
)

#: Contrast conjunctions terminate a negation's scope: "no medial tear but lateral is torn".
CONTRAST = _f(
    "but ", "however", "although", "pero ", "aunque", "sin embargo", "mais ", "cependant",
    "aber ", "jedoch", "maar ", "echter", "ancak", "fakat", "ali ", "αλλα", "ομως", "но ", "обаче",
)

#: Severity modifiers, mapping to a graded target. The gold labels use severity-thresholded
#: criteria that the reports do not share, so the grade is emitted and the cut fitted later.
SEVERITY: dict[float, tuple[str, ...]] = {
    # Romance forms are enumerated with their vowel rather than left as a bare stem. A stem is
    # anchored at its *start* only, so `sever` matches "several", `extens` matches "extension"
    # (knee extension — a movement, not a severity) and `leve` matches the English "level". Since
    # the strongest grade wins, any one of those promotes a mild finding to 0.95.
    0.35: _f(
        "minimal", "mild", "slight", "small", "trace", "low grade", "low-grade", "grade 1", "grade i",
        "minim", "leve ", "leves", "levemente", "ligero", "pequen", "discreto", "leger", "legere",
        "faible", "bas grade",
        "gering", "diskret", "leicht", "licht", "hafif", "az ", "blag", "ηπι", "μικρ", "лек",
    ),
    0.65: _f(
        "moderate", "grade 2", "grade ii", "moderad", "modere", "massig", "maessig", "matig",
        "orta ", "umjeren", "μετρι", "умерен",
    ),
    0.95: _f(
        "severe", "marked", "advanced", "extensive", "large", "complete", "full thickness",
        "full-thickness", "grade 3", "grade 4", "grade iii", "grade iv",
        # "severa " takes the exact-boundary marker because it is itself a prefix of "several".
        "severo", "severos", "severa ", "severas", "grave", "avanzad",
        "extenso", "extensa", "complet", "espesor total",
        "importante",
        "ausgepragt", "schwer", "hochgradig", "ernstig", "uitgebreid",
        "ileri", "belirgin", "tam kat", "izrazen", "tesk", "σοβαρ", "εκτεταμεν", "тежк", "изразен",
    ),
}

#: Pathology terms that turn a mention of a *structure* into a positive assertion.
PATHOLOGY = _f(
    # tears / injury
    "tear", "torn", "rupture", "ruptured", "sprain", "injury", "lesion", "disruption", "discontinu",
    "rotura", "roto", "desgarro", "lesion", "esguince", "rotur",
    "dechirure", "dechirement", "rupture", "entorse", "fissure",
    # Unanchored: German fuses these too — "Innenmeniskusriss", "Kreuzbandruptur".
    "*riss", "*ruptur", "laesion", "lasion", "zerrung",
    "scheur", "ruptuur", "letsel",
    "yirtik", "rupture", "yaralanma", "lezyon", "burkulma",
    "puknu", "ozljed", "lezij", "ruptur",
    "ρηξη", "ρηγμα", "βλαβη", "θλαση",
    "разкъсв", "руптура", "увреда", "лезия",
    # Intrasubstance degeneration. Turkish reports describe the menisci almost exclusively this way
    # — "grade II dejenerasyon" rather than "yırtık" — which left Turkish medial meniscus at 0.12
    # coverage against 0.63 lateral. Degeneration is *not* a tear: the grade decides, and grades are
    # picked up by SEVERITY so the cut can be fitted against the gold labels rather than guessed.
    "degenerat", "degenerac", "dejenerasyon", "dejeneratif", "degeneraz", "degenerescence",
    "εκφυλισ", "дегенерац", "maceration", "fibrilacion", "fibrillation",
    # signal abnormality used as evidence of injury
    "increased signal", "high signal", "signal alteration", "abnormal signal",
    "aumento de senal", "hiperintens", "alteracion de senal",
    "signal augmente", "hypersignal",
    "signalanhebung", "signalalteration", "signalsteigerung",
    "signaalverhoging", "sinyal artis", "povisen signal",
    "αυξημενο σημα", "повишен сигнал",
)


#: Default ceiling on the distance between a finding's two cue sets. Suits the noun-adjective
#: bindings — "medial meniscus", "menisco interno", "Innenmeniskus", "koštani edem" — which are
#: short in every language here.
DEFAULT_MAX_GAP = 30

#: Osteoarthritis is stated far more diffusely than a meniscus is: "thinning of **lateral**
#: femorotibial articular **cartilage**" already spans 31 characters, and the corpus puts the 95th
#: percentile of genuine compartment-to-degenerative gaps at 86. A window tight enough for the
#: menisci silently drops half of the OA assertions, so these three carry their own.
#:
#: Swept against the gold set at 30/60/90/120/200/400: macro label AUC runs
#: 0.779 / 0.792 / 0.806 / 0.806 / 0.806 / 0.806 and OA coverage 0.31 / 0.41 / 0.44 / 0.44 / 0.45 /
#: 0.45. It saturates at 90 — beyond that the clause boundary binds before the window does — so 90
#: buys the full coverage without leaving the constraint purely decorative. The cross-compartment
#: confusion the window was meant to stop is handled by ``labeller._resolve_laterality`` instead,
#: which is the part that actually discriminates.
OA_MAX_GAP = 90


@dataclass(frozen=True)
class Finding:
    label: str
    cues: tuple[str, ...]
    qualifiers: tuple[str, ...] = field(default=())
    needs_pathology: bool = False
    max_gap: int = DEFAULT_MAX_GAP


_DEGENERATIVE = _f(
    "osteoarthritis", "osteoarthrosis", "arthrosis", "degenerative", "chondral", "chondropathy",
    "chondromalacia", "cartilage loss", "cartilage thinning", "osteophyte", "joint space narrowing",
    "artrosis", "gonartrosis", "condropatia", "condromalacia", "degenerativ", "osteofito",
    "cartilag", "condral", "ulceras condrales",
    "arthrose", "gonarthrose", "chondropathie", "chondral", "osteophyte", "cartilage",
    "gonarthrose", "chondropathie", "chondromalazie", "knorpel", "osteophyt", "degenerativ",
    "artrose", "kraakbeen", "chondropathie",
    "osteoartrit", "kondromalazi", "kikirdak", "dejeneratif", "osteofit",
    "artroza", "hondromalacij", "hrskavic", "degenerativ", "osteofit",
    "αρθριτιδα", "χονδροπαθεια", "χονδρ", "οστεοφυτ", "εκφυλιστικ",
    "артроза", "хондропатия", "хрущял", "остеофит", "дегенератив",
)

# Cues are stems rather than full words wherever a language declines or agglutinates: Croatian
# writes "medijalnog meniskusa", Turkish "medial menisküste", Bulgarian "медиалния менискус". A
# full-word cue matches none of those, which is how Croatian scored 0.00 coverage on both menisci
# across 406 reports in the first audit.
_MEDIAL = _f(
    "medial", "internal", "inner", "intern", "medi", "innen", "mediaal", "binnenst",
    "ic ", "unutarnj", "medijaln", "εσω", "медиал", "вътреш",
)
_LATERAL = _f(
    "lateral", "external", "outer", "extern", "aussen", "ausen", "lateraal", "buitenst",
    "dis ", "vanjsk", "lateraln", "εξω", "латерал", "външ",
)

#: Meniscus stem across all nine languages. Unanchored (``*``) because German fuses it into
#: ``Innenmeniskus`` / ``Außenmeniskus``, which 70 % of the German reports use.
_MENISCUS = _f("*menisc", "*menisk", "*μηνισκ", "*мениск")

#: Bone-marrow oedema and bruising, split into the abnormality and the tissue it sits in. Requiring
#: both is what separates a bone contusion from Hoffa fat-pad oedema, which shares the first cue and
#: is not a target here. Splitting also survives Croatian declension — ``koštani edem``,
#: ``koštanog edema``, ``kostanim edemom`` — where a fixed two-word cue matches none of the forms.
_EDEMA = _f("edema", "edem", "oedem", "*odem", "*oedeem", "bruise", "contusion", "contus", "kontuz",
            "οιδημα", "оток")
_BONE = _f("bone", "marrow", "osseous", "osse", "ose", "*knochen", "*bot", "*been", "kemik",
           "kostan", "kosti", "οστ", "кост")
_PATELLOFEMORAL = _f(
    "patellofemoral", "femoropatellar", "retropatellar", "patellar", "patella", "trochlea",
    "rotulian", "rotula", "femoropatelar", "retrorrotulian",
    "femoropatellaire", "retropatellaire", "patellaire", "rotulien",
    "femoropatellar", "retropatellar", "patella",
    "patellofemoraal", "patella", "patellar",
    "patellofemoral", "patella", "diz kapagi",
    "patelofemoraln", "patela", "επιγονατιδομηριαι", "επιγονατιδ", "пателофеморал", "пател",
)

FINDINGS: tuple[Finding, ...] = (
    Finding("ACL", needs_pathology=True, cues=_f(
        "anterior cruciate ligament", "acl ", "lca ", "ligamento cruzado", "ligamentos cruzados",
        "ligament croise", "vorderes kreuzband", "vkb ", "kreuzband", "voorste kruisband",
        "kruisband", "on capraz bag", "ocb ", "capraz bag", "prednj", "ukrizen",
        "προσθιος χιαστος", "χιαστ", "предна кръстна", "кръстна връзка",
    )),
    Finding("MCL", needs_pathology=True, cues=_f(
        "medial collateral ligament", "mcl ", "lcm ", "ligamento colateral medial",
        "ligamento colateral interno", "ligament collateral medial", "ligament collateral interne",
        "mediales kollateralband", "innenband", "mediale collaterale band", "mediale band",
        # Croatian "medijaln" is *medial*, not *medial collateral*, so it matched "medijalnog
        # meniskusa" and made every Croatian medial meniscus tear an MCL tear too — Croatian MCL
        # reached 0.96 coverage at a 0.57 positive share (polarity z = +29) against a gold MCL
        # prevalence of 0.16. "kolateraln" reaches the ligament without the collision.
        "medial kollateral", "ic yan bag", "kolateraln", "collaterale band",
        # Greek and Bulgarian reports assess the collateral ligaments as a pair — "πλαγίων
        # συνδέσμων", "колатералните лигаменти" — and never name the medial one alone, which left
        # MCL at 0.02 and 0.01 coverage. The generic cue mostly harvests negatives, which is the
        # point; watch MCL positive share in the polarity audit for the LCL-only false positive.
        "πλαγι", "колатерал", "вътрешна странична",
    )),
    # Both menisci are compound rather than structure findings: laterality and the meniscus stem are
    # matched independently so declined and suffixed forms are reached in every language.
    Finding("Medial Meniscus", _MEDIAL, _MENISCUS, needs_pathology=True),
    Finding("Lateral Meniscus", _LATERAL, _MENISCUS, needs_pathology=True),
    Finding("Medial OA", _MEDIAL, _DEGENERATIVE, max_gap=OA_MAX_GAP),
    Finding("Lateral OA", _LATERAL, _DEGENERATIVE, max_gap=OA_MAX_GAP),
    Finding("PF OA", _PATELLOFEMORAL, _DEGENERATIVE, max_gap=OA_MAX_GAP),
    Finding("Effusion", _f(
        "joint effusion", "effusion", "joint fluid", "derrame", "epanchement",
        "*erguss", "gewrichtsvocht", "hydrops", "effusie",
        # Turkish reports overwhelmingly say "mayi/mayii artışı" (increased fluid) rather than
        # "efüzyon"; the first audit scored Turkish effusion at 0.07 coverage against 0.93 English.
        "efuzyon", "mayi", "mayii", "eklem ici sivi", "izljev", "αρθρικη συλλογη", "συλλογη",
        "излив",
    )),
    Finding("Synovitis", _f(
        "synovitis", "synovial thickening", "synovial proliferation", "sinovitis",
        "engrosamiento sinovial", "synovite", "synovialitis", "synoviale verdickung",
        "synoviitis", "sinovit", "sinovijitis", "υμενιτιδα", "υμενικ", "синовит",
    )),
    Finding("Baker's", _f(
        "baker", "popliteal cyst", "quiste popliteo", "kyste poplite", "poplitealzyste",
        "bakerzyste", "baker-zyste", "bakercyste", "popliteale cyste", "popliteal kist",
        "poplitealn", "bakerova", "κυστη baker", "κυστη κατα baker", "поплитеал", "бейкър",
        # Qualified by a cyst stem so that Bulgarian "поплитеална ямка" (popliteal *fossa*) does not
        # read as a cyst. Stems are unanchored for the German fusions "Bakerzyste"/"Poplitealzyste".
    ), _f("*cyst", "*cist", "*kist", "*quist", "*kyst", "*zyste", "*cyste", "*κυστ", "*киста")),
    Finding("Contusion", _EDEMA, _BONE),
    Finding("Fracture", _f(
        "fracture", "fractura", "fraktur", "fractuur", "kirik", "prijelom", "καταγμα", "фрактур",
        "счупван",
    )),
)

FINDINGS_BY_LABEL = {f.label: f for f in FINDINGS}
