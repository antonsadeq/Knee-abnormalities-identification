"""Normalising and segmenting the multilingual report corpus.

Everything the labeller does depends on this module cutting the text correctly. Three properties of
this corpus make naive handling wrong, and each is fixed here rather than worked around downstream:

1. **Hard wrapping.** Many reports are wrapped at a fixed column, so a sentence breaks across two
   lines with no punctuation at the break. Splitting on newlines severs a finding from its anatomy —
   ``"Rotura compleja degenerativa del cuerno anterior, cuerpo, cuerno\\nposterior"`` becomes two
   fragments, neither of which states the finding.
2. **Headings carry the subject.** A French report reads ``Fractures :`` on one line and ``Aucune.``
   on the next; a clause of ``Aucune.`` alone asserts nothing. Headings are attached to the clauses
   beneath them.
3. **Codepoint drift.** Greek reports frequently spell mu with MICRO SIGN (U+00B5) rather than GREEK
   SMALL LETTER MU (U+03BC). NFKD maps one onto the other, so normalise before matching or the two
   spellings never compare equal.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Terminators that reliably end a clause across all nine languages in the corpus.
_CLAUSE_BREAK = re.compile(r"[.;!?•·]+|\s+[-–—>]\s+")

#: A heading is a short line ending in a colon, optionally numbered or bulleted.
_HEADING = re.compile(r"^\s*[\d>\-•*]*\s*([^:\n]{1,60}):\s*$")

#: An inline heading: "Ligament croisé postérieur : Normal." — subject and value on one line.
_INLINE_HEADING = re.compile(r"^\s*[\d>\-•*]*\s*([^:\n]{1,60}?)\s*:\s*(.+)$")

#: Leading list markers to strip from a line before use.
_BULLET = re.compile(r"^\s*(?:[>\-–—•*]+|\d+[.)])\s*")


def fold(text: str) -> str:
    """Case-fold, strip diacritics, and unify codepoint variants.

    NFKD decomposition does double duty: it separates combining marks so they can be dropped, and it
    maps compatibility characters (MICRO SIGN → mu, non-breaking space → space) onto their canonical
    forms. Greek loses its tonos and Latin its accents, which is what we want for cue matching —
    reports are inconsistent about both.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace(" ", " ").replace("’", "'")
    return text.casefold()


def unwrap(text: str) -> str:
    """Join lines broken by hard wrapping, preserving genuine line structure.

    A newline is treated as a soft wrap when the line before it does not end in a terminator, a
    colon, or a comma-free heading, *and* the next line continues rather than starting a new item.
    Anything else — bullets, numbered items, headings — keeps its break.
    """
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        if not out or not out[-1]:
            out.append(stripped)
            continue

        prev = out[-1]
        starts_new_item = bool(_BULLET.match(line)) or bool(_HEADING.match(line))
        prev_is_complete = prev.endswith((".", ";", ":", "!", "?"))
        # A wrapped continuation begins mid-sentence: no bullet, no heading, and the previous line
        # stopped without closing punctuation.
        if not starts_new_item and not prev_is_complete:
            out[-1] = f"{prev} {stripped}"
        else:
            out.append(stripped)
    return "\n".join(out)


@dataclass(frozen=True)
class Clause:
    """One assertable unit of a report.

    ``text`` is the clause itself and ``heading`` the section it fell under, if any. Cue matching
    runs against ``subject``, which concatenates the two — a clause of "Normal." means nothing until
    its heading says what is normal.
    """

    text: str
    heading: str | None
    line: int

    @property
    def subject(self) -> str:
        return f"{self.heading}. {self.text}" if self.heading else self.text


def segment(text: str) -> list[Clause]:
    """Split a report into clauses, attaching each to the heading that governs it.

    Runs on raw text; callers fold the clause text themselves so that both the folded and original
    forms remain available for debugging.
    """
    clauses: list[Clause] = []
    heading: str | None = None

    for lineno, line in enumerate(unwrap(text).split("\n")):
        if not line.strip():
            continue

        bare = _HEADING.match(line)
        if bare:
            heading = bare.group(1).strip()
            continue

        body = line
        inline = _INLINE_HEADING.match(line)
        if inline:
            # "Ligament croisé postérieur : Normal." — the subject governs only this line, so it is
            # attached here and does not become the running heading for what follows.
            local_heading, body = inline.group(1).strip(), inline.group(2).strip()
        else:
            local_heading = heading
            body = _BULLET.sub("", body)

        for part in _CLAUSE_BREAK.split(body):
            part = part.strip()
            if len(part) > 1:
                clauses.append(Clause(text=part, heading=local_heading, line=lineno))

    return clauses


def detect_language(text: str) -> str:
    """Best-effort ISO-639-1 code for a report, or ``"unk"``.

    Used for *auditing* coverage per language, not for routing cue matching. Routing would mean
    committing to a guess before any evidence is read, and the cheap guess fails badly here — cue
    vocabulary is shared enough across the Latin-script languages that pooling costs little, while a
    misrouted report loses every cue at once.
    """
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return "unk"
