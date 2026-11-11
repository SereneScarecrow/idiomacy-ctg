"""
Alignment of idiom expressions with the text they come from.

The same code is used for the human annotation (dataset) and for the expressions
returned by the LLM judge, so both are measured in exactly the same way.

All offsets refer to the NORMALIZED text (see normalize_text).
"""

import re
from collections import Counter
from dataclasses import dataclass

from idiomacy.metrics.text import normalize_text, tokenize

# [word]: a clarification added by the annotator, e.g. 'тычет им [недостатком] в глаза'
_BRACKET_GROUP = re.compile(r"\[[^\]]*\]")


@dataclass(frozen=True)
class IdiomMatch:
    idiom: str          # normalized idiom (as annotated)
    start: int | None   # character offsets of the occurrence this entry is assigned to
    end: int | None     # (None if the idiom was not found)
    method: str         # exact | casefold | bracket | not_found
    n_occurrences: int  # how many times the idiom occurs in the text
    occurrence: int     # 0-based: how many identical entries were listed before this one

    @property
    def found(self) -> bool:
        return self.start is not None


def parse_idioms(raw) -> list[str]:
    """
    Splits the annotation cell (one idiom per line) into a list of normalized idioms.
    Empty cells (NaN) give an empty list.
    """
    if not isinstance(raw, str):
        return []
    idioms = (normalize_text(line) for line in raw.split("\n"))
    return [idiom for idiom in idioms if idiom]


def _occurrences(text: str, needle: str):
    """All non-overlapping occurrences: (method, [(start, end), ...]) or None.
    Exact (case-sensitive) matches are preferred, case-insensitive ones are the fallback."""
    if not needle:
        return None
    spans, pos = [], text.find(needle)
    while pos >= 0:
        spans.append((pos, pos + len(needle)))
        pos = text.find(needle, pos + len(needle))
    if spans:
        return "exact", spans
    spans = [m.span() for m in re.finditer(re.escape(needle), text, flags=re.IGNORECASE)]
    return ("casefold", spans) if spans else None


def _variants(idiom: str):
    """Strings to look for: the idiom itself, then versions without the annotator's [brackets]."""
    yield idiom, None
    if "[" in idiom:
        yield normalize_text(_BRACKET_GROUP.sub("", idiom)), "bracket"  # bracketed word is not in the text
        yield idiom.replace("[", "").replace("]", ""), "bracket"  # bracketed word is in the text


def align_idioms(text: str, idioms: list[str]) -> list[IdiomMatch]:
    """
    Finds each idiom in the text. `text` must be normalized (offsets depend on it);
    idioms are normalized here, so raw annotation lines are fine.

    If the same idiom is listed several times, the k-th entry is assigned to the k-th
    occurrence in the text, so every listed repetition covers its own words.
    Idioms that cannot be found are returned with method='not_found'.
    """
    if text != normalize_text(text):
        raise ValueError("text must be normalized before aligning idioms")

    listed_before = Counter()
    matches = []
    for idiom in (normalize_text(i) for i in idioms):
        if not idiom:
            continue
        k = listed_before[idiom.casefold()]
        listed_before[idiom.casefold()] += 1

        match = None
        for variant, tag in _variants(idiom):
            found = _occurrences(text, variant)
            if found:
                method, spans = found
                # more entries than occurrences: the extra entries reuse the last occurrence
                start, end = spans[min(k, len(spans) - 1)]
                match = IdiomMatch(idiom, start, end, tag or method, len(spans), k)
                break
        matches.append(match or IdiomMatch(idiom, None, None, "not_found", 0, k))
    return matches


def occurrence_mismatches(matches: list[IdiomMatch]) -> list[tuple[str, int, int]]:
    """
    (idiom, listed, occurs) for found idioms that are listed a different number of times
    than they occur in the text. listed < occurs: the remaining occurrences are not counted;
    listed > occurs: the extra entries cover nothing new.
    """
    listed = Counter(m.idiom.casefold() for m in matches if m.found)
    occurs = {m.idiom.casefold(): (m.idiom, m.n_occurrences) for m in matches if m.found}
    return [(idiom, listed[key], n) for key, (idiom, n) in occurs.items() if listed[key] != n]


def covered_tokens(text: str, matches: list[IdiomMatch]) -> set[tuple[int, int]]:
    """
    (start, end) of every word of the text that overlaps at least one found idiom.
    A set, so overlapping idioms never count a word twice. Two such sets (human vs judge)
    over the same text can be intersected directly.
    """
    spans = [(m.start, m.end) for m in matches if m.found]
    return {
        (token.start, token.end)
        for token in tokenize(text)
        if token.is_word and any(token.start < end and token.end > start for start, end in spans)
    }


def covered_words(text: str, matches: list[IdiomMatch]) -> int:
    """Number of words of the text covered by idioms (the basis of len_idioms)."""
    return len(covered_tokens(text, matches))