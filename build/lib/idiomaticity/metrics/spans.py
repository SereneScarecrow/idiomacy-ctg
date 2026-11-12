"""
Alignment of idiom expressions with the text they come from.

The same code is used for the human annotation (dataset) and for the expressions
returned by the LLM judge, so both are measured in exactly the same way.

All offsets refer to the NORMALIZED text (see normalize_text).
"""

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from idiomaticity.metrics.text import get_nlp, normalize_text, tokenize

# [word]: a clarification added by the annotator, e.g. 'тычет им [недостатком] в глаза'
_BRACKET_GROUP = re.compile(r"\[[^\]]*\]")


@dataclass(frozen=True)
class IdiomMatch:
    idiom: str          # normalized idiom (as annotated)
    start: int | None   # character offsets of the occurrence this entry is assigned to
    end: int | None     # (None if the idiom was not found)
    method: str         # exact | casefold | bracket | lemma | not_found
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


# an idiom must not run over a sentence border
_SENTENCE_END = re.compile(r"[.!?…\n]")


def _is_word(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _lemma(token) -> str:
    return (token.lemma_ or token.text).lower().replace("ё", "е")


@lru_cache(maxsize=256)
def _text_lemmas(text: str) -> tuple[tuple[str, int, int], ...]:
    """(lemma, start, end) of every word of the text. Cached: the same text is aligned several times."""
    return tuple((_lemma(t), t.idx, t.idx + len(t.text)) for t in get_nlp()(text) if _is_word(t.text))


def _lemma_occurrences(text: str, needle: str):
    """
    All non-overlapping occurrences of `needle` found by lemmas: ('lemma', [(start, end), ...]) or None.
    Fallback for idioms that the judge gives in another word form than the text has
    ('королеву игрового трона' is found in 'королевой игрового трона').
    The words must follow each other without a sentence border between them.
    """
    target = [_lemma(t) for t in get_nlp()(needle) if _is_word(t.text)]
    if not target:
        return None
    words = _text_lemmas(text)
    spans, i, n = [], 0, len(target)
    while i + n <= len(words):
        window = words[i:i + n]
        same = [w[0] for w in window] == target
        if same and not any(_SENTENCE_END.search(text[a[2]:b[1]]) for a, b in zip(window, window[1:])):
            spans.append((window[0][1], window[-1][2]))
            i += n
        else:
            i += 1
    return ("lemma", spans) if spans else None


def align_idioms(text: str, idioms: list[str]) -> list[IdiomMatch]:
    """
    Finds each idiom in the text. `text` must be normalized (offsets depend on it);
    idioms are normalized here, so raw annotation lines are fine.

    If the same idiom is listed several times, the k-th entry is assigned to the k-th
    occurrence in the text, so every listed repetition covers its own words.
    Search order: the exact string, the case-insensitive string, then the lemmas of the words
    (the idiom is given in another word form). Idioms that cannot be found are returned with
    method='not_found'.
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
        for finder in (_occurrences, _lemma_occurrences):   # surface forms first, then lemmas
            for variant, tag in _variants(idiom):
                found = finder(text, variant)
                if found:
                    method, spans = found
                    # more entries than occurrences: the extra entries reuse the last occurrence
                    start, end = spans[min(k, len(spans) - 1)]
                    match = IdiomMatch(idiom, start, end, method if method == "lemma" else (tag or method),
                                       len(spans), k)
                    break
            if match:
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