"""
Text processing module.

Handles everything related to basic text processing:
normalization, tokenization (with character offsets) and word counting.
"""

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from typing import List, Union

import spacy
from spacy.tokens import Doc


# updated when rules of text processing change
# directly affects every numeric metric in the dataset
TEXT_PIPELINE_VERSION = "1"

DEFAULT_MODEL = "ru_core_news_sm"

# a single letter (no digits, no underscore)
LETTER = r"[^\W\d_]"

# whole token made of letters/digits joined by hyphens: как-то, по-русски, 1990-е
HYPHENATED_WORD = re.compile(r"[^\W_]+(?:-[^\W_]+)+$")

_INVISIBLE = re.compile("[\u00ad\u200b\u200c\u200d\u2060\ufeff]")
_SPECIAL_SPACES = re.compile("[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")

# "вырос-\nли": a word hyphenated at a line break
_LINEBREAK_HYPHEN = re.compile(rf"({LETTER}+)-[ \t]*\n[ \t]*({LETTER}+)")

# a single line break before a dialogue dash is a paragraph break:
# "...очки.\n– Да уж верно!"
_DIALOGUE_BREAK = re.compile(r"(?<=[.!?…»”\":])[ \t]*\n[ \t]*(?=[–—]|-[ \t])")

# blank line(s) = paragraph break
_PARAGRAPH_SPLIT = re.compile(r"\n(?:[ \t]*\n)+")

# straight double quote at the start of a paragraph / after a space or an opening bracket
_OPENING_QUOTE = re.compile(r'(?:(?<=\s)|(?<=[(\[—])|^)"', re.MULTILINE)

# spaced-out words ("с т р а н н о"): letters separated by exactly one space.
# Wider gaps (2+ spaces) are word boundaries, so this must run before spaces are collapsed.
# long form: 4+ letters in a row
_SCATTERED_LONG = re.compile(rf"(?<!{LETTER})(?:{LETTER} ){{3,}}{LETTER}(?!{LETTER})")
# short form (2-3 letters, e.g. "н а"): only when flanked by 2+ spaces on both sides,
# otherwise it would glue ordinary one-letter words ("а и", "я у")
_SCATTERED_SHORT = re.compile(rf"(?<=  )(?:{LETTER} ){{1,2}}{LETTER}(?=  )")


@dataclass(frozen=True)
class Token:
    """A token with its position in the (normalized) text it was taken from."""
    text: str
    start: int
    end: int
    is_word: bool  # contains at least one letter or digit (punctuation is not a word)


@lru_cache(maxsize=None)
def _morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer()


def _join_linebreak_hyphen(match: re.Match) -> str:
    """
    Decides what to do with 'left-\\nright'.
    Usually it is a hyphenation of a single word ('вырос-\\nли' -> 'выросли'),
    but sometimes the hyphen is real ('квадратно-\\nбетонных'). The joined form
    is accepted only if the pymorphy3 dictionary knows it; capitalised
    continuations ('Санкт-\\nПетербург') always keep the hyphen.
    """
    left, right = match.group(1), match.group(2)
    if right[0].isupper():
        return f"{left}-{right}"
    joined = left + right
    if _morph().word_is_known(joined.lower()):
        return joined
    return f"{left}-{right}"


def normalize_text(text: str) -> str:
    """
    Normalizes text (the function is idempotent):
        - unicode NFC, removes invisible characters, unifies special spaces;
        - joins words hyphenated at line breaks;
        - paragraphs: a blank line or a line break before a dialogue dash
          starts a new paragraph (separated by '\\n\\n'); any other line break
          is a hard wrap and becomes a space;
        - unifies quotes to «», ellipsis to '…';
        - unifies dashes to '—' (en dash between digits is kept as a range),
          dialogue dash is always followed by one space;
        - removes spaces before punctuation, squashes scattered words.

    Scattered words ('с т р а н н о') are glued if they have 4+ letters, or 2-3 letters
    surrounded by 2+ spaces ('брал   н а   к а р а у л' -> 'брал на караул').
    Known limitation: if the source separates words by a single space too
    ('и с т р а н н о'), the preceding one-letter word is glued as well.
    Not supported: „…“ nested quotation style.
    """
    if not text or not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE.sub("", text)
    text = _SPECIAL_SPACES.sub(" ", text)
    text = text.replace("\u2010", "-").replace("\u2011", "-")

    # words hyphenated at line breaks (before line breaks are removed)
    text = _LINEBREAK_HYPHEN.sub(_join_linebreak_hyphen, text)

    # scattered words (before spaces are collapsed: wide gaps mark word boundaries)
    text = _SCATTERED_LONG.sub(lambda m: m.group(0).replace(" ", ""), text)
    text = _SCATTERED_SHORT.sub(lambda m: m.group(0).replace(" ", ""), text)

    # paragraphs
    text = _DIALOGUE_BREAK.sub("\n\n", text)
    paragraphs = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = " ".join(paragraph.split())  # hard wraps and repeated spaces -> one space
        if paragraph:
            paragraphs.append(paragraph)
    text = "\n\n".join(paragraphs)

    # ellipsis and quotes
    text = text.replace("...", "…")
    if "„" in text:  # „open“ style: “ closes
        text = text.replace("“", "»")
    else:  # “open” style: “ opens
        text = text.replace("“", "«")
    text = text.replace("„", "«").replace("‟", "«").replace("”", "»")
    text = _OPENING_QUOTE.sub("«", text).replace('"', "»")

    # dashes
    text = re.sub(r"(?<!\d)–|–(?!\d)", "—", text)  # en dash -> em dash, except ranges like 1990–2000
    text = re.sub(r"(?<= )-(?= )", "—", text)  # ' - ' -> ' — '
    text = re.sub(rf"(?<={LETTER})—(?={LETTER})", " — ", text)  # 'слово—слово' -> 'слово — слово'
    text = re.sub(r"(?m)^[–—-][ ]*(?=[^\s\d])", "— ", text)  # dialogue dash: '— ' + text

    # spaces before punctuation
    text = re.sub(r"(?<=[\w»)]) +(?=[,;:!?])", "", text)
    text = re.sub(r"(?<=[\w»)]) +(?=\.(?!\.))", "", text)

    return text.strip()


@lru_cache(maxsize=None)
def get_nlp(model: str = DEFAULT_MODEL):
    nlp = spacy.load(model)
    previous = nlp.tokenizer.token_match

    def token_match(text):
        return HYPHENATED_WORD.match(text) or (previous(text) if previous else None)

    nlp.tokenizer.token_match = token_match
    return nlp


def _is_word(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def tokenize(normalized_text: str, model: str = DEFAULT_MODEL) -> list[Token]:
    """
    Returns tokens of an already normalized text with character offsets
    (normalized_text[token.start:token.end] == token.text).
    Whitespace tokens are skipped; punctuation tokens are kept with is_word=False.
    A word with a hyphen ('как-то', 'по-русски') is a single token.
    """
    doc = get_nlp(model).make_doc(normalized_text)
    return [
        Token(token.text, token.idx, token.idx + len(token.text), _is_word(token.text))
        for token in doc
        if not token.is_space
    ]


def count_words(text: str, model: str = DEFAULT_MODEL) -> int:
    """
    The only way to count words in the project.
    Normalizes the text first (normalization is idempotent, so it does not matter
    whether the text has been normalized before). Numbers count as words,
    punctuation does not, a hyphenated word is one word.
    """
    return sum(token.is_word for token in tokenize(normalize_text(text), model))


def content_lemmas(doc: Doc) -> list[str]:
    """
    Filters lemmas, returns only meaningful lemmas from a spacy Doc object.
    """
    return [token.lemma_ for token in doc if not token.is_stop]


def sentences():
    """
    Divides text into sentences.
    """
    pass


