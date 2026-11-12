"""
Statistical metrics for figurative text analysis.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union
from collections import Counter

from spacy.tokens import Doc
from wordfreq import zipf_frequency

from idiomaticity.metrics.text import get_content_tokens


@lru_cache(maxsize=None)
def _load_abstractness_dict(dict_path: Union[str, Path]) -> dict:
    with open(dict_path, encoding='utf-8') as f:
        return json.load(f)


def calculate_mean_abstractness(doc: Doc, dict_path: Union[str, Path]) -> Optional[float]:
    """
    Calculates the mean abstractness score using an already parsed spacy Doc.
    """
    dict_path = Path(dict_path)
    if not dict_path.exists():
        raise FileNotFoundError(f"Dictionary not found at {dict_path}")
        
    abstractness_dict = _load_abstractness_dict(dict_path)

    lemmas = [token.lemma_ for token in get_content_tokens(doc)]
    if not lemmas:
        return 0.0

    values = []
    for l in lemmas:
        if l in abstractness_dict:
            values.append(abstractness_dict[l])
        
    return sum(values) / len(values) if values else None


def calculate_mean_frequency_with_coverage(doc: Doc):
    """
    Calculates the mean Zipf-frequency for unique and all lemmas, as well as coverage.
    """
    lemmas = [token.lemma_ for token in get_content_tokens(doc)]

    if not lemmas:
        return {
            'mean_frequency_with_repeats': 0.0,
            'mean_frequency_unique': 0.0,
            'coverage': 0.0
        }
        
    lemma_counts = Counter(lemmas)
    total_tokens = len(lemmas)

    freqs_with_repeats = []
    freqs_unique = []
    known_tokens_count = 0
    
    for word, count in lemma_counts.items():
        freq = zipf_frequency(word, 'ru')
        if freq > 0:
            freqs_unique.append(freq)
            freqs_with_repeats.extend([freq] * count)
            known_tokens_count += count
    
    coverage = known_tokens_count / total_tokens if total_tokens else 0.0
    
    return {
        'mean_frequency_with_repeats': sum(freqs_with_repeats) / len(freqs_with_repeats) if freqs_with_repeats else 0.0,
        'mean_frequency_unique': sum(freqs_unique) / len(freqs_unique) if freqs_unique else 0.0,
        'coverage': coverage
    }