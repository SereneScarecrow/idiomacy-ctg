"""
Measures distributional semantic similarity.
"""
import numpy as np
from gensim.models import KeyedVectors
from spacy.tokens import Doc

from idiomaticity.metrics.text import get_content_tokens


def load_model(path: str) -> KeyedVectors:
    return KeyedVectors.load_word2vec_format(path, binary=path.endswith('.bin'))


def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1 is None or v2 is None:
        return 0.0
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def _verb_to_context_sim(verb_key: str, context_keys: list[str], model: KeyedVectors) -> float:
    """
    Measures mean cosine simularity between the verb and its context window.
    """
    if verb_key not in model:
        return 0.0

    verb_vec = model[verb_key]
    context_vecs = [model[k] for k in context_keys if k in model]

    if not context_vecs:
        return 0.0

    sims = [_cosine(verb_vec, ctx_vec) for ctx_vec in context_vecs]
    return float(np.mean(sims))


def calculate_distributional_semantic_similarity(
    doc: Doc, 
    model: KeyedVectors, 
    window: int = 5,
    use_pos_tags: bool = True
) -> float:
    """
    Calculates distributional semantic similarity 
    between verbs and their context window (Badrylova, 2019)
    """
    content_tokens = [(token.lemma_.lower(), token.pos_) for token in get_content_tokens(doc)]

    if not content_tokens:
        return 0.0

    if use_pos_tags:
        keys = [f"{lemma}_{pos}" for lemma, pos in content_tokens]
    else:
        keys = [lemma for lemma, _ in content_tokens]

    verb_indices = [i for i, (_, pos) in enumerate(content_tokens) if pos == 'VERB']

    if not verb_indices:
        return 0.0

    verb_similarities = []

    for idx in verb_indices:
        left_bound = max(0, idx - window)
        right_bound = min(len(keys), idx + window + 1)

        context_keys = [
            keys[j] for j in range(left_bound, right_bound) if j != idx
        ]

        sim = _verb_to_context_sim(keys[idx], context_keys, model)
        verb_similarities.append(sim)

    return float(np.mean(verb_similarities))


