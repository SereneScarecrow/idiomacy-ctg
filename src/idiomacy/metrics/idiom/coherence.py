"""
Measures coherence.
"""
import numpy as np
from gensim.models import KeyedVectors
from gensim.downloader import load

from spacy.tokens import Doc

from idiomacy.metrics.text import get_content_tokens


def load_model(path):
    return KeyedVectors.load_word2vec_format(path, binary=path.endswith('.bin'))

def cosine(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def mean_sim(lemmas, model):
    # words уже должны быть леммами
    vecs = [model[l] for l in lemmas if l in model]
    if len(vecs) < 2:
        return 0.0
    sims = [cosine(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i+1, len(vecs))]
    return np.mean(sims)

def calculate_linear_semantic_coherence():
    """
    """
    return

def calculate_distributional_semantic_similarity(doc, model, window=5):
    """
    """
    content_tokens = [(token.lemma_, token.pos_) for token in get_content_tokens(doc)]

    if not content_tokens:
        return 0.0

    verb_idx = []
    for i, (lemma, pos) in enumerate(content_tokens):
        if pos in ('VERB', 'INFN'):
            verb_idx.append(i)
    
    if not verb_idx:
        return 0.0
