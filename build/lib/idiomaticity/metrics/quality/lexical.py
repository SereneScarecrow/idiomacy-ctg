"""
METEOR + WMD
"""

import nltk
from nltk.translate.meteor_score import meteor_score
import gensim.downloader as api

from typing import List

nltk.download('wordnet')
nltk.download('omw-1.4')

MODEL = api.load("word2vec-ruscorpora-300")
MODEL.fill_norms()


def calculate_meteor(reference: List[str], candidate: List[str]) -> float:
    """Calculates METEOR score""" # normalization needed
    return meteor_score(reference, candidate)


def calculate_wmd(reference: List[str], candidate: List[str]) -> float:
    """Calculates WMD""" # normalization needed
    return MODEL.wmdistance(reference, candidate)
