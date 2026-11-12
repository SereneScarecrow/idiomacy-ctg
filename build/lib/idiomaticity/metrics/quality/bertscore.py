"""
BERTScore for texts that are longer than the context window of the encoder.

BERTScore encodes the candidate and the reference separately and only then matches their
token embeddings, so long texts can be handled without aligning the two texts:

  1. each text is cut into windows that fit the model (at sentence boundaries; a sentence
     longer than a window is cut by tokens);
  2. every window is encoded on its own;
  3. the token embeddings of all windows of a text are pooled into one set;
  4. precision / recall / F1 are computed by greedy matching over the pooled sets,
     exactly as in the original metric (Zhang et al., 2020).

The only difference from encoding the whole text at once is that a token does not see the
context outside its window. Texts that fit into one window give the usual BERTScore.

Differences from the bert_score library (the numbers can differ in the 2nd-3rd decimal):
special tokens ([CLS], [SEP]) are excluded from the matching, no idf weighting,
no baseline rescaling. The layer is the one the library uses for the model
(layer 9 for bert-base-multilingual-cased).

Texts are normalized with normalize_text first (the project-wide text processing).
"""

import re
from collections import OrderedDict
from dataclasses import dataclass

from idiomaticity.metrics.text import normalize_text

DEFAULT_MODEL = "bert-base-multilingual-cased"
# layer taken for the model (the one used by the bert_score library); other models need num_layers
DEFAULT_LAYERS = {"bert-base-multilingual-cased": 9}

# end of a sentence: . ! ? … (maybe followed by a closing quote or bracket), a space,
# and then a capital letter / digit (maybe after an opening quote, bracket or dash)
_SENTENCE_BOUNDARY = re.compile(
    r"(?:(?<=[.!?…])|(?<=[.!?…][»”\")]))\s+(?=[«\"“(—]?\s*[А-ЯЁA-Z\d])"
)


@dataclass(frozen=True)
class BertScoreResult:
    precision: float
    recall: float
    f1: float
    candidate_windows: int   # how many windows the candidate was cut into (1 = fits at once)
    reference_windows: int


def split_sentences(text: str) -> list[str]:
    """
    Rough sentence splitter, enough to choose window borders.
    Paragraphs (line breaks) are always borders.
    """
    sentences = []
    for paragraph in re.split(r"\n+", text):
        sentences.extend(s.strip() for s in _SENTENCE_BOUNDARY.split(paragraph) if s.strip())
    return sentences


def pack_windows(sentences_ids: list[list[int]], window: int) -> list[list[int]]:
    """
    Packs sentences (lists of token ids) into windows of at most `window` tokens, in order.
    A sentence is moved whole to the next window when it does not fit; a sentence that is
    longer than a window is cut into pieces of `window` tokens.
    """
    windows: list[list[int]] = []
    current: list[int] = []
    for ids in sentences_ids:
        for start in range(0, len(ids), window):
            piece = ids[start:start + window]
            if current and len(current) + len(piece) > window:
                windows.append(current)
                current = []
            current.extend(piece)
    if current:
        windows.append(current)
    return windows


class BertScorer:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        num_layers: int | None = None,
        max_length: int | None = None,
        batch_size: int = 16,
        device: str | None = None,
        cache_size: int = 128,
    ):
        # heavy dependencies are imported only when the scorer is created
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        if num_layers is None:
            if model_name not in DEFAULT_LAYERS:
                raise ValueError(f"Set num_layers for {model_name!r} (known: {sorted(DEFAULT_LAYERS)})")
            num_layers = DEFAULT_LAYERS[model_name]

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)

        if not 1 <= num_layers <= self.model.config.num_hidden_layers:
            raise ValueError(f"num_layers must be in 1..{self.model.config.num_hidden_layers}")
        self.num_layers = num_layers

        if max_length is None:
            declared = self.tokenizer.model_max_length
            max_length = declared if declared < 100_000 else self.model.config.max_position_embeddings
        # room for [CLS]/[SEP] (or <s>/</s>)
        self.window = max_length - len(self.tokenizer.build_inputs_with_special_tokens([]))

        self._cache: OrderedDict[str, tuple] = OrderedDict()  # text -> (embeddings, n_windows)
        self._cache_size = cache_size

    def describe(self) -> dict:
        """Settings of the metric for the run manifest."""
        return {
            "metric": "bertscore-windowed",
            "model": self.model_name,
            "num_layers": self.num_layers,
            "window_tokens": self.window,
            "idf": False,
            "baseline_rescaling": False,
        }

    # ------------------------------------------------------------------ encoding

    def _windows(self, text: str) -> list[list[int]]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        ids = self.tokenizer(sentences, add_special_tokens=False)["input_ids"]
        return pack_windows(ids, self.window)

    def _embed_windows(self, windows: list[list[int]]):
        """Normalized embeddings of the real tokens of every window (special tokens dropped)."""
        torch = self._torch
        results = [None] * len(windows)
        order = sorted(range(len(windows)), key=lambda i: len(windows[i]))  # similar lengths per batch
        pad_id = self.tokenizer.pad_token_id or 0

        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                indices = order[start:start + self.batch_size]
                batch = []
                for i in indices:
                    full = self.tokenizer.build_inputs_with_special_tokens(windows[i])
                    special = self.tokenizer.get_special_tokens_mask(windows[i], already_has_special_tokens=False)
                    batch.append((full, [m == 0 for m in special]))

                width = max(len(full) for full, _ in batch)
                input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
                attention = torch.zeros((len(batch), width), dtype=torch.long)
                for row, (full, _) in enumerate(batch):
                    input_ids[row, :len(full)] = torch.tensor(full)
                    attention[row, :len(full)] = 1

                output = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention.to(self.device),
                    output_hidden_states=True,
                )
                hidden = output.hidden_states[self.num_layers]
                for row, i in enumerate(indices):
                    full, keep = batch[row]
                    keep = torch.tensor(keep, dtype=torch.bool, device=hidden.device)
                    embeddings = hidden[row, :len(full)][keep]
                    results[i] = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        return results

    def embed(self, text: str):
        """Pooled token embeddings [n_tokens, hidden] of a whole text, and the number of windows."""
        text = normalize_text(text)
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]

        windows = self._windows(text)
        if not windows:
            raise ValueError("The text has no tokens")
        pooled = self._torch.cat(self._embed_windows(windows), dim=0)
        self._cache[text] = (pooled, len(windows))
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)  # references are reused by many candidates, the rest is dropped
        return self._cache[text]

    # ------------------------------------------------------------------ scoring

    def score_pair(self, candidate: str, reference: str) -> BertScoreResult:
        torch = self._torch
        cand, cand_windows = self.embed(candidate)
        ref, ref_windows = self.embed(reference)

        with torch.inference_mode():
            similarity = cand.to(self.device) @ ref.to(self.device).T   # cosine: vectors are normalized
            precision = similarity.max(dim=1).values.mean().item()      # each candidate token -> best reference token
            recall = similarity.max(dim=0).values.mean().item()         # each reference token -> best candidate token
        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if denominator > 0 else 0.0
        return BertScoreResult(precision, recall, f1, cand_windows, ref_windows)

    def score(self, candidates: list[str], references: list[str]) -> list[BertScoreResult]:
        if len(candidates) != len(references):
            raise ValueError("candidates and references must have the same length")
        return [self.score_pair(c, r) for c, r in zip(candidates, references)]