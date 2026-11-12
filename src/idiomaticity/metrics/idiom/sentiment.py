"""
Sentiment (negative / neutral / positive) of a text with a fine-tuned Russian BERT.

The model is a BertForSequenceClassification checkpoint: the classification head is already
trained and stored in it, nothing has to be attached. AutoModelForSequenceClassification
loads the encoder together with the head; the label names are read from the model config.

BERT reads at most 512 tokens, the texts are longer, so a text is cut into windows (at
sentence boundaries, see bertscore.py), every window is classified, and the class
probabilities of the windows are averaged with the number of tokens as weights. The label of
the text is the class with the highest mean probability.

Texts are normalized with normalize_text first (the project-wide text processing).
"""

from collections import Counter
from dataclasses import dataclass

from idiomaticity.metrics.quality.bertscore import pack_windows, split_sentences
from idiomaticity.metrics.text import normalize_text

DEFAULT_MODEL = "blanchefort/rubert-base-cased-sentiment"


@dataclass(frozen=True)
class SentimentResult:
    label: str                         # negative | neutral | positive (as named by the model)
    probabilities: dict[str, float]    # mean over the windows, sums to 1
    n_windows: int                     # 1 = the text fitted into one window


def aggregate_windows(probabilities: list[list[float]], lengths: list[int], labels: list[str]) -> SentimentResult:
    """Mean of the window probabilities weighted by the window lengths (in tokens)."""
    total = sum(lengths)
    mean = [
        sum(p[k] * n for p, n in zip(probabilities, lengths)) / total
        for k in range(len(labels))
    ]
    best = max(range(len(labels)), key=mean.__getitem__)
    return SentimentResult(labels[best], dict(zip(labels, mean)), len(probabilities))


def class_shares(results: list[SentimentResult]) -> dict[str, float]:
    """Share of the texts per label (the 'negative/neutral/positive' triple of the results table)."""
    counts = Counter(r.label for r in results)
    return {label: counts[label] / len(results) for label in sorted(counts)}


class SentimentClassifier:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int | None = None,
        batch_size: int = 16,
        device: str | None = None,
    ):
        # heavy dependencies are imported only when the classifier is created
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(self.device)

        # label names come from the model, in the order of its outputs
        id2label = self.model.config.id2label
        self.labels = [str(id2label[i]).lower() for i in range(self.model.config.num_labels)]

        if max_length is None:
            declared = self.tokenizer.model_max_length
            max_length = declared if declared < 100_000 else self.model.config.max_position_embeddings
        self.window = max_length - len(self.tokenizer.build_inputs_with_special_tokens([]))

    def describe(self) -> dict:
        """Settings of the metric for the run manifest."""
        return {
            "metric": "sentiment-windowed",
            "model": self.model_name,
            "labels": self.labels,
            "window_tokens": self.window,
        }

    def _windows(self, text: str) -> list[list[int]]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        ids = self.tokenizer(sentences, add_special_tokens=False)["input_ids"]
        return pack_windows(ids, self.window)

    def classify(self, texts: list[str]) -> list[SentimentResult]:
        """Classifies the texts; the windows of all texts are batched together."""
        torch = self._torch

        owners, windows = [], []   # window k belongs to texts[owners[k]]
        for index, text in enumerate(texts):
            text_windows = self._windows(normalize_text(text))
            if not text_windows:
                raise ValueError(f"Text {index} has no tokens")
            owners += [index] * len(text_windows)
            windows += text_windows

        probabilities = [None] * len(windows)
        order = sorted(range(len(windows)), key=lambda k: len(windows[k]))  # similar lengths per batch
        pad_id = self.tokenizer.pad_token_id or 0

        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                indices = order[start:start + self.batch_size]
                batch = [self.tokenizer.build_inputs_with_special_tokens(windows[k]) for k in indices]
                width = max(len(ids) for ids in batch)
                input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
                attention = torch.zeros((len(batch), width), dtype=torch.long)
                for row, ids in enumerate(batch):
                    input_ids[row, :len(ids)] = torch.tensor(ids)
                    attention[row, :len(ids)] = 1
                logits = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention.to(self.device),
                ).logits
                for row, k in enumerate(indices):
                    probabilities[k] = torch.softmax(logits[row].float(), dim=-1).cpu().tolist()

        results = []
        for index in range(len(texts)):
            mine = [k for k, owner in enumerate(owners) if owner == index]
            results.append(aggregate_windows(
                [probabilities[k] for k in mine],
                [len(windows[k]) for k in mine],
                self.labels,
            ))
        return results

    def classify_one(self, text: str) -> SentimentResult:
        return self.classify([text])[0]