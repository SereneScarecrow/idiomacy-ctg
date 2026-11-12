"""
Perplexity of a text and surprisal of its idiom spans, from one pass of a causal language model.

One forward pass gives the surprisal (negative log-probability, in nats) of every token:
  - perplexity of the text = exp(mean surprisal over all tokens);
  - surprisal of an idiom = mean surprisal of the tokens that overlap its span;
  - surprisal of all idioms of the text = the same over the union of their tokens
    (a token is never counted twice), compared with the rest of the text.

Idioms are given as strings (the human annotation or the judge's list) and are found in the
text by align_idioms (spans.py), so human and judge idioms are measured in the same way.

Details that matter:
  - a start token (BOS, or <|endoftext|> for models without BOS) is put in front of the
    text, so that the first real token is scored too;
  - texts longer than the context are scored with a sliding window: each token gets as much
    left context as possible and is scored exactly once;
  - the loss is computed in float32 even if the model runs in bfloat16;
  - a fast tokenizer is required (character offsets are used to map spans to tokens).

Texts are normalized with normalize_text first (the project-wide text processing).
"""

import math
from dataclasses import dataclass

from idiomaticity.metrics.spans import IdiomMatch, align_idioms
from idiomaticity.metrics.text import normalize_text

DEFAULT_MODEL = "Qwen/Qwen2-1.5B"
DEFAULT_MAX_LENGTH = 4096   # tokens per forward pass (capped by the model's own limit)


@dataclass(frozen=True)
class IdiomSurprisal:
    idiom: str
    start: int | None            # character offsets in the normalized text (None: not found)
    end: int | None
    method: str                  # how the idiom was found: exact | casefold | bracket | not_found
    n_tokens: int
    surprisal: float | None      # mean surprisal of its tokens, nats (None: not found)


@dataclass(frozen=True)
class PerplexityResult:
    perplexity: float                    # exp(mean surprisal): the standard definition
    mean_token_perplexity: float         # mean of exp(surprisal) over tokens (mean of per-token PPL)
    mean_surprisal: float                # mean surprisal over all tokens, nats
    n_tokens: int
    n_windows: int                       # 1 = the text fitted into the context at once
    idiom_surprisal: float | None        # tokens covered by the idioms (None: no idioms given / found)
    other_surprisal: float | None        # all the other tokens
    n_idiom_tokens: int
    idioms: tuple[IdiomSurprisal, ...]


# ----------------------------------------------------------------------------
# pure functions (no model): window plan and aggregation
# ----------------------------------------------------------------------------

def plan_windows(n_tokens: int, max_length: int, context: int) -> list[tuple[int, int, int]]:
    """
    Sliding-window plan for a sequence of n_tokens (start token included).
    Returns (begin, end, first_scored): the window is seq[begin:end] and the tokens at positions
    first_scored .. end-1 are scored in it. Position 0 (the start token) is never scored; every
    other position is scored exactly once; the next window re-reads `context` tokens as left context.
    """
    if not 0 < context < max_length:
        raise ValueError("need 0 < context < max_length")
    windows = []
    begin, scored = 0, 1
    while scored < n_tokens:
        end = min(begin + max_length, n_tokens)
        windows.append((begin, end, scored))
        scored = end
        begin = end - context
    return windows


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize(
    surprisals: list[float],
    offsets: list[tuple[int, int]],
    matches: list[IdiomMatch] | None,
    n_windows: int,
) -> PerplexityResult:
    """
    Turns per-token surprisals (and the character offsets of the tokens) into the result.
    `matches` come from align_idioms; None means that no idioms were given.
    """
    if not surprisals:
        raise ValueError("The text has no tokens")
    mean_surprisal = sum(surprisals) / len(surprisals)

    idioms = []
    covered: set[int] = set()
    for match in matches or []:
        if not match.found:
            idioms.append(IdiomSurprisal(match.idiom, None, None, match.method, 0, None))
            continue
        tokens = [i for i, (s, e) in enumerate(offsets) if e > s and s < match.end and e > match.start]
        covered.update(tokens)
        idioms.append(IdiomSurprisal(
            match.idiom, match.start, match.end, match.method, len(tokens),
            _mean(surprisals[i] for i in tokens),
        ))

    return PerplexityResult(
        perplexity=math.exp(mean_surprisal),
        mean_token_perplexity=sum(math.exp(x) for x in surprisals) / len(surprisals),
        mean_surprisal=mean_surprisal,
        n_tokens=len(surprisals),
        n_windows=n_windows,
        idiom_surprisal=_mean(surprisals[i] for i in sorted(covered)),
        other_surprisal=_mean(s for i, s in enumerate(surprisals) if i not in covered) if matches is not None else None,
        n_idiom_tokens=len(covered),
        idioms=tuple(idioms),
    )


# ----------------------------------------------------------------------------
# scorer
# ----------------------------------------------------------------------------

class PerplexityScorer:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        """dtype: 'float32', 'bfloat16', 'float16'; default: bfloat16 on cuda, float32 on cpu."""
        # heavy dependencies are imported only when the scorer is created
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or ("bfloat16" if self.device.startswith("cuda") else "float32")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.is_fast:
            raise ValueError(f"{model_name}: a fast tokenizer is needed (character offsets)")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, self.dtype))
        self.model.eval().to(self.device)

        self.bos_id = self.tokenizer.bos_token_id
        if self.bos_id is None:
            self.bos_id = self.tokenizer.eos_token_id
        if self.bos_id is None:
            raise ValueError(f"{model_name}: no BOS/EOS token to start the text with")

        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        self.max_length = min(max_length, model_limit) if model_limit else max_length
        self.context = self.max_length // 2   # left context re-read by the next window

    def describe(self) -> dict:
        """Settings of the metric for the run manifest."""
        return {
            "metric": "perplexity-surprisal",
            "model": self.model_name,
            "dtype": self.dtype,
            "max_length": self.max_length,
            "window_context": self.context,
            "unit": "nats",
        }

    def token_surprisals(self, text: str) -> tuple[list[float], list[tuple[int, int]], int]:
        """
        Surprisal (nats) and character offsets of every token of a normalized text,
        and the number of windows used. The start token is not part of the result.
        """
        torch = self._torch
        encoding = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoding["input_ids"]
        offsets = [tuple(offset) for offset in encoding["offset_mapping"]]
        if not ids:
            raise ValueError("The text has no tokens")

        sequence = [self.bos_id] + ids
        surprisals = [0.0] * len(ids)   # surprisals[j] belongs to sequence[j + 1]
        windows = plan_windows(len(sequence), self.max_length, self.context)

        with torch.inference_mode():
            for begin, end, first in windows:
                window = torch.tensor([sequence[begin:end]], device=self.device)
                logits = self.model(input_ids=window).logits[0]
                predictions = logits[first - begin - 1:end - begin - 1].float()   # predict window[first-begin:]
                targets = window[0, first - begin:end - begin]
                loss = torch.nn.functional.cross_entropy(predictions, targets, reduction="none")
                surprisals[first - 1:end - 1] = loss.tolist()
        return surprisals, offsets, len(windows)

    def score(self, text: str, idioms: list[str] | None = None) -> PerplexityResult:
        """
        Perplexity of the text and, if `idioms` is given, the surprisal of those idioms
        (raw annotation lines or the judge's expressions: they are normalized and aligned here).
        """
        text = normalize_text(text)
        surprisals, offsets, n_windows = self.token_surprisals(text)
        matches = align_idioms(text, idioms) if idioms is not None else None
        return summarize(surprisals, offsets, matches, n_windows)