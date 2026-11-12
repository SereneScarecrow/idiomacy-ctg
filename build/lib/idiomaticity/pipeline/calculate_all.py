"""
Computes all metrics for a file of generations: jsonl -> jsonl.

Every line of the input file is one generation with metadata. For every record:
  1. the judge's idiom list is normalized (the rest of the judge dict is taken as is,
     the schema is assumed to be strict);
  2. the generated text is normalized (normalize_text) and its words are counted (count_words);
  3. the judge's idioms are aligned with the text (spans.py);
  4. the metrics are computed stage by stage - every stage goes over the WHOLE file, so a model
     is loaded once and released after its stage.
The output has the chosen input fields plus the new ones. An error in one record or one metric does
not stop the run: it is written to `metric_errors` and the value of the metric is null.

Usage:
    python -m idiomaticity.metrics.calculate_all in.jsonl out.jsonl \
        --abstractness-dict data/abstractness.json [--w2v-path vectors.bin] [--skip perplexity]

Next to out.jsonl a out.manifest.json is written (metric settings, text pipeline version).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from idiomaticity.metrics.spans import align_idioms, covered_words
from idiomaticity.metrics.text import TEXT_PIPELINE_VERSION, count_words, get_nlp, normalize_text

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(iterable, **_kwargs):
        return iterable

log = logging.getLogger("calculate_all")


# ----------------------------------------------------------------------------
# config and data row
# ----------------------------------------------------------------------------

@dataclass
class Config:
    # field names in the input jsonl
    text_field: str = "text"            # the text to evaluate
    judge_field: str = "answer"                   # the judge's dict
    judge_idioms_key: str = "idiomatic_expressions"   # key of the idiom list inside the judge's dict
    drop_fields: Optional[list[str]] = None
    reference_csv: Optional[Path] = None
    reference_id_field: str = "text_id"
    reference_csv_field: str = "literal_version"
    record_id_field: str = "text_id"
    # resources
    abstractness_dict: Optional[Path] = None
    w2v_path: Optional[str] = None                # lemma_POS vectors; default: the model from lexical.py
    window: int = 5                               # context window of distributional semantic similarity
    device: Optional[str] = None
    ppl_model: Optional[str] = None               # None -> DEFAULT_MODEL of llm_based.py
    ppl_dtype: Optional[str] = None
    # filled in while running
    manifest: dict = field(default_factory=dict)


@dataclass
class Row:
    rec: dict                                     # the input record (never modified)
    text: str = ""                                # normalized generated text
    reference: str = ""
    idioms: Optional[list[str]] = None            # normalized idioms of the judge (None: record has an error)
    doc: Any = None                               # spaCy Doc of the text
    ref_doc: Any = None                           # spaCy Doc of the reference
    out: dict = field(default_factory=dict)       # new fields
    errors: dict = field(default_factory=dict)    # stage -> error message


# ----------------------------------------------------------------------------
# preprocessing of a record: text, words, idioms
# ----------------------------------------------------------------------------

def normalize_idioms(raw: Any) -> list[str]:
    """Normalizes the judge's idioms (a list, or one idiom per line); empty entries are dropped.
    Repeated idioms are kept on purpose: align_idioms assigns them to successive occurrences."""
    lines = raw.split("\n") if isinstance(raw, str) else raw
    cleaned = (normalize_text(line) for line in lines)
    return [idiom for idiom in cleaned if idiom]


def preprocess(rec: dict, cfg: Config) -> Row:
    row = Row(rec)

    text = rec.get(cfg.text_field)
    row.text = normalize_text(text) if isinstance(text, str) else ""
    reference = None
    if cfg.reference_csv is not None:
        rid = rec.get(cfg.record_id_field)
        reference = _load_reference_csv(cfg).get(str(rid))
    row.reference = normalize_text(reference) if isinstance(reference, str) else ""
    if cfg.reference_csv is not None and not row.reference:
        row.errors["reference"] = f"no reference in the csv for {cfg.record_id_field}={rec.get(cfg.record_id_field)!r}"
    row.out["text_norm"] = row.text
    row.out["n_words"] = count_words(row.text) if row.text else 0
    row.out["text_pipeline_version"] = TEXT_PIPELINE_VERSION
    if not row.text:
        row.errors["text"] = f"empty or missing field {cfg.text_field!r}"

    # judge: the schema is strict, only the idiom list is normalized
    try:
        judge = rec[cfg.judge_field]
        row.idioms = normalize_idioms(judge[cfg.judge_idioms_key])
        row.out["judge_norm"] = {**judge, cfg.judge_idioms_key: row.idioms}   # schema keys are kept as they are
    except (KeyError, TypeError) as exc:
        row.errors["judge"] = f"no judge idioms: {type(exc).__name__}: {exc}"

    # idiom spans and the share of the text they cover
    if row.idioms is not None and row.text:
        matches = align_idioms(row.text, row.idioms)
        n_idiom_words = covered_words(row.text, matches)
        row.out["idiom_spans"] = [
            {"idiom": m.idiom, "start": m.start, "end": m.end, "method": m.method,
             "n_occurrences": m.n_occurrences}
            for m in matches
        ]
        row.out["n_idioms"] = len(matches)
        row.out["n_idioms_not_found"] = sum(not m.found for m in matches)
        row.out["n_idiom_words"] = n_idiom_words
        # share of words covered by idioms (len_idiom / len_original in the dataset)
        row.out["idiomatic_density_judge"] = n_idiom_words / row.out["n_words"] if row.out["n_words"] else None
    return row


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _free_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _each(rows: list[Row], stage: str, fn: Callable[[Row], None], need_reference: bool = False) -> None:
    """Applies fn to every row; an error of a row goes to row.errors and the run continues."""
    failed = 0
    for row in tqdm(rows, desc=stage):
        if not row.text or (need_reference and not row.reference):
            continue
        try:
            fn(row)
        except Exception as exc:  # noqa: BLE001 - one bad row must not stop the whole file
            failed += 1
            row.errors[stage] = f"{type(exc).__name__}: {exc}"
    if failed:
        log.warning("%s: %d rows failed", stage, failed)


def ensure_docs(rows: list[Row], reference: bool = False) -> None:
    """Parses texts (and references) with spaCy once; the result is cached in the row."""
    nlp = get_nlp()
    todo = [r for r in rows if r.text and r.doc is None]
    if todo:
        docs = nlp.pipe([r.text for r in todo], batch_size=32)
        for row, doc in zip(todo, tqdm(docs, total=len(todo), desc="spaCy: texts")):
            row.doc = doc
    if reference:
        todo = [r for r in rows if r.reference and r.ref_doc is None]
        if todo:
            docs = nlp.pipe([r.reference for r in todo], batch_size=32)
            for row, doc in zip(todo, tqdm(docs, total=len(todo), desc="spaCy: references")):
                row.ref_doc = doc


def _lemmas(doc) -> list[str]:
    """Tokens for METEOR: lemmas without punctuation."""
    return [t.lemma_.lower() for t in doc if not (t.is_punct or t.is_space)]

_ref_csv_cache: dict = {}

def _load_reference_csv(cfg: Config) -> dict[str, str]:
    key = (str(cfg.reference_csv), cfg.reference_csv_field, cfg.reference_id_field)
    if key not in _ref_csv_cache:
        import csv
        with open(cfg.reference_csv, encoding="utf-8-sig", newline="") as f:
            _ref_csv_cache[key] = {
                str(r[cfg.reference_id_field]): r[cfg.reference_csv_field]
                for r in csv.DictReader(f)
            }
    return _ref_csv_cache[key]


def _w2v_tokens(doc) -> list[str]:
    """Tokens for WMD: lemma_POS (the word2vec-ruscorpora key format)."""
    return [f"{t.lemma_.lower()}_{t.pos_}" for t in doc if not (t.is_punct or t.is_space)]


_w2v_cache: dict = {}


def _get_w2v(cfg: Config):
    """Vectors for distributional similarity: a given file or the model already loaded in lexical.py."""
    if "model" not in _w2v_cache:
        if cfg.w2v_path:
            from idiomaticity.metrics.idiom.coherence import load_model
            _w2v_cache["model"] = load_model(cfg.w2v_path)
        else:
            from idiomaticity.metrics.quality.lexical import MODEL
            _w2v_cache["model"] = MODEL
    return _w2v_cache["model"]


# ----------------------------------------------------------------------------
# stages: every stage goes over the whole file
# ----------------------------------------------------------------------------

def stage_perplexity(rows: list[Row], cfg: Config) -> None:
    """Perplexity of the text + surprisal of the idioms (spans are aligned again inside the scorer)."""
    from idiomaticity.metrics.idiom.llm_based import PerplexityScorer

    kwargs = {k: v for k, v in
              {"model_name": cfg.ppl_model, "device": cfg.device, "dtype": cfg.ppl_dtype}.items() if v}
    scorer = PerplexityScorer(**kwargs)
    cfg.manifest["perplexity"] = scorer.describe()

    def run(row: Row) -> None:
        res = scorer.score(row.text, row.idioms)   # idioms=None -> perplexity only
        row.out.update(
            perplexity=res.perplexity,
            mean_token_perplexity=res.mean_token_perplexity,
            mean_surprisal=res.mean_surprisal,
            n_tokens=res.n_tokens,
            ppl_n_windows=res.n_windows,
            idiom_surprisal=res.idiom_surprisal,
            other_surprisal=res.other_surprisal,
            n_idiom_tokens=res.n_idiom_tokens,
        )
        # surprisal of every idiom goes to the same span list (the order is the same)
        for span, item in zip(row.out.get("idiom_spans", []), res.idioms):
            span["n_tokens"] = item.n_tokens
            span["surprisal"] = item.surprisal

    _each(rows, "perplexity", run)


def stage_bertscore(rows: list[Row], cfg: Config) -> None:
    from idiomaticity.metrics.quality.bertscore import BertScorer

    scorer = BertScorer(device=cfg.device)
    cfg.manifest["bertscore"] = scorer.describe()

    def run(row: Row) -> None:
        res = scorer.score_pair(row.text, row.reference)
        row.out.update(bertscore_precision=res.precision, bertscore_recall=res.recall, bertscore_f1=res.f1)

    _each(rows, "bertscore", run, need_reference=True)


def calculate_bleurt(candidates: list[str], references: list[str]) -> list[Optional[float]]:
    """STUB. TODO: bleurt_pytorch, model lucadiliello/BLEURT-20-D12."""
    return [None] * len(candidates)


def calculate_comet(candidates: list[str], references: list[str]) -> list[Optional[float]]:
    """STUB. TODO: comet, model Unbabel/wmt22-comet-da."""
    return [None] * len(candidates)


def _stub_stage(name: str, fn: Callable[[list[str], list[str]], list[Optional[float]]]):
    def stage(rows: list[Row], cfg: Config) -> None:
        log.warning("%s: stub, the values will be null", name)
        cfg.manifest[name] = {"metric": name, "status": "stub"}
        subset = [r for r in rows if r.text and r.reference]
        for row, value in zip(subset, fn([r.text for r in subset], [r.reference for r in subset])):
            row.out[name] = value
    return stage


def stage_meteor(rows: list[Row], cfg: Config) -> None:
    from idiomaticity.metrics.quality.lexical import calculate_meteor

    ensure_docs(rows, reference=True)
    # nltk's meteor_score expects a LIST of references, so the reference is wrapped in a list
    _each(rows, "meteor",
          lambda row: row.out.__setitem__("meteor", calculate_meteor([_lemmas(row.ref_doc)], _lemmas(row.doc))),
          need_reference=True)


def stage_wmd(rows: list[Row], cfg: Config) -> None:
    from idiomaticity.metrics.quality.lexical import calculate_wmd

    ensure_docs(rows, reference=True)

    def run(row: Row) -> None:
        distance = calculate_wmd(_w2v_tokens(row.ref_doc), _w2v_tokens(row.doc))
        ok = distance is not None and math.isfinite(distance)   # inf: no words found in the vectors
        row.out["wmd"] = distance if ok else None
        row.out["wmd_similarity"] = 1 / (1 + distance) if ok else None   # as in the handout

    _each(rows, "wmd", run, need_reference=True)


def stage_abstractness(rows: list[Row], cfg: Config) -> None:
    from idiomaticity.metrics.idiom.statistical import calculate_mean_abstractness

    if cfg.abstractness_dict is None:
        raise ValueError("--abstractness-dict is not set")
    ensure_docs(rows)
    _each(rows, "abstractness",
          lambda row: row.out.__setitem__("abstractness",
                                          calculate_mean_abstractness(row.doc, cfg.abstractness_dict)))


def stage_frequency(rows: list[Row], cfg: Config) -> None:
    from idiomaticity.metrics.idiom.statistical import calculate_mean_frequency_with_coverage

    ensure_docs(rows)

    def run(row: Row) -> None:
        res = calculate_mean_frequency_with_coverage(row.doc)
        row.out.update(
            freq_with_repeats=res["mean_frequency_with_repeats"],
            freq_unique=res["mean_frequency_unique"],
            freq_coverage=res["coverage"],
        )

    _each(rows, "frequency", run)


def stage_dist_sem(rows: list[Row], cfg: Config) -> None:
    from idiomaticity.metrics.idiom.coherence import calculate_distributional_semantic_similarity

    ensure_docs(rows)
    model = _get_w2v(cfg)
    cfg.manifest["dist_sem"] = {"metric": "distributional-semantic-similarity", "window": cfg.window,
                                "vectors": cfg.w2v_path or "word2vec-ruscorpora-300 (lexical.py)"}
    _each(rows, "dist_sem",
          lambda row: row.out.__setitem__(
              "dist_sem_similarity",
              calculate_distributional_semantic_similarity(row.doc, model, window=cfg.window)))


def stage_sentiment(rows: list[Row], cfg: Config) -> None:
    from idiomaticity.metrics.idiom.sentiment import SentimentClassifier

    classifier = SentimentClassifier(device=cfg.device)
    cfg.manifest["sentiment"] = classifier.describe()
    subset = [r for r in rows if r.text]
    results = classifier.classify([r.text for r in subset])   # windows of all texts are batched together
    for row, res in zip(subset, results):
        row.out["sentiment_label"] = res.label
        row.out["sentiment_n_windows"] = res.n_windows
        for label, probability in res.probabilities.items():
            row.out[f"sentiment_p_{label}"] = probability


# order = order of execution
STAGES: dict[str, Callable[[list[Row], Config], None]] = {
    "perplexity": stage_perplexity,
    "bertscore": stage_bertscore,
    "bleurt": _stub_stage("bleurt", calculate_bleurt),
    "comet": _stub_stage("comet", calculate_comet),
    "meteor": stage_meteor,
    "wmd": stage_wmd,
    "abstractness": stage_abstractness,
    "frequency": stage_frequency,
    "dist_sem": stage_dist_sem,
    "sentiment": stage_sentiment,
}


def run_stage(name: str, rows: list[Row], cfg: Config) -> None:
    log.info("== %s", name)
    try:
        STAGES[name](rows, cfg)
    except Exception as exc:  # noqa: BLE001 - a failed stage must not lose the other metrics
        log.exception("stage %s failed completely", name)
        for row in rows:
            row.errors[name] = f"{type(exc).__name__}: {exc}"
    finally:
        _free_memory()


# ----------------------------------------------------------------------------
# input / output
# ----------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON ({exc})") from exc
    return records


def _clean(obj: Any) -> Any:
    """NaN / inf are not valid JSON -> null."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _json_default(obj: Any):
    return obj.item() if hasattr(obj, "item") else str(obj)   # numpy numbers


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(_clean(rec), ensure_ascii=False, default=_json_default) + "\n")


def calculate_all(records: list[dict], cfg: Config, stages: Optional[list[str]] = None) -> list[dict]:
    """records -> records with the new fields (the input fields are not changed)."""
    stages = stages if stages is not None else list(STAGES)
    rows = [preprocess(rec, cfg) for rec in tqdm(records, desc="preprocess")]
    for name in stages:
        run_stage(name, rows, cfg)

    results = []
    for row in rows:
        rec = {**row.rec, **row.out}
        if row.errors:
            rec["metric_errors"] = row.errors
        if cfg.drop_fields:
            for key in cfg.drop_fields:
                rec.pop(key, None)                # a missing key is fine
        results.append(rec)
    return results


def process_file(input_path: Path, output_path: Path, cfg: Config, stages: Optional[list[str]] = None) -> None:
    records = read_jsonl(input_path)
    log.info("records: %d", len(records))
    results = calculate_all(records, cfg, stages)
    write_jsonl(output_path, results)

    manifest = {
        "text_pipeline_version": TEXT_PIPELINE_VERSION,
        "input": str(input_path),
        "n_records": len(records),
        "n_records_with_errors": sum("metric_errors" in r for r in results),
        "fields": {"text": cfg.text_field,
           "reference": f"csv:{cfg.reference_csv}" if cfg.reference_csv else None,
           "judge": cfg.judge_field},
        "metrics": cfg.manifest,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("done: %s (+ %s)", output_path, manifest_path.name)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--drop-fields", nargs="+", help="remove these keys from the output jsonl")
    p.add_argument("--only", nargs="+", choices=list(STAGES), help="compute only these metrics")
    p.add_argument("--skip", nargs="+", choices=list(STAGES), default=[], help="skip these metrics")
    p.add_argument("--text-field", default=Config.text_field)
    p.add_argument("--reference-csv", type=Path)
    p.add_argument("--reference-csv-field", default="literal_version")
    p.add_argument("--reference-csv-id", default="text_id")
    p.add_argument("--record-id-field", default="text_id")
    p.add_argument("--judge-field", default=Config.judge_field)
    p.add_argument("--abstractness-dict", type=Path)
    p.add_argument("--w2v-path", help="lemma_POS vectors for dist_sem (default: the model of lexical.py)")
    p.add_argument("--window", type=int, default=Config.window)
    p.add_argument("--device")
    p.add_argument("--ppl-model")
    p.add_argument("--ppl-dtype", choices=["float32", "bfloat16", "float16"])
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(
        text_field=args.text_field, judge_field=args.judge_field,
        abstractness_dict=args.abstractness_dict, w2v_path=args.w2v_path, window=args.window,
        device=args.device, ppl_model=args.ppl_model, ppl_dtype=args.ppl_dtype, drop_fields=args.drop_fields,
        reference_csv=args.reference_csv, reference_csv_field=args.reference_csv_field,
        reference_id_field=args.reference_csv_id, record_id_field=args.record_id_field,
    )
    stages = [s for s in (args.only or list(STAGES)) if s not in args.skip]
    process_file(args.input, args.output, cfg, stages)


if __name__ == "__main__":
    main()