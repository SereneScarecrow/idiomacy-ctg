"""
This module creates the dataset.

    python build_dataset.py --input data/raw/dataset_maxi.csv --output-dir data/processed

Outputs (in --output-dir):
    dataset_maxi.csv       the dataset (columns are described in DATASET.md)
    titr_thresholds.json   TITR thresholds of the idiomaticity levels (used by generation)
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from idiomacy.metrics.spans import align_idioms, covered_words, occurrence_mismatches, parse_idioms
from idiomacy.metrics.text import TEXT_PIPELINE_VERSION, count_words, normalize_text

# columns that must be in the raw file (text_id is created from the row order if absent)
REQUIRED_COLUMNS = [
    "domain", "author", "source", "link", "year",
    "original_text", "literal_text", "literal_model",
    "idiomatic_expressions",
]

OUTPUT_COLUMNS = [
    "text_id", "domain", "author", "source", "link", "year",
    "original_text", "literal_text", "literal_model",
    "idiomatic_expressions", "idiom_spans",
    "len_original", "len_literal", "len_idioms", "TITR", "titr_level", "idiom_count", "len_ratio",
]

# expectations that are checked by validate_dataset (reported as warnings)
EXPECTED_TEXTS_PER_DOMAIN = 42
MAX_TEXTS_PER_AUTHOR = 2
LENGTH_RANGE = (100, 400)


# ---------------------------------------------------------------- 
# loading
# ----------------------------------------------------------------

def read_raw(path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Сhecks that all required columns are present."""
    df = raw.copy()

    if "text_id" not in df.columns:
        print("WARNING: no text_id column, using row order (1..n)")
        df.insert(0, "text_id", range(1, len(df) + 1))

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in the raw data: {missing}")

    return df


# ---------------------------------------------------------------- 
# processing
# ----------------------------------------------------------------

def normalize_entries(df: pd.DataFrame, columns=("original_text", "literal_text")) -> pd.DataFrame:
    """
    Normalizes all text entries in dataset.
    """
    df = df.copy()
    for column in columns:
        df[column] = df[column].apply(normalize_text)
    return df


def process_idioms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses the annotated idioms and aligns them with the (normalized) original text.
    Adds `idioms` (list of str) and `idiom_matches` (list of IdiomMatch).
    """
    df = df.copy()
    df["idioms"] = df["idiomatic_expressions"].apply(parse_idioms)
    df["idiom_matches"] = [
        align_idioms(text, idioms) for text, idioms in zip(df["original_text"], df["idioms"])
    ]
    return df


def count_dataset_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all dataset metrics (the level needs the thresholds, see assign_titr_levels).
    Lengths are numbers of words, counted only by count_words / covered_words.

        len_idioms   words of the original covered by idioms (a word is counted once)
        TITR         text idiom token ratio = len_idioms / len_original
        len_ratio    len_literal / len_original
    """
    df = df.copy()
    df["len_original"] = df["original_text"].apply(count_words)
    df["len_literal"] = df["literal_text"].apply(count_words)
    df["len_idioms"] = [
        covered_words(text, matches) for text, matches in zip(df["original_text"], df["idiom_matches"])
    ]
    df["TITR"] = df["len_idioms"] / df["len_original"]
    df["idiom_count"] = df["idioms"].apply(len)
    df["len_ratio"] = df["len_literal"] / df["len_original"]
    return df


# ---------------------------------------------------------------- 
# levels
# ----------------------------------------------------------------

def resolve_thresholds(titr: pd.Series, low_max=None, high_min=None) -> dict:
    """
    Thresholds of the idiomaticity levels. By default they are the 25% and 75% quantiles
    of TITR; explicitly given numbers replace the computed ones.
    """
    q = titr.quantile([0.25, 0.5, 0.75])
    thresholds = {
        "low_max": float(q[0.25]) if low_max is None else float(low_max),
        "high_min": float(q[0.75]) if high_min is None else float(high_min),
        "source": {
            "low_max": "computed" if low_max is None else "override",
            "high_min": "computed" if high_min is None else "override",
        },
        "computed_quantiles": {"q25": float(q[0.25]), "q50": float(q[0.5]), "q75": float(q[0.75])},
        "n_texts": int(titr.notna().sum()),
        "rule": "low: TITR <= low_max; high: TITR > high_min; medium otherwise",
        "method": "pandas quantile (linear interpolation) over all texts of both domains",
        "text_pipeline_version": TEXT_PIPELINE_VERSION,  # the thresholds depend on how words are counted
    }
    if thresholds["low_max"] >= thresholds["high_min"]:
        raise ValueError(f"low_max must be smaller than high_min: {thresholds}")
    return thresholds


def load_thresholds(path, low_max=None, high_min=None) -> dict:
    """
    Reads titr_thresholds.json (for generation). Numbers given explicitly replace
    the stored ones, so a run can use other thresholds without rebuilding the dataset.
    """
    thresholds = json.loads(Path(path).read_text(encoding="utf-8"))
    if low_max is not None:
        thresholds["low_max"], thresholds["source"]["low_max"] = float(low_max), "override"
    if high_min is not None:
        thresholds["high_min"], thresholds["source"]["high_min"] = float(high_min), "override"
    return thresholds


def level_from_titr(value: float, thresholds: dict) -> str:
    if value <= thresholds["low_max"]:
        return "low"
    if value > thresholds["high_min"]:
        return "high"
    return "medium"


def assign_titr_levels(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    df = df.copy()
    df["titr_level"] = df["TITR"].apply(lambda v: level_from_titr(v, thresholds))
    return df


# ---------------------------------------------------------------- 
# validation
# ----------------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> dict:
    """Errors stop the build, warnings are only reported."""
    errors, warnings = [], []

    if df["text_id"].duplicated().any():
        errors.append(f"duplicated text_id: {df.loc[df['text_id'].duplicated(), 'text_id'].tolist()}")
    for column in ("original_text", "literal_text"):
        empty = df.loc[df[column].str.strip() == "", "text_id"].tolist()
        if empty:
            errors.append(f"empty {column}: {empty}")
    if not df["TITR"].between(0, 1).all():
        errors.append(f"TITR outside [0, 1]: {df.loc[~df['TITR'].between(0, 1), 'text_id'].tolist()}")
    if (df["len_idioms"] > df["len_original"]).any():
        errors.append("len_idioms > len_original")

    for domain, n in df["domain"].value_counts().items():
        if n != EXPECTED_TEXTS_PER_DOMAIN:
            warnings.append(f"domain '{domain}': {n} texts, expected {EXPECTED_TEXTS_PER_DOMAIN}")
    per_author = df["author"].dropna().value_counts()
    too_many = per_author[per_author > MAX_TEXTS_PER_AUTHOR].to_dict()
    if too_many:
        warnings.append(f"more than {MAX_TEXTS_PER_AUTHOR} texts per author: {too_many}")
    low, high = LENGTH_RANGE
    outside = df.loc[~df["len_original"].between(low, high), ["text_id", "len_original"]]
    if len(outside):
        warnings.append(f"len_original outside {low}-{high}: {dict(zip(outside.text_id, outside.len_original))}")

    for text_id, matches in zip(df["text_id"], df["idiom_matches"]):
        for match in matches:
            if not match.found:
                warnings.append(f"text {text_id}: idiom not found in the text: {match.idiom[:80]!r}")
        for idiom, listed, occurs in occurrence_mismatches(matches):
            consequence = "extra entries cover nothing new" if listed > occurs else "the other occurrences are not counted"
            warnings.append(
                f"text {text_id}: {idiom[:60]!r} is listed {listed}x but occurs {occurs}x ({consequence})"
            )

    return {"errors": errors, "warnings": warnings}


# ---------------------------------------------------------------- 
# build and save
# ----------------------------------------------------------------

def export_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Final columns in a fixed order; idioms and their positions are stored as JSON."""
    out = df.copy()
    out["idiomatic_expressions"] = out["idioms"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    out["idiom_spans"] = out["idiom_matches"].apply(
        lambda ms: json.dumps([asdict(m) for m in ms], ensure_ascii=False)
    )
    return out[OUTPUT_COLUMNS]


def build_dataset(raw: pd.DataFrame, low_max=None, high_min=None):
    """
    Returns (dataset, thresholds, warnings).
    Raises ValueError if the checks find errors; warnings are returned for the caller to show.
    """
    df = prepare_raw(raw)
    df = normalize_entries(df)
    df = process_idioms(df)
    df = count_dataset_metrics(df)
    thresholds = resolve_thresholds(df["TITR"], low_max, high_min)
    df = assign_titr_levels(df, thresholds)

    problems = validate_dataset(df)
    if problems["errors"]:
        raise ValueError("Dataset checks failed:\n  " + "\n  ".join(problems["errors"]))

    return export_frame(df), thresholds, problems["warnings"]


def write_outputs(dataset: pd.DataFrame, thresholds: dict, output_dir) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_dir / "dataset_maxi.csv", index=False, encoding="utf-8-sig")
    (output_dir / "titr_thresholds.json").write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="Build the dataset")
    parser.add_argument("--input", required=True, help="raw CSV with texts, idioms and literal versions")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--low-max", type=float, help="use this threshold instead of the 25%% quantile")
    parser.add_argument("--high-min", type=float, help="use this threshold instead of the 75%% quantile")
    args = parser.parse_args()

    dataset, thresholds, warnings = build_dataset(read_raw(args.input), args.low_max, args.high_min)
    write_outputs(dataset, thresholds, args.output_dir)

    print(f"{len(dataset)} texts -> {args.output_dir}")
    print(f"thresholds: low_max={thresholds['low_max']:.4f}, high_min={thresholds['high_min']:.4f} "
          f"({thresholds['source']['low_max']}/{thresholds['source']['high_min']})")
    print(f"levels: {dataset['titr_level'].value_counts().to_dict()}")
    for warning in warnings:
        print("WARNING:", warning)


if __name__ == "__main__":
    main()