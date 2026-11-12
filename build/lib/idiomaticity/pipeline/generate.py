"""
Runs the full generation cycle for an experiment.
The parameters of the experiment are stated in the config (configs/generation.yaml).

Every run lives in its own folder results/<run_id>/:
    manifest.json      facts about the run: config, models, prompts, dataset, git, status
    generations.jsonl  one line per generated text, appended (and fsynced) right after
                       the generation, so a crash never loses finished work

If a run fails or is interrupted, it can be continued: the texts that are already in
generations.jsonl are skipped.

Shared helpers (paths, errors, manifest, JSONL log, retry) live in common.py.

Usage:
    python -m idiomacy.pipeline.generate configs/generation.yaml
    python -m idiomacy.pipeline.generate configs/generation.yaml --resume <run_id>
"""

import argparse
import csv
import itertools
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

from idiomaticity.llm.base import LLMClient, LLMError, Request, Response, create_client
from idiomaticity.llm.prompts import Prompt
from idiomaticity.pipeline.common import (
    MANIFEST_NAME, PROJECT_ROOT, RESULTS_DIR,
    ConfigError, JsonlWriter, ResumeError,
    check_llm_options, generate_with_retry, git_state, is_number, load_checked_prompt,
    make_run_id, now_iso, read_jsonl, read_yaml_mapping, resolve_path, write_json_atomic,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "generation.yaml"
GENERATIONS_NAME = "generations.jsonl"

# columns of the processed dataset: the model gets the literal version of the text
ID_COLUMN = "text_id"
TEXT_COLUMN = "literal_text"


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

_ALLOWED_KEYS = {
    "experiment", "backend", "temperature", "low_max", "high_min",
    "prompts", "models", "dataset", "seeds", "max_tokens", "schema", "extra",
}


@dataclass(frozen=True)
class GenerationConfig:
    experiment: str
    backend: str
    temperature: float
    low_max: float | None
    high_min: float | None
    prompts: tuple[str, ...]
    models: tuple[str, ...]
    dataset_path: str
    seeds: tuple[int, ...]
    max_tokens: int | None = None
    schema: dict | None = None
    extra: dict = field(default_factory=dict)   # backend options, e.g. {num_ctx: 8192}

    def to_dict(self) -> dict:
        """JSON-compatible form (lists instead of tuples), used in the manifest."""
        return json.loads(json.dumps(asdict(self)))


def load_and_validate_config(config_path: Path) -> GenerationConfig:
    """
    Loads the config and validates existing fields.
    All problems are collected and reported together.
    """
    raw = read_yaml_mapping(config_path)
    errors: list[str] = []

    unknown = sorted(set(raw) - _ALLOWED_KEYS)
    if unknown:
        errors.append(f"unknown keys: {unknown} (typo?)")

    # experiment, backend, temperature, max_tokens, extra
    options = check_llm_options(raw, errors)

    def string_list(key: str) -> tuple[str, ...]:
        value = raw.get(key)
        if not isinstance(value, list) or not value or not all(isinstance(v, str) and v.strip() for v in value):
            errors.append(f"'{key}' must be a non-empty list of strings")
            return ()
        if len(set(value)) != len(value):
            errors.append(f"'{key}' contains duplicates")
        return tuple(value)

    thresholds = {}
    for key in ("low_max", "high_min"):
        value = raw.get(key)
        if value is not None and not is_number(value):
            errors.append(f"'{key}' must be a number or empty")
            value = None
        thresholds[key] = value
    if None not in thresholds.values() and thresholds["low_max"] >= thresholds["high_min"]:
        errors.append("'low_max' must be smaller than 'high_min'")

    prompts = string_list("prompts")
    models = string_list("models")

    dataset = raw.get("dataset")
    dataset_path = dataset.get("path") if isinstance(dataset, dict) else None
    if not isinstance(dataset_path, str) or not dataset_path:
        errors.append("'dataset.path' must be a string")
        dataset_path = ""

    seeds = raw.get("seeds")
    if (not isinstance(seeds, list) or not seeds
            or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds)):
        errors.append("'seeds' must be a non-empty list of integers")
        seeds = []
    elif len(set(seeds)) != len(seeds):
        errors.append("'seeds' contains duplicates")

    schema = raw.get("schema")
    if schema is not None and not isinstance(schema, dict):
        errors.append("'schema' must be a mapping (valid JSON Schema object)")
        schema = None

    if errors:
        raise ConfigError(f"Invalid config {config_path}:\n  - " + "\n  - ".join(errors))

    return GenerationConfig(
        experiment=options["experiment"],
        backend=options["backend"],
        temperature=options["temperature"],
        low_max=thresholds["low_max"],
        high_min=thresholds["high_min"],
        prompts=prompts,
        models=models,
        dataset_path=dataset_path,
        seeds=tuple(seeds),
        max_tokens=options["max_tokens"],
        schema=schema,
        extra=options["extra"],
    )


# ----------------------------------------------------------------------------
# inputs: dataset and prompts
# ----------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    """Reads the dataset; returns [{'id': str, 'text': str}, ...] in file order."""
    path = resolve_path(path)
    if not path.is_file():
        raise ConfigError(f"Dataset not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = [c for c in (ID_COLUMN, TEXT_COLUMN) if c not in (reader.fieldnames or [])]
        if missing:
            raise ConfigError(f"Dataset {path} has no columns {missing}")
        items = [{"id": row[ID_COLUMN].strip(), "text": row[TEXT_COLUMN]} for row in reader]
    ids = [item["id"] for item in items]
    if not items:
        raise ConfigError(f"Dataset {path} is empty")
    if len(set(ids)) != len(ids):
        raise ConfigError(f"Dataset {path}: ids are not unique")
    empty = [item["id"] for item in items if not item["text"].strip()]
    if empty:
        raise ConfigError(f"Dataset {path}: empty '{TEXT_COLUMN}' for ids {empty}")
    return items


def prompt_values(cfg: GenerationConfig, text: str) -> dict:
    """Values that can be used as {{placeholders}} in the prompt files."""
    values = {"text": text}
    if cfg.low_max is not None:
        values["low_max"] = cfg.low_max
    if cfg.high_min is not None:
        values["high_min"] = cfg.high_min
    return values


def load_prompts(cfg: GenerationConfig) -> dict[str, Prompt]:
    """
    Loads all prompts of the config and checks that every placeholder will get a value
    (low_max/high_min are available only if they are set in the config).
    """
    available = set(prompt_values(cfg, ""))
    return {name: load_checked_prompt(name, available) for name in cfg.prompts}


# ----------------------------------------------------------------------------
# run folder: manifest and generations
# ----------------------------------------------------------------------------

def build_manifest_header(
    cfg: GenerationConfig,
    config_path: Path,
    client: LLMClient,
    prompts: dict[str, Prompt],
    items: list[dict],
) -> dict:
    """
    Generates manifest header before the actual experiment is started:
    everything that is known before the first generation.
    model_info() raises LLMError if a model is not installed, so a run fails before it starts.
    Prompt texts are stored in full: the manifest alone says what the models were asked.
    """
    return {
        "experiment": cfg.experiment,
        "started_at": now_iso(),
        "status": "running",
        "config_path": str(config_path),
        "config": cfg.to_dict(),
        "git": git_state(),
        "backend": client.describe(),
        "models": {model: client.model_info(model) for model in cfg.models},
        "prompts": {
            name: {"variables": list(p.variables), "template": p.template}
            for name, p in prompts.items()
        },
        "dataset": {"path": cfg.dataset_path, "n_items": len(items)},
        "seeds": list(cfg.seeds),
        "n_planned": total_generations(cfg, items),
        "resumes": [],
    }


def check_resume_compatible(manifest: dict, header: dict) -> None:
    """
    A run is continued only if nothing that affects the generations has changed:
    the config, the prompt texts and the models (by digest).
    """
    problems = []
    if manifest.get("config") != header["config"]:
        problems.append("the config differs from the one the run was started with")
    for name, info in header["prompts"].items():
        old = manifest.get("prompts", {}).get(name)
        if old is None or old["template"] != info["template"]:
            problems.append(f"prompt {name!r} has changed")
    for model, info in header["models"].items():
        old = manifest.get("models", {}).get(model)
        if old is None or old.get("digest") != info.get("digest"):
            problems.append(f"model {model!r} is not the same build (digest differs)")
    if problems:
        raise ResumeError("Cannot resume:\n  - " + "\n  - ".join(problems))


def read_done_keys(path: Path) -> set[tuple]:
    """Keys (model, prompt, seed, text_id) of the generations that are already saved."""
    return {
        (r["model"], r["prompt"], r["seed"], r["text_id"])
        for r in read_jsonl(path)
        if all(key in r for key in ("model", "prompt", "seed", "text_id"))
    }


# ----------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------

def total_generations(cfg: GenerationConfig, items: list[dict]) -> int:
    return len(cfg.models) * len(cfg.prompts) * len(cfg.seeds) * len(items)


def make_record(model: str, prompt: Prompt, seed: int, item: dict, request: Request, response: Response) -> dict:
    """One line of generations.jsonl: the full result of one generation."""
    valid = bool(response.text.strip()) and response.finish_reason == "stop"
    return {
        "model": model,
        "prompt": prompt.name,
        "seed": seed,
        "text_id": item["id"],
        "temperature": request.temperature,
        "text": response.text,
        "valid": valid,  # False: empty answer or cut off by the token limit
        "finish_reason": response.finish_reason,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "latency_s": round(response.latency_s, 3),
        "backend_model": response.model,
        "reasoning": response.reasoning,
        "created_at": now_iso(),
        "raw": response.raw,
    }


def run_single_experiment(
    client: LLMClient,
    cfg: GenerationConfig,
    model: str,
    prompt: Prompt,
    seed: int,
    items: list[dict],
    done: set[tuple],
    writer: JsonlWriter,
    bar: tqdm,
) -> None:
    """
    Runs a single experiment in a config grid: one (model, prompt, seed) over all texts.
    Texts that are already saved are skipped.
    """
    bar.set_postfix(model=model, prompt=prompt.name, seed=seed)
    for item in items:
        key = (model, prompt.name, seed, item["id"])
        if key in done:
            continue
        request = Request(
            model=model,
            prompt=prompt.render(**prompt_values(cfg, item["text"])),
            temperature=cfg.temperature,
            seed=seed,
            max_tokens=cfg.max_tokens,
            schema=cfg.schema,
            extra=dict(cfg.extra),
        )
        response = generate_with_retry(client, request)
        record = make_record(model, prompt, seed, item, request, response)
        writer.write(record)
        done.add(key)
        bar.update(1)
        if not record["valid"]:
            tqdm.write(f"[warning] {model} / {prompt.name} / seed {seed} / text {item['id']}: "
                       f"finish_reason={response.finish_reason!r}, {len(response.text)} chars")


def runner(config_path: Path, resume_run_id: str | None = None, results_dir: Path = RESULTS_DIR) -> Path:
    """Runs (or continues) an experiment; returns the run folder."""
    config_path = Path(config_path)
    cfg = load_and_validate_config(config_path)
    items = load_dataset(Path(cfg.dataset_path))
    prompts = load_prompts(cfg)
    client = create_client(cfg.backend)

    # everything above can fail without leaving an empty run folder behind
    header = build_manifest_header(cfg, config_path, client, prompts, items)

    if resume_run_id:
        run_id = resume_run_id
        out = results_dir / run_id
        if not (out / MANIFEST_NAME).is_file():
            raise ResumeError(f"Run {run_id!r} not found in {results_dir}")
        manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
        check_resume_compatible(manifest, header)
        manifest["status"] = "running"
        manifest.pop("error", None)
        manifest["resumes"].append({"at": now_iso(), "backend": header["backend"]})
    else:
        run_id = make_run_id(cfg.experiment)
        out = results_dir / run_id
        out.mkdir(parents=True)
        manifest = {"run_id": run_id, **header}

    generations_path = out / GENERATIONS_NAME
    done = read_done_keys(generations_path)
    manifest["n_done"] = len(done)
    write_json_atomic(out / MANIFEST_NAME, manifest)

    status, error = "completed", None
    try:
        with JsonlWriter(generations_path) as writer, tqdm(
            total=manifest["n_planned"], initial=len(done), desc=run_id, unit="text"
        ) as bar:
            for model, prompt_name, seed in itertools.product(cfg.models, cfg.prompts, cfg.seeds):
                run_single_experiment(client, cfg, model, prompts[prompt_name], seed, items, done, writer, bar)
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["status"] = status
        manifest["n_done"] = len(done)
        manifest["updated_at"] = now_iso()
        if error:
            manifest["error"] = error
        write_json_atomic(out / MANIFEST_NAME, manifest)
        if status != "completed":
            print(f"\nProgress saved ({len(done)}/{manifest['n_planned']}). To continue:\n"
                  f"  python -m idiomacy.pipeline.generate {config_path} --resume {run_id}")

    print(f"Done: {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate texts for an idiomacy experiment.")
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG,
                        help=f"path to the config (default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)})")
    parser.add_argument("--resume", metavar="RUN_ID",
                        help="continue an existing run (folder name in results/)")
    args = parser.parse_args(argv)
    try:
        runner(args.config, resume_run_id=args.resume)
    except (ConfigError, ResumeError, LLMError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())