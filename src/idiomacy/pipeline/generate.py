"""
Runs the full generation cycle for an experiment.
The parameters of the experiment are stated in the config (configs/generation.yaml).

Every run lives in its own folder results/<run_id>/:
    manifest.json      facts about the run: config, models, prompts, dataset, git, status
    generations.jsonl  one line per generated text, appended (and fsynced) right after
                       the generation, so a crash never loses finished work

If a run fails or is interrupted, it can be continued: the texts that are already in
generations.jsonl are skipped.

Usage:
    python -m idiomacy.pipeline.generate configs/generation.yaml
    python -m idiomacy.pipeline.generate configs/generation.yaml --resume <run_id>
"""

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from idiomacy.llm.base import LLMClient, LLMError, Request, Response, create_client
from idiomacy.llm.prompts import Prompt, load_prompt

# src/idiomacy/pipeline/generate.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "generation.yaml"

# columns of the processed dataset: the model gets the literal version of the text
ID_COLUMN = "text_id"
TEXT_COLUMN = "literal_text"

MANIFEST_NAME = "manifest.json"
GENERATIONS_NAME = "generations.jsonl"

MAX_ATTEMPTS = 3        # per generation, only for backend errors (server down, ...)
RETRY_DELAY_S = 5       # grows linearly: 5 s, 10 s, ...


class ConfigError(ValueError):
    """The config (or a file it refers to) is invalid."""


class ResumeError(RuntimeError):
    """The run cannot be continued (not found, or something changed since it started)."""


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


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_and_validate_config(config_path: Path) -> GenerationConfig:
    """
    Loads the config and validates existing fields.
    All problems are collected and reported together.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ConfigError(f"Config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: expected a mapping at the top level")

    errors: list[str] = []

    unknown = sorted(set(raw) - _ALLOWED_KEYS)
    if unknown:
        errors.append(f"unknown keys: {unknown} (typo?)")

    def string_list(key: str) -> tuple[str, ...]:
        value = raw.get(key)
        if not isinstance(value, list) or not value or not all(isinstance(v, str) and v.strip() for v in value):
            errors.append(f"'{key}' must be a non-empty list of strings")
            return ()
        if len(set(value)) != len(value):
            errors.append(f"'{key}' contains duplicates")
        return tuple(value)

    experiment = raw.get("experiment")
    if not isinstance(experiment, str) or not re.fullmatch(r"[\w.-]+", experiment):
        errors.append("'experiment' must be a string of letters, digits, '_', '-', '.' (it becomes a folder name)")
        experiment = ""

    backend = raw.get("backend", "ollama")
    if not isinstance(backend, str) or not backend:
        errors.append("'backend' must be a string (default: ollama)")
        backend = ""

    temperature = raw.get("temperature")
    if not _is_number(temperature) or temperature < 0:
        errors.append("'temperature' must be a number >= 0")
        temperature = 0.0

    thresholds = {}
    for key in ("low_max", "high_min"):
        value = raw.get(key)
        if value is not None and not _is_number(value):
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

    max_tokens = raw.get("max_tokens")
    if max_tokens is not None and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0):
        errors.append("'max_tokens' must be a positive integer or empty")
        max_tokens = None

    schema = raw.get("schema")
    if schema is not None:
        if not isinstance(schema, dict):
            errors.append("'schema' must be a mapping (valid JSON Schema object)")
            schema = None 

    extra = raw.get("extra") or {}
    if not isinstance(extra, dict):
        errors.append("'extra' must be a mapping")
        extra = {}

    if errors:
        raise ConfigError(f"Invalid config {config_path}:\n  - " + "\n  - ".join(errors))

    return GenerationConfig(
        experiment=experiment,
        backend=backend,
        temperature=float(temperature),
        low_max=thresholds["low_max"],
        high_min=thresholds["high_min"],
        prompts=prompts,
        models=models,
        dataset_path=dataset_path,
        seeds=tuple(seeds),
        max_tokens=max_tokens,
        schema=schema,
        extra=extra,
    )


# ----------------------------------------------------------------------------
# inputs: dataset and prompts
# ----------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    """Reads the dataset; returns [{'id': str, 'text': str}, ...] in file order."""
    if not path.is_absolute():
        path = PROJECT_ROOT / path
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
    """Loads all prompts of the config and checks that every placeholder will get a value."""
    prompts = {}
    available = set(prompt_values(cfg, ""))
    for name in cfg.prompts:
        path = PROMPTS_DIR / f"{name}.txt"
        if not path.is_file():
            raise ConfigError(f"Prompt file not found: {path}")
        prompt = load_prompt(path)
        no_value = [v for v in prompt.variables if v not in available]
        if no_value:
            raise ConfigError(
                f"Prompt {name!r} uses {no_value}, which have no value "
                f"(available: {sorted(available)}; low_max/high_min must be set in the config to be used)"
            )
        prompts[name] = prompt
    return prompts


# ----------------------------------------------------------------------------
# run folder: manifest and generations
# ----------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_run_id(experiment: str) -> str:
    return f"{experiment}_{datetime.now():%Y%m%d-%H%M%S}"


def git_state() -> dict | None:
    """Commit of the code the run was started from (None outside a git repo)."""
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json_atomic(path: Path, data: dict) -> None:
    """A crash in the middle of writing never leaves a broken manifest."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


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


class JsonlWriter:
    """Append-only JSON lines file; every record is flushed to disk immediately."""

    def __init__(self, path: Path):
        needs_newline = False
        if path.exists() and path.stat().st_size > 0:
            with path.open("rb") as file:  # a crash can leave the last line unfinished
                file.seek(-1, os.SEEK_END)
                needs_newline = file.read(1) != b"\n"
        self._file = path.open("a", encoding="utf-8", newline="\n")
        if needs_newline:
            self._file.write("\n")
            self._file.flush()

    def write(self, record: dict) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._file.close()


def read_done_keys(path: Path) -> set[tuple]:
    """Keys (model, prompt, seed, text_id) of the generations that are already saved."""
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
                done.add((record["model"], record["prompt"], record["seed"], record["text_id"]))
            except (json.JSONDecodeError, KeyError):
                continue  # unfinished line left by a crash: this text will be generated again
    return done


# ----------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------

def total_generations(cfg: GenerationConfig, items: list[dict]) -> int:
    return len(cfg.models) * len(cfg.prompts) * len(cfg.seeds) * len(items)


def generate_with_retry(client: LLMClient, request: Request) -> Response:
    """Backend errors are retried a few times; if the backend stays down, the error is raised."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return client.generate(request)
        except LLMError as error:
            if attempt == MAX_ATTEMPTS:
                raise
            delay = RETRY_DELAY_S * attempt
            tqdm.write(f"{error} - retry {attempt}/{MAX_ATTEMPTS - 1} in {delay} s")
            time.sleep(delay)


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