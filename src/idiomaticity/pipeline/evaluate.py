"""
Runs the LLM judge over the texts of an experiment.
The parameters are stated in the config (configs/evaluation.yaml).

What is judged, in this order:
  1. baseline texts from the dataset (marked with prompt="literal"), then
  2. every text of the generations.jsonl of a generation run.

Every run lives in its own folder results/<experiment>_<time>/:
    manifest.json     facts about the run: config, judge model, prompt, inputs, git, status
    judgments.jsonl   one line per judged text, appended (and fsynced) right after the call

Each line keeps the metadata of the generation (model, prompt, seed, text_id, temperature),
the judged text and the judge's answer. If a run fails or is interrupted, it can be
continued: the texts that are already in judgments.jsonl are skipped.

Usage:
    python -m idiomacy.pipeline.evaluate configs/evaluation.yaml
    python -m idiomacy.pipeline.evaluate configs/evaluation.yaml --resume <run_id>
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

from idiomaticity.llm.base import LLMClient, LLMError, Request, create_client
from idiomaticity.llm.prompts import Prompt
from idiomaticity.pipeline.common import (
    MANIFEST_NAME, PROJECT_ROOT, RESULTS_DIR,
    ConfigError, JsonlWriter, ResumeError,
    check_llm_options, generate_with_retry, git_state, is_number, load_checked_prompt,
    make_run_id, now_iso, read_jsonl, read_yaml_mapping, resolve_path, write_json_atomic,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "evaluation.yaml"
JUDGMENTS_NAME = "judgments.jsonl"

# dataset column with the id of a text (the same as in generation)
ID_COLUMN = "text_id"
# baseline texts to judge: marker (goes to the 'prompt' field of the result) -> dataset column
BASELINE_TEXTS = {
    "literal": "literal_text",
    "original": "original_text"
}

# metadata of a generation that is copied to the result
GENERATION_META = ("model", "prompt", "seed", "text_id", "temperature")


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

_ALLOWED_KEYS = {
    "experiment", "backend", "model", "temperature", "max_tokens", "extra",
    "prompt", "baseline", "generated", "schema",
}


@dataclass(frozen=True)
class EvaluationConfig:
    experiment: str
    backend: str
    model: str
    temperature: float
    prompt: str
    baseline_path: str
    generated_path: str
    schema: dict
    max_tokens: int | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-compatible form, used in the manifest."""
        return json.loads(json.dumps(asdict(self)))


def load_and_validate_config(config_path: Path) -> EvaluationConfig:
    """
    Loads the config and validates existing fields.
    All problems are collected and reported together.
    """
    raw = read_yaml_mapping(config_path)
    errors: list[str] = []

    unknown = sorted(set(raw) - _ALLOWED_KEYS)
    if unknown:
        errors.append(f"unknown keys: {unknown} (typo?)")

    options = check_llm_options(raw, errors)

    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        errors.append("'model' must be a non-empty string")
        model = ""

    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        errors.append("'prompt' must be a prompt name (prompts/<name>.txt)")
        prompt = ""

    paths = {}
    for key in ("baseline", "generated"):
        section = raw.get(key)
        path = section.get("path") if isinstance(section, dict) else None
        if not isinstance(path, str) or not path:
            errors.append(f"'{key}.path' must be a string")
            path = ""
        paths[key] = path

    schema = raw.get("schema")
    if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict) or not schema["properties"]:
        errors.append("'schema' must be a JSON Schema object with 'properties'")
        schema = {}

    if errors:
        raise ConfigError(f"Invalid config {config_path}:\n  - " + "\n  - ".join(errors))

    return EvaluationConfig(
        experiment=options["experiment"],
        backend=options["backend"],
        model=model,
        temperature=options["temperature"],
        prompt=prompt,
        baseline_path=paths["baseline"],
        generated_path=paths["generated"],
        schema=schema,
        max_tokens=options["max_tokens"],
        extra=options["extra"],
    )


# ----------------------------------------------------------------------------
# inputs:  dataset and generated texts
# ----------------------------------------------------------------------------
# A task is one text to judge:
#   {"source": "baseline" | "generated", "meta": {...}, "text": str | None,
#    "generator_percentage": int | None, "skip_reason": str | None}

def task_key(meta: dict) -> tuple:
    """Identity of a task; the same for the task and for the line saved for it."""
    return (meta.get("model"), meta["prompt"], meta.get("seed"), meta["text_id"])


def load_baseline_tasks(path: Path) -> list[dict]:
    if not path.is_file():
        raise ConfigError(f"Baseline dataset not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        wanted = [ID_COLUMN, *BASELINE_TEXTS.values()]
        missing = [c for c in wanted if c not in (reader.fieldnames or [])]
        if missing:
            raise ConfigError(f"Dataset {path} has no columns {missing}")
        rows = list(reader)
    ids = [row[ID_COLUMN].strip() for row in rows]
    if not rows:
        raise ConfigError(f"Dataset {path} is empty")
    if len(set(ids)) != len(ids):
        raise ConfigError(f"Dataset {path}: ids are not unique")

    tasks = []
    for marker, column in BASELINE_TEXTS.items():
        for text_id, row in zip(ids, rows):
            text = row[column]
            tasks.append({
                "source": "baseline",
                "meta": {"prompt": marker, "text_id": text_id},
                "text": text,
                "generator_percentage": None,
                "skip_reason": None if text.strip() else f"empty '{column}'",
            })
    return tasks


def extract_generated_text(record: dict) -> tuple[str | None, int | None, str | None]:
    """
    The generator answered with JSON {"generated_text": ..., "idiomatic_percentage": ...}
    (see the schema in generation.yaml). Returns (text, reported percentage, skip reason).
    """
    if record.get("valid") is False:
        return None, None, "generation is marked invalid (empty or cut off)"
    try:
        data = json.loads(record["text"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None, None, "generation is not valid JSON"
    text = data.get("generated_text") if isinstance(data, dict) else None
    if not isinstance(text, str) or not text.strip():
        return None, None, "no 'generated_text' in the generation"
    percentage = data.get("idiomatic_percentage")
    if not isinstance(percentage, int) or isinstance(percentage, bool):
        percentage = None
    return text, percentage, None


def load_generated_tasks(path: Path) -> list[dict]:
    if not path.is_file():
        raise ConfigError(f"Generations not found: {path}")
    latest = {}  # a text generated twice (e.g. after a repeated run): the last record counts
    for record in read_jsonl(path):
        if all(key in record for key in ("model", "prompt", "seed", "text_id")):
            latest[(record["model"], record["prompt"], record["seed"], record["text_id"])] = record
    if not latest:
        raise ConfigError(f"No generations in {path}")

    tasks = []
    for record in latest.values():
        text, percentage, skip_reason = extract_generated_text(record)
        tasks.append({
            "source": "generated",
            "meta": {key: record.get(key) for key in GENERATION_META},
            "text": text,
            "generator_percentage": percentage,
            "skip_reason": skip_reason,
        })
    return tasks


def build_tasks(cfg: EvaluationConfig) -> list[dict]:
    """Baseline texts first, then the generations in the order of the file."""
    return load_baseline_tasks(resolve_path(cfg.baseline_path)) + load_generated_tasks(resolve_path(cfg.generated_path))


# ----------------------------------------------------------------------------
# judge answer
# ----------------------------------------------------------------------------

def validate_answer(answer, schema: dict) -> list[str]:
    """Checks the answer against the (flat) JSON schema: required, types, ranges, extra keys."""
    if not isinstance(answer, dict):
        return ["the answer is not a JSON object"]
    problems = []
    properties = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in answer:
            problems.append(f"missing '{key}'")
    if schema.get("additionalProperties") is False:
        unexpected = sorted(set(answer) - set(properties))
        if unexpected:
            problems.append(f"unexpected keys {unexpected}")
    for key, spec in properties.items():
        if key not in answer:
            continue
        value, kind = answer[key], spec.get("type")
        if kind == "integer":
            right_type = isinstance(value, int) and not isinstance(value, bool)
        elif kind == "number":
            right_type = is_number(value)
        elif kind == "string":
            right_type = isinstance(value, str)
        else:
            right_type = True
        if not right_type:
            problems.append(f"'{key}' must be {kind}")
            continue
        if kind in ("integer", "number"):
            if "minimum" in spec and value < spec["minimum"]:
                problems.append(f"'{key}'={value} is below {spec['minimum']}")
            if "maximum" in spec and value > spec["maximum"]:
                problems.append(f"'{key}'={value} is above {spec['maximum']}")
    return problems


def parse_answer(text: str, schema: dict) -> tuple[dict | None, str | None]:
    """Returns (answer, None) or (None, what is wrong)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):  # some models wrap JSON in a code fence
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
    try:
        answer = json.loads(cleaned)
    except json.JSONDecodeError as error:
        return None, f"the answer is not valid JSON ({error.msg})"
    problems = validate_answer(answer, schema)
    if problems:
        return None, "; ".join(problems)
    return answer, None


def judge_task(client: LLMClient, cfg: EvaluationConfig, prompt: Prompt, task: dict) -> dict:
    """One line of judgments.jsonl: metadata, the judged text and the judge's answer."""
    record = {
        **task["meta"],
        "text": task["text"],
        "generator_percentage": task["generator_percentage"],
    }
    if task["skip_reason"]:
        return {**record, "status": "skipped", "reason": task["skip_reason"], "answer": None,
                "judge_text": None, "created_at": now_iso()}

    request = Request(
        model=cfg.model,
        prompt=prompt.render(text=task["text"]),
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        schema=cfg.schema,
        extra=dict(cfg.extra),
    )
    response = generate_with_retry(client, request)
    answer, problem = parse_answer(response.text, cfg.schema)
    if problem and response.finish_reason != "stop":
        problem += f" (finish_reason={response.finish_reason!r})"
    return {
        **record,
        "status": "ok" if answer is not None else "invalid_answer",
        "reason": problem,
        "answer": answer,
        "judge_text": response.text,
        "finish_reason": response.finish_reason,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "latency_s": round(response.latency_s, 3),
        "judge_model": response.model,
        "created_at": now_iso(),
        "raw": response.raw,
    }


# ----------------------------------------------------------------------------
# run folder: manifest and results
# ----------------------------------------------------------------------------

def read_done_keys(path: Path) -> set[tuple]:
    """Keys of the texts that already have a line in judgments.jsonl."""
    return {task_key(r) for r in read_jsonl(path) if "prompt" in r and "text_id" in r}


def build_manifest_header(
    cfg: EvaluationConfig,
    config_path: Path,
    client: LLMClient,
    prompt: Prompt,
    tasks: list[dict],
) -> dict:
    """
    Everything that is known before the first judgment.
    model_info() raises LLMError if the judge model is not available, so a run fails before it starts.
    """
    generated_path = resolve_path(cfg.generated_path)
    return {
        "experiment": cfg.experiment,
        "started_at": now_iso(),
        "status": "running",
        "config_path": str(config_path),
        "config": cfg.to_dict(),
        "git": git_state(),
        "backend": client.describe(),
        "judge_model": client.model_info(cfg.model),
        "prompt": {"name": prompt.name, "variables": list(prompt.variables), "template": prompt.template},
        "inputs": {
            "baseline": {
                "path": cfg.baseline_path,
                "texts": BASELINE_TEXTS,
                "n_tasks": sum(t["source"] == "baseline" for t in tasks),
            },
            "generated": {
                "path": cfg.generated_path,
                "generation_run": generated_path.parent.name,
                "n_tasks": sum(t["source"] == "generated" for t in tasks),
            },
        },
        "n_planned": len(tasks),
        "resumes": [],
    }


def check_resume_compatible(manifest: dict, header: dict) -> None:
    """A run is continued only if the config, the prompt text and the inputs are the same."""
    problems = []
    if manifest.get("config") != header["config"]:
        problems.append("the config differs from the one the run was started with")
    if manifest.get("prompt", {}).get("template") != header["prompt"]["template"]:
        problems.append(f"prompt {header['prompt']['name']!r} has changed")
    if manifest.get("inputs") != header["inputs"]:
        problems.append("the number of texts to judge has changed (the input files are not the same)")
    if problems:
        raise ResumeError("Cannot resume:\n  - " + "\n  - ".join(problems))


# ----------------------------------------------------------------------------
# runner
# ----------------------------------------------------------------------------

def runner(config_path: Path, resume_run_id: str | None = None, results_dir: Path = RESULTS_DIR) -> Path:
    """Runs (or continues) the evaluation; returns the run folder."""
    config_path = Path(config_path)
    cfg = load_and_validate_config(config_path)
    prompt = load_checked_prompt(cfg.prompt, available={"text"})
    tasks = build_tasks(cfg)
    client = create_client(cfg.backend)

    # everything above can fail without leaving an empty run folder behind
    header = build_manifest_header(cfg, config_path, client, prompt, tasks)

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
        run_id = make_run_id(f"{cfg.experiment}_judge")
        out = results_dir / run_id
        out.mkdir(parents=True)
        manifest = {"run_id": run_id, **header}

    judgments_path = out / JUDGMENTS_NAME
    done = read_done_keys(judgments_path)
    manifest["n_done"] = len(done)
    write_json_atomic(out / MANIFEST_NAME, manifest)

    status, error = "completed", None
    try:
        with JsonlWriter(judgments_path) as writer, tqdm(
            total=len(tasks), initial=len(done), desc=run_id, unit="text"
        ) as bar:
            for task in tasks:
                key = task_key(task["meta"])
                if key in done:
                    continue
                bar.set_postfix(model=task["meta"].get("model"), prompt=task["meta"]["prompt"])
                record = judge_task(client, cfg, prompt, task)
                writer.write(record)
                done.add(key)
                bar.update(1)
                if record["status"] != "ok":
                    tqdm.write(f"[{record['status']}] {key}: {record['reason']}")
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["status"] = status
        manifest["n_done"] = len(done)
        manifest["status_counts"] = dict(Counter(r.get("status") for r in read_jsonl(judgments_path)))
        manifest["updated_at"] = now_iso()
        if error:
            manifest["error"] = error
        write_json_atomic(out / MANIFEST_NAME, manifest)
        if status != "completed":
            print(f"\nProgress saved ({len(done)}/{len(tasks)}). To continue:\n"
                  f"  python -m idiomacy.pipeline.evaluate {config_path} --resume {run_id}")

    print(f"Done: {out}  {manifest['status_counts']}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LLM judge over baseline and generated texts.")
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