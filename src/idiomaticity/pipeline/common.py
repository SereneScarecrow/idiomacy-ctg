"""
Helpers shared by the pipeline runners (evaluate.py; generate.py has its own copies of
most of them and is left untouched): paths, errors, config checks, the run manifest,
the append-only JSONL log, prompt loading and the retry around a backend call.
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from idiomaticity.llm.base import LLMClient, LLMError, Request, Response
from idiomaticity.llm.prompts import Prompt, load_prompt

# src/idiomacy/pipeline/common.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"

MANIFEST_NAME = "manifest.json"

MAX_ATTEMPTS = 3        # per backend call, only for backend errors (server down, rate limit, ...)
RETRY_DELAY_S = 5       # grows linearly: 5 s, 10 s, ...


class ConfigError(ValueError):
    """The config (or a file it refers to) is invalid."""


class ResumeError(RuntimeError):
    """The run cannot be continued (not found, or something changed since it started)."""


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def resolve_path(path: str | Path) -> Path:
    """Relative paths in configs are relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_yaml_mapping(config_path: Path) -> dict:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ConfigError(f"Config not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"{config_path}: cannot parse YAML: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: expected a mapping at the top level")
    return raw


def check_llm_options(raw: dict, errors: list[str]) -> dict:
    """
    Validates the fields every config has (experiment, backend, temperature, max_tokens, extra).
    Problems are appended to `errors`; the returned values are safe placeholders in that case.
    """
    experiment = raw.get("experiment")
    if not isinstance(experiment, str) or not re.fullmatch(r"[\w.-]+", experiment):
        errors.append("'experiment' must be a string of letters, digits, '_', '-', '.' (it becomes a folder name)")
        experiment = ""

    backend = raw.get("backend", "ollama")
    if not isinstance(backend, str) or not backend:
        errors.append("'backend' must be a string (default: ollama)")
        backend = ""

    temperature = raw.get("temperature")
    if not is_number(temperature) or temperature < 0:
        errors.append("'temperature' must be a number >= 0")
        temperature = 0.0

    max_tokens = raw.get("max_tokens")
    if max_tokens is not None and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0):
        errors.append("'max_tokens' must be a positive integer or empty")
        max_tokens = None

    extra = raw.get("extra") or {}
    if not isinstance(extra, dict):
        errors.append("'extra' must be a mapping")
        extra = {}

    return {
        "experiment": experiment,
        "backend": backend,
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        "extra": extra,
    }


def load_checked_prompt(name: str, available: set[str]) -> Prompt:
    """Loads prompts/<name>.txt and checks that every {{placeholder}} will get a value."""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.is_file():
        raise ConfigError(f"Prompt file not found: {path}")
    prompt = load_prompt(path)
    no_value = [v for v in prompt.variables if v not in available]
    if no_value:
        raise ConfigError(f"Prompt {name!r} uses {no_value}, which have no value (available: {sorted(available)})")
    return prompt


# ----------------------------------------------------------------------------
# run folder
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


def read_jsonl(path: Path) -> list[dict]:
    """All records of a JSON lines file; unfinished or broken lines (left by a crash) are skipped."""
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


# ----------------------------------------------------------------------------
# backend call
# ----------------------------------------------------------------------------

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