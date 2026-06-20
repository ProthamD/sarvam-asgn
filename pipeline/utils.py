"""
pipeline/utils.py — Shared helpers: logging, config loading, retry, file ops.
"""
from __future__ import annotations

import json
import os
import sys
import hashlib
import time
import logging
import functools
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests

load_dotenv()

# ── Console & Logging ─────────────────────────────────────────────────────────
console = Console()

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    return logging.getLogger(name)

logger = get_logger("utils")

# ── Config ────────────────────────────────────────────────────────────────────
_CONFIG: dict | None = None

def load_config(path: str = "config.yaml") -> dict:
    global _CONFIG
    if _CONFIG is None:
        with open(path, "r", encoding="utf-8") as f:
            _CONFIG = yaml.safe_load(f)
    return _CONFIG


def cfg(path: str, config: dict | None = None) -> Any:
    """
    Dot-path accessor: cfg("quality.min_snr_db") → 18.0
    """
    c = config or load_config()
    for key in path.split("."):
        c = c[key]
    return c


# ── Paths ─────────────────────────────────────────────────────────────────────
def ensure_dirs(config: dict | None = None) -> None:
    c = config or load_config()
    for p in c["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)


def get_path(key: str, config: dict | None = None) -> Path:
    c = config or load_config()
    return Path(c["paths"][key])


# ── File helpers ──────────────────────────────────────────────────────────────
def file_md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(data: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(record: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Env helpers ───────────────────────────────────────────────────────────────
def get_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(
            f"Missing environment variable: {key}\n"
            f"Add it to your .env file."
        )
    return val


# ── Retry decorator ───────────────────────────────────────────────────────────
T = TypeVar("T")

def with_retry(
    max_attempts: int = 4,
    wait_min: int = 2,
    wait_max: int = 30,
) -> Callable:
    """Decorator: retry on transient HTTP / connection errors."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=wait_min, max=wait_max),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError,
                                       requests.exceptions.Timeout,
                                       requests.exceptions.HTTPError)),
        reraise=True,
    )


# ── Progress helpers ──────────────────────────────────────────────────────────
def seconds_to_hms(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:05.2f}"


def format_duration_table(label_durations: dict[str, float]) -> str:
    """Pretty-print a dict of {label: seconds} as a table."""
    lines = [f"{'Stage':<30} {'Duration':>10}", "-" * 42]
    total = 0.0
    for label, dur in label_durations.items():
        lines.append(f"{label:<30} {seconds_to_hms(dur):>10}")
        total += dur
    lines.append("-" * 42)
    lines.append(f"{'TOTAL':<30} {seconds_to_hms(total):>10}")
    return "\n".join(lines)
