#!/usr/bin/env python
"""Static release check for secrets and private artifacts."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".parquet", ".npy", ".pkl", ".joblib", ".pyc", ".pdf", ".docm"}
FORBIDDEN_NAMES = {".env", "id_rsa", "id_ed25519", "hosts.yml"}
FORBIDDEN_JSON_KEYS = {"records", "top_examples", "examples", "prompt_excerpt", "prompt", "judge_reason"}
SKIP_PARTS = {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
FORBIDDEN_TERMS = [
    "black" + "leg",
    "black-" + "leg-" + "nameko",
    "yosi" + "zuka",
    "yoshi" + "zuka",
    "\u5409\u585a",
]
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----"),
    re.compile(r"/(?:home|Users)/[^/\s\"']+(?:/[^\s\"']*)?"),
    re.compile(r"<local-" + r"home>/[^\s\"']+"),
    re.compile(r"github\.com/[^/\s\"']+/"),
]


def iter_files() -> list[Path]:
    if (ROOT / ".git").exists():
        proc = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            check=True,
            stdout=subprocess.PIPE,
        )
        return [ROOT / name.decode() for name in proc.stdout.split(b"\0") if name]
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not (set(path.parts) & SKIP_PARTS)
    ]


def walk_json(value: Any, path: Path) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert key not in FORBIDDEN_JSON_KEYS, f"{path}: forbidden JSON key {key!r}"
            walk_json(item, path)
    elif isinstance(value, list):
        for item in value:
            walk_json(item, path)
    elif isinstance(value, str):
        for pattern in SECRET_PATTERNS:
            assert not pattern.search(value), f"{path}: private path or secret leaked"
        for term in FORBIDDEN_TERMS:
            assert term not in value, f"{path}: identifying term leaked"


def main() -> None:
    for path in iter_files():
        assert path.name not in FORBIDDEN_NAMES, f"forbidden file name: {path}"
        assert path.suffix not in FORBIDDEN_SUFFIXES, f"forbidden artifact type: {path}"
        if path.stat().st_size > 5_000_000:
            raise AssertionError(f"unexpectedly large tracked file: {path}")

        text = None
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for pattern in SECRET_PATTERNS:
            assert not pattern.search(text), f"possible secret matched {pattern.pattern!r} in {path}"
        for term in FORBIDDEN_TERMS:
            assert term not in text, f"identifying term leaked in {path}"

        if path.suffix == ".json":
            walk_json(json.loads(text), path)

    print("release check passed")


if __name__ == "__main__":
    main()
