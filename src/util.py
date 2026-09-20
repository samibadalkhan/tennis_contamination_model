"""Shared paths and durable-IO helpers.

Everything a stage produces is written to ``results/`` as it is computed, so a
session that dies mid-run (out of usage, crash) loses nothing: a fresh run reads
what is on disk and continues. Writes are atomic (temp file + rename) so a
half-written JSON can never be mistaken for a completed checkpoint.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
STAGE0 = RESULTS / "stage0"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj) -> Path:
    """Atomically write ``obj`` as pretty JSON to ``path``."""
    ensure(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def append_jsonl(path: Path, obj) -> None:
    """Append one JSON record as a line. Append-only; never rewrites the file."""
    ensure(path.parent)
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, default=str) + "\n")


def exists_nonempty(path: Path) -> bool:
    p = Path(path)
    return p.exists() and p.stat().st_size > 0
