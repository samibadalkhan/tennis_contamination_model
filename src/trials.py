"""Append-only log of every model variant tried.

Trial count is required for any honest significance claim, and it is what lets a
negative result be reported without anyone asking whether the search stopped
early. The log is ``results/trials.jsonl`` (one JSON record per line). It is
**append-only and never reset** -- there is no function here that truncates it.

A "trial" is any fit/evaluation whose result could influence a modeling choice:
a model variant, a hyperparameter setting, a diagnostic with a tunable knob.
Log it when you run it, not later.

    from src import trials
    trials.log(stage="0", kind="iid_given_server",
               params={"C": 1.0, "features": ["serve", "return", "slam_year"]},
               metrics={"val_logloss": 0.612},
               note="baseline i.i.d. fit on train, evaluated on validation")
"""

from __future__ import annotations

from pathlib import Path

from src.util import RESULTS, append_jsonl, utcnow

TRIAL_LOG = RESULTS / "trials.jsonl"


def log(*, stage: str, kind: str, params: dict | None = None,
        metrics: dict | None = None, note: str = "") -> int:
    """Append one trial record. Returns the new trial count."""
    n = count() + 1
    append_jsonl(TRIAL_LOG, {
        "trial": n,
        "utc": utcnow(),
        "stage": stage,
        "kind": kind,
        "params": params or {},
        "metrics": metrics or {},
        "note": note,
    })
    return n


def count() -> int:
    p = Path(TRIAL_LOG)
    if not p.exists():
        return 0
    with open(p) as fh:
        return sum(1 for line in fh if line.strip())


def all_trials() -> list[dict]:
    import json
    p = Path(TRIAL_LOG)
    if not p.exists():
        return []
    with open(p) as fh:
        return [json.loads(line) for line in fh if line.strip()]
