"""The train / validation / test split -- declared ONCE, up front, by TIME.

Non-negotiable (CLAUDE.md): split by match and by time, never by point; ability
drifts, so train early and test late; declare the split once and never re-split
after seeing results. Because year boundaries never cut through a match (a match
lives in one slam-year), a by-year split is automatically match-clean.

The dataset is FROZEN (slam point-by-point ends Oct 2024), so the TEST fold here
is the only out-of-sample period this project will ever have. Stage 0 does NOT
touch it -- Stage 0 fits on TRAIN and measures the noise floor on VALIDATION.
Test is reserved for the final full-mixture-vs-single-component comparison.

``declare()`` persists this to ``results/splits.json`` exactly once and refuses
to overwrite it, so the declaration is a durable fact, not something re-derived
(and silently changed) each run.
"""

from __future__ import annotations

from src.util import RESULTS, read_json, utcnow, write_json

SPLITS_FILE = RESULTS / "splits.json"

# --- the declaration -----------------------------------------------------------
TRAIN_YEARS = list(range(2011, 2019))   # 2011-2018, all four slams each year
VAL_YEARS = [2019, 2020]                # 2020 missing Wimbledon (cancelled)
TEST_YEARS = [2021, 2022, 2023, 2024]   # 2022-2024 carry only two slams each

RATIONALE = (
    "Chronological split, train-early/test-late, to respect ability drift. "
    "Boundaries fall on year gaps so no match spans folds (match-clean by "
    "construction). Coverage is uneven: TRAIN 2011-2018 has all four slams per "
    "year; VAL 2019-2020 is missing 2020 Wimbledon; TEST 2021-2024 has all four "
    "in 2021 but only two slams each in 2022-2024, skewing the late test years "
    "toward grass/hard. Consequence: never compare raw rates across folds "
    "without conditioning on slam x year fixed effects, and report per-slam "
    "cross-checks. The split is frozen once written; the test fold is the only "
    "out-of-sample period available and Stage 0 does not touch it."
)


def _decl() -> dict:
    return {
        "declared_utc": utcnow(),
        "unit": "by year (match-clean) and by time (train early, test late)",
        "train_years": TRAIN_YEARS,
        "val_years": VAL_YEARS,
        "test_years": TEST_YEARS,
        "rationale": RATIONALE,
        "frozen_dataset": (
            "slam point-by-point ends 2024-10; test fold is the only OOS period "
            "this project will ever have -- declared once, never re-split."
        ),
    }


def declare(force: bool = False) -> dict:
    """Write the split declaration once. Refuses to overwrite unless ``force``.

    Re-splitting after seeing results is forbidden, so overwriting requires an
    explicit, logged decision (``force=True``) -- never a silent re-run.
    """
    existing = read_json(SPLITS_FILE)
    if existing is not None and not force:
        return existing
    d = _decl()
    write_json(SPLITS_FILE, d)
    return d


def load() -> dict:
    d = read_json(SPLITS_FILE)
    if d is None:
        raise FileNotFoundError(
            "results/splits.json missing; call splits.declare() first "
            "(the split must be declared before any fitting).")
    return d


def fold_of(year: int, decl: dict | None = None) -> str:
    d = decl or load()
    if year in d["train_years"]:
        return "train"
    if year in d["val_years"]:
        return "val"
    if year in d["test_years"]:
        return "test"
    return "unassigned"
