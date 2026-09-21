"""The develop / test split -- declared ONCE, up front, chronologically.

New design (Ingram forecasting protocol; see tennis-contamination.md
"Frozen-data protocol"): develop on everything <= 2024, test on the 2025 ATP
season, once. This retires the earlier slam-year split (train 2011-2018 /
val 2019-2020 / test 2021-2024), which belonged to the superseded
contamination-detection design; that split never touched a 2025 fold, so
nothing about the new test is compromised by the change.

Why this split:
  * Mirrors Ingram (2019): hold out a full ATP season, forecast its matches.
  * Chronological and match-clean -- season boundaries never cut a match.
  * Slam point-by-point ends Oct 2024, so ALL point data sits in development.
    The hybrid arm (point terms replacing the ATP match term for slam matches)
    is therefore a development-time signal; the 2025 test is match-level.
  * The corpus is frozen and the test is SINGLE-USE. Validation is done inside
    development by rolling-origin evaluation, never on 2025.

Uncertainty is clustered by PLAYER-TOURNAMENT (tournament-level as a
conservative check) -- not by match.

``declare()`` persists this to ``results/splits.json``. Re-running is a no-op;
changing the declared design requires ``force=True`` (a logged, deliberate act,
as when this replaced the slam-year split), never a silent re-run.
"""

from __future__ import annotations

from src.util import RESULTS, read_json, utcnow, write_json

SPLITS_FILE = RESULTS / "splits.json"

# --- the declaration -----------------------------------------------------------
DEV_MAX_YEAR = 2024          # develop on everything up to and including 2024
TEST_SEASON = 2025           # hold out the 2025 ATP season, forecast it once
# 2026 exists in the ATP mirror but is partial; excluded from both.
CLUSTER_UNIT = "player_tournament"

RATIONALE = (
    "Ingram protocol: develop on <=2024, test on the 2025 ATP season once. "
    "Chronological and match-clean (season boundaries never split a match). "
    "Slam point-by-point ends Oct 2024, so all point data is in development and "
    "feeds the hybrid arm; the 2025 test is match-level. Validation is by "
    "rolling-origin within development -- the 2025 test is single-use and never "
    "tuned on. Uncertainty clustered by player-tournament. This replaces the "
    "superseded slam-year split (train 2011-2018 / val 2019-2020 / test "
    "2021-2024) from the contamination-detection design."
)


def _decl() -> dict:
    return {
        "declared_utc": utcnow(),
        "design": "ingram_forecasting_robust_vs_ordinary",
        "unit": "chronological by season (match-clean); test = one held-out ATP season",
        "dev_max_year": DEV_MAX_YEAR,
        "test_season": TEST_SEASON,
        "validation": "rolling-origin within development (<=2024); never on test",
        "cluster_unit": CLUSTER_UNIT,
        "rationale": RATIONALE,
        "frozen_dataset": (
            "corpus frozen (ATP to Jun 2026, slam points to Oct 2024); the 2025 "
            "test season is the single-use out-of-sample period. An underpowered "
            "result is permanent."
        ),
        "supersedes": "slam-year split (train 2011-2018 / val 2019-2020 / test 2021-2024)",
    }


def declare(force: bool = False) -> dict:
    """Write the split declaration once. Refuses to overwrite unless ``force``."""
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
    """Map a calendar year to its fold under the frozen split."""
    d = decl or load()
    y = int(year)
    if y <= d["dev_max_year"]:
        return "develop"
    if y == d["test_season"]:
        return "test"
    return "excluded"        # e.g. partial 2026
