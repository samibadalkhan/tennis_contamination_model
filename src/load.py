"""Ingest + schema harmonization for the slam point-by-point corpus.

Responsibilities (see tennis-contamination.md "Data" and D6 "data artifacts"):
  * Filter slam point files to SINGLES (never glob -doubles / -mixed).
  * Harmonize columns by MEANING, not name; verify against
    data/slam_pointbypoint/data_dictionary.txt. Required core columns are
    asserted; genuinely optional ones (e.g. ServeNumber, absent in 2011) are
    carried as reduced-covariate (NaN), never treated as contamination and
    never a reason to drop a row.
  * Filter SENTINELS: PointServer / PointWinner use 0 for warmup / unknown
    (PointNumber "0X"/"0Y" rows). Those rows are dropped for rate estimation;
    the count dropped is reported by the audit, not silently swallowed.
  * Keep retirement / walkover truncation -- it is the Stage 0.5 control. Loading
    never drops a match for being short.
  * Player identity: slam matches files do NOT populate player ids, so identity
    within this corpus is the player NAME string (player1 / player2). Cross-
    corpus joins to ATP (accent/hyphenation normalization) are a Stage 0.5
    concern, not done here.

Snapshot-date mismatch to assert later (slam frozen Oct 2024; ATP/WTA to Jun
2026): the retirement-label join slam -> atp is safe because ATP is a superset,
but ASSERT it -- every slam match_id must resolve to an ATP match; investigate
the unmatched set, do NOT drop. (Deferred to Stage 0.5.)

Output: one tidy point-level DataFrame, one row per *played* point, from the
server's perspective, with columns:
    match_id, slam, year, fold, set_no, game_no, point_no,
    server, returner,            # player-name strings
    server_wins,                 # 1 if server won the point
    serve_number,                # 1/2 or NaN (reduced-covariate)
    is_tiebreak                  # game played at 6-6
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src import splits
from src.util import DATA

SLAM_DIR = DATA / "slam_pointbypoint"

# Core columns that MUST be present (assert). Harmonized by meaning; the slam
# corpus uses these names consistently across years for the core.
CORE = ["match_id", "SetNo", "GameNo", "PointNumber", "PointServer", "PointWinner"]
# Optional (reduced-covariate where missing, e.g. ServeNumber absent in 2011).
OPTIONAL = ["ServeNumber", "P1GamesWon", "P2GamesWon"]

_POINTS_RE = re.compile(r"^(\d{4})-([a-z]+)-points\.csv$")


def singles_point_files() -> list[Path]:
    """All singles ``*-points.csv`` (doubles/mixed excluded by the regex)."""
    return sorted(p for p in SLAM_DIR.glob("*-points.csv")
                  if _POINTS_RE.match(p.name))


def _match_meta(slam: str, year: int) -> dict[str, tuple[str, str, str]]:
    """match_id -> (player1, player2, tour).

    These files carry BOTH men's and women's singles. Tour is read from the
    stable ``match_num`` convention (1xxx = men, 2xxx = women), which is present
    every year; ``event_name`` ("Men's Singles"/"Women's Singles") is only
    populated in early years and is used to cross-check when available. Tour
    matters: serve dominance (~64-65% men vs high-50s women) and match length
    (best-of-5 vs best-of-3) differ, so it must not load onto other effects.
    """
    mf = SLAM_DIR / f"{year}-{slam}-matches.csv"
    m = pd.read_csv(mf, dtype=str)
    out = {}
    for r in m.itertuples():
        tour = "M" if str(r.match_num).startswith("1") else (
            "W" if str(r.match_num).startswith("2") else "?")
        ev = getattr(r, "event_name", None)
        if isinstance(ev, str) and ev.strip():
            tour = "M" if "Men" in ev else ("W" if "Women" in ev else tour)
        out[r.match_id] = (r.player1, r.player2, tour)
    return out


def load_points_file(path: Path, decl: dict | None = None) -> pd.DataFrame:
    """Load and harmonize a single singles points file. Returns played points."""
    decl = decl or splits.declare()
    m = _POINTS_RE.match(path.name)
    year, slam = int(m.group(1)), m.group(2)

    head = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in CORE if c not in head]
    if missing:
        raise ValueError(f"{path.name}: missing required core columns {missing}")
    usecols = [c for c in CORE + OPTIONAL if c in head]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    meta = _match_meta(slam, year)

    # Sentinel filter: keep only played points with a real server and winner.
    server = pd.to_numeric(df["PointServer"], errors="coerce")
    winner = pd.to_numeric(df["PointWinner"], errors="coerce")
    played = server.isin([1, 2]) & winner.isin([1, 2])
    df = df.loc[played].copy()
    server = server[played].astype(int)
    winner = winner[played].astype(int)

    import numpy as np
    p1 = df["match_id"].map(lambda mid: meta.get(mid, (None, None, "?"))[0]).values
    p2 = df["match_id"].map(lambda mid: meta.get(mid, (None, None, "?"))[1]).values
    tour = df["match_id"].map(lambda mid: meta.get(mid, (None, None, "?"))[2]).values
    srv = server.values
    server_name = np.where(srv == 1, p1, p2)
    returner_name = np.where(srv == 1, p2, p1)

    out = pd.DataFrame({
        "match_id": df["match_id"].values,
        "slam": slam,
        "year": year,
        "tour": tour,
        "fold": splits.fold_of(year, decl),
        "set_no": pd.to_numeric(df["SetNo"], errors="coerce").astype("Int64").values,
        "game_no": pd.to_numeric(df["GameNo"], errors="coerce").astype("Int64").values,
        "point_no": df["PointNumber"].astype(str).values,
        "server": server_name,
        "returner": returner_name,
        "server_wins": (winner.values == srv).astype(int),
    })
    if "ServeNumber" in df:
        sn = pd.to_numeric(df["ServeNumber"], errors="coerce")
        out["serve_number"] = sn.where(sn.isin([1, 2])).values
    else:
        out["serve_number"] = pd.NA
    if "P1GamesWon" in df and "P2GamesWon" in df:
        g1 = pd.to_numeric(df["P1GamesWon"], errors="coerce")
        g2 = pd.to_numeric(df["P2GamesWon"], errors="coerce")
        out["is_tiebreak"] = ((g1 == 6) & (g2 == 6)).values
    else:
        out["is_tiebreak"] = False
    # Player-name resolution can fail if a match_id is absent from the matches
    # file; drop those points (they cannot be attributed) but they are rare.
    return out.dropna(subset=["server", "returner"]).reset_index(drop=True)


def load_points(folds: tuple[str, ...] | None = None,
                years: tuple[int, ...] | None = None) -> pd.DataFrame:
    """Load all singles points, optionally restricted to folds and/or years.

    ``folds`` is the guard that keeps Stage 0 off the test set: pass
    ``("train",)`` or ``("train","val")`` and test-year files are never read.
    """
    decl = splits.declare()
    frames = []
    for path in singles_point_files():
        yr = int(_POINTS_RE.match(path.name).group(1))
        if years is not None and yr not in years:
            continue
        if folds is not None and splits.fold_of(yr, decl) not in folds:
            continue
        frames.append(load_points_file(path, decl))
    if not frames:
        raise ValueError("no point files matched the requested folds/years")
    return pd.concat(frames, ignore_index=True)
