"""Materialize a clean, relabeled analysis layer under data/processed/.

WHY THIS EXISTS. Raw player names are recorded inconsistently across years
("Dominic Thiem" pre-2019, "D. Thiem" after; accents, nicknames,
transliterations, maiden/married names), which fragments a player across years
and breaks name-based joins. `load.canonical_name` fixes this at read time, but
anything that touches the raw CSVs directly hits the fragmentation again. This
script bakes the normalization into a derived layer so downstream work reads one
clean, canonical dataset.

WHAT IT DOES NOT DO. It never modifies the raw files. Raw stays byte-for-byte as
fetched and SHA-256-verifiable (`python -m src.fetch --verify`); the processed
layer is DERIVED and fully regenerable from it. Both live under the gitignored
data/ tree — neither raw nor processed data is committed.

Outputs (data/processed/):
  points.parquet             tidy point-level data, canonical names, ALL folds
                             (fold column preserved; test rows labeled, not used)
  slam_matches.parquet       slam match index, canonical players, tour, fold
  tour_slam_matches.parquet  ATP+WTA slam results: canonical winner/loser, RET/WO
                             flags, round, ranks, surface (labels for Stage 0.5,
                             the weak-player test, external validation)
  name_crosswalk.parquet     every raw name -> canonical, with sources + counts
  BUILD.json                 provenance: source SHAs (from raw MANIFEST) + counts
  README.md                  data dictionary for the layer
A small committed summary (incl. collision flags) is written to
results/processed_build.json.

    python -m src.build_processed
"""

from __future__ import annotations

import glob
import re
import sys
from collections import defaultdict

import pandas as pd

from src import load, splits
from src.util import DATA, RESULTS, ensure, read_json, utcnow, write_json

PROC = DATA / "processed"
SLAM_MAP = {"Australian Open": "ausopen", "Roland Garros": "frenchopen",
            "Wimbledon": "wimbledon", "US Open": "usopen"}
_MATCHES_RE = re.compile(r"^(\d{4})-([a-z]+)-matches\.csv$")

# Canonical keys (first-initial + surname) that genuinely collide TWO DISTINCT
# PLAYERS -- twins, siblings, or same-initial opposite-gender pairs. Curated from
# the crosswalk collision audit (the rest of the flagged keys are same-person
# spelling/nickname/transliteration variants that canonicalization correctly
# merges). For these, initial-only records ("K. Pliskova") are INHERENTLY
# ambiguous -- the corpus dropped the first name that distinguishes them -- so
# rows on these keys are marked `ambiguous_identity` rather than silently merged.
# Re-review the audit's flagged list on each build in case new names appear.
KNOWN_COLLISIONS = {
    "a beck",        # Andreas (M) / Annika (W) Beck
    "a kuznetsov",   # Alex / Andrey Kuznetsov
    "a rodionova",   # Anastasia / Arina Rodionova (sisters)
    "c harrison",    # Christian (M) / Catherine (W) Harrison
    "e nava",        # Eduardo / Emilio Nava
    "k pliskova",    # Karolina / Kristyna Pliskova (twins)
    "m gonzalez",    # Maximo (M) / Montserrat (W) Gonzalez
    "x wang",        # Xinyu / Xiyu Wang
    "z zhang",       # Ze / Zhizhen Zhang
}


def build_points() -> dict:
    pts = load.load_points()                         # all folds, canonical names
    pts["ambiguous_identity"] = (pts.server.isin(KNOWN_COLLISIONS)
                                 | pts.returner.isin(KNOWN_COLLISIONS))
    ensure(PROC)
    pts.to_parquet(PROC / "points.parquet")
    return {"rows": int(len(pts)),
            "by_fold": pts.groupby("fold", observed=True).size().to_dict(),
            "by_tour": pts.groupby("tour", observed=True).size().to_dict(),
            "ambiguous_identity_rows": int(pts.ambiguous_identity.sum()),
            "distinct_players_canonical": int(pd.unique(pts[["server", "returner"]].values.ravel()).size)}


def build_slam_matches() -> dict:
    decl = splits.declare()
    rows = []
    for f in sorted(glob.glob(str(DATA / "slam_pointbypoint" / "*-matches.csv"))):
        m = _MATCHES_RE.match(f.split("/")[-1])
        if not m or "doubles" in f or "mixed" in f:
            continue
        year, slam = int(m.group(1)), m.group(2)
        df = pd.read_csv(f, dtype=str)
        for r in df.itertuples():
            rows.append({
                "match_id": r.match_id, "year": year, "slam": slam,
                "fold": splits.fold_of(year, decl),
                "tour": load.tour_of(r.match_num, getattr(r, "event_name", None)),
                "player1": load.canonical_name(r.player1),
                "player2": load.canonical_name(r.player2),
                "player1_raw": r.player1, "player2_raw": r.player2,
            })
    out = pd.DataFrame(rows)
    out["ambiguous_identity"] = (out.player1.isin(KNOWN_COLLISIONS)
                                 | out.player2.isin(KNOWN_COLLISIONS))
    out.to_parquet(PROC / "slam_matches.parquet")
    return {"rows": int(len(out)), "by_tour": out.groupby("tour").size().to_dict(),
            "ambiguous_identity_rows": int(out.ambiguous_identity.sum())}


def build_tour_matches() -> dict:
    decl = splits.declare()
    keep = ["surface", "round", "best_of", "minutes",
            "winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points"]
    rows = []
    for tour, pat in (("M", DATA / "atp" / "atp_matches_*.csv"),
                      ("W", DATA / "wta" / "wta_matches_*.csv")):
        for f in glob.glob(str(pat)):
            y = int(re.search(r"(\d{4})", f).group(1))
            df = pd.read_csv(f, dtype=str)
            if "tourney_level" in df:
                df = df[df.tourney_level == "G"]
            df = df[df.tourney_name.isin(SLAM_MAP)]
            if df.empty:
                continue
            sc = df.score.astype(str)
            for r in df.itertuples():
                s = str(r.score)
                rows.append({
                    "year": y, "slam": SLAM_MAP[r.tourney_name], "tour": tour,
                    "fold": splits.fold_of(y, decl),
                    "winner": load.canonical_name(r.winner_name),
                    "loser": load.canonical_name(r.loser_name),
                    "winner_raw": r.winner_name, "loser_raw": r.loser_name,
                    "is_ret": "RET" in s, "is_walkover": bool(re.search(r"W/O|WO|DEF", s)),
                    "score": s,
                    **{k: getattr(r, k, None) for k in keep},
                })
    out = pd.DataFrame(rows)
    out.to_parquet(PROC / "tour_slam_matches.parquet")
    return {"rows": int(len(out)), "retirements": int(out.is_ret.sum()),
            "walkovers": int(out.is_walkover.sum()),
            "by_tour": out.groupby("tour").size().to_dict()}


def build_crosswalk() -> tuple[dict, list]:
    """raw name -> canonical, across all sources, with counts and a collision audit."""
    counts = defaultdict(lambda: defaultdict(int))     # raw -> source -> count
    # slam
    for f in glob.glob(str(DATA / "slam_pointbypoint" / "*-matches.csv")):
        if "doubles" in f or "mixed" in f:
            continue
        df = pd.read_csv(f, dtype=str)
        for c in ("player1", "player2"):
            for v in df[c].dropna():
                counts[v]["slam"] += 1
    # atp / wta
    for src, pat in (("atp", DATA / "atp" / "atp_matches_*.csv"),
                     ("wta", DATA / "wta" / "wta_matches_*.csv")):
        for f in glob.glob(str(pat)):
            df = pd.read_csv(f, dtype=str)
            if "tourney_level" in df:
                df = df[df.tourney_level == "G"]
            df = df[df.tourney_name.isin(SLAM_MAP)] if "tourney_name" in df else df
            for c in ("winner_name", "loser_name"):
                if c in df:
                    for v in df[c].dropna():
                        counts[v][src] += 1

    rows = []
    variants = defaultdict(list)
    for raw, src_counts in counts.items():
        can = load.canonical_name(raw)
        rows.append({"raw_name": raw, "canonical": can,
                     "sources": ",".join(sorted(src_counts)),
                     "count": int(sum(src_counts.values()))})
        variants[can].append(raw)
    cw = pd.DataFrame(rows).sort_values(["canonical", "raw_name"])
    cw.to_parquet(PROC / "name_crosswalk.parquet")

    # collision audit: canonical keys mapping to >=2 distinct spelled-out first
    # names (a hint at two different people; most are same-person spelling variants).
    def full_firsts(vs):
        fs = set()
        for v in vs:
            toks = v.replace(".", "").split()
            if toks and len(toks[0]) > 1:
                fs.add(toks[0].lower())
        return fs
    flagged = []
    for can, vs in variants.items():
        ff = full_firsts(vs)
        if len(ff) >= 2:
            flagged.append({"canonical": can, "raw_variants": sorted(set(vs)),
                            "distinct_full_first_names": sorted(ff),
                            "known_collision": can in KNOWN_COLLISIONS})
    flagged.sort(key=lambda d: (not d["known_collision"], d["canonical"]))
    summary = {"distinct_raw_names": int(len(cw)),
               "distinct_canonical": int(cw.canonical.nunique()),
               "canonical_keys_with_multiple_raw_variants": int((cw.groupby("canonical").size() > 1).sum()),
               "flagged_possible_collisions": len(flagged),
               "true_collisions_curated": sorted(KNOWN_COLLISIONS),
               "flagged_not_curated_review": [f["canonical"] for f in flagged if not f["known_collision"]]}
    return summary, flagged


def build_readme(reports: dict) -> None:
    md = [
        "# data/processed — derived analysis layer",
        "",
        "**Derived, regenerable, not committed.** Built from the raw corpus by "
        "`python -m src.build_processed`; raw is never modified and stays "
        "SHA-256-verifiable (`python -m src.fetch --verify`). Provenance in "
        "`BUILD.json`.",
        "",
        "The one job of this layer: player names are **canonical** everywhere "
        "(`load.canonical_name`: accent-stripped, lowercased, first-initial + "
        "surname), so a player is never fragmented across years and every join "
        "uses the same key. Raw spellings are retained in the `*_raw` columns and "
        "the crosswalk.",
        "",
        "## Files",
        "| file | rows | what |",
        "|---|---|---|",
        f"| `points.parquet` | {reports['points']['rows']:,} | tidy point-level data, one row per played point, canonical `server`/`returner`; `fold` labels train/val/test (test present but not to be used until final) |",
        f"| `slam_matches.parquet` | {reports['slam_matches']['rows']:,} | slam match index; canonical `player1`/`player2` + `*_raw` |",
        f"| `tour_slam_matches.parquet` | {reports['tour_matches']['rows']:,} | ATP+WTA slam results; canonical `winner`/`loser`, `is_ret`/`is_walkover`, `round`, ranks, `surface` |",
        f"| `name_crosswalk.parquet` | {reports['crosswalk']['distinct_raw_names']:,} | every `raw_name` → `canonical`, with `sources` and `count` |",
        "",
        "## Columns — points.parquet",
        "`match_id, slam, year, tour (M/W), fold, set_no, game_no, point_no, "
        "server, returner, server_wins, serve_number, is_tiebreak`",
        "",
        "## Name normalization audit",
        f"- {reports['crosswalk']['distinct_raw_names']:,} distinct raw names → "
        f"{reports['crosswalk']['distinct_canonical']:,} canonical keys "
        f"({reports['crosswalk']['canonical_keys_with_multiple_raw_variants']:,} keys "
        f"unify 2+ raw spellings).",
        f"- {reports['crosswalk']['flagged_possible_collisions']} keys map to 2+ "
        "spelled-out first names (`results/processed_build.json`). Most are "
        "same-person variants (nicknames/transliterations/maiden names) that "
        "canonicalization correctly merges.",
        "",
        "### True collisions (distinct players sharing initial + surname)",
        f"- {len(reports['crosswalk']['true_collisions_curated'])} curated keys are "
        "genuine two-player collisions (twins/siblings/same-initial pairs): "
        + ", ".join(f"`{k}`" for k in reports['crosswalk']['true_collisions_curated']) + ".",
        "- Full-name records for these still merge under one key, and initial-only "
        "records ('K. Pliskova') are **inherently ambiguous** — the corpus dropped "
        "the distinguishing first name. Rows on these keys carry "
        f"`ambiguous_identity = True` ({reports['points']['ambiguous_identity_rows']:,} "
        "point rows) so downstream can filter or special-case them; they are NOT "
        "silently merged into one player's stats without warning.",
    ]
    (PROC / "README.md").write_text("\n".join(md) + "\n")


def run():
    ensure(PROC)
    reports = {}
    reports["points"] = build_points()
    reports["slam_matches"] = build_slam_matches()
    reports["tour_matches"] = build_tour_matches()
    cw_summary, flagged = build_crosswalk()
    reports["crosswalk"] = cw_summary

    raw_manifest = read_json(DATA / "MANIFEST.json") or {}
    build = {
        "built_utc": utcnow(),
        "generator": "src.build_processed",
        "canonical_name": "accent-stripped, lowercased, first-initial + surname",
        "raw_source_pins": raw_manifest.get("pins"),
        "raw_manifest_generated": raw_manifest.get("generated_utc"),
        "reports": reports,
    }
    write_json(PROC / "BUILD.json", build)
    build_readme(reports)

    # committed summary (small; names limited to the flagged review list)
    write_json(RESULTS / "processed_build.json",
               {"built_utc": utcnow(), "reports": reports,
                "flagged_possible_collisions": flagged})
    print("processed layer written to data/processed/ (gitignored):")
    for k, v in reports.items():
        print(f"  {k}: {v}")
    print(f"committed summary: results/processed_build.json "
          f"({cw_summary['flagged_possible_collisions']} flagged names for review)")


if __name__ == "__main__":
    sys.exit(run())
