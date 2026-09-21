"""Blocking and the per-slam-year audit.

Default block = a player's service points WITHIN A SET (~30-35 points): at
p~0.63 the block-rate SD is ~0.08, so a 20pp collapse is ~2.5 SD -- detectable
in aggregate. Game-level blocks (~6-8 points, SD ~0.18) are for diagnostics
only and exclude the tiebreak, which breaks the one-server-per-game invariant
(serve alternates within it). Set-level blocks keep tiebreak points, since the
block is keyed on the actual server of each point, not on a constant game server.

Per-block labels are NOT the deliverable -- EM later recovers epsilon from the
aggregate shape of the block-rate distribution, not by classifying blocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def set_blocks(points: pd.DataFrame) -> pd.DataFrame:
    """Per-(match, set, server) service block. The Stage 0 default unit."""
    g = (points.groupby(["match_id", "slam", "year", "tour", "fold", "set_no", "server"],
                         observed=True)["server_wins"]
         .agg(n="count", wins="sum").reset_index())
    g["rate"] = g["wins"] / g["n"]
    return g


def game_blocks(points: pd.DataFrame) -> pd.DataFrame:
    """Per-(match, set, game, server) block, tiebreaks excluded. Diagnostics only."""
    pts = points.loc[~points["is_tiebreak"].astype(bool)]
    g = (pts.groupby(["match_id", "slam", "year", "tour", "fold", "set_no", "game_no", "server"],
                     observed=True)["server_wins"]
         .agg(n="count", wins="sum").reset_index())
    g["rate"] = g["wins"] / g["n"]
    return g


def audit(points: pd.DataFrame) -> pd.DataFrame:
    """Per-slam-year audit: coverage, sizes, missingness, serve rate.

    The signals that matter for D6: is serve_number present (schema grows over
    time), how big are matches/sets, and does the raw serve rate move by
    slam x year (surface / recording differences that must not load onto the
    contamination component).
    """
    rows = []
    for (slam, year, tour), d in points.groupby(["slam", "year", "tour"], observed=True):
        setb = set_blocks(d)
        ppmatch = d.groupby("match_id", observed=True).size()
        ppset = setb["n"]
        rows.append({
            "slam": slam,
            "year": int(year),
            "tour": tour,
            "fold": d["fold"].iloc[0],
            "n_matches": int(d["match_id"].nunique()),
            "n_points": int(len(d)),
            "n_set_blocks": int(len(setb)),
            "serve_win_rate": round(float(d["server_wins"].mean()), 4),
            "serve_number_missing_frac": round(float(d["serve_number"].isna().mean()), 4),
            "tiebreak_point_frac": round(float(d["is_tiebreak"].astype(bool).mean()), 4),
            "points_per_match_median": float(np.median(ppmatch)),
            "points_per_set_block_median": float(np.median(ppset)),
            "points_per_set_block_p10": float(np.percentile(ppset, 10)),
            "points_per_set_block_p90": float(np.percentile(ppset, 90)),
        })
    return pd.DataFrame(rows).sort_values(["year", "slam", "tour"]).reset_index(drop=True)
