"""Stage 0.5 -- retirement positive control. RUN BEFORE BUILDING ANYTHING ELSE.

The points in the games/sets preceding a retirement are the one near-certain
contaminated sample available. If the detector cannot see service-win-rate
degradation on KNOWN-contaminated data, it does not work -- and we learn that in
hours, not weeks. So: does a retiree's service-win rate visibly degrade in the
lead-up to the retirement, relative to that same player's baseline earlier in
the same match -- and by MORE than an ordinary match-loser (who also tends to
serve a bit worse late)?

Labels come from OUTSIDE the point model: RET in the ATP/WTA score strings, the
loser being the retiree. Joined to the slam point corpus by the canonical
name key (see load.canonical_name) -- the corpus's full-name/initial-surname
split otherwise breaks the join for 2019+. The unmatched set is audited, not
ignored (D6: coverage is Hawkeye-court-only and misses first-rounders, which is
where retirements cluster -- this biases the available sample, so we report it).

Uses TRAIN+VAL years only; the test fold is never touched. Every result is
checkpointed to results/stage05/ with match-clustered bootstrap CIs, and every
variant is appended to the trial log.

    python -m src.stage05            # run/resume
    python -m src.stage05 --fresh
"""

from __future__ import annotations

import argparse
import glob
import re
import sys

import numpy as np
import pandas as pd

from src import evaluate as ev, load, splits, trials
from src.util import DATA, RESULTS, ensure, exists_nonempty, read_json, utcnow, write_json

STAGE05 = RESULTS / "stage05"
CACHE = DATA / "cache"
SLAM_MAP = {"Australian Open": "ausopen", "Roland Garros": "frenchopen",
            "Wimbledon": "wimbledon", "US Open": "usopen"}
MIN_BASE, MIN_LEAD, LASTK = 10, 5, 15   # min service points to include a match; last-k window
STEPS = ["labels", "degradation", "report"]


def _out(step):
    return STAGE05 / f"{step}.json"


def _done(step):
    return exists_nonempty(_out(step))


# --- labels from ATP / WTA -----------------------------------------------------

def _scan(paths, years, tour, want):
    """want='RET' -> retirements (loser=retiree); want='completed' -> plain losers."""
    rows = []
    for p in paths:
        m0 = re.search(r"(\d{4})", p)
        y = int(m0.group(1))
        if y not in years:
            continue
        m = pd.read_csv(p, dtype=str)
        if "tourney_level" in m:
            m = m[m.tourney_level == "G"]
        m = m[m.tourney_name.isin(SLAM_MAP)]
        sc = m.score.astype(str)
        is_ret = sc.str.contains("RET", na=False)
        is_wo = sc.str.contains(r"W/O|WO|DEF", na=False, regex=True)
        sel = is_ret if want == "RET" else (~is_ret & ~is_wo)
        for t in m[sel].itertuples():
            rows.append({
                "year": y, "slam": SLAM_MAP[t.tourney_name], "tour": tour,
                "retiree": load.canonical_name(t.loser_name),
                "opponent": load.canonical_name(t.winner_name),
                "round": getattr(t, "round", None),
                "retiree_rank": getattr(t, "loser_rank", None),
                "opp_rank": getattr(t, "winner_rank", None),
            })
    return rows


def load_labels(years):
    atp = glob.glob(str(DATA / "atp" / "atp_matches_*.csv"))
    wta = glob.glob(str(DATA / "wta" / "wta_matches_*.csv"))
    ret = _scan(atp, years, "M", "RET") + _scan(wta, years, "W", "RET")
    comp = _scan(atp, years, "M", "completed") + _scan(wta, years, "W", "completed")
    return pd.DataFrame(ret), pd.DataFrame(comp)


def step_labels():
    decl = splits.declare()
    years = set(decl["train_years"] + decl["val_years"])          # never test
    ret, comp = load_labels(years)
    summary = {"utc": utcnow(), "years": sorted(years),
               "n_retirements": int(len(ret)), "n_completed_losers": int(len(comp)),
               "retirements_by_tour": ret.groupby("tour").size().to_dict(),
               "retirements_by_year": ret.groupby("year").size().to_dict()}
    ret.to_parquet(CACHE / "ret_labels.parquet")
    comp.to_parquet(CACHE / "completed_labels.parquet")
    write_json(_out("labels"), summary)
    trials.log(stage="0.5", kind="labels",
               metrics={"n_ret": len(ret), "n_completed": len(comp)},
               note="RET from ATP/WTA score strings; loser=retiree; canonical-name join")
    return summary


# --- join + degradation --------------------------------------------------------

def _pair_map(pts):
    """(year, slam, frozenset(players)) -> match_id, from the point corpus."""
    out = {}
    for mid, g in pts.groupby("match_id", observed=True):
        players = frozenset(pd.unique(g[["server", "returner"]].values.ravel()))
        out[(int(g.year.iloc[0]), g.slam.iloc[0], players)] = mid
    return out


def _degrade(gm: pd.DataFrame, player: str):
    """Serve-win rate for `player` in this match: last-set and last-k vs baseline.

    gm must be in chronological order (row order in the corpus). Returns None if
    there are too few service points to compare.
    """
    s = gm[gm.server == player]
    if len(s) < MIN_BASE + MIN_LEAD:
        return None
    out = {}
    # last-set window
    last = int(s.set_no.max())
    lead_ls = s[s.set_no == last].server_wins
    base_ls = s[s.set_no < last].server_wins
    if len(base_ls) >= MIN_BASE and len(lead_ls) >= MIN_LEAD:
        out["last_set"] = (float(base_ls.mean()), float(lead_ls.mean()),
                           int(len(base_ls)), int(len(lead_ls)))
    # last-k window
    y = s.server_wins.to_numpy()
    if len(y) >= LASTK + MIN_BASE:
        out["last_k"] = (float(y[:-LASTK].mean()), float(y[-LASTK:].mean()),
                         len(y) - LASTK, LASTK)
    return out or None


def _aggregate(deltas: list[float], seed=0):
    arr = np.array(deltas, dtype=float)
    est, lo, hi, se = ev.clustered_ci({"sum_x": arr, "n": np.ones_like(arr)}, ev._mean,
                                      seed=seed)
    return {"mean_degradation": round(est, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n_matches": int(len(arr))}


def step_degradation():
    pts = pd.read_parquet(CACHE / "points_trainval.parquet")
    ret = pd.read_parquet(CACHE / "ret_labels.parquet")
    comp = pd.read_parquet(CACHE / "completed_labels.parquet")
    pm = _pair_map(pts)

    def resolve(df):
        mids, players, unmatched = [], [], 0
        for r in df.itertuples():
            key = (int(r.year), r.slam, frozenset([r.retiree, r.opponent]))
            mid = pm.get(key)
            if mid is None:
                unmatched += 1
            else:
                mids.append(mid); players.append(r.retiree)
        return mids, players, unmatched

    ret_mids, ret_players, ret_unmatched = resolve(ret)
    comp_mids, comp_players, comp_unmatched = resolve(comp)

    # index points by match once for speed
    by_match = {mid: g for mid, g in pts.groupby("match_id", observed=True)}

    def collect(mids, players, window):
        deltas, base_r, lead_r = [], [], []
        for mid, pl in zip(mids, players):
            gm = by_match.get(mid)
            if gm is None:
                continue
            d = _degrade(gm, pl)
            if d and window in d:
                b, l, nb, nl = d[window]
                deltas.append(b - l); base_r.append(b); lead_r.append(l)
        return deltas, base_r, lead_r

    result = {"utc": utcnow(),
              "join": {"retirements_matched": len(ret_mids),
                       "retirements_unmatched": ret_unmatched,
                       "completed_matched": len(comp_mids),
                       "completed_unmatched": comp_unmatched},
              "windows": {}}
    for window in ("last_set", "last_k"):
        rd, rb, rl = collect(ret_mids, ret_players, window)
        cd, cb, cl = collect(comp_mids, comp_players, window)
        ret_agg = _aggregate(rd)
        comp_agg = _aggregate(cd)
        excess = ret_agg["mean_degradation"] - comp_agg["mean_degradation"]
        result["windows"][window] = {
            "retirement": {**ret_agg,
                           "baseline_serve_rate": round(float(np.mean(rb)), 4),
                           "leadup_serve_rate": round(float(np.mean(rl)), 4)},
            "control_losers": {**comp_agg,
                               "baseline_serve_rate": round(float(np.mean(cb)), 4),
                               "leadup_serve_rate": round(float(np.mean(cl)), 4)},
            "excess_over_control": round(excess, 4),
        }
        trials.log(stage="0.5", kind="degradation", params={"window": window},
                   metrics={"ret_degradation": ret_agg["mean_degradation"],
                            "control_degradation": comp_agg["mean_degradation"],
                            "excess": excess})

    # verdict: retirement degradation CI clears 0 AND exceeds the control
    ls = result["windows"]["last_set"]["retirement"]
    fires = ls["ci95"][0] > 0 and result["windows"]["last_set"]["excess_over_control"] > 0
    result["verdict"] = ("FIRES: retirees degrade in the lead-up beyond ordinary "
                         "losers -> the detector works on known-contaminated data; "
                         "proceed to Stage 1." if fires else
                         "DOES NOT FIRE: no clear degradation beyond the loser "
                         "control -> per the spec, STOP and rethink the detector "
                         "before building Stages 1+.")
    write_json(_out("degradation"), result)
    return result


# --- report --------------------------------------------------------------------

def step_report():
    lab = read_json(_out("labels")); deg = read_json(_out("degradation"))
    j = deg["join"]
    ls = deg["windows"]["last_set"]; lk = deg["windows"]["last_k"]

    def line(w, name):
        r, c = w["retirement"], w["control_losers"]
        return (f"- **{name}**: retiree serve-win {r['baseline_serve_rate']} → "
                f"{r['leadup_serve_rate']} (Δ={r['mean_degradation']}, CI {r['ci95']}, "
                f"n={r['n_matches']}); control losers Δ={c['mean_degradation']} "
                f"CI {c['ci95']} (n={c['n_matches']}); **excess = {w['excess_over_control']}**.")

    md = [
        "# Stage 0.5 report — retirement positive control",
        "", f"_generated {utcnow()} · trials logged: {trials.count()}_", "",
        "**Does the detector see degradation on known-contaminated data?** Points "
        "preceding a retirement are the one near-certain contaminated sample. If "
        "service-win rate does not visibly degrade there, the detector does not "
        "work and we stop. Labels are external (RET in ATP/WTA scores); test fold "
        "untouched.", "",
        "## Labels & join",
        f"- {lab['n_retirements']} retirements (by tour {lab['retirements_by_tour']}) "
        f"and {lab['n_completed_losers']} completed-match losers (control) in "
        f"{lab['years'][0]}–{lab['years'][-1]}.",
        f"- Joined to point corpus by canonical name: **{j['retirements_matched']} "
        f"retirements matched**, {j['retirements_unmatched']} unmatched "
        f"(coverage: Hawkeye-court only, first-rounders missing — biases the "
        f"available retirement sample, a known D6 limitation).",
        "",
        "## Degradation (retiree serve-win rate, lead-up vs earlier; clustered CI)",
        line(ls, "last set played"),
        line(lk, f"last {LASTK} service points"),
        "",
        f"## Verdict",
        f"**{deg['verdict']}**",
        "",
        "## Caveats",
        "- Within-match baseline (earlier serve points) vs lead-up; the loser "
        "control absorbs the generic 'losers serve worse late' effect, so the "
        "excess is the contamination-specific signal.",
        "- Retirement coverage is biased by Hawkeye-only recording (first-round "
        "retirements missing). The positive control still validates the detector; "
        "it does not estimate population ε.",
        "- This is detector validation, not a test of the contamination hypothesis "
        "(that is Stages 2+).",
    ]
    (STAGE05 / "stage05_report.md").write_text("\n".join(md) + "\n")
    write_json(_out("report"), {"utc": utcnow(), "verdict": deg["verdict"],
                                "fires": deg["verdict"].startswith("FIRES")})
    return {"report": str(STAGE05 / "stage05_report.md")}


def run(fresh=False):
    ensure(STAGE05)
    if not exists_nonempty(CACHE / "points_trainval.parquet"):
        raise SystemExit("run Stage 0 first (need data/cache/points_trainval.parquet)")
    if fresh:
        for s in STEPS:
            _out(s).unlink(missing_ok=True)
    if not _done("labels"):
        step_labels()
    if not _done("degradation"):
        step_degradation()
    step_report()
    print("Stage 0.5 complete. See results/stage05/stage05_report.md")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args(argv)
    run(fresh=args.fresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
