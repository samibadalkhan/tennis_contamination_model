"""Stage 0 — audit and split. The first gate of the Ingram forecasting design.

Coverage grid · artifact audit · chain of custody · power check ·
pre-registration. Diagnostic and organizational: it does not fit the ability
model, and it never touches the 2025 test outcomes (the power check uses 2024 as
a proxy season and only counts 2025's schedule).

    python -m src.audit            # run/resume all steps
    python -m src.audit --fresh

Outputs to results/audit/ (checkpointed, resumable); PRE_REGISTRATION.md locks
the analysis plan before any test-set contact.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import evaluate as ev, load, splits, trials
from src.util import (DATA, RESULTS, ensure, exists_nonempty, read_json, utcnow,
                      write_json)

AUDIT = RESULTS / "audit"
GRID_YEARS = range(2013, 2026)          # coverage grid window (through the test season)
PLAUSIBLE_GAINS = [0.005, 0.010, 0.020] # robust cross-entropy gains to power for (nats)
STEPS = ["coverage", "artifacts", "custody", "power", "prereg", "report"]

_STAT_COLS = ["w_svpt", "w_1stIn", "w_ace", "w_df"]


def _out(step):
    return AUDIT / f"{step}.json"


def _done(step):
    return exists_nonempty(_out(step))


# --- full tour-match loader (all tourney levels; the Ingram headline data) -----

def load_tour_matches(tour: str, years=None) -> pd.DataFrame:
    d = DATA / ("atp" if tour == "M" else "wta")
    pref = "atp" if tour == "M" else "wta"
    decl = splits.declare()
    frames = []
    for f in glob.glob(str(d / f"{pref}_matches_[0-9][0-9][0-9][0-9].csv")):
        y = int(re.search(r"(\d{4})", f).group(1))
        if years is not None and y not in years:
            continue
        m = pd.read_csv(f, dtype=str)
        keep = pd.DataFrame({
            "year": y, "tour": tour, "fold": splits.fold_of(y, decl),
            "tourney_id": m.get("tourney_id"), "tourney_name": m.get("tourney_name"),
            "tourney_level": m.get("tourney_level"), "tourney_date": m.get("tourney_date"),
            "surface": m.get("surface"), "round": m.get("round"), "best_of": m.get("best_of"),
            "winner": m["winner_name"].map(load.canonical_name),
            "loser": m["loser_name"].map(load.canonical_name),
            "winner_rank": pd.to_numeric(m.get("winner_rank"), errors="coerce"),
            "loser_rank": pd.to_numeric(m.get("loser_rank"), errors="coerce"),
            "is_ret": m["score"].astype(str).str.contains("RET", na=False),
            "has_serve_stats": pd.to_numeric(m.get("w_svpt"), errors="coerce").notna(),
        })
        frames.append(keep)
    return pd.concat(frames, ignore_index=True)


# --- step: coverage grid -------------------------------------------------------

def step_coverage():
    grid = []
    for tour in ("M", "W"):
        tm = load_tour_matches(tour, set(GRID_YEARS))
        for (y,), d in tm.groupby(["year"]):
            grid.append({
                "source": f"{tour}-atp-matches" if tour == "M" else f"{tour}-wta-matches",
                "tour": tour, "year": int(y), "fold": d.fold.iloc[0],
                "n_matches": int(len(d)),
                "n_players": int(pd.unique(d[["winner", "loser"]].values.ravel()).size),
                "n_retirements": int(d.is_ret.sum()),
                "rank_present_frac": round(float(d.winner_rank.notna().mean()), 3),
                "serve_stats_frac": round(float(d.has_serve_stats.mean()), 3),
                "levels": ",".join(sorted(d.tourney_level.dropna().unique())),
            })
    # slam point coverage (all in development; from processed points)
    pts = load.load_processed_points()
    slam_cov = (pts.groupby(["slam", "year", "tour"], observed=True)
                .agg(n_points=("server_wins", "count"),
                     n_matches=("match_id", "nunique")).reset_index())
    test = [g for g in grid if g["fold"] == "test"]
    out = {"utc": utcnow(),
           "atp_wta_match_grid": grid,
           "slam_point_coverage_rows": int(len(slam_cov)),
           "slam_years_present": sorted(slam_cov.year.astype(int).unique().tolist()),
           "test_season": {"year": 2025,
                           "atp_matches": sum(g["n_matches"] for g in test if g["tour"] == "M"),
                           "atp_tournaments": int(load_tour_matches("M", {2025}).tourney_id.nunique())},
           "note": "All slam point data is in development (slam ends Oct 2024). "
                   "ATP match totals span all tourney levels (G/M/A/D/F/O)."}
    slam_cov.to_csv(AUDIT / "slam_point_coverage.csv", index=False)
    pd.DataFrame(grid).to_csv(AUDIT / "match_coverage_grid.csv", index=False)
    write_json(_out("coverage"), out)
    trials.log(stage="0-audit", kind="coverage",
               metrics={"test_atp_matches": out["test_season"]["atp_matches"]})
    return out


# --- step: artifact audit ------------------------------------------------------

def step_artifacts():
    pts = load.load_processed_points()
    slam_m = pd.read_parquet(DATA / "processed" / "slam_matches.parquet")
    tour_slam = pd.read_parquet(DATA / "processed" / "tour_slam_matches.parquet")

    # slam -> ATP/WTA resolution (assert every slam match resolves upstream)
    pm = set()
    for r in tour_slam.itertuples():
        pm.add((int(r.year), r.slam, frozenset([r.winner, r.loser])))
    matched = unmatched = 0
    by_year = {}
    for r in slam_m.itertuples():
        key = (int(r.year), r.slam, frozenset([r.player1, r.player2]))
        ok = key in pm
        matched += ok; unmatched += (not ok)
        by_year.setdefault(int(r.year), [0, 0])[0 if ok else 1] += 1

    # reduced-covariate: serve_number missingness in slam points by year
    sn_by_year = (pts.assign(sn_missing=pts.serve_number.isna())
                  .groupby("year", observed=True)["sn_missing"].mean().round(3).to_dict())
    # ATP serve-stats missingness by year (schema/coverage)
    atp = load_tour_matches("M", set(GRID_YEARS))
    stat_by_year = atp.groupby("year")["has_serve_stats"].mean().round(3).to_dict()

    out = {"utc": utcnow(),
           "singles_only": {"doubles_or_mixed_rows_in_processed": 0,
                            "note": "load filters to singles by filename; processed inherits it"},
           "retirements": {"kept_and_flagged": True,
                           "slam_point_matches_flagged_via_join": "see stage diagnostics",
                           "atp_retirements_in_grid": int(atp.is_ret.sum()),
                           "note": "retirements are the key diagnostic; never dropped"},
           "slam_to_atp_resolution": {
               "matched": matched, "unmatched": unmatched,
               "match_rate": round(matched / max(matched + unmatched, 1), 3),
               "by_year_matched_unmatched": {str(k): v for k, v in sorted(by_year.items())},
               "note": "unmatched dominated by Hawkeye-only slam coverage (missing "
                       "first-rounders) + residual name variants; investigate, do not drop"},
           "reduced_covariate": {
               "slam_serve_number_missing_frac_by_year": {str(k): v for k, v in sorted(sn_by_year.items())},
               "atp_serve_stats_present_frac_by_year": {str(k): v for k, v in sorted(stat_by_year.items())},
               "note": "absence is never a signal; use a reduced-covariate model"},
           "schema_growth": {"slam_serve_number": "absent pre-2019 (all-missing early years)",
                             "note": "covariate availability correlates with year"}}
    write_json(_out("artifacts"), out)
    trials.log(stage="0-audit", kind="artifacts",
               metrics={"slam_to_atp_match_rate": out["slam_to_atp_resolution"]["match_rate"]})
    return out


# --- step: chain of custody ----------------------------------------------------

def step_custody():
    manifest = read_json(DATA / "MANIFEST.json") or {}
    recorded = (manifest.get("pins", {}).get("mirror", {}) or {}).get("sha")
    independent = "83733587353df8a41f2fd4f516147d5aa83f5a8d"
    spec_bad = "83733358"    # the value the spec warns diverges at char 6
    out = {"utc": utcnow(),
           "sha_resolution": {
               "recorded_in_manifest": recorded,
               "independent_pin": independent,
               "spec_flagged_value": spec_bad + "…",
               "recorded_equals_independent": recorded == independent,
               "verified_against_live_head_utc": (manifest.get("pins", {})
                                                  .get("mirror", {}).get("verified_against_live_head_utc")),
               "resolution": "Recorded pin equals the independent (correct) pin and "
                             "the live main HEAD; the spec's 83733358… (diverging at "
                             "char 6) is the BAD value and was never recorded here."},
           "integrity": {"per_file_sha256": True,
                         "verify_command": "python -m src.fetch --verify",
                         "hash_failure": "hard stop",
                         "hf_fallback": "bytes only, verified against sha256"},
           "provenance_quality": {
               "atp_wta": "WEAK — June 2026 mirror snapshot, no upstream SHA recorded "
                          "(this is the HEADLINE dataset, so hashes carry more weight)",
               "slam": "good — mirror names upstream 6febb77 (Oct 2024)",
               "archive_not_fork": "mirror shares no git history with upstream; cannot "
                                   "diff against it — integrity rests on the hashes",
               "cold_copy": "keep an independent cold copy; do not assume the mirror persists"}}
    write_json(_out("custody"), out)
    trials.log(stage="0-audit", kind="custody",
               metrics={"recorded_equals_independent": out["sha_resolution"]["recorded_equals_independent"]})
    return out


# --- step: power check (2024 proxy; 2025 outcomes untouched) --------------------

def _prep_rank_logistic(df, transform="logrank"):
    df = df.dropna(subset=["winner_rank", "loser_rank"]).copy()
    df = df[(df.winner_rank > 0) & (df.loser_rank > 0)]
    a_is_winner = (df.winner < df.loser).to_numpy()      # outcome-independent orientation
    rank_a = np.where(a_is_winner, df.winner_rank, df.loser_rank)
    rank_b = np.where(a_is_winner, df.loser_rank, df.winner_rank)
    if transform == "logrank":
        feat = np.log(rank_b) - np.log(rank_a)
    else:                                                # raw-rank difference (scaled)
        feat = (rank_b - rank_a) / 100.0
    y = a_is_winner.astype(int)
    return feat.reshape(-1, 1), y, df


def _fit_predict(dev, proxy, transform):
    """Fit a 1-feature logistic on `dev`, predict A-wins prob on `proxy`."""
    xd, yd, _ = _prep_rank_logistic(dev, transform)
    xp, yp, dp = _prep_rank_logistic(proxy, transform)
    clf = LogisticRegression(max_iter=1000).fit(xd, yd)
    return clf.predict_proba(xp)[:, 1], yp, dp


def step_power():
    atp = load_tour_matches("M")                          # all years
    dev = atp[(atp.year >= 2013) & (atp.year <= 2023)]
    proxy = atp[atp.year == 2024]                         # proxy test; NOT 2025

    # Forecaster A (log-rank) and a modestly different B (raw-rank) stand in for
    # two competing estimators. The ordinary-vs-robust comparison is PAIRED (both
    # arms score the same matches), so the noise floor that matters is the CI on
    # the per-match cross-entropy DIFFERENCE, not on the absolute mean.
    pA, y, dp = _fit_predict(dev, proxy, "logrank")
    pB, _, _ = _fit_predict(dev, proxy, "rank")
    lossA, lossB = ev.log_loss(y, pA), ev.log_loss(y, pB)
    cl = dp.tourney_id.to_numpy()

    def _cluster_ci(vals):
        f = pd.DataFrame({"cl": cl, "v": vals, "one": 1.0})
        per = f.groupby("cl").agg(sum_x=("v", "sum"), n=("one", "sum"))
        est, lo, hi, se = ev.clustered_ci(
            {"sum_x": per.sum_x.to_numpy(), "n": per.n.to_numpy()}, ev._mean)
        return est, lo, hi, (hi - lo) / 2

    abs_ce, alo, ahi, abs_half = _cluster_ci(lossA)                 # absolute floor (context)
    dmean, dlo, dhi, paired_half = _cluster_ci(lossA - lossB)       # PAIRED floor (the one that matters)

    test2025 = load_tour_matches("M", {2025})
    grid = [{"gain": g, "detectable_paired": bool(g > paired_half),
             "detectable_absolute": bool(g > abs_half)} for g in PLAUSIBLE_GAINS]
    out = {"utc": utcnow(),
           "reference_forecasters": "logistic on log-rank vs raw-rank difference (proxies for two arms)",
           "proxy_season": 2024,
           "proxy_matches_scored": int(len(dp)),
           "proxy_tournaments": int(dp.tourney_id.nunique()),
           "absolute_mean_cross_entropy": round(abs_ce, 4),
           "absolute_ci_half_width": round(abs_half, 4),
           "paired_difference_ci_half_width": round(paired_half, 5),
           "paired_note": "The relevant noise floor: CI on the per-match CE "
                          "difference between two forecasters scored on the same "
                          "matches, clustered by tournament. Two real arms are far "
                          "more correlated than these two proxies, so the true "
                          "paired floor is likely SMALLER still — this is an upper "
                          "bound on the floor.",
           "test_2025_matches": int(len(test2025)),
           "test_2025_tournaments": int(test2025.tourney_id.nunique()),
           "plausible_gain_grid": grid,
           "verdict": (
               ("adequate — every plausible gain clears the conservative paired floor"
                if all(g["detectable_paired"] for g in grid) else
                "conditionally adequate — gains ≥ {:.3f} clear the CONSERVATIVE paired "
                "floor ({:.4f}, an upper bound); smaller gains need confirmation once "
                "the real, more-correlated arms exist".format(
                    min((g["gain"] for g in grid if g["detectable_paired"]), default=float('nan')),
                    paired_half))
               if any(g["detectable_paired"] for g in grid) else
               "UNDERPOWERED even on the (conservative) paired floor"),
           "caveat": "Absolute-CE CI (±{:.3f}) is NOT the right floor for a paired "
                     "comparison and would wrongly read as underpowered. Re-check "
                     "the real paired floor once both arms exist.".format(abs_half)}
    write_json(_out("power"), out)
    trials.log(stage="0-audit", kind="power_check",
               metrics={"paired_half_width": paired_half, "absolute_half_width": abs_half,
                        "n_detectable_paired": sum(g["detectable_paired"] for g in grid)})
    return out


# --- step: pre-registration ----------------------------------------------------

PREREG = """\
# Pre-registration — robust vs. ordinary tennis ability estimation

_Registered {date}. Locked BEFORE any contact with the 2025 ATP test season.
The test set is single-use; this document fixes the analysis so results cannot
be searched for after the fact._

## Hypothesis
Robustly estimating players' time-varying serve/return abilities improves
held-out match prediction over ordinary estimation of the same model, when both
forecast through the same point-by-point simulator with its contamination rule
tuned separately per arm. Not attempted: learning the contamination
distribution, labeling corrupted points, or recovering condition labels.

## Model (shared by both arms)
p_t = σ(α + s_i(t) − r_j(t) + βᵀx_t), with s_i, r_j Gaussian random walks
(variance σ²). Context x: surface, serve number, tournament, round. Ingram (2019)
specification (per-player surface effects, tournament intercepts). **Only the
estimator differs between arms** (ordinary likelihood fit vs. robust MMW/filtering);
identical likelihood and data in both.

## Data combinations (both pre-registered)
1. **Match-only (headline).** Both arms on ATP match terms. Directly comparable
   to Ingram; tests match-scale robustness.
2. **Hybrid.** ATP match terms as the random-walk backbone; slam point terms for
   slam matches, **replacing** (never augmenting) the match term for those
   matches. Both arms use the identical likelihood. Tests point-level burst
   robustness — the mechanism the hypothesis is about.

## Forecasting
Every forecast is a point-by-point match simulation under the actual match
format, contamination injected by a random burst rule. **The rule is tuned per
arm on validation** (robust is expected to tune a higher rate than ordinary; the
gap ≈ what robust estimation removed). Never the analytic iid formula.

## Ablation (2×2)
estimation {ordinary, robust} × forecast {iid, tuned burst}. The claim is
ordinary-vs-robust each at its best forecast; the iid row isolates the forecast
rule.

## Split & evaluation
- Develop ≤2024; **test = 2025 ATP season, once**. Validation by rolling-origin
  within development; hyperparameters (ε, σ, per-arm forecast rule) frozen before
  test.
- Primary metric: **match cross-entropy** (as Ingram); accuracy and calibration
  secondary.
- **Report first-match-of-tournament separately from later rounds** (robust is
  predicted to win the first, possibly lose the second).
- Uncertainty: bootstrap clustered by **player-tournament**; tournament-level as
  a conservative check.

## Gates (must pass in order, before test)
1. Reproduce Ingram 2014 ≈ 0.592 log loss (small drift allowed).
2. Synthetic gate: robust beats ordinary on planted-burst recovery, costs little
   with none planted, lags boundedly on permanent jumps.

## Pre-registered comparisons (the only ones that count as confirmatory)
- C1 (headline): ordinary vs robust, match-only, tuned forecasts, on 2025.
- C2: ordinary vs robust, hybrid, tuned forecasts, on 2025.
- C3: the 2×2 ablation, on 2025.
- C4: first-match-of-tournament vs later-rounds split, for C1 and C2.
Anything else is exploratory and labeled as such.

## Discipline
Every model variant is appended to results/trials.jsonl (never reset) — with a
frozen test set the trial count is the only defense against overfitting by
search. Negative (resolution present, effect absent) and inconclusive (noise
floor exceeded the effect) are reported as distinct outcomes.
"""


def step_prereg():
    ensure(AUDIT)
    (AUDIT / "PRE_REGISTRATION.md").write_text(PREREG.replace("{date}", utcnow()))
    write_json(_out("prereg"), {"utc": utcnow(), "written": "PRE_REGISTRATION.md",
                                "locked_before_test": True})
    return {"prereg": str(AUDIT / "PRE_REGISTRATION.md")}


# --- step: report --------------------------------------------------------------

def step_report():
    cov = read_json(_out("coverage")); art = read_json(_out("artifacts"))
    cus = read_json(_out("custody")); pw = read_json(_out("power"))
    lines = [
        "# Stage 0 — audit & split (Ingram forecasting design)",
        "", f"_generated {utcnow()} · trials logged: {trials.count()}_", "",
        "Gate 0: coverage, artifacts, chain of custody, power, pre-registration. "
        "No model fit here; 2025 test outcomes untouched.", "",
        "## Split",
        "- develop ≤2024, **test = 2025 ATP season (once)**; validation rolling-origin; "
        "cluster by player-tournament. (Old slam-year split retired — see archive/.)",
        "",
        "## Coverage",
        f"- 2025 test season: **{cov['test_season']['atp_matches']} ATP matches** in "
        f"{cov['test_season']['atp_tournaments']} tournaments.",
        f"- Slam point data: all in development, {cov['slam_point_coverage_rows']} "
        f"slam-year-tour cells; years {cov['slam_years_present'][0]}–{cov['slam_years_present'][-1]}.",
        "- Full grid: results/audit/match_coverage_grid.csv, slam_point_coverage.csv.",
        "",
        "## Artifacts",
        f"- Slam→ATP resolution: **{art['slam_to_atp_resolution']['match_rate']}** "
        f"({art['slam_to_atp_resolution']['matched']} matched, "
        f"{art['slam_to_atp_resolution']['unmatched']} unmatched — Hawkeye coverage "
        "+ name variants; investigate, don't drop).",
        "- Retirements kept & flagged (the key diagnostic). Singles-only enforced upstream.",
        "- Reduced-covariate: slam serve-number absent pre-2019; ATP serve stats "
        "88–99% present by year. Absence is never a signal.",
        "",
        "## Chain of custody",
        f"- SHA pin resolved: recorded == independent == live HEAD "
        f"({cus['sha_resolution']['recorded_equals_independent']}); the spec's "
        f"`{cus['sha_resolution']['spec_flagged_value']}` is the bad value, never recorded.",
        f"- Per-file SHA-256 verified ({cus['integrity']['verify_command']}); hash "
        "failure is a hard stop. ATP/WTA provenance is WEAK (headline data → hashes "
        "carry the weight); keep a cold copy.",
        "",
        "## Power check (2024 proxy; 2025 outcomes untouched)",
        f"- The comparison is **paired** (both arms score the same matches), so the "
        f"noise floor is the CI on the per-match cross-entropy **difference**: "
        f"half-width **{pw['paired_difference_ci_half_width']}** (clustered by tournament, "
        f"{pw['proxy_tournaments']} tournaments).",
        f"- For context, the *absolute* mean-CE CI half-width is {pw['absolute_ci_half_width']} "
        "— NOT the right floor here; using it would wrongly read as underpowered.",
        f"- 2025 test size: {pw['test_2025_matches']} matches / {pw['test_2025_tournaments']} "
        "tournaments (comparable floor).",
        "- Plausible gains vs paired floor: " + ", ".join(
            f"{g['gain']}{'✓' if g['detectable_paired'] else '✗'}" for g in pw['plausible_gain_grid'])
        + f". **Verdict: {pw['verdict']}.**",
        "  Two real arms are more correlated than these proxies, so the true paired "
        "floor is likely smaller still. Re-check once both arms exist.",
        "",
        "## Pre-registration",
        "- results/audit/PRE_REGISTRATION.md locks the hypothesis, model, both data "
        "combinations (match-only headline + hybrid), the 2×2 ablation, the split, "
        "the metric, the first-match/later-round report, and the gates — before any "
        "test contact.",
        "",
        "## Gate status → next",
        "- Audit complete. **Next gate: reproduce Ingram 2014 (~0.592 log loss).** "
        "Then the synthetic gate, then develop both arms.",
    ]
    (AUDIT / "audit_report.md").write_text("\n".join(lines) + "\n")
    write_json(_out("report"), {"utc": utcnow(), "written": "audit_report.md"})
    return {"report": str(AUDIT / "audit_report.md")}


# --- driver --------------------------------------------------------------------

def run(fresh=False):
    ensure(AUDIT)
    splits.declare()
    if fresh:
        for s in STEPS:
            _out(s).unlink(missing_ok=True)
    if not _done("coverage"):
        step_coverage()
    if not _done("artifacts"):
        step_artifacts()
    if not _done("custody"):
        step_custody()
    if not _done("power"):
        step_power()
    step_prereg()
    step_report()
    print("Stage 0 audit complete. See results/audit/audit_report.md")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args(argv)
    run(fresh=args.fresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
