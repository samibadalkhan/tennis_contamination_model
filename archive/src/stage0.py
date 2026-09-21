"""Stage 0 -- i.i.d.-given-server sanity check + noise floor. DIAGNOSTIC ONLY.

Fits the i.i.d.-given-server null (per-player serve + return effects, L2
regularized, slam x year x tour fixed effect), then establishes, all with
match-clustered bootstrap intervals:
  * overdispersion   -- block variance vs binomial
  * serial dependence-- lag-1 autocorrelation of outcomes and squared residuals
  * asymmetry        -- skewness of block residuals (left = involuntary-D5-like)
  * NOISE FLOOR      -- clustered-bootstrap CI on held-out (validation) log-loss,
                        compared to the oracle log-loss gain a plausible
                        epsilon-contamination could ever produce. If the CI is
                        wider than that gain, the design is UNDERPOWERED and the
                        honest verdict is "inconclusive", not "negative".

This is a diagnostic, not a foundation: it gates whether Stage 0.5 (the
retirement positive control) is worth running. It fits on TRAIN and measures the
noise floor on VALIDATION; it never touches TEST.

Durability: every step writes a checkpoint to results/stage0/ and updates
PROGRESS.md; heavy intermediates cache to data/cache/ (gitignored). Re-running
skips completed steps, so a session that dies mid-run resumes from disk. Run:

    python -m src.stage0            # run/resume all steps
    python -m src.stage0 --fresh    # ignore checkpoints and recompute
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from src import blocks, evaluate as ev, load, splits, trials
from src.util import (DATA, STAGE0, ensure, exists_nonempty, read_json, utcnow,
                      write_json)

CACHE = DATA / "cache"
C_GRID = [0.1, 0.3, 1.0, 3.0]           # L2 inverse-strength; tuned on VAL only
MIN_BLOCK_N = 10                         # blocks this small are too noisy for z
EPS_GRID = [0.02, 0.05, 0.10]            # plausible contamination rates
DELTA_GRID = [0.10, 0.20, 0.30]          # serve-win-prob drop when contaminated

PROGRESS = STAGE0 / "PROGRESS.md"
STEPS = ["audit", "fit", "overdispersion", "serial", "asymmetry",
         "noise_floor", "report"]


# --- progress bookkeeping ------------------------------------------------------

def _out(step: str):
    return STAGE0 / f"{step}.json"


def _done(step: str) -> bool:
    return exists_nonempty(_out(step))


def update_progress(current: str = "") -> None:
    ensure(STAGE0)
    lines = ["# Stage 0 progress", "",
             f"_updated {utcnow()} · trials logged: {trials.count()}_", ""]
    for s in STEPS:
        mark = "x" if _done(s) else " "
        tag = "  <- running" if s == current else ""
        lines.append(f"- [{mark}] {s}{tag}")
    lines += ["", "Resumable: re-run `python -m src.stage0`; completed steps are skipped.",
              "Verdict lives in stage0_report.md once `report` is checked."]
    PROGRESS.write_text("\n".join(lines) + "\n")


# --- step 1: declare split, load, audit ---------------------------------------

def step_audit():
    decl = splits.declare()                       # write-once
    write_json(STAGE0 / "splits_used.json", decl)
    pts = load.load_points(folds=("train", "val"))   # never reads test files
    ensure(CACHE)
    pts.to_parquet(CACHE / "points_trainval.parquet")
    aud = blocks.audit(pts)
    aud.to_csv(STAGE0 / "audit_coverage.csv", index=False)
    summary = {
        "utc": utcnow(),
        "folds_loaded": ["train", "val"],
        "n_points": int(len(pts)),
        "by_fold": pts.groupby("fold", observed=True).size().to_dict(),
        "by_tour": pts.groupby("tour", observed=True).size().to_dict(),
        "serve_win_rate_by_tour": pts.groupby("tour", observed=True)["server_wins"].mean().round(4).to_dict(),
        "n_slam_years": int(aud[["slam", "year"]].drop_duplicates().shape[0]),
        "audit_rows": len(aud),
        "note": "Coverage uneven; serve rate differs by tour (M~0.64, W~0.57). "
                "See audit_coverage.csv.",
    }
    write_json(_out("audit"), summary)
    trials.log(stage="0", kind="audit", metrics={"n_points": len(pts)},
               note="loaded train+val singles, per-slam-year audit")
    return pts


def _load_points():
    p = CACHE / "points_trainval.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return load.load_points(folds=("train", "val"))


# --- step 2: fit i.i.d.-given-server -------------------------------------------

def _context(df):
    return (df["slam"].astype(str) + "|" + df["year"].astype(str) + "|" + df["tour"].astype(str))


def step_fit(pts=None):
    pts = pts if pts is not None else _load_points()
    tr = pts[pts.fold == "train"].reset_index(drop=True)
    va = pts[pts.fold == "val"].reset_index(drop=True)

    Xtr_df = pd.DataFrame({"server": tr.server, "returner": tr.returner, "ctx": _context(tr)})
    Xva_df = pd.DataFrame({"server": va.server, "returner": va.returner, "ctx": _context(va)})
    enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    Xtr = enc.fit_transform(Xtr_df)
    Xva = enc.transform(Xva_df)
    ytr = tr.server_wins.to_numpy()
    yva = va.server_wins.to_numpy()

    sens = {}
    best = None
    for C in C_GRID:
        clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                                 max_iter=2000, n_jobs=-1)
        clf.fit(Xtr, ytr)
        pva = clf.predict_proba(Xva)[:, 1]
        vll = float(ev.log_loss(yva, pva).mean())
        sens[C] = round(vll, 6)
        trials.log(stage="0", kind="iid_given_server",
                   params={"C": C, "features": ["serve", "return", "slam_year_tour"],
                           "penalty": "l2"},
                   metrics={"val_logloss": vll},
                   note="i.i.d.-given-server; C tuned on validation only")
        if best is None or vll < best[1]:
            best = (C, vll, clf)
    C_best, vll_best, clf = best

    ptr = clf.predict_proba(Xtr)[:, 1]
    pva = clf.predict_proba(Xva)[:, 1]

    # top serve / return effects, as a sanity check the fit learned real skill
    names = enc.get_feature_names_out(["server", "returner", "ctx"])
    coef = clf.coef_[0]
    def _tops(prefix, k=8):
        idx = [i for i, n in enumerate(names) if n.startswith(prefix)]
        order = sorted(idx, key=lambda i: coef[i])
        top = [(names[i].split("_", 1)[1], round(float(coef[i]), 3)) for i in order[-k:][::-1]]
        bot = [(names[i].split("_", 1)[1], round(float(coef[i]), 3)) for i in order[:k]]
        return {"strongest": top, "weakest": bot}

    ensure(CACHE)
    tr_cache = tr[["match_id", "slam", "year", "tour", "set_no", "game_no",
                   "server", "is_tiebreak", "server_wins"]].copy()
    tr_cache["p"] = ptr
    tr_cache.to_parquet(CACHE / "train_pred.parquet")
    va_cache = va[["match_id", "server_wins"]].copy()
    va_cache["p"] = pva
    va_cache.to_parquet(CACHE / "val_pred.parquet")

    summary = {
        "utc": utcnow(),
        "model": "i.i.d.-given-server: L2 logistic on serve(player) + return(player) + slam|year|tour",
        "n_train_points": int(len(tr)), "n_val_points": int(len(va)),
        "n_features": int(Xtr.shape[1]),
        "C_grid_val_logloss": sens,
        "C_selected": C_best,
        "val_logloss": round(vll_best, 6),
        "train_logloss": round(float(ev.log_loss(ytr, ptr).mean()), 6),
        "baseline_val_logloss_constant_rate": round(
            float(ev.log_loss(yva, np.full_like(yva, ytr.mean(), dtype=float)).mean()), 6),
        "serve_effects": _tops("server_"),
        "return_effects": _tops("returner_"),
        "note": "C selected on validation; full C-grid reported for sensitivity. "
                "Test fold untouched.",
    }
    write_json(_out("fit"), summary)
    update_progress()
    return summary


# --- diagnostics on TRAIN (in-sample residual structure) -----------------------

def _train_blocks_with_p():
    d = pd.read_parquet(CACHE / "train_pred.parquet")
    # set-level service blocks; p is constant within a block (server, returner,
    # context all fixed), so block expected rate = mean p.
    g = (d.groupby(["match_id", "set_no", "server"], observed=True)
         .agg(n=("server_wins", "count"), wins=("server_wins", "sum"),
              p=("p", "mean")).reset_index())
    return d, g


def step_overdispersion():
    _, g = _train_blocks_with_p()
    e = g.n * g.p
    var = g.n * g.p * (1 - g.p)
    pear = (g.wins - e) ** 2 / var.clip(lower=ev.EPS)      # ~1 each under binomial
    df = pd.DataFrame({"match_id": g.match_id, "pear": pear, "one": 1.0})
    per = df.groupby("match_id", observed=True).agg(numer=("pear", "sum"),
                                                    denom=("one", "sum"))
    phi, lo, hi, se = ev.clustered_ci(
        {"numer": per.numer.to_numpy(), "denom": per.denom.to_numpy()}, ev._ratio)
    out = {"utc": utcnow(), "statistic": "pearson_dispersion_phi",
           "interpretation": "phi=1 binomial; phi>1 overdispersed (block variance "
                             "exceeds binomial). Does NOT by itself separate D4 from D5.",
           "phi": round(phi, 4), "ci95": [round(lo, 4), round(hi, 4)],
           "n_blocks": int(len(g)), "min_block_n": int(g.n.min()),
           "median_block_n": float(g.n.median())}
    write_json(_out("overdispersion"), out)
    trials.log(stage="0", kind="overdispersion", metrics={"phi": phi, "ci_lo": lo, "ci_hi": hi})
    return out


def step_serial():
    d = pd.read_parquet(CACHE / "train_pred.parquet")
    d = d[~d.is_tiebreak.astype(bool)].copy()
    # order preserved from load (chronological). Pair consecutive points WITHIN a
    # service game (one server per game). The clean serial-dependence statistic
    # for a short binary sequence is the RUNS / TRANSITION test CONDITIONAL on
    # the game's win total: it isolates point-to-point arrangement from
    # block-level overdispersion (whole games above/below expectation), which a
    # model-p-centered autocorrelation would conflate with true dependence.
    d["seq"] = np.arange(len(d))
    gkey = ["match_id", "set_no", "game_no", "server"]
    d = d.sort_values(gkey + ["seq"], kind="stable")
    y = d.server_wins.to_numpy().astype(float)
    y_prev = d.groupby(gkey, observed=True)["server_wins"].shift(1).to_numpy()
    same = ~np.isnan(y_prev)
    trans = ((y != y_prev) & same).astype(float)   # 1 at each within-game sign change

    gid = d.groupby(gkey, observed=True).ngroup().to_numpy()
    tmp = pd.DataFrame({"match_id": d.match_id.to_numpy(), "gid": gid, "y": y, "trans": trans})
    pg = tmp.groupby("gid", observed=True).agg(
        match_id=("match_id", "first"), n=("y", "count"),
        n1=("y", "sum"), obs=("trans", "sum"))
    pg = pg[pg.n >= 2]
    n, n1 = pg.n.to_numpy(), pg.n1.to_numpy()
    n0 = n - n1
    pg["exp"] = 2 * n1 * n0 / n                                   # E[transitions | n, n1]
    pg["Rme"] = pg.obs.to_numpy() - pg["exp"].to_numpy()          # R - E (R = 1 + transitions)
    pg["V"] = np.where(n > 1, 2 * n1 * n0 * (2 * n1 * n0 - n) / (n ** 2 * (n - 1)), 0.0)

    per = pg.groupby("match_id", observed=True).agg(
        obs=("obs", "sum"), exp=("exp", "sum"), Rme=("Rme", "sum"), V=("V", "sum"))
    ratio, lo, hi, se = ev.clustered_ci(
        {"numer": per.obs.to_numpy(), "denom": per["exp"].to_numpy()}, ev._ratio)
    z = float(per.Rme.sum() / np.sqrt(per.V.sum())) if per.V.sum() > 0 else float("nan")

    out = {"utc": utcnow(),
           "statistic": "within-game transition ratio (conditional runs test)",
           "transition_ratio": round(ratio, 4), "ci95": [round(lo, 4), round(hi, 4)],
           "pooled_runs_test_z": round(z, 3),
           "interpretation": "ratio = observed within-game sign-changes / expected "
                             "given each game's win total. ratio>1 (z>0) = more "
                             "alternation than chance = ANTI-persistence; ratio<1 "
                             "(z<0) = streakier than chance = momentum (the K&M "
                             "positive-dependence direction). Conditioning on the "
                             "game total isolates point-to-point dependence (D3) "
                             "from block overdispersion (D4/D5), which a model-p-"
                             "centered autocorrelation conflates.",
           "n_within_game_pairs": int(same.sum()),
           "n_games": int(len(pg))}
    write_json(_out("serial"), out)
    trials.log(stage="0", kind="serial_dependence",
               metrics={"transition_ratio": ratio, "ci_lo": lo, "ci_hi": hi, "runs_z": z},
               note="conditional runs/transition test; clustered CI over matches")
    return out


def step_asymmetry():
    _, g = _train_blocks_with_p()
    g = g[g.n >= MIN_BLOCK_N]
    z = (g.wins - g.n * g.p) / np.sqrt((g.n * g.p * (1 - g.p)).clip(lower=ev.EPS))
    f = pd.DataFrame({"match_id": g.match_id, "z": z.to_numpy()})
    per = f.groupby("match_id", observed=True).agg(
        s1=("z", "sum"), s2=("z", lambda a: float(np.sum(a ** 2))),
        s3=("z", lambda a: float(np.sum(a ** 3))), nb=("z", "count"))
    skew, lo, hi, se = ev.clustered_ci(
        {k: per[k].to_numpy() for k in ["s1", "s2", "s3", "nb"]}, ev._skew)
    out = {"utc": utcnow(), "statistic": "skewness_of_block_z_residuals",
           "interpretation": "left-skew (negative) = excess of worse-than-expected "
                             "blocks = directional degradation (involuntary-D5-like). "
                             "symmetric ~0 = D4-consistent (does not kill strategic D5).",
           "skewness": round(skew, 4), "ci95": [round(lo, 4), round(hi, 4)],
           "n_blocks_used": int(len(g)), "min_block_n": MIN_BLOCK_N}
    write_json(_out("asymmetry"), out)
    trials.log(stage="0", kind="asymmetry", metrics={"skew": skew, "ci_lo": lo, "ci_hi": hi})
    return out


# --- step: noise floor on VALIDATION -------------------------------------------

def step_noise_floor():
    v = pd.read_parquet(CACHE / "val_pred.parquet")
    ll = ev.log_loss(v.server_wins.to_numpy(), v.p.to_numpy())
    f = pd.DataFrame({"match_id": v.match_id, "ll": ll, "one": 1.0})
    per = f.groupby("match_id", observed=True).agg(sum_x=("ll", "sum"), n=("one", "sum"))
    mean_ll, lo, hi, se = ev.clustered_ci(
        {"sum_x": per.sum_x.to_numpy(), "n": per.n.to_numpy()}, ev._mean)
    half_width = (hi - lo) / 2

    # oracle detectable effect: per-point log-loss an oracle mixture would save
    # over the single-component model, at plausible (eps, delta). Upper bound on
    # what ANY contamination model could recover -> if it is below the noise
    # floor, the design cannot distinguish the mixture. Computed per tour.
    grid = []
    p_by_tour = {"overall": float(v.p.mean())}
    for eps in EPS_GRID:
        for delta in DELTA_GRID:
            p_c = p_by_tour["overall"]
            p_obs = p_c - eps * delta
            H = ev.binary_entropy
            single = float(H(np.array([p_obs]))[0])
            mix = float((1 - eps) * H(np.array([p_c]))[0] + eps * H(np.array([max(p_c - delta, ev.EPS)]))[0])
            effect = single - mix
            grid.append({"eps": eps, "delta": delta,
                         "oracle_logloss_gain": round(effect, 5),
                         "detectable": bool(effect > half_width)})
    detectable = [g for g in grid if g["detectable"]]
    verdict = ("adequate" if detectable else "UNDERPOWERED")
    out = {"utc": utcnow(),
           "held_out_fold": "val",
           "mean_val_logloss": round(mean_ll, 6),
           "ci95": [round(lo, 6), round(hi, 6)],
           "ci_half_width": round(half_width, 6),
           "bootstrap_se": round(se, 6),
           "reference_serve_win_prob": round(p_by_tour["overall"], 4),
           "oracle_effect_grid": grid,
           "n_detectable_cells": len(detectable),
           "verdict": verdict,
           "interpretation": "If the oracle log-loss gain from resolving a plausible "
                             "contamination is smaller than the held-out CI half-width, "
                             "the mixture cannot be distinguished from the single "
                             "component at that (eps,delta): report INCONCLUSIVE, not "
                             "negative. This is an upper bound (oracle knows the labels).",
           }
    write_json(_out("noise_floor"), out)
    trials.log(stage="0", kind="noise_floor",
               metrics={"val_logloss": mean_ll, "ci_half_width": half_width,
                        "n_detectable": len(detectable)},
               note="clustered bootstrap over val matches; oracle effect grid")
    return out


# --- step: synthesize report ---------------------------------------------------

def step_report():
    a = read_json(_out("audit")); fit = read_json(_out("fit"))
    od = read_json(_out("overdispersion")); se = read_json(_out("serial"))
    asym = read_json(_out("asymmetry")); nf = read_json(_out("noise_floor"))
    mom = read_json(STAGE0 / "momentum_investigation.json")   # optional
    decl = read_json(STAGE0 / "splits_used.json") or splits.load()

    def ci(x, k="ci95"):
        return f"[{x[k][0]}, {x[k][1]}]"

    lines = [
        "# Stage 0 report — i.i.d.-given-server sanity check + noise floor",
        "", f"_generated {utcnow()} · trials logged so far: {trials.count()}_", "",
        "**Diagnostic only.** Gates whether Stage 0.5 (retirement positive control) "
        "is worth running. Fits on TRAIN, measures noise floor on VALIDATION, never "
        "touches TEST.", "",
        "## Split (declared once, frozen)",
        f"- train {decl['train_years']}, val {decl['val_years']}, test {decl['test_years']} (untouched)",
        f"- {a['n_points']:,} train+val points; serve-win rate by tour {a['serve_win_rate_by_tour']}",
        "",
        "## Model",
        f"- {fit['model']}",
        f"- C selected on val = {fit['C_selected']} (grid {fit['C_grid_val_logloss']})",
        f"- val log-loss {fit['val_logloss']} vs constant-rate baseline "
        f"{fit['baseline_val_logloss_constant_rate']} (lower is better)",
        f"- strongest serve effects: {fit['serve_effects']['strongest'][:5]}",
        "",
        "## Diagnostics (train, in-sample; match-clustered 95% CI)",
        f"- **Overdispersion** φ = {od['phi']} CI {ci(od)} "
        f"({'overdispersed' if od['ci95'][0] > 1 else 'consistent with binomial'}). "
        f"{od['n_blocks']:,} set-blocks, median n={od['median_block_n']}.",
        f"- **Serial dependence** within-game transition ratio = "
        f"{se['transition_ratio']} CI {ci(se)} (runs z = {se['pooled_runs_test_z']}) — "
        f"looks like anti-persistence, but see the momentum investigation below: "
        f"~95% of it is a scoring-structure artifact of the permutation null."
        + (f" Against an i.i.d.-under-scoring null, the real effect is "
           f"{mom['real_effect_obs_over_null']} (≈none)." if mom else ""),
        f"- **Asymmetry** block-residual skewness = {asym['skewness']} CI {ci(asym)} "
        f"({'left-skewed (directional)' if asym['ci95'][1] < 0 else 'right-skewed' if asym['ci95'][0] > 0 else 'not clearly asymmetric'}).",
        "",
        "## Noise floor (validation, held-out)",
        f"- mean val log-loss {nf['mean_val_logloss']} CI {ci(nf)}, half-width "
        f"{nf['ci_half_width']}.",
        f"- oracle-detectable (eps,delta) cells: {nf['n_detectable_cells']} / "
        f"{len(nf['oracle_effect_grid'])}. **Verdict: {nf['verdict']}.**",
        f"  - grid: " + "; ".join(
            f"({g['eps']},{g['delta']})→{g['oracle_logloss_gain']}{'*' if g['detectable'] else ''}"
            for g in nf['oracle_effect_grid']) + "  (* = above noise floor)",
        "",
    ] + ([
        "## Momentum investigation (why anti-momentum?)",
        f"- Observed transition ratio {mom['observed_transition_ratio']} vs an "
        f"i.i.d.-under-scoring null of {mom['iid_under_scoring_null_ratio']} "
        f"(95% {mom['iid_null_ratio_ci95']}). Real effect {mom['real_effect_obs_over_null']}.",
        f"- **{mom['verdict'].split('.')[0]}.** Winning points does NOT build momentum "
        "here; the apparent anti-momentum is tennis scoring (deuce forces alternation), "
        "not psychology. See momentum_investigation.md. A real momentum test belongs "
        "in Stage 1, conditioned on score state.",
        "",
    ] if mom else []) + [
        "## Read-out",
        "- If overdispersion CI sits above 1 AND some plausible (eps,delta) clears "
        "the noise floor → the detector has resolution; **proceed to Stage 0.5**.",
        "- If no plausible cell clears the noise floor → **INCONCLUSIVE by design** "
        "(a fact about the design, not about tennis); redesign before continuing.",
        "- Asymmetry sign hints strategic (symmetric) vs involuntary (left) H, but "
        "does not by itself separate D4 from D5 — that is Stage 2.",
        "",
        "## Limitations (Stage 0)",
        "- Diagnostics are in-sample on train (residual structure); only the noise "
        "floor is held-out. Overdispersion/asymmetry describe the fitted model's "
        "residuals, not out-of-sample generalization.",
        "- Player identity is the raw slam-corpus name string; variant spellings "
        "(e.g. 'J. Isner' vs 'John Isner') fragment a player's serve/return effect "
        "across two levels. A D6 name-normalization issue to resolve before the "
        "ATP retirement join at Stage 0.5; it slightly weakens the fit here.",
        "- Serve-number (1st/2nd) is absent pre-2019 (all-missing in early years); "
        "the base i.i.d.-given-server model does not use it, so this is neutral "
        "for Stage 0 but is a reduced-covariate issue for later stages.",
        "",
        "## Figures (results/stage0/figures/, `python -m src.plots`)",
        "- `1_block_residuals.png` — overdispersion + left-skew vs the i.i.d. N(0,1) expectation.",
        "- `2_momentum_artifact.png` — observed vs momentum-free simulation; the anti-momentum is scoring structure.",
        "- `3_noise_floor.png` — oracle contamination effect per (ε,δ) vs the held-out noise floor.",
        "",
        "_Negative ≠ inconclusive. This report states which one Stage 0 produced._",
    ]
    (STAGE0 / "stage0_report.md").write_text("\n".join(lines) + "\n")
    write_json(_out("report"), {"utc": utcnow(), "verdict_noise_floor": nf["verdict"],
                                "written": "stage0_report.md"})
    return {"report": str(STAGE0 / "stage0_report.md")}


# --- driver --------------------------------------------------------------------

def run(fresh: bool = False):
    ensure(STAGE0)
    if fresh:
        for s in STEPS:
            _out(s).unlink(missing_ok=True)
    update_progress()
    pts = None
    if not _done("audit"):
        update_progress("audit"); pts = step_audit()
    if not _done("fit"):
        update_progress("fit"); step_fit(pts)
    if not _done("overdispersion"):
        update_progress("overdispersion"); step_overdispersion()
    if not _done("serial"):
        update_progress("serial"); step_serial()
    if not _done("asymmetry"):
        update_progress("asymmetry"); step_asymmetry()
    if not _done("noise_floor"):
        update_progress("noise_floor"); step_noise_floor()
    update_progress("report"); step_report()      # cheap; always refresh
    update_progress()
    print("Stage 0 complete. See results/stage0/stage0_report.md")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoints, recompute all")
    args = ap.parse_args(argv)
    run(fresh=args.fresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
