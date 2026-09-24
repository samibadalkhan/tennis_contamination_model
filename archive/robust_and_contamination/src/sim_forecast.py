"""The full-method read the design's ablation calls for: robust (and ordinary)
abilities forecast by POINT-BY-POINT SIMULATION with a per-arm-tuned contamination
rule, on a development season. NOT the 2025 test.

Why this matters: the iid-forecast read (robust_check.py) is the ablation's "iid
column", where robust is expected to look overconfident. The full method gives
each arm its own tuned burst forecast (tennis-contamination.md "Forecasting"):
the simulator injects bursts that pull match-win probs toward 0.5, and the rule
is tuned SEPARATELY per arm (a shared rule favors one). Diagnostic: the robust
arm should tune a HIGHER rate than ordinary; the gap ~ what robust removed.

Protocol (mirrors Gate 1): tune the contamination RATE per arm on `tune_year`
(default eval-1), score `eval_year`. Severity/length fixed (1-D tune) and frozen.
Same sigma/prior for every arm — only the estimator and its tuned rate differ.
2025 is never read.

    python -m src.sim_forecast --year 2014
"""

from __future__ import annotations

import argparse

import numpy as np

from src import evaluate as ev, load, simulate, trials
from src.models import ordinary, robust
from src.models.ability import alpha_by_surface
from src.util import RESULTS, ensure, utcnow, write_json

OUT = RESULTS / "sim_forecast"
SIGMA, PRIOR = 0.03, 0.25                 # corrected Gate-1 choice; frozen, shared
SEV, LEN_FRAC = 1.5, 0.35                 # forecast-burst severity/length (fixed, frozen)
RATE_GRID = [0.0, 0.10, 0.20, 0.35]       # forecast-contamination rate, tuned per arm
S_SIMS = 200


def _make(alpha, robust_c):
    return (ordinary.make(alpha, SIGMA, PRIOR) if robust_c is None
            else robust.make(alpha, SIGMA, robust_c, PRIOR))


def forecast_season(matches, alpha, eval_year, robust_c, rate, S, seed):
    """Filter chronologically; forecast each eval-year match by simulation
    (abilities read BEFORE the match's update). Returns preds + tournament ids."""
    f = _make(alpha, robust_c)
    rng = np.random.default_rng(seed)
    preds, tourneys = [], []
    for row in matches.itertuples():
        if row.year < eval_year:
            f.process(row, forecast_first=False)
        elif row.year == eval_year:
            t = row.date
            f._ensure(row.p1, t); f._ensure(row.p2, t)
            f._diffuse(row.p1, t); f._diffuse(row.p2, t)
            pa = f.serve_prob(row.p1, row.p2, row.surface)
            pb = f.serve_prob(row.p2, row.p1, row.surface)
            preds.append(simulate.simulate_winprob(pa, pb, int(row.best_of), S, rng,
                                                   rate=rate, sev=SEV, len_frac=LEN_FRAC))
            tourneys.append(row.tourney_id)
            f._update_serve_obs(row.p1, row.p2, int(row.p1_spw), int(row.p1_svpt), row.surface)
            f._update_serve_obs(row.p2, row.p1, int(row.p2_spw), int(row.p2_svpt), row.surface)
        else:
            break
    # Laplace (add-one) smoothing stabilizes log loss under finite sims and
    # bounds the tails; the ~1/S shrink is identical across arms (same S), so it
    # does not favor either. wins = preds*S.
    p = (np.array(preds) * S + 1.0) / (S + 2.0)
    return p, np.array(tourneys, dtype=object)


def _ll(preds):
    return float(-np.log(preds).mean())        # y = 1 (p1 is the winner)


def tune_arm(matches, alpha, tune_year, robust_c):
    """Pick the contamination rate minimizing tune-year log loss (per arm)."""
    curve = []
    for rate in RATE_GRID:
        preds, _ = forecast_season(matches, alpha, tune_year, robust_c, rate, S_SIMS, seed=7)
        ll = _ll(preds)
        curve.append({"rate": rate, "log_loss": ll})
        trials.log(stage="3-sim-forecast", kind=f"tune_{tune_year}/{'ord' if robust_c is None else f'rob{robust_c}'}",
                   params={"rate": rate, "sev": SEV, "len_frac": LEN_FRAC, "robust_c": robust_c,
                           "S": S_SIMS, "tune_year": tune_year},
                   metrics={"val_log_loss": ll})
    best = min(curve, key=lambda c: c["log_loss"])
    return best["rate"], curve


def run(eval_year=2014, tune_year=None):
    ensure(OUT)
    tune_year = tune_year or (eval_year - 1)
    matches = load.load_matches("M")
    alpha_tune = alpha_by_surface(matches[matches.year < tune_year])
    alpha_eval = alpha_by_surface(matches[matches.year < eval_year])

    arms = {"ordinary": None, "robust_1.345": 1.345, "robust_2.5": 2.5}
    out = {"utc": utcnow(), "eval_year": eval_year, "tune_year": tune_year,
           "sigma": SIGMA, "prior_var": PRIOR, "sev": SEV, "len_frac": LEN_FRAC,
           "rate_grid": RATE_GRID, "S": S_SIMS, "arms": {}}

    # ordinary + iid analytic, as the fixed reference point
    from src.reproduce_ingram import run_filter
    iid = run_filter(matches, alpha_eval, SIGMA, PRIOR, eval_year)
    out["ordinary_iid_reference"] = {"log_loss": iid["log_loss"], "accuracy": iid["accuracy"]}

    preds_by_arm = {}
    for name, rc in arms.items():
        rate, curve = tune_arm(matches, alpha_tune, tune_year, rc)
        preds, tourneys = forecast_season(matches, alpha_eval, eval_year, rc, rate, S_SIMS, seed=11)
        preds_by_arm[name] = (preds, tourneys)
        out["arms"][name] = {"tuned_rate": rate, "tune_curve": curve,
                             "log_loss": _ll(preds),
                             "accuracy": float((preds > 0.5).mean()),
                             "mean_pred": float(preds.mean())}
        trials.log(stage="3-sim-forecast", kind=f"score_{eval_year}/{name}",
                   params={"tuned_rate": rate, "robust_c": rc, "sev": SEV, "len_frac": LEN_FRAC},
                   metrics={"log_loss": _ll(preds), "accuracy": float((preds > 0.5).mean())})

    # paired gains vs the ordinary SIM arm (both use the tuned sim forecast)
    op, ot = preds_by_arm["ordinary"]
    base_ll = -np.log(op)
    for name in ["robust_1.345", "robust_2.5"]:
        rp, _ = preds_by_arm[name]
        diff = base_ll - (-np.log(rp))              # + => robust better
        order = {t: i for i, t in enumerate(dict.fromkeys(ot))}
        G = len(order); sd = np.zeros(G); cn = np.zeros(G)
        for d, t in zip(diff, ot):
            g = order[t]; sd[g] += d; cn[g] += 1
        pt, lo, hi, se = ev.clustered_ci({"sum_x": sd, "n": cn},
                                         lambda s: s["sum_x"] / max(s["n"], 1e-12), B=2000)
        out["arms"][name]["paired_gain_vs_ordinary_sim"] = {"mean": pt, "lo": lo, "hi": hi}

    write_json(OUT / f"sim_forecast_{eval_year}.json", out)
    _report(out)
    return out


def _report(o):
    y = o["eval_year"]
    L = [f"# Full method: simulated forecast, per-arm tuned — {y} (development, NOT test)", "",
         f"_generated {o['utc']} · tuned on {o['tune_year']} · S={o['S']} sims · "
         f"sev={o['sev']} len_frac={o['len_frac']} · sigma={o['sigma']} prior={o['prior_var']} · "
         f"trials {trials.count()}_", "",
         f"Reference — ordinary + iid analytic forecast: log loss "
         f"**{o['ordinary_iid_reference']['log_loss']:.4f}**", "",
         "Diagnostic (design): the robust arm should tune a HIGHER contamination "
         "rate than ordinary; equal rates mean robustness removed nothing.", "",
         "| arm | tuned rate | log loss | acc | mean pred | paired gain vs ordinary-sim |",
         "|---|---|---|---|---|---|"]
    for name, a in o["arms"].items():
        g = a.get("paired_gain_vs_ordinary_sim")
        gs = f"{g['mean']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]" if g else "—"
        L.append(f"| {name} | {a['tuned_rate']} | {a['log_loss']:.4f} | {a['accuracy']:.3f} | "
                 f"{a['mean_pred']:.3f} | {gs} |")
    L += ["", "## Tuning curves (log loss vs contamination rate, on " f"{o['tune_year']})", ""]
    for name, a in o["arms"].items():
        pts = "  ".join(f"{c['rate']}:{c['log_loss']:.4f}" for c in a["tune_curve"])
        L.append(f"- **{name}**: {pts}")
    (OUT / f"sim_forecast_{y}_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2014)
    ap.add_argument("--tune-year", type=int, default=None)
    a = ap.parse_args()
    if a.year >= 2025:
        raise SystemExit("refusing: 2025+ is the single-use test season")
    run(a.year, a.tune_year)
