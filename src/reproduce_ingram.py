"""Gate 1 — reproduce Ingram (2019): ordinary fit + iid analytic forecast.

Ingram reported ~0.592 log loss and 68.8% accuracy holding out the 2014 ATP
season. The ordinary arm (online random-walk ability filter) with the iid
analytic forecast IS that model, so hitting ~0.592 validates the whole baseline
before anything robust is added. Small drift is expected (data revisions).

2014 is a development-year checkpoint, NOT the 2025 test. Hyperparameters
(sigma drift, prior variance) are tuned on the 2013 season and the chosen
setting is scored on 2014. The per-surface intercept is computed from pre-season
matches only.

    python -m src.reproduce_ingram
"""

from __future__ import annotations

import sys

import numpy as np

from src import evaluate as ev, load, trials
from src.models.ability import OnlineAbility, alpha_by_surface
from src.util import RESULTS, ensure, utcnow, write_json

INGRAM = RESULTS / "ingram"
SIGMA_GRID = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18]
PRIOR_GRID = [0.25, 0.5, 1.0]
INGRAM_LOGLOSS, INGRAM_ACC = 0.592, 0.688


def run_filter(matches, alpha, sigma, prior_var, eval_year):
    """Filter chronologically; forecast each eval_year match before updating."""
    f = OnlineAbility(alpha, sigma=sigma, prior_var=prior_var)
    preds = []
    for row in matches.itertuples():
        if row.year < eval_year:
            f.process(row, forecast_first=False)
        elif row.year == eval_year:
            preds.append(f.process(row, forecast_first=True))
        else:
            break
    preds = np.clip(np.array(preds), 1e-6, 1 - 1e-6)
    y = np.ones_like(preds)                       # p1 is always the actual winner
    return {"n": int(len(preds)),
            "log_loss": float(ev.log_loss(y, preds).mean()),
            "accuracy": float((preds > 0.5).mean()),
            "mean_pred": float(preds.mean())}


def score_season(matches, eval_year, tune_year=None):
    """Tune sigma/prior_var on ``tune_year`` (default eval_year-1), score eval_year.
    Never tunes on the evaluation season; per-surface intercept from pre-season data."""
    tune_year = tune_year or (eval_year - 1)
    alpha_tune = alpha_by_surface(matches[matches.year < tune_year])
    tuning, best = [], None
    for sigma in SIGMA_GRID:
        for pv in PRIOR_GRID:
            m = run_filter(matches, alpha_tune, sigma, pv, eval_year=tune_year)
            tuning.append({"sigma": sigma, "prior_var": pv, **m})
            trials.log(stage="1-ingram", kind=f"tune_{tune_year}",
                       params={"sigma": sigma, "prior_var": pv},
                       metrics={"val_log_loss": m["log_loss"], "val_acc": m["accuracy"]})
            if best is None or m["log_loss"] < best["log_loss"]:
                best = {"sigma": sigma, "prior_var": pv, "log_loss": m["log_loss"]}
    alpha_eval = alpha_by_surface(matches[matches.year < eval_year])
    test = run_filter(matches, alpha_eval, best["sigma"], best["prior_var"], eval_year)
    trials.log(stage="1-ingram", kind=f"score_{eval_year}",
               params={"sigma": best["sigma"], "prior_var": best["prior_var"], "tuned_on": tune_year},
               metrics={"log_loss": test["log_loss"], "accuracy": test["accuracy"]})
    return {"utc": utcnow(), "eval_year": eval_year, "tuned_on": tune_year,
            "ingram_target": {"log_loss": INGRAM_LOGLOSS, "accuracy": INGRAM_ACC},
            "chosen_hyperparams": {"sigma": best["sigma"], "prior_var": best["prior_var"]},
            "result": test, "log_loss_gap_vs_ingram": round(test["log_loss"] - INGRAM_LOGLOSS, 4),
            "tuning": tuning,
            "note": "Ordinary random-walk ability filter + iid analytic forecast, "
                    "match-only tour-level singles (G/M/A/F). Hyperparameters tuned "
                    "on the prior season, never the eval season."}


def run(eval_year=2014, tune_year=None):
    ensure(INGRAM)
    matches = load.load_matches("M")
    out = score_season(matches, eval_year, tune_year)
    t = out["result"]
    is_gate = (eval_year == 2014)
    passed = abs(out["log_loss_gap_vs_ingram"]) <= 0.02
    out["gate_passed"] = bool(passed) if is_gate else None
    fname = "reproduce" if is_gate else f"score_{eval_year}"
    write_json(INGRAM / f"{fname}.json", out)
    lines = [
        f"# {'Gate 1 — reproduce Ingram (2019)' if is_gate else f'Season score — {eval_year}'}",
        "", f"_generated {utcnow()} · trials logged: {trials.count()}_", "",
        f"Ingram 2014 reference: log loss **{INGRAM_LOGLOSS}**, accuracy **{INGRAM_ACC}**.",
        f"Hyperparameters tuned on {out['tuned_on']}: sigma={out['chosen_hyperparams']['sigma']}, "
        f"prior_var={out['chosen_hyperparams']['prior_var']}.", "",
        f"## {eval_year} result",
        f"- log loss **{t['log_loss']:.4f}** (gap {out['log_loss_gap_vs_ingram']:+.4f} vs Ingram), "
        f"accuracy **{t['accuracy']:.4f}**, mean forecast {t['mean_pred']:.3f}, on {t['n']} matches.",
    ]
    if is_gate:
        lines.append(f"- **Gate {'PASSED' if passed else 'NOT passed'}** "
                     f"({'within' if passed else 'outside'} 0.02 of Ingram).")
    lines += ["", f"## {out['tuned_on']} tuning (log loss)",
              *[f"- sigma={x['sigma']}, prior_var={x['prior_var']}: {x['log_loss']:.4f} "
                f"(acc {x['accuracy']:.3f})" for x in sorted(out['tuning'], key=lambda z: z['log_loss'])]]
    (INGRAM / f"{fname}_report.md").write_text("\n".join(lines) + "\n")
    print(f"{eval_year}: log loss {t['log_loss']:.4f} (gap {out['log_loss_gap_vs_ingram']:+.4f}), "
          f"acc {t['accuracy']:.4f}" + (f" | gate {'PASSED' if passed else 'NOT passed'}" if is_gate else ""))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2014, help="season to score (default 2014, the Ingram gate)")
    ap.add_argument("--tune-year", type=int, default=None, help="season to tune on (default: year-1)")
    a = ap.parse_args()
    run(a.year, a.tune_year)
    sys.exit(0)
