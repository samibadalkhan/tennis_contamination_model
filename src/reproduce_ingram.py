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


def run():
    ensure(INGRAM)
    matches = load.load_matches("M")               # tour-level singles, serve stats
    # per-surface intercept from pre-2013 data only (no leakage into tuning)
    alpha = alpha_by_surface(matches[matches.year < 2013])

    # tune sigma / prior_var on 2013 (development), pick best log loss
    tuning = []
    best = None
    for sigma in SIGMA_GRID:
        for pv in PRIOR_GRID:
            m = run_filter(matches, alpha, sigma, pv, eval_year=2013)
            tuning.append({"sigma": sigma, "prior_var": pv, **m})
            trials.log(stage="1-ingram", kind="tune_2013",
                       params={"sigma": sigma, "prior_var": pv},
                       metrics={"val_log_loss": m["log_loss"], "val_acc": m["accuracy"]})
            if best is None or m["log_loss"] < best["log_loss"]:
                best = {"sigma": sigma, "prior_var": pv, "log_loss": m["log_loss"]}

    # reproduce on 2014 with the chosen setting
    alpha14 = alpha_by_surface(matches[matches.year < 2014])
    test = run_filter(matches, alpha14, best["sigma"], best["prior_var"], eval_year=2014)
    trials.log(stage="1-ingram", kind="reproduce_2014",
               params={"sigma": best["sigma"], "prior_var": best["prior_var"]},
               metrics={"log_loss": test["log_loss"], "accuracy": test["accuracy"]})

    ll_gap = test["log_loss"] - INGRAM_LOGLOSS
    passed = abs(ll_gap) <= 0.02                   # "near 0.592", small drift allowed
    out = {"utc": utcnow(),
           "ingram_target": {"log_loss": INGRAM_LOGLOSS, "accuracy": INGRAM_ACC, "season": 2014},
           "chosen_hyperparams": {"sigma": best["sigma"], "prior_var": best["prior_var"],
                                  "tuned_on": 2013},
           "reproduction_2014": test,
           "log_loss_gap_vs_ingram": round(ll_gap, 4),
           "gate_passed": bool(passed),
           "tuning_2013": tuning,
           "note": "Ordinary random-walk ability filter + iid analytic forecast. "
                   "Match-only, tour-level singles (G/M/A/F). If off, check surface "
                   "handling, Davis Cup inclusion, per-surface skills, first/second "
                   "serve split, and data revisions."}
    write_json(INGRAM / "reproduce.json", out)

    lines = [
        "# Gate 1 — reproduce Ingram (2019)", "",
        f"_generated {utcnow()} · trials logged: {trials.count()}_", "",
        f"Target (2014): log loss **{INGRAM_LOGLOSS}**, accuracy **{INGRAM_ACC}**.",
        f"Chosen on 2013: sigma={best['sigma']}, prior_var={best['prior_var']}.", "",
        f"## 2014 reproduction",
        f"- log loss **{test['log_loss']:.4f}** (gap {ll_gap:+.4f}), "
        f"accuracy **{test['accuracy']:.4f}**, on {test['n']} matches.",
        f"- **Gate {'PASSED' if passed else 'NOT passed'}** "
        f"({'within' if passed else 'outside'} 0.02 of Ingram).", "",
        "## 2013 tuning (log loss)",
        *[f"- sigma={t['sigma']}, prior_var={t['prior_var']}: {t['log_loss']:.4f} "
          f"(acc {t['accuracy']:.3f})" for t in tuning],
    ]
    (INGRAM / "reproduce_report.md").write_text("\n".join(lines) + "\n")
    print(f"2014: log loss {test['log_loss']:.4f} (gap {ll_gap:+.4f}), "
          f"acc {test['accuracy']:.4f} | gate {'PASSED' if passed else 'NOT passed'}")
    return out


if __name__ == "__main__":
    run()
    sys.exit(0)
