"""Real-data read: ordinary vs. robust on a DEVELOPMENT season (default 2014).

Not the synthetic gate and NOT the 2025 test. This asks the direct question the
synthetic world can only approximate: on a real ATP season that already contains
the contamination (injury/tilt/retirement-truncated matches), does the Huber cap
change held-out match forecasting at all?

Guardrails (CLAUDE.md non-negotiables):
  * Same likelihood, same data, both arms — only the estimator (robust_c) differs.
  * Same frozen sigma/prior from Gate 1 for BOTH arms (sigma is NOT tuned per arm
    here; that is a Stage-4 develop step). This isolates the cap's effect.
  * Same iid analytic forecast for both arms — isolates ESTIMATION from any
    per-arm forecast-contamination rule (that rule is Stage 3/4).
  * Fixed, pre-declared caps (robust.DEFAULT_CAPS) — nothing tuned here.
  * 2025 test season is never read.
  * Paired per-match log-loss difference (both arms score the SAME matches), with
    a bootstrap clustered by tournament (conservative).
  * First-match-of-tournament reported separately from later rounds.

    python -m src.robust_check              # 2014
    python -m src.robust_check --year 2019
"""

from __future__ import annotations

import argparse

import numpy as np

from src import evaluate as ev, load, trials
from src.models import ordinary, robust
from src.util import RESULTS, ensure, utcnow, write_json

OUT = RESULTS / "robust_check"
SIGMA, PRIOR = 0.03, 1.0                       # Gate-1 chosen; frozen, shared by both arms


def run_arm(matches, alpha, eval_year, robust_c):
    """Filter chronologically; forecast each eval-year match before updating.
    Returns aligned lists (same match order for every arm)."""
    f = (ordinary.make(alpha, SIGMA, PRIOR) if robust_c is None
         else robust.make(alpha, SIGMA, robust_c, PRIOR))
    preds, tourneys, rounds = [], [], []
    for row in matches.itertuples():
        if row.year < eval_year:
            f.process(row, forecast_first=False)
        elif row.year == eval_year:
            preds.append(f.process(row, forecast_first=True))
            tourneys.append(row.tourney_id)
            rounds.append(int(row.round_order) if row.round_order == row.round_order else 0)
        else:
            break
    return (np.clip(np.array(preds), 1e-6, 1 - 1e-6), np.array(tourneys, dtype=object),
            np.array(rounds))


def _paired_ci(diff, tourneys, seed=0):
    """Mean paired per-match log-loss difference (ordinary - robust; + = robust
    better), bootstrap CLUSTERED BY TOURNAMENT."""
    order = {t: i for i, t in enumerate(dict.fromkeys(tourneys))}
    G = len(order)
    sum_d = np.zeros(G); cnt = np.zeros(G)
    for d, t in zip(diff, tourneys):
        g = order[t]; sum_d[g] += d; cnt[g] += 1
    per = {"sum_x": sum_d, "n": cnt}
    return ev.clustered_ci(per, lambda s: s["sum_x"] / max(s["n"], 1e-12), B=2000, seed=seed)


def run(eval_year=2014):
    ensure(OUT)
    from src.models.ability import alpha_by_surface
    matches = load.load_matches("M")
    alpha = alpha_by_surface(matches[matches.year < eval_year])
    caps = list(robust.DEFAULT_CAPS)

    ord_p, tourneys, rounds = run_arm(matches, alpha, eval_year, None)
    y = np.ones_like(ord_p)
    ll_ord = ev.log_loss(y, ord_p)

    # first match of tournament = opening round present in that tournament
    min_round = {}
    for t, r in zip(tourneys, rounds):
        min_round[t] = min(min_round.get(t, r), r)
    is_first = np.array([rounds[i] == min_round[tourneys[i]] for i in range(len(rounds))])

    arms = {"ordinary": {"log_loss": float(ll_ord.mean()),
                         "accuracy": float((ord_p > 0.5).mean()),
                         "mean_pred": float(ord_p.mean())}}
    comparisons = {}
    for c in caps:
        rob_p, _, _ = run_arm(matches, alpha, eval_year, c)
        ll_rob = ev.log_loss(y, rob_p)
        diff = ll_ord - ll_rob                       # + => robust better
        pt, lo, hi, se = _paired_ci(diff, tourneys)
        pt_f, lo_f, hi_f, _ = _paired_ci(diff[is_first], tourneys[is_first])
        pt_l, lo_l, hi_l, _ = _paired_ci(diff[~is_first], tourneys[~is_first])
        arms[f"robust_{c}"] = {"log_loss": float(ll_rob.mean()),
                               "accuracy": float((rob_p > 0.5).mean()),
                               "mean_pred": float(rob_p.mean())}
        comparisons[f"robust_{c}"] = {
            "cap": c,
            "paired_ll_gain": {"mean": pt, "lo": lo, "hi": hi, "se": se,
                               "sig": bool(lo > 0 or hi < 0)},
            "first_match": {"n": int(is_first.sum()), "mean": pt_f, "lo": lo_f, "hi": hi_f},
            "later_round": {"n": int((~is_first).sum()), "mean": pt_l, "lo": lo_l, "hi": hi_l},
            "n_capped_updates": None,
        }
        trials.log(stage="2b-robust-real", kind=f"{eval_year}/robust_{c}_vs_ordinary",
                   params={"eval_year": eval_year, "cap": c, "sigma": SIGMA, "prior_var": PRIOR,
                           "forecast": "iid_analytic", "same_sigma_both_arms": True},
                   metrics={"ordinary_ll": arms["ordinary"]["log_loss"],
                            "robust_ll": arms[f"robust_{c}"]["log_loss"],
                            "paired_ll_gain": pt, "ci_lo": lo, "ci_hi": hi})

    result = {"utc": utcnow(), "eval_year": eval_year, "n_matches": int(len(ord_p)),
              "sigma": SIGMA, "prior_var": PRIOR, "caps": caps,
              "note": "Development season, not test. Same sigma/forecast both arms; "
                      "only the Huber cap differs. Positive paired_ll_gain = robust better. "
                      "Tournament-clustered bootstrap.",
              "arms": arms, "comparisons": comparisons}
    write_json(OUT / f"compare_{eval_year}.json", result)
    _report(result)
    return result


def _report(r):
    y = r["eval_year"]
    L = [f"# Ordinary vs. robust on {y} (development read — NOT test)", "",
         f"_generated {r['utc']} · {r['n_matches']} matches · sigma={r['sigma']} "
         f"prior_var={r['prior_var']} (frozen, shared) · iid forecast both arms · "
         f"trials {trials.count()}_", "",
         "Same likelihood, same data, same forecast; only the Huber cap differs. "
         "Positive gain = robust better. Bootstrap clustered by tournament.", "",
         f"- **ordinary**: log loss **{r['arms']['ordinary']['log_loss']:.4f}**, "
         f"acc {r['arms']['ordinary']['accuracy']:.3f}"]
    for k, c in r["comparisons"].items():
        a = r["arms"][k]; g = c["paired_ll_gain"]
        L.append(f"- **{k}** (cap {c['cap']}): log loss **{a['log_loss']:.4f}**, "
                 f"acc {a['accuracy']:.3f} — paired gain vs ordinary "
                 f"**{g['mean']:+.4f}** [{g['lo']:+.4f}, {g['hi']:+.4f}] "
                 f"({'significant' if g['sig'] else 'within noise'})")
    L += ["", "## First match of tournament vs. later rounds (paired gain)", ""]
    for k, c in r["comparisons"].items():
        f_, l_ = c["first_match"], c["later_round"]
        L.append(f"- **{k}**: first ({f_['n']}) {f_['mean']:+.4f} "
                 f"[{f_['lo']:+.4f}, {f_['hi']:+.4f}]  ·  "
                 f"later ({l_['n']}) {l_['mean']:+.4f} [{l_['lo']:+.4f}, {l_['hi']:+.4f}]")
    (OUT / f"compare_{y}_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2014, help="development season to read (never 2025)")
    a = ap.parse_args()
    if a.year >= 2025:
        raise SystemExit("refusing: 2025+ is the single-use test season")
    run(a.year)
