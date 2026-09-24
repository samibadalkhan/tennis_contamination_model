"""Stage 2 — the synthetic burst gate.

A PASS/FAIL gate, run BEFORE any real development. It never selects a
hyperparameter (that would encode our assumptions into the model); it only
certifies that the robust filter can recover planted trajectories before we
trust it on real matches. See tennis-contamination.md "Synthetic data".

What it does
------------
1. Generate a synthetic tennis world with a KNOWN latent truth: every player
   carries true serve/return skill trajectories s_p(t), r_p(t).
   * Two independent GENERATORS of truth (multiple, never only the robust fit's
     own assumption — CLAUDE.md trap):
        "rw"  — Gaussian random walk (matches the model class)
        "arc" — smooth deterministic career arcs (model-mismatched stress test)
   * A simulated single-elimination BRACKET each week: winners advance, so
     impaired players exit early and selection-on-outcome is reproduced.
   * Fresh outcomes; nothing is ever spliced into real data.

2. Plant one of four CONDITIONS (tennis-contamination.md:149):
        none            — efficiency cost of robustness with nothing to resist
        point_burst     — a contiguous run of impaired points inside a match
        tournament      — a player impaired for a whole tournament (match scale)
        permanent_jump  — a lasting true-skill step (genuine change, not corruption)
   Random bursts and STRUCTURED bursts (tied to late rounds / fatigue) are both
   available.

3. Fit BOTH arms on the identical observations — ordinary (no cap) and robust
   at each PRE-DECLARED cap in robust.DEFAULT_CAPS — and measure recovery of the
   latent trajectory (RMSE vs. truth, gauge-fixed by centering skills across
   players each week). Per-player recovery is reported, not just the aggregate.

Gate (tennis-contamination.md:165): robust beats ordinary on recovery under
planted bursts, costs little with none planted, and lags boundedly on permanent
jumps.

    python -m src.synthetic          # run the full gate
    python -m src.synthetic --smoke  # tiny fast config, wiring check only
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

import numpy as np

from src import forecast, trials
from src.models import ordinary, robust
from src.util import RESULTS, ensure, utcnow, write_json

SYN = RESULTS / "synthetic"
_LOG_TRIALS = True                            # disabled in --smoke (wiring check only)
SURFACE = "hard"
ALPHA0 = float(np.log(0.64 / 0.36))          # logit(0.64): men's serve dominance
ALPHA = {SURFACE: ALPHA0}
BASE_DATE = date(2001, 1, 8)                  # arbitrary; only spacing matters


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --- configuration ------------------------------------------------------------

@dataclass
class GenConfig:
    n_players: int = 48
    draw: int = 16                 # players per weekly tournament (power of 2)
    n_weeks: int = 26
    best_of: int = 3
    base_svpt: int = 65            # mean service points per player per match
    skill_sd: float = 0.33         # spread of baseline serve/return skills (logit)
    sigma_true: float = 0.06       # rw drift per year (generator "rw")
    arc_amp: float = 0.30          # career-arc amplitude (generator "arc")
    generator: str = "rw"          # "rw" | "arc"
    # contamination
    condition: str = "none"        # none | point_burst | tournament | permanent_jump
    burst_rate: float = 0.15       # per-match (point_burst) or per player-week (tournament)
    severity: float = 2.0          # logit depression of an impaired player
    burst_len_frac: float = 0.25   # fraction of a match's points in a point burst
    jump_size: float = 0.5         # permanent-jump magnitude (logit)
    jump_frac: float = 0.4         # fraction of players who jump
    structured: bool = False       # raise burst rate in later rounds (fatigue)
    # filter dynamics (fixed, well-specified; NEVER tuned on synthetic)
    filter_sigma: float = 0.06     # set to sigma_true for rw; a fixed value for arc
    prior_var: float = 1.0
    seed: int = 0


ROUND_NAMES = {16: ["R16", "QF", "SF", "F"], 8: ["QF", "SF", "F"], 4: ["SF", "F"]}


# --- ground-truth trajectories ------------------------------------------------

def generate_truth(cfg: GenConfig, rng):
    """True serve/return skill arrays s[P, W], r[P, W] plus permanent-jump info."""
    P, W = cfg.n_players, cfg.n_weeks
    t = np.arange(W) / 52.0                                   # years
    s0 = rng.normal(0, cfg.skill_sd, P)
    r0 = rng.normal(0, cfg.skill_sd, P)
    if cfg.generator == "rw":
        step = cfg.sigma_true * np.sqrt(1 / 52.0)
        ds = np.cumsum(rng.normal(0, step, (P, W)), axis=1)
        dr = np.cumsum(rng.normal(0, step, (P, W)), axis=1)
        s = s0[:, None] + ds - ds[:, :1]                      # start at s0
        r = r0[:, None] + dr - dr[:, :1]
    elif cfg.generator == "arc":
        period = rng.uniform(1.0, 2.5, P)                     # multi-year arcs
        phase = rng.uniform(0, 2 * np.pi, P)
        amp = np.abs(rng.normal(0, cfg.arc_amp, P))
        s = s0[:, None] + amp[:, None] * np.sin(2 * np.pi * t[None, :] / period[:, None] + phase[:, None])
        amp2 = np.abs(rng.normal(0, cfg.arc_amp, P))
        phase2 = rng.uniform(0, 2 * np.pi, P)
        r = r0[:, None] + amp2[:, None] * np.sin(2 * np.pi * t[None, :] / period[:, None] + phase2[:, None])
    else:
        raise ValueError(f"unknown generator {cfg.generator!r}")

    jumped = np.zeros(P, bool)
    jump_week = np.full(P, -1)
    if cfg.condition == "permanent_jump":
        who = rng.random(P) < cfg.jump_frac
        for p in np.where(who)[0]:
            w = int(rng.integers(W // 4, 3 * W // 4))         # jump in the middle
            sign = -1.0 if rng.random() < 0.7 else 1.0        # mostly declines
            s[p, w:] += sign * cfg.jump_size
            r[p, w:] += sign * cfg.jump_size
            jumped[p] = True
            jump_week[p] = w
    return s, r, jumped, jump_week


# --- one match: observation + winner under contamination ----------------------

def _serve_segments(base_logit, c_self, f_self, c_other, f_other):
    """(prob, frac) segments for a server's point-win prob.
    c_self depresses this server's serve (impaired server); c_other is the
    returner's impairment, which RAISES this server's prob (impaired returner)."""
    f_clean = max(0.0, 1.0 - f_self - f_other)
    segs = []
    if f_self > 0:
        segs.append((float(sigmoid(base_logit - c_self)), f_self))
    if f_other > 0:
        segs.append((float(sigmoid(base_logit + c_other)), f_other))
    segs.append((float(sigmoid(base_logit)), f_clean))
    return segs


def _counts(segs, n, rng):
    k, used = 0, 0
    for i, (p, frac) in enumerate(segs):
        nb = n - used if i == len(segs) - 1 else int(round(frac * n))
        nb = max(0, min(nb, n - used))
        k += int(rng.binomial(nb, p))
        used += nb
    mean_p = sum(p * frac for p, frac in segs)
    return k, mean_p


def sim_match(sA, rA, sB, rB, cA, fA, cB, fB, cfg, rng):
    """Simulate one match. Returns (a_wins, kA, nA, kB, nB).
    cA/cB = impairment magnitude for A/B (logit); fA/fB = fraction of points affected."""
    lA = ALPHA0 + sA - rB                       # A serving vs B returning
    lB = ALPHA0 + sB - rA                       # B serving vs A returning
    segA = _serve_segments(lA, cA, fA, cB, fB)  # A's serve hurt by cA; helped by B's cB
    segB = _serve_segments(lB, cB, fB, cA, fA)
    nA = max(20, int(rng.normal(cfg.base_svpt, 8)))
    nB = max(20, int(rng.normal(cfg.base_svpt, 8)))
    kA, pA = _counts(segA, nA, rng)
    kB, pB = _counts(segB, nB, rng)
    a_wins = rng.random() < forecast.p_match(pA, pB, cfg.best_of)
    return a_wins, kA, nA, kB, nB


# --- season: bracket schedule + observations ----------------------------------

class Row:
    """Minimal match row with the attributes OnlineAbility.process expects."""
    __slots__ = ("p1", "p2", "surface", "best_of", "date", "year",
                 "p1_svpt", "p1_spw", "p2_svpt", "p2_spw", "week", "round_idx")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def build_season(cfg: GenConfig, s, r, jumped, jump_week, rng):
    """Play the weekly brackets; return chronological rows (winner = p1)."""
    rows = []
    nrounds = int(np.log2(cfg.draw))
    for w in range(cfg.n_weeks):
        field_players = rng.choice(cfg.n_players, size=cfg.draw, replace=False)
        rng.shuffle(field_players)
        alive = list(field_players)
        # tournament-scale impairment: chosen once per player-week
        impaired = {}
        if cfg.condition == "tournament":
            for p in field_players:
                if rng.random() < cfg.burst_rate:
                    impaired[p] = cfg.severity
        d = BASE_DATE + timedelta(weeks=w)
        for rnd in range(nrounds):
            names = ROUND_NAMES.get(cfg.draw, [str(i) for i in range(nrounds)])
            fatigue = (1.0 + 1.5 * rnd) if cfg.structured else 1.0
            nxt = []
            for i in range(0, len(alive), 2):
                A, B = alive[i], alive[i + 1]
                cA, fA, cB, fB = 0.0, 0.0, 0.0, 0.0
                if cfg.condition == "tournament":
                    if A in impaired: cA, fA = impaired[A], 1.0
                    if B in impaired: cB, fB = impaired[B], 1.0
                elif cfg.condition == "point_burst":
                    if rng.random() < min(1.0, cfg.burst_rate * fatigue):
                        if rng.random() < 0.5:
                            cA, fA = cfg.severity, cfg.burst_len_frac
                        else:
                            cB, fB = cfg.severity, cfg.burst_len_frac
                a_wins, kA, nA, kB, nB = sim_match(
                    s[A, w], r[A, w], s[B, w], r[B, w], cA, fA, cB, fB, cfg, rng)
                win, lose = (A, B) if a_wins else (B, A)
                kw, nw = (kA, nA) if a_wins else (kB, nB)
                kl, nl = (kB, nB) if a_wins else (kA, nA)
                rows.append(Row(p1=int(win), p2=int(lose), surface=SURFACE,
                                best_of=cfg.best_of, date=d, year=d.year,
                                p1_svpt=nw, p1_spw=kw, p2_svpt=nl, p2_spw=kl,
                                week=w, round_idx=rnd))
                nxt.append(win)
            alive = nxt
    return rows


# --- fit an arm and score trajectory recovery ---------------------------------

def recover(cfg: GenConfig, rows, s, r, arm):
    """Fit ``arm`` (an OnlineAbility) on rows; snapshot skills at each week end;
    return recovery RMSE (serve/return/combined, gauge-fixed by centering) plus
    per-player errors and, for jumps, per-week centered error of jumped players."""
    by_week = {}
    for row in rows:
        by_week.setdefault(row.week, []).append(row)

    sq_s, sq_r = [], []                       # (player, week) squared centered errors
    per_player = {}                           # player -> [combined sq errors]
    jump_curve = {}                           # week -> list of |centered err| for jumped players
    appeared = set()

    for w in range(cfg.n_weeks):
        for row in by_week.get(w, []):
            arm.process(row, forecast_first=False)
            appeared.add(row.p1); appeared.add(row.p2)
        ps = sorted(appeared)
        if len(ps) < 2:
            continue
        est_s = np.array([arm.sm[p] for p in ps]); est_s -= est_s.mean()
        est_r = np.array([arm.rm[p] for p in ps]); est_r -= est_r.mean()
        tru_s = np.array([s[p, w] for p in ps]);   tru_s -= tru_s.mean()
        tru_r = np.array([r[p, w] for p in ps]);   tru_r -= tru_r.mean()
        for j, p in enumerate(ps):
            es, er = (est_s[j] - tru_s[j]) ** 2, (est_r[j] - tru_r[j]) ** 2
            sq_s.append(es); sq_r.append(er)
            per_player.setdefault(p, []).append(es + er)
    return {
        "rmse_serve": float(np.sqrt(np.mean(sq_s))),
        "rmse_return": float(np.sqrt(np.mean(sq_r))),
        "rmse_combined": float(np.sqrt(np.mean(np.array(sq_s) + np.array(sq_r)))),
        "per_player_rmse": {int(p): float(np.sqrt(np.mean(v))) for p, v in per_player.items()},
    }


def realism_logloss(cfg: GenConfig, rows, arm_factory):
    """Ordinary iid-forecast log loss on this world (winner = p1, so y=1).
    The sim world should give Ingram roughly its real ~0.59 (calibrating the SIM,
    not the model)."""
    arm = arm_factory()
    preds = []
    for row in rows:
        preds.append(arm.process(row, forecast_first=True))
    preds = np.clip(np.array(preds), 1e-6, 1 - 1e-6)
    return float(-np.log(preds).mean())


# --- jump lag -----------------------------------------------------------------

def jump_lag(cfg: GenConfig, rows, s, r, jumped, jump_week, arm, band=0.15):
    """For jumped players, weeks from the jump until the centered estimate is
    within ``band`` of truth (censored at horizon end), and asymptotic error."""
    by_week = {}
    for row in rows:
        by_week.setdefault(row.week, []).append(row)
    appeared = set()
    err_by_week = {p: {} for p in np.where(jumped)[0]}
    for w in range(cfg.n_weeks):
        for row in by_week.get(w, []):
            arm.process(row, forecast_first=False)
            appeared.add(row.p1); appeared.add(row.p2)
        ps = sorted(appeared)
        if len(ps) < 2:
            continue
        est_s = np.array([arm.sm[p] for p in ps]); est_s -= est_s.mean()
        tru_s = np.array([s[p, w] for p in ps]);   tru_s -= tru_s.mean()
        idx = {p: j for j, p in enumerate(ps)}
        for p in err_by_week:
            if p in idx:
                err_by_week[p][w] = abs(est_s[idx[p]] - tru_s[idx[p]])
    lags, asymp = [], []
    for p, curve in err_by_week.items():
        jw = int(jump_week[p])
        after = {w: e for w, e in curve.items() if w >= jw}
        if not after:
            continue
        recovered = [w for w, e in sorted(after.items()) if e < band]
        lags.append((recovered[0] - jw) if recovered else (cfg.n_weeks - jw))
        tail = [e for w, e in after.items() if w >= cfg.n_weeks - 4]
        if tail:
            asymp.append(float(np.mean(tail)))
    return {"mean_lag_weeks": float(np.mean(lags)) if lags else float("nan"),
            "asymptotic_err": float(np.mean(asymp)) if asymp else float("nan"),
            "n_jumped": len(lags)}


# --- run one (generator, condition) cell across seeds -------------------------

def run_cell(cfg: GenConfig, caps, seeds):
    """Aggregate recovery for ordinary + each robust cap over ``seeds`` worlds."""
    keys = ["ordinary"] + [f"robust_{c}" for c in caps]
    acc = {k: {"serve": [], "return": [], "combined": []} for k in keys}
    lag = {k: {"lag": [], "asymp": []} for k in keys}
    realism = []
    worst_player = {k: [] for k in keys}     # worst per-player combined RMSE per seed
    for sd in range(seeds):
        c = replace(cfg, seed=sd)
        rng = np.random.default_rng(1000 + sd)
        s, r, jumped, jw = generate_truth(c, rng)
        rows = build_season(c, s, r, jumped, jw, rng)
        if c.condition == "none":
            realism.append(realism_logloss(c, rows, lambda: ordinary.make(ALPHA, c.filter_sigma, c.prior_var)))
        arms = {"ordinary": lambda: ordinary.make(ALPHA, c.filter_sigma, c.prior_var)}
        for cap in caps:
            arms[f"robust_{cap}"] = (lambda cap=cap: robust.make(ALPHA, c.filter_sigma, cap, c.prior_var))
        for k, factory in arms.items():
            rec = recover(c, rows, s, r, factory())
            acc[k]["serve"].append(rec["rmse_serve"])
            acc[k]["return"].append(rec["rmse_return"])
            acc[k]["combined"].append(rec["rmse_combined"])
            worst_player[k].append(max(rec["per_player_rmse"].values()))
            if c.condition == "permanent_jump":
                jl = jump_lag(c, rows, s, r, jumped, jw, factory())
                lag[k]["lag"].append(jl["mean_lag_weeks"])
                lag[k]["asymp"].append(jl["asymptotic_err"])
            if _LOG_TRIALS:
                trials.log(stage="2-synthetic",
                       kind=f"{c.generator}/{c.condition}/{k}",
                       params={"generator": c.generator, "condition": c.condition, "arm": k,
                               "severity": c.severity, "burst_rate": c.burst_rate,
                               "burst_len_frac": c.burst_len_frac, "structured": c.structured,
                               "filter_sigma": c.filter_sigma, "seed": sd},
                       metrics={"rmse_combined": rec["rmse_combined"],
                                "rmse_serve": rec["rmse_serve"],
                                "worst_player_rmse": max(rec["per_player_rmse"].values())})

    def summ(vals):
        a = np.array(vals, float)
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}

    out = {"generator": cfg.generator, "condition": cfg.condition,
           "arms": {k: {"combined": summ(acc[k]["combined"]),
                        "serve": summ(acc[k]["serve"]),
                        "return": summ(acc[k]["return"]),
                        "worst_player": summ(worst_player[k])} for k in keys}}
    if realism:
        out["realism_logloss"] = summ(realism)
    if cfg.condition == "permanent_jump":
        out["lag"] = {k: {"weeks": summ(lag[k]["lag"]), "asymptotic_err": summ(lag[k]["asymp"])}
                      for k in keys}
    return out


# --- gate logic ---------------------------------------------------------------

def evaluate_gate(cells, caps, weeks):
    """Apply the three pass/fail criteria across all (generator, condition) cells."""
    rob = [f"robust_{c}" for c in caps]
    verdict = {"recovery": {}, "efficiency": {}, "lag": {}}

    for cell in cells:
        tag = f"{cell['generator']}/{cell['condition']}"
        arms = cell["arms"]
        ordm = arms["ordinary"]["combined"]["mean"]
        if cell["condition"] in ("point_burst", "tournament"):
            # robust must beat ordinary for BOTH caps, by more than a seed-noise band
            checks = {}
            for k in rob:
                rm = arms[k]["combined"]["mean"]
                noise = arms[k]["combined"]["std"] + arms["ordinary"]["combined"]["std"]
                checks[k] = {"ordinary": ordm, "robust": rm,
                             "improvement": ordm - rm,
                             "clear": bool(ordm - rm > 0.5 * noise)}
            verdict["recovery"][tag] = {"pass": all(c["clear"] for c in checks.values()),
                                        "caps": checks}
        elif cell["condition"] == "none":
            checks = {}
            for k in rob:
                rm = arms[k]["combined"]["mean"]
                checks[k] = {"ordinary": ordm, "robust": rm,
                             "cost_ratio": rm / ordm,
                             "ok": bool(rm <= ordm * 1.10)}      # <=10% efficiency cost
            verdict["efficiency"][tag] = {"pass": all(c["ok"] for c in checks.values()),
                                          "caps": checks}
        elif cell["condition"] == "permanent_jump":
            lag = cell["lag"]
            checks = {}
            ord_asymp = lag["ordinary"]["asymptotic_err"]["mean"]
            for k in rob:
                lw = lag[k]["weeks"]["mean"]
                as_ = lag[k]["asymptotic_err"]["mean"]
                bounded = bool(lw < 0.9 * weeks) and bool(as_ <= max(ord_asymp * 1.6, 0.15))
                checks[k] = {"mean_lag_weeks": lw, "asymptotic_err": as_,
                             "ordinary_asymptotic_err": ord_asymp, "bounded": bounded}
            verdict["lag"][tag] = {"pass": all(c["bounded"] for c in checks.values()),
                                   "caps": checks}

    passed = {crit: all(v["pass"] for v in d.values()) if d else None
              for crit, d in verdict.items()}
    verdict["passed"] = passed
    verdict["gate_passed"] = bool(all(p for p in passed.values() if p is not None))
    return verdict


# --- sweeps -------------------------------------------------------------------

def severity_sweep(base: GenConfig, caps, seeds):
    """point_burst recovery advantage vs. burst severity and length."""
    out = []
    for sev in [1.0, 2.0, 3.0]:
        for flen in [0.15, 0.30]:
            cfg = replace(base, condition="point_burst", severity=sev, burst_len_frac=flen)
            cell = run_cell(cfg, caps, seeds)
            adv = {k: cell["arms"]["ordinary"]["combined"]["mean"] - cell["arms"][k]["combined"]["mean"]
                   for k in [f"robust_{c}" for c in caps]}
            out.append({"severity": sev, "burst_len_frac": flen, "advantage": adv})
    return out


def timescale_map(base: GenConfig, caps, seeds):
    """Recovery advantage as burst length is swept against the filter's drift σ.
    Maps where timescale separation (bursts faster than drift) holds or fails."""
    grid = []
    cap = caps[0]
    for fsig in [0.03, 0.06, 0.12]:
        for flen in [0.10, 0.25, 0.50]:
            cfg = replace(base, condition="point_burst", filter_sigma=fsig,
                          burst_len_frac=flen, severity=2.0)
            cell = run_cell(cfg, caps, seeds)
            adv = cell["arms"]["ordinary"]["combined"]["mean"] - cell["arms"][f"robust_{cap}"]["combined"]["mean"]
            grid.append({"filter_sigma": fsig, "burst_len_frac": flen,
                         "robust_advantage": adv})
    return grid


# --- driver -------------------------------------------------------------------

def run(smoke=False):
    global _LOG_TRIALS
    _LOG_TRIALS = not smoke
    ensure(SYN)
    caps = list(robust.DEFAULT_CAPS)
    seeds = 3 if smoke else 6
    weeks = 12 if smoke else 26
    conditions = ["none", "point_burst", "tournament", "permanent_jump"]
    generators = ["rw", "arc"]

    base = {"rw": GenConfig(n_weeks=weeks, generator="rw", sigma_true=0.06, filter_sigma=0.06),
            "arc": GenConfig(n_weeks=weeks, generator="arc", filter_sigma=0.10)}

    cells = []
    for gen in generators:
        for cond in conditions:
            cfg = replace(base[gen], condition=cond)
            cells.append(run_cell(cfg, caps, seeds))

    verdict = evaluate_gate(cells, caps, weeks)
    result = {"utc": utcnow(), "caps": caps, "seeds": seeds, "weeks": weeks,
              "trials_logged": trials.count(), "cells": cells, "gate": verdict}
    if not smoke:
        result["severity_sweep"] = severity_sweep(base["rw"], caps, seeds)
        result["timescale_map"] = timescale_map(base["rw"], caps, seeds)

    write_json(SYN / ("gate_smoke.json" if smoke else "gate.json"), result)
    _write_report(result, smoke)
    g = verdict["passed"]
    print(f"\nGate: recovery={g['recovery']} efficiency={g['efficiency']} lag={g['lag']} "
          f"=> {'PASSED' if verdict['gate_passed'] else 'NOT passed'}")
    return result


def _write_report(result, smoke):
    v = result["gate"]
    L = [f"# Stage 2 — synthetic burst gate{' (SMOKE)' if smoke else ''}", "",
         f"_generated {result['utc']} · {result['seeds']} seeds × {result['weeks']} weeks · "
         f"caps {result['caps']} · trials {result['trials_logged']}_", "",
         "Robust method: **Huber bounded-influence filter** (per-match innovation "
         "clipped at `robust_c·√H`). MMW / gradient-covariance filtering is a "
         "documented upgrade, not what this gate certifies.", "",
         "## Gate verdict", "",
         f"- **Recovery** (robust beats ordinary under bursts): "
         f"**{_yn(v['passed']['recovery'])}**",
         f"- **Efficiency** (robust costs <10% with nothing planted): "
         f"**{_yn(v['passed']['efficiency'])}**",
         f"- **Lag** (bounded tracking of permanent jumps): "
         f"**{_yn(v['passed']['lag'])}**",
         f"- **Overall: {'PASSED' if v['gate_passed'] else 'NOT passed'}**", "",
         "## Recovery RMSE by cell (combined serve+return, mean ± seed sd)", "",
         "| generator / condition | ordinary | " +
         " | ".join(f"robust {c}" for c in result["caps"]) + " |",
         "|---|---|" + "---|" * len(result["caps"])]
    for cell in result["cells"]:
        a = cell["arms"]
        row = [f"{cell['generator']} / {cell['condition']}",
               f"{a['ordinary']['combined']['mean']:.3f} ± {a['ordinary']['combined']['std']:.3f}"]
        for c in result["caps"]:
            row.append(f"{a[f'robust_{c}']['combined']['mean']:.3f} ± {a[f'robust_{c}']['combined']['std']:.3f}")
        L.append("| " + " | ".join(row) + " |")
    # realism
    rl = [c for c in result["cells"] if "realism_logloss" in c]
    if rl:
        L += ["", "## Realism check (sim should give Ingram ~0.59 log loss)", ""]
        for c in rl:
            L.append(f"- {c['generator']}: ordinary iid forecast log loss "
                     f"**{c['realism_logloss']['mean']:.3f} ± {c['realism_logloss']['std']:.3f}**")
    # lag detail
    lag_cells = [c for c in result["cells"] if "lag" in c]
    if lag_cells:
        L += ["", "## Permanent-jump lag (weeks to recover; asymptotic error)", ""]
        for c in lag_cells:
            parts = [f"{c['generator']}:"]
            for k, d in c["lag"].items():
                parts.append(f"{k} lag={d['weeks']['mean']:.1f}w asymp={d['asymptotic_err']['mean']:.3f}")
            L.append("- " + "  ".join(parts))
    if "timescale_map" in result:
        L += ["", "## Timescale map (robust advantage; + = robust better)", "",
              "| filter σ | burst len frac | robust advantage |", "|---|---|---|"]
        for g in result["timescale_map"]:
            L.append(f"| {g['filter_sigma']} | {g['burst_len_frac']} | {g['robust_advantage']:+.3f} |")
    if "severity_sweep" in result:
        cap0 = result["caps"][0]
        rk = f"robust_{cap0}"
        L += ["", f"## Severity × length sweep (robust advantage, tight cap {cap0})", "",
              "| severity | burst len frac | advantage |", "|---|---|---|"]
        for s in result["severity_sweep"]:
            L.append(f"| {s['severity']} | {s['burst_len_frac']} | {s['advantage'][rk]:+.3f} |")
    (SYN / ("gate_smoke_report.md" if smoke else "gate_report.md")).write_text("\n".join(L) + "\n")


def _yn(x):
    return "PASS" if x else ("n/a" if x is None else "FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny fast config (wiring check)")
    a = ap.parse_args()
    run(smoke=a.smoke)
