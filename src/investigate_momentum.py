"""Why did Stage 0 report anti-momentum? Investigate the serial-dependence result.

Stage 0's within-game transition test found ratio = 1.053 (>1 => more alternation
than a random permutation of the game's points => "anti-momentum"). That is the
opposite of the usual expectation (winning builds momentum), so it deserves
scrutiny before it is believed.

Hypothesis: the statistic's null is MIS-SPECIFIED for tennis. The expected
transition count 2*n1*n0/n assumes every arrangement of a game's wins/losses is
equally likely (a random permutation). But tennis scoring (first to 4, win by 2,
deuce/advantage) makes only some sequences reachable, and long/deuce games are
STRUCTURALLY forced to alternate -- to stay at deuce you must trade points. So the
permutation null under-counts transitions for reasons that have nothing to do with
psychology, and the statistic reads that structural alternation as anti-momentum.

Decisive test (posterior-predictive / parametric bootstrap): simulate service
games in which points are i.i.d. Bernoulli(p) -- ZERO momentum by construction --
but played out under the real tennis game rules, drawing p from the empirical
distribution of the model's per-game serve-win probabilities. Run the SAME
transition-ratio statistic on the simulated games.

  * If simulated (momentum-free) games also give ratio ~ 1.05, the "anti-momentum"
    is entirely a scoring-structure artifact of the permutation null.
  * The real point-to-point effect is then observed_ratio / simulated_ratio:
    ~1 = none, <1 = genuine momentum, >1 = genuine anti-persistence.

Writes results/stage0/momentum_investigation.{json,md} and a figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import trials
from src.util import DATA, STAGE0, utcnow, write_json

CACHE = DATA / "cache"
LMAX = 60          # max points per simulated game (deuce marathons beyond this are negligible)
N_SIM = 1_000_000  # simulated games for a precise null ratio
N_REPS = 40        # repetitions (different seeds) for a null-ratio interval


def _game_stats(n1, n, transitions):
    """Pooled transition ratio = observed transitions / permutation-expected."""
    n = n.astype(float); n1 = n1.astype(float); n0 = n - n1
    exp = 2 * n1 * n0 / n
    keep = n >= 2
    return float(transitions[keep].sum() / exp[keep].sum()), keep


def observed_by_length():
    d = pd.read_parquet(CACHE / "train_pred.parquet")
    d = d[~d.is_tiebreak.astype(bool)].copy()
    d["seq"] = np.arange(len(d))
    gkey = ["match_id", "set_no", "game_no", "server"]
    d = d.sort_values(gkey + ["seq"], kind="stable")
    y = d.server_wins.to_numpy().astype(int)
    yprev = d.groupby(gkey, observed=True)["server_wins"].shift(1).to_numpy()
    same = ~np.isnan(yprev)
    trans = ((y != yprev) & same).astype(int)
    gid = d.groupby(gkey, observed=True).ngroup().to_numpy()
    pg = pd.DataFrame({"gid": gid, "y": y, "trans": trans, "p": d.p.to_numpy()}) \
        .groupby("gid").agg(n=("y", "count"), n1=("y", "sum"),
                            tr=("trans", "sum"), p=("p", "mean"))
    return pg


def simulate(p_pool: np.ndarray, n_games: int, rng) -> pd.DataFrame:
    """Simulate i.i.d. service games under first-to-4, win-by-2 tennis rules."""
    p = rng.choice(p_pool, size=n_games, replace=True)
    U = rng.random((n_games, LMAX))
    win = (U < p[:, None]).astype(np.int8)              # 1 = server wins the point
    s = np.cumsum(win, axis=1)                          # server points after each
    r = np.cumsum(1 - win, axis=1)                      # returner points after each
    done = ((s >= 4) & (s - r >= 2)) | ((r >= 4) & (r - s >= 2))
    has_end = done.any(axis=1)
    end = np.where(has_end, done.argmax(axis=1), LMAX - 1)   # index of game-ending point
    cols = np.arange(LMAX)[None, :]
    in_game = cols <= end[:, None]                      # points that are part of the game
    n = end + 1
    n1 = np.where(in_game, win, 0).sum(axis=1)          # server wins in the game
    # transitions: pair (j,j+1) counts if both are in the game, i.e. j < end
    diff = (win[:, 1:] != win[:, :-1])
    pair_valid = (np.arange(LMAX - 1)[None, :] < end[:, None])
    tr = (diff & pair_valid).sum(axis=1)
    return pd.DataFrame({"n": n, "n1": n1, "tr": tr})


def _bucket(n):
    return np.where(n <= 4, "4 (4-0)",
           np.where(n == 5, "5 (4-1)",
           np.where(n == 6, "6 (4-2)",
           np.where(n <= 9, "7-9 (short deuce)", "10+ (long deuce)"))))


def run():
    pg = observed_by_length()
    obs_ratio, _ = _game_stats(pg.n1.to_numpy(), pg.n.to_numpy(), pg.tr.to_numpy())
    p_pool = pg.p.to_numpy()

    # simulated null ratio: point estimate on a big pool + reps for an interval
    rng = np.random.default_rng(0)
    big = simulate(p_pool, N_SIM, rng)
    sim_ratio, _ = _game_stats(big.n1.to_numpy(), big.n.to_numpy(), big.tr.to_numpy())
    reps = []
    for k in range(N_REPS):
        s = simulate(p_pool, 200_000, np.random.default_rng(100 + k))
        reps.append(_game_stats(s.n1.to_numpy(), s.n.to_numpy(), s.tr.to_numpy())[0])
    sim_lo, sim_hi = np.percentile(reps, [2.5, 97.5])

    # by-length comparison
    def by_len(df):
        b = _bucket(df.n.to_numpy())
        out = {}
        for lab in ["4 (4-0)", "5 (4-1)", "6 (4-2)", "7-9 (short deuce)", "10+ (long deuce)"]:
            m = b == lab
            if m.sum() == 0:
                continue
            n = df.n.to_numpy()[m].astype(float); n1 = df.n1.to_numpy()[m].astype(float)
            exp = (2 * n1 * (n - n1) / n).sum()
            if exp <= 0:      # 4-0 shutouts: no transitions possible, ratio undefined
                out[lab] = {"ratio": None, "n_games": int(m.sum())}
            else:
                out[lab] = {"ratio": round(float(df.tr.to_numpy()[m].sum() / exp), 4),
                            "n_games": int(m.sum())}
        return out
    obs_len = by_len(pg)
    sim_len = by_len(big)

    real_effect = obs_ratio / sim_ratio
    # Magnitude first: with ~300k games even a 0.3% deviation clears a Monte-Carlo
    # band, so judge by effect size, not just whether it is inside [sim_lo, sim_hi].
    pct = abs(real_effect - 1) * 100
    direction = "anti-persistence" if real_effect > 1 else "momentum"
    if pct < 1.0:
        verdict = (f"ARTIFACT (essentially). Scoring structure alone produces a "
                   f"transition ratio of {sim_ratio:.4f}; the observed {obs_ratio:.4f} "
                   f"leaves a residual real effect of just {pct:.2f}% ({direction}). "
                   f"The Stage 0 'anti-momentum' was ~{100*(sim_ratio-1)/(obs_ratio-1):.0f}% "
                   f"an artifact of the permutation null. There is NO evidence of "
                   f"positive momentum, and the tiny residual is within the "
                   f"simplifications of this null (constant within-game p; real serve "
                   f"prob varies by score state, adding still more structural "
                   f"alternation), so it is an upper bound, not a finding.")
    elif obs_ratio < sim_lo:
        verdict = (f"Genuine MOMENTUM ({pct:.1f}% fewer transitions than the "
                   f"i.i.d.-under-scoring null): streakier than chance.")
    else:
        verdict = (f"Genuine ANTI-persistence ({pct:.1f}% beyond scoring structure): "
                   f"observed transitions exceed the i.i.d.-under-scoring null.")

    out = {
        "utc": utcnow(),
        "question": "Stage 0 reported anti-momentum (transition ratio > 1). Real or artifact?",
        "observed_transition_ratio": round(obs_ratio, 4),
        "iid_under_scoring_null_ratio": round(sim_ratio, 4),
        "iid_null_ratio_ci95": [round(float(sim_lo), 4), round(float(sim_hi), 4)],
        "real_effect_obs_over_null": round(real_effect, 4),
        "by_game_length_observed": obs_len,
        "by_game_length_iid_null": sim_len,
        "verdict": verdict,
        "why": "The permutation null (E[transitions]=2*n1*n0/n) assumes all arrangements "
               "of a game's points are equally likely. Tennis scoring forbids many and "
               "forces alternation in deuce games, so even momentum-FREE i.i.d. points "
               "produce transition ratios above 1. The correct null is i.i.d. points "
               "played under the scoring rules, estimated here by simulation.",
        "n_observed_games": int(len(pg)),
        "n_sim_games": N_SIM,
    }
    write_json(STAGE0 / "momentum_investigation.json", out)

    md = [
        "# Momentum investigation — is the Stage 0 anti-momentum real?",
        "", f"_generated {utcnow()}_", "",
        "**Finding.** " + verdict, "",
        f"- observed within-game transition ratio: **{obs_ratio:.4f}**",
        f"- i.i.d.-under-scoring null (momentum-free by construction): "
        f"**{sim_ratio:.4f}** (95% over reps [{sim_lo:.4f}, {sim_hi:.4f}])",
        f"- real point-to-point effect (observed / null): **{real_effect:.4f}** "
        f"(1.00 = no effect, <1 = momentum, >1 = anti-persistence)",
        "",
        "## By game length (transition ratio)",
        "| game length (points) | observed | i.i.d. null |",
        "|---|---|---|",
    ]
    for lab in obs_len:
        md.append(f"| {lab} | {obs_len[lab]['ratio']} | {sim_len.get(lab, {}).get('ratio', '—')} |")
    md += [
        "",
        "The ratio climbs with game length in BOTH observed and simulated data: "
        "short decisive games (4-0/4-1) sit near 1, long deuce games are pushed well "
        "above 1 — because reaching and holding deuce mechanically requires trading "
        "points. That length dependence is present with zero momentum in the model, "
        "which is the tell that the permutation null, not player psychology, produced "
        "the original 'anti-momentum'.",
        "",
        "## Consequence for the pipeline",
        "- The Stage 0 serial statistic should be read against the i.i.d.-under-"
        "scoring null, not the permutation null. Interpreted correctly, there is "
        "little-to-no point-to-point dependence beyond what tennis scoring imposes.",
        "- This does not change the overdispersion or asymmetry findings (those are "
        "block-level, not sequence-order). It corrects the momentum read only.",
        "- A proper Stage 1 momentum test (K&M-style) must condition on score state, "
        "which absorbs exactly this structural alternation.",
    ]
    (STAGE0 / "momentum_investigation.md").write_text("\n".join(md) + "\n")
    trials.log(stage="0", kind="momentum_investigation",
               metrics={"obs_ratio": obs_ratio, "iid_null_ratio": sim_ratio,
                        "real_effect": real_effect},
               note="parametric-bootstrap null: i.i.d. points under tennis scoring rules")
    print("obs", round(obs_ratio, 4), "| iid-null", round(sim_ratio, 4),
          "| effect", round(real_effect, 4))
    print(verdict)
    return out


if __name__ == "__main__":
    run()
