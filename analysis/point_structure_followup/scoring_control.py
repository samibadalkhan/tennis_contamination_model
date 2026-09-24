"""Stationary iid tennis control for the descriptive sequence statistic."""
from __future__ import annotations

import json
import os
import sys
import time

from analyze import HERE, ROOT, LAGS, VIEWS, LABELS, components, digest, ratio, utc, write_json
import numpy as np

sys.path.insert(0, str(ROOT))
from src.forecast import p_match

CELLS = [(0.65, 0.65, 3), (0.65, 0.65, 5), (0.65, 0.60, 3), (0.65, 0.60, 5)]
N = 512
SEED = 912703


def simulate(pa, pb, best_of, rng):
    """Record a complete iid match with seven-point tiebreaks in every set."""
    if not (0 <= pa <= 1 and 0 <= pb <= 1 and best_of in (3, 5)):
        raise ValueError("invalid match settings")
    sets_a = sets_b = games_a = games_b = a = b = 0
    need = best_of // 2 + 1
    server = int(rng.integers(2))
    tb = False
    tb_first = tb_index = 0
    servers, outcomes, sets, games, tiebreaks = [], [], [], [], []
    for point in range(100000):
        effective = (tb_first ^ (((tb_index + 1) // 2) % 2)) if tb else server
        won_serve = int(rng.random() < (pa if effective == 0 else pb))
        servers.append(effective)
        outcomes.append(won_serve)
        sets.append(sets_a + sets_b + 1)
        games.append(games_a + games_b + 1)
        tiebreaks.append(tb)
        won_a = won_serve if effective == 0 else 1 - won_serve
        a += won_a
        b += 1 - won_a
        if tb:
            tb_index += 1
            if max(a, b) >= 7 and abs(a - b) >= 2:
                sets_a += int(a > b)
                sets_b += int(b > a)
                server = 1 - tb_first
                tb = False
                games_a = games_b = a = b = 0
        elif max(a, b) >= 4 and abs(a - b) >= 2:
            games_a += int(a > b)
            games_b += int(b > a)
            a = b = 0
            server = 1 - server
            if max(games_a, games_b) >= 6 and abs(games_a - games_b) >= 2:
                sets_a += int(games_a > games_b)
                sets_b += int(games_b > games_a)
                games_a = games_b = 0
            elif games_a == games_b == 6:
                tb = True
                tb_first = server
                tb_index = 0
        if sets_a == need or sets_b == need:
            arrays = [np.asarray(x) for x in (servers, outcomes, sets, games, tiebreaks)]
            return arrays, int(sets_a > sets_b)
    raise RuntimeError("simulation exceeded point budget")


def run():
    started = time.monotonic()
    if (HERE / "SCORING_FROZEN.json").exists():
        raise FileExistsError("Preserve existing control; use a new directory for reruns")
    try:
        os.nice(10)
    except OSError:
        pass
    protected = {str(path): digest(path) for path in
                 [HERE / "analyze.py", HERE / "PROTOCOL.md", ROOT / "src/forecast.py"]}
    freeze = {"utc": utc(), "cells": CELLS, "matches_per_cell": N, "seed": SEED,
              "source_sha256": digest(__file__),
              "protocol_sha256": digest(HERE / "SCORING_CONTROL.md"),
              "protected_sha256": protected, "bootstraps": 1000,
              "status": "post_hoc_scoring_artifact_control"}
    write_json("SCORING_FROZEN.json", freeze)
    with (HERE / "trials.jsonl").open("a") as handle:
        handle.write(json.dumps({"utc": utc(), "stage": "scoring-only-control",
                                 "params": freeze, "note": "No physical impairment planted; local ledger only."}) + "\n")
    rng = np.random.default_rng(SEED)
    results = []
    for pa, pb, best_of in CELLS:
        values, wins = [], []
        for _ in range(N):
            (servers, x, sets, games, tiebreak), winner = simulate(pa, pb, best_of, rng)
            total = np.zeros((len(VIEWS), len(LAGS), 3))
            for player in (0, 1):
                use = servers == player
                total += components(x[use], sets[use], games[use],
                                    np.zeros(use.sum()), tiebreak[use])
            values.append(total.reshape(len(LABELS), 3))
            wins.append(winner)
        values = np.stack(values)
        num, den = values[:, :, 0], values[:, :, 1]
        observed = ratio(num.sum(axis=0), den.sum(axis=0))
        boot_rng = np.random.default_rng(SEED + best_of)
        boot = np.empty((1000, len(LABELS)))
        for k in range(len(boot)):
            ids = boot_rng.integers(N, size=N)
            boot[k] = ratio(num[ids].sum(axis=0), den[ids].sum(axis=0))
        if not np.isfinite(boot).all():
            raise ValueError("undefined bootstrap statistic")
        estimates = {}
        for j, label in enumerate(LABELS):
            lo, hi = np.quantile(boot[:, j], [.025, .975])
            estimates[label] = {"value": float(observed[j]), "lo": float(lo), "hi": float(hi)}
        expected = p_match(pa, pb, best_of)
        results.append({"pa": pa, "pb": pb, "best_of": best_of, "matches": N,
                        "simulated_win_fraction": float(np.mean(wins)),
                        "analytic_win_probability": float(expected),
                        "win_fraction_mc_standard_error": float(np.sqrt(expected * (1 - expected) / N)),
                        "estimates": estimates})
        print(f"Finished iid scoring control {pa}/{pb} best-of-{best_of}", flush=True)
    if any(digest(path) != sha for path, sha in protected.items()):
        raise ValueError("protected code changed during control")
    result = {"utc": utc(), "freeze": freeze, "cells": results,
              "seconds": time.monotonic() - started, "protected_inputs_unchanged": True}
    write_json("scoring_control.json", result)
    lines = ["# Can scoring alone create the apparent pattern?", "",
             "These simulations contain **no bursts, fatigue, injury, momentum or ability changes**. "
             "Each point is independent conditional on its server. Tennis scoring and stopping remain in place.", "",
             "| Serve probabilities | Format | Within-game lag 1 [Monte Carlo interval] | Across-game lag 1 [Monte Carlo interval] | Same-set lag 1 [Monte Carlo interval] |",
             "|---|---|---:|---:|---:|"]
    for cell in results:
        cells = []
        for view in ["within_game_set_centered", "across_games_set_centered", "within_set_set_centered"]:
            value = cell["estimates"][f"{view}:lag1"]
            cells.append(f"{value['value']:+.5f} [{value['lo']:+.5f}, {value['hi']:+.5f}]")
        lines.append(f"| {cell['pa']:.2f}, {cell['pb']:.2f} | Best of {cell['best_of']} | " + " | ".join(cells) + " |")
    lines += ["", "The simulations demonstrate which qualitative patterns can be produced by the "
              "measurement and scoring structure alone. They are illustrative matchups, not a matched "
              "null for the real cohort. Subtracting their estimates from the real data would not "
              "give a valid physical-effect estimate.", "",
              "## Simulator check", "",
              "| Serve probabilities / format | Simulated A win fraction | Exact iid probability | Monte Carlo SE |",
              "|---|---:|---:|---:|"]
    for cell in results:
        lines.append(f"| {cell['pa']:.2f}/{cell['pb']:.2f}, bo{cell['best_of']} | "
                     f"{cell['simulated_win_fraction']:.4f} | {cell['analytic_win_probability']:.4f} | "
                     f"{cell['win_fraction_mc_standard_error']:.4f} |")
    lines += ["", "All five lags, six views, seeds, and source hashes are in scoring_control.json. "
              "See SCORING_CONTROL.md for the protocol and report.md for the observed-data decomposition."]
    (HERE / "scoring_control.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
