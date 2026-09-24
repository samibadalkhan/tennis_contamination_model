"""Isolated descriptive point-structure analysis; see PROTOCOL.md."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLCONFIGDIR"] = str(HERE / ".mplconfig")

import numpy as np
import pandas as pd

LAGS = (1, 2, 4, 8, 16)
VIEWS = ("match", "within_set_match_centered", "within_set_set_centered",
         "within_game_set_centered", "across_games_set_centered",
         "within_set_no_tiebreak")
BOOTSTRAPS = 1000
SEED = 73019
LABELS = [f"{view}:lag{lag}" for view in VIEWS for lag in LAGS]


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(name, value):
    target = HERE / name
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(target)


def components(x, sets, games, segments, tiebreak, min_match=20, min_set=12):
    """Sum adjusted covariance, variance and eligible pairs for one server.

    Segments are derived from the full match stream before selecting a server,
    so an opponent's intervening service game is not treated as a data gap.
    """
    x = np.asarray(x, dtype=float)
    sets, games, segments, tiebreak = map(np.asarray, (sets, games, segments, tiebreak))
    if not all(len(a) == len(x) for a in (sets, games, segments, tiebreak)):
        raise ValueError("unaligned point arrays")
    if not np.isin(x, [0, 1]).all():
        raise ValueError("non-binary outcomes")
    total = np.zeros((len(VIEWS), len(LAGS), 3))
    if len(x) < min_match or x.min() == x.max():
        return total
    p = x.mean()
    v = p * (1 - p)
    residual = x - p
    set_p = np.zeros(len(x))
    set_n = np.zeros(len(x), dtype=int)
    for value in np.unique(sets):
        mask = sets == value
        set_p[mask] = x[mask].mean()
        set_n[mask] = mask.sum()
    set_v = set_p * (1 - set_p)
    for j, lag in enumerate(LAGS):
        if len(x) <= lag:
            continue
        left, right = np.arange(len(x) - lag), np.arange(lag, len(x))
        contiguous = segments[left] == segments[right]
        matched = (contiguous & (sets[left] == sets[right]) &
                   (set_n[left] >= min_set) & (set_v[left] > 0))
        same_game = games[left] == games[right]
        masks = (contiguous, matched, matched, matched & same_game,
                 matched & ~same_game, matched & ~tiebreak[left] & ~tiebreak[right])
        base_num = residual[left] * residual[right] + v / (len(x) - 1)
        local_num = ((x[left] - set_p[left]) * (x[right] - set_p[right]) +
                     np.divide(set_v[left], set_n[left] - 1,
                               out=np.zeros(len(left)), where=set_n[left] > 1))
        for i, mask in enumerate(masks):
            num = base_num if i < 2 else local_num
            den = np.full(len(left), v) if i < 2 else set_v[left]
            total[i, j] = num[mask].sum(), den[mask].sum(), mask.sum()
    return total


def build_components(points):
    records, values = [], []
    audit = {"source_rows": len(points), "excluded_matches_inconsistent_players": 0,
             "point_number_gaps": 0}
    points = points.copy()
    points["point_no"] = pd.to_numeric(points.point_no, errors="raise")
    if points.point_no.isna().any() or (points.point_no % 1 != 0).any():
        raise ValueError("invalid source point order")
    for _, group in points.groupby("match_id", sort=False):
        if np.any(np.diff(group.point_no.to_numpy()) <= 0):
            raise ValueError("source points are not strictly ordered")
    keep = ((points.year <= 2024) & points.tour.isin(["M", "W"]) &
            ~points.ambiguous_identity & points.server.notna() & points.returner.notna() &
            points.server_wins.isin([0, 1]) & (points.set_no >= 1) &
            (points.game_no >= 1) & (points.point_no >= 1) & points.is_tiebreak.notna())
    audit["excluded_rows"] = int((~keep).sum())
    points = points.loc[keep]
    for match_id, group in points.groupby("match_id", sort=False):
        players = sorted(set(group.server) | set(group.returner))
        if (len(players) != 2 or (group.server == group.returner).any() or
                group.tour.nunique() != 1 or group.year.nunique() != 1 or group.slam.nunique() != 1):
            audit["excluded_matches_inconsistent_players"] += 1
            continue
        segment = np.r_[0, np.cumsum(np.diff(group.point_no.to_numpy()) != 1)]
        audit["point_number_gaps"] += int(segment[-1])
        totals = np.zeros((len(VIEWS), len(LAGS), 3))
        for server in players:
            mask = group.server.to_numpy() == server
            totals += components(group.server_wins.to_numpy()[mask],
                                 group.set_no.to_numpy()[mask], group.game_no.to_numpy()[mask],
                                 segment[mask], group.is_tiebreak.to_numpy(dtype=bool)[mask])
        first = group.iloc[0]
        records.append({"match_id": match_id, "tour": first.tour,
                        "event": f"{first.tour}-{first.year}-{first.slam}",
                        "p1": players[0], "p2": players[1], "n_points": len(group)})
        values.append(totals.reshape(len(LABELS), 3))
    meta = pd.DataFrame(records)
    audit.update(retained_matches=len(meta), retained_points=int(meta.n_points.sum()))
    return meta, np.stack(values), audit


def ratio(num, den):
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)


def summarize(meta, sums):
    num, den = sums[:, :, 0], sums[:, :, 1]
    observed = ratio(num.sum(axis=0), den.sum(axis=0))
    pair_counts = sums[:, :, 2].sum(axis=0).astype(int)
    event_code, events = pd.factorize(meta.event)
    one = (meta.event + "|" + meta.p1).to_numpy()
    two = (meta.event + "|" + meta.p2).to_numpy()
    nodes, node_codes = np.unique(np.r_[one, two], return_inverse=True)
    a, b = node_codes[:len(meta)], node_codes[len(meta):]
    estimates = {key: {"association": float(observed[i]), "pairs": int(pair_counts[i]),
                       "intervals": {}} for i, key in enumerate(LABELS)}
    differences = {str(lag): {"change": float(observed[len(LAGS) + j] -
                                                observed[2 * len(LAGS) + j]),
                              "intervals": {}} for j, lag in enumerate(LAGS)}
    for scheme in ("match", "event", "player_event"):
        rng = np.random.default_rng(SEED)
        draws = np.zeros((BOOTSTRAPS, len(LABELS)))
        for replicate in range(BOOTSTRAPS):
            if scheme == "match":
                w = np.bincount(rng.integers(len(meta), size=len(meta)), minlength=len(meta))
            elif scheme == "event":
                counts = np.bincount(rng.integers(len(events), size=len(events)), minlength=len(events))
                w = counts[event_code]
            else:
                counts = rng.poisson(1.0, len(nodes))
                w = counts[a] * counts[b]
            draws[replicate] = ratio(w @ num, w @ den)
        if not np.isfinite(draws).all():
            raise ValueError("undefined bootstrap ratio; do not silently drop replicates")
        for i, key in enumerate(LABELS):
            lo, hi = np.quantile(draws[:, i], [.025, .975])
            estimates[key]["intervals"][scheme] = {"lo": float(lo), "hi": float(hi)}
        for j, lag in enumerate(LAGS):
            delta = draws[:, len(LAGS) + j] - draws[:, 2 * len(LAGS) + j]
            lo, hi = np.quantile(delta, [.025, .975])
            differences[str(lag)]["intervals"][scheme] = {"lo": float(lo), "hi": float(hi)}
    return {"matches": len(meta), "events": len(events), "player_events": len(nodes),
            "points": int(meta.n_points.sum()), "estimates": estimates,
            "paired_change_after_set_centering": differences}


def protect_inputs():
    paths = list((ROOT / "src").rglob("*.py"))
    paths += [ROOT / "CLAUDE.md", ROOT / "data/processed/points.parquet",
              ROOT / "data/processed/BUILD.json", ROOT / "data/MANIFEST.json"]
    paths += list((ROOT / "results").rglob("*PROTOCOL*.md"))
    paths += list((ROOT / "results").rglob("*FROZEN*.json"))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def verify_raw():
    manifest = json.loads((ROOT / "data/MANIFEST.json").read_text())
    selected = [entry for entry in manifest["files"]
                if entry["file"].startswith("slam_pointbypoint/") and
                entry["file"].endswith(("-points.csv", "-matches.csv")) and
                "doubles" not in entry["file"] and "mixed" not in entry["file"]]
    if not selected:
        raise ValueError("no Slam source manifest entries")
    for entry in selected:
        if digest(ROOT / "data" / entry["file"]) != entry["sha256"]:
            raise ValueError(f"raw source hash mismatch: {entry['file']}")
    return len(selected)


def report(result):
    lines = ["# Point structure: exploratory follow-up", "",
             "This describes where observed dependence appears. Its permutation reference "
             "does not reproduce tennis scoring or stopping, so it does not establish fatigue, "
             "injury, momentum, or a physical burst duration.", "",
             f"Retained {result['audit']['retained_points']:,} points in "
             f"{result['audit']['retained_matches']:,} matches; no pairs bridge recorded gaps.", "",
             "## Lag-one decomposition", "",
             "Intervals below use both canonical-player/event memberships. "
             "Match and event intervals, all five lags, and paired differences are in results.json.", "",
             "| View | Men: association [95% interval] | Women: association [95% interval] |",
             "|---|---:|---:|"]
    for view in VIEWS:
        cells = []
        for tour in ("M", "W"):
            item = result["groups"][tour]["estimates"][f"{view}:lag1"]
            interval = item["intervals"]["player_event"]
            cells.append(f"{item['association']:+.5f} [{interval['lo']:+.5f}, {interval['hi']:+.5f}]")
        lines.append(f"| {view} | " + " | ".join(cells) + " |")
    lines += ["", "## Matched change after accounting for set-specific serve rates", "",
              "Positive values mean the association statistic decreases after set centering. "
              "Both estimates use exactly the same pairs. This is not explained variance.", ""]
    for tour in ("M", "W"):
        item = result["groups"][tour]["paired_change_after_set_centering"]["1"]
        interval = item["intervals"]["player_event"]
        lines.append(f"- {tour}: {item['change']:+.5f} "
                     f"[{interval['lo']:+.5f}, {interval['hi']:+.5f}].")
    lines += ["", "![Association over service-point lags](lag_structure.png)", "",
              "## Reading the evidence", "",
              "Set centering uses future outcomes from the same set, so these are descriptive "
              "statistics, not out-of-sample prediction scores. Within-game and across-game pair "
              "selection can itself induce structure under iid tennis scoring. In particular, "
              "a bootstrap interval excluding zero is not a scoring-aware rejection of iid points.", "",
              "Large changes after set centering motivate studying slower set-level changes and "
              "selection. Residual association between games motivates a scoring-aware null check. "
              "Neither observation identifies physiological fatigue or temporary injury. A separate "
              "prospective forecasting comparison would be required to claim predictive value.", "",
              "The processed identities are canonical names with known ambiguous keys removed; "
              "the player/event bootstrap is a sensitivity analysis and does not resolve every "
              "cross-event dependency. Intervals are unadjusted exploratory comparisons.", "",
              "## Reproduce", "", "```bash",
              ".venv-independent/bin/python analysis/point_structure_followup/analyze.py", "```", "",
              "Execution refuses to overwrite a completed run. Review PROTOCOL.md, FROZEN.json, "
              "audit.json, results.json and the local trials.jsonl for provenance. "
              "All new files stay in this directory; active model files and shared trial log are untouched."]
    (HERE / "report.md").write_text("\n".join(lines) + "\n")


def plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)
    chosen = [("within_set_match_centered", "Same-set pairs, match centering"),
              ("within_set_set_centered", "Same pairs, set centering"),
              ("across_games_set_centered", "Across games, set centering")]
    for ax, tour, title in zip(axes, ("M", "W"), ("Men", "Women")):
        for view, label in chosen:
            items = [result["groups"][tour]["estimates"][f"{view}:lag{lag}"] for lag in LAGS]
            y = [item["association"] for item in items]
            lo = [item["intervals"]["player_event"]["lo"] for item in items]
            hi = [item["intervals"]["player_event"]["hi"] for item in items]
            line, = ax.plot(LAGS, y, "o-", label=label, linewidth=1.5, markersize=4)
            ax.fill_between(LAGS, lo, hi, alpha=.13, color=line.get_color())
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(LAGS, [str(x) for x in LAGS])
        ax.set_xlabel("Lag in a player's service points")
        ax.set_title(title)
    axes[0].set_ylabel("Adjusted descriptive association")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Point structure after accounting for set-level rates\n"
                 "Shading: player/event 95% intervals; scoring-aware null not tested", fontsize=11)
    fig.savefig(HERE / "lag_structure.png", dpi=180)
    plt.close(fig)


def main():
    started = time.monotonic()
    if (HERE / "results.json").exists() or (HERE / "FROZEN.json").exists():
        raise FileExistsError("Existing run: preserve its frozen evidence; use a new directory")
    try:
        os.nice(10)
    except OSError:
        pass
    protected = protect_inputs()
    verified = verify_raw()
    freeze = {"utc": utc(), "source_sha256": digest(__file__),
              "protocol_sha256": digest(HERE / "PROTOCOL.md"), "protected_sha256": protected,
              "raw_files_verified": verified, "lags": LAGS, "views": VIEWS,
              "bootstraps": BOOTSTRAPS, "seed": SEED, "threads": 1,
              "status": "exploratory_direct_measurement"}
    write_json("FROZEN.json", freeze)
    with (HERE / "trials.jsonl").open("a") as ledger:
        ledger.write(json.dumps({"utc": utc(), "stage": "direct-point-structure-followup",
                                 "params": {k: freeze[k] for k in ["lags", "views", "bootstraps", "seed"]},
                                 "note": "Local ledger avoids writes to live shared trial log; no model fitting."}) + "\n")
    write_json("status.json", {"utc": utc(), "state": "reading_processed_points"})
    columns = ["match_id", "slam", "year", "tour", "set_no", "game_no", "point_no",
               "server", "returner", "server_wins", "is_tiebreak", "ambiguous_identity"]
    points = pd.read_parquet(ROOT / "data/processed/points.parquet", columns=columns)
    meta, sums, audit = build_components(points)
    del points
    write_json("audit.json", audit)
    # Non-observation aggregate components only; retained for independent checks.
    packed = meta.copy()
    for j, label in enumerate(LABELS):
        for k, suffix in enumerate(("numerator", "denominator", "pairs")):
            packed[f"{label}:{suffix}"] = sums[:, j, k]
    packed.to_csv(HERE / "match_components.csv", index=False)
    result = {"utc": utc(), "audit": audit, "freeze": freeze, "groups": {}}
    for tour in ("M", "W"):
        write_json("status.json", {"utc": utc(), "state": "bootstrap", "tour": tour})
        print(f"Bootstrap {tour}", flush=True)
        selected = meta.tour == tour
        result["groups"][tour] = summarize(meta.loc[selected], sums[selected])
    changed = [name for name, sha in protected.items() if digest(ROOT / name) != sha]
    result["protected_inputs_unchanged"] = not changed
    result["changed_protected_paths"] = changed
    if changed:
        raise ValueError(f"Inputs changed during analysis: {changed}")
    result["seconds"] = time.monotonic() - started
    write_json("results.json", result)
    report(result)
    plot(result)
    write_json("status.json", {"utc": utc(), "state": "complete", "seconds": result["seconds"]})
    print(json.dumps({"state": "complete", "seconds": result["seconds"], "audit": audit}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json("status.json", {"utc": utc(), "state": "failed", "error": repr(exc)})
        raise
