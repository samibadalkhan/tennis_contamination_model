"""Direct within-match dependence measurements in the Slam point data."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import trials
from src.util import DATA, RESULTS, utcnow, write_json


SOURCE = DATA / "processed" / "points.parquet"
OUT = RESULTS / "point_dependence"
BOOTSTRAPS = 3000
SEED = 2907


def sequence_components(frame: pd.DataFrame, minimum_points: int = 20) -> pd.DataFrame:
    """Components for a permutation-adjusted lag-one association.

    The conditional null randomly orders the observed wins and losses within
    each match-server sequence, preserving its length and success rate.
    """
    records = []
    for (match_id, server), group in frame.groupby(["match_id", "server"], sort=False):
        x = group.server_wins.to_numpy(dtype=float)
        n = len(x)
        k = int(x.sum())
        if n < minimum_points or k == 0 or k == n:
            continue
        p = k / n
        pairs = n - 1
        observed_agree = float(np.sum(x[:-1] == x[1:]))
        expected_agree = pairs * (k * (k - 1) + (n - k) * (n - k - 1)) / (n * (n - 1))
        observed_cross = float(np.sum((x[:-1] - p) * (x[1:] - p)))
        expected_cross = -p * (1 - p)
        records.append({
            "match_id": match_id, "server": server, "n": n,
            "observed_agree": observed_agree, "expected_agree": expected_agree,
            "pairs": pairs, "observed_cross": observed_cross,
            "expected_cross": expected_cross,
            "rho_denominator": pairs * p * (1 - p),
        })
    return pd.DataFrame(records)


def set_components(frame: pd.DataFrame, minimum_set_points: int = 12) -> pd.DataFrame:
    """Conditional Pearson dispersion of set rates within match-server."""
    counts = (frame.groupby(["match_id", "server", "set_no"], sort=False)
                   .server_wins.agg(["sum", "count"]).reset_index())
    counts = counts[counts["count"] >= minimum_set_points]
    records = []
    for (match_id, server), group in counts.groupby(["match_id", "server"], sort=False):
        if len(group) < 2:
            continue
        n = group["count"].to_numpy(dtype=float)
        k = group["sum"].to_numpy(dtype=float)
        total_n, total_k = n.sum(), k.sum()
        p = total_k / total_n
        if p <= 0 or p >= 1:
            continue
        observed = float(np.sum((k - n * p) ** 2 / (n * p * (1 - p))))
        # Exact expectation after randomly allocating a fixed number of wins
        # among the observed set sizes (multivariate hypergeometric null).
        expected = float(np.sum((total_n - n) / (total_n - 1)))
        records.append({"match_id": match_id, "server": server,
                        "sets": len(group), "points": int(total_n),
                        "observed": observed, "expected": expected})
    return pd.DataFrame(records)


def summarize(sequence: pd.DataFrame, sets: pd.DataFrame) -> dict:
    def calculate(seq: pd.DataFrame, st: pd.DataFrame) -> dict:
        return {
            "excess_adjacent_agreement": float(
                (seq.observed_agree.sum() - seq.expected_agree.sum()) / seq.pairs.sum()),
            "permutation_adjusted_lag1_rho": float(
                (seq.observed_cross.sum() - seq.expected_cross.sum()) /
                seq.rho_denominator.sum()),
            "set_rate_dispersion_ratio": float(st.observed.sum() / st.expected.sum()),
        }

    estimate = calculate(sequence, sets)
    match_ids = np.array(sorted(set(sequence.match_id) | set(sets.match_id)))
    seq_by_match = sequence.groupby("match_id").sum(numeric_only=True)
    set_by_match = sets.groupby("match_id").sum(numeric_only=True)
    rng = np.random.default_rng(SEED)
    boot = {key: np.empty(BOOTSTRAPS) for key in estimate}
    for b in range(BOOTSTRAPS):
        sampled = rng.choice(match_ids, len(match_ids), replace=True)
        multiplicity = pd.Series(sampled).value_counts()
        seq = seq_by_match.reindex(multiplicity.index).fillna(0).mul(multiplicity, axis=0).sum()
        st = set_by_match.reindex(multiplicity.index).fillna(0).mul(multiplicity, axis=0).sum()
        boot["excess_adjacent_agreement"][b] = (
            (seq.observed_agree - seq.expected_agree) / seq.pairs)
        boot["permutation_adjusted_lag1_rho"][b] = (
            (seq.observed_cross - seq.expected_cross) / seq.rho_denominator)
        boot["set_rate_dispersion_ratio"][b] = st.observed / st.expected
    return {
        "n_matches": int(len(match_ids)),
        "n_match_server_sequences": int(len(sequence)),
        "n_match_server_set_profiles": int(len(sets)),
        "estimates": {
            key: {"value": value,
                  "lo": float(np.quantile(boot[key], 0.025)),
                  "hi": float(np.quantile(boot[key], 0.975))}
            for key, value in estimate.items()
        },
    }


def run() -> dict:
    points = pd.read_parquet(SOURCE).reset_index(names="source_order")
    point_number = pd.to_numeric(points.point_no, errors="coerce")
    if point_number.isna().any():
        raise ValueError("missing or nonnumeric source point order")
    bad_order = 0
    for indices in points.groupby("match_id", sort=False).groups.values():
        bad_order += int(np.any(np.diff(point_number.loc[indices].to_numpy()) <= 0))
    if bad_order:
        raise ValueError(f"non-increasing point order in {bad_order} matches")
    points = points[(~points.ambiguous_identity) & points.server.notna() &
                    points.set_no.notna() & points.server_wins.isin([0, 1])].copy()
    if points.empty or points.match_id.isna().any():
        raise ValueError("invalid point cohort")
    result = {
        "utc": utcnow(), "source": str(SOURCE), "n_points": len(points),
        "methods": {
            "sequence": "Within each match-server, preserve n and k and compare adjacent outcomes with their random-permutation expectation; require at least 20 service points.",
            "sets": "Within each match-server, compare per-set counts with the exact multivariate-hypergeometric Pearson expectation; require at least 12 service points per set and two sets.",
            "intervals": f"{BOOTSTRAPS} match-cluster bootstrap replicates, seed {SEED}.",
            "chronology": "point_no is numeric and strictly increasing within every source match.",
        },
        "groups": {},
    }
    group_masks = {"men": points.tour == "M", "women": points.tour == "W",
                   "all": points.tour.isin(["M", "W"])}
    for name, mask in group_masks.items():
        frame = points.loc[mask].sort_values("source_order")
        sequence = sequence_components(frame)
        sets = set_components(frame)
        result["groups"][name] = summarize(sequence, sets)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "results.json", result)
    report(result)
    plot(result)
    men = result["groups"]["men"]["estimates"]
    trials.log(
        stage="direct-point-dependence",
        kind="conditional-within-match-dependence",
        params={"minimum_sequence_points": 20, "minimum_set_points": 12,
                "bootstraps": BOOTSTRAPS, "seed": SEED},
        metrics={"men_lag1_rho": men["permutation_adjusted_lag1_rho"]["value"],
                 "men_set_dispersion": men["set_rate_dispersion_ratio"]["value"],
                 "n_points": result["n_points"]},
        note="Descriptive direct measurement; conditions on match-server totals.",
    )
    return result


def report(result: dict) -> None:
    lines = ["# Direct within-match point dependence", "",
             f"Analyzed {result['n_points']:,} unambiguous Slam points. Intervals resample whole matches.", "",
             "| Tour | Matches | Excess adjacent agreement | Adjusted lag-1 rho | Set-rate dispersion |",
             "|---|---:|---:|---:|---:|"]
    for name in ["men", "women", "all"]:
        item = result["groups"][name]
        e = item["estimates"]
        a, r, d = (e["excess_adjacent_agreement"],
                   e["permutation_adjusted_lag1_rho"],
                   e["set_rate_dispersion_ratio"])
        lines.append(
            f"| {name.title()} | {item['n_matches']} | {a['value']:+.4f} "
            f"[{a['lo']:+.4f}, {a['hi']:+.4f}] | {r['value']:+.4f} "
            f"[{r['lo']:+.4f}, {r['hi']:+.4f}] | {d['value']:.3f} "
            f"[{d['lo']:.3f}, {d['hi']:.3f}] |")
    lines += ["", "Under conditionally iid points, both dependence measures equal zero and the "
              "set-rate dispersion ratio equals one. Conditioning on each match-server's total "
              "wins removes stable player, opponent, and surface differences. The estimates remain "
              "descriptive because tennis scoring and match stopping constrain observed sequences."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def plot(result: dict) -> None:
    groups = ["men", "women", "all"]
    metrics = ["permutation_adjusted_lag1_rho", "set_rate_dispersion_ratio"]
    titles = ["Permutation-adjusted lag-1 association", "Per-set rate dispersion"]
    nulls = [0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, title, null in zip(axes, metrics, titles, nulls):
        values = [result["groups"][g]["estimates"][metric] for g in groups]
        mean = np.array([x["value"] for x in values])
        lo = np.array([x["lo"] for x in values])
        hi = np.array([x["hi"] for x in values])
        ax.errorbar(np.arange(3), mean, yerr=[mean - lo, hi - mean], fmt="o",
                    capsize=4, color="#2457A6")
        ax.axhline(null, color="black", linewidth=1)
        ax.set_xticks(np.arange(3), [x.title() for x in groups])
        ax.set_title(title)
    fig.suptitle("Within-match dependence in Slam point data")
    fig.tight_layout()
    fig.savefig(OUT / "point_dependence.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
