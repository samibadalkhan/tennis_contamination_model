"""Exploratory bimonthly online updates on the already-opened 2025 holdout."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=/private/tmp/tennis_pytensor_2025")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tennis_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/tennis_cache")

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src import trials
from src.independent.run import ci, score, strata
from src.independent.run_2025 import (
    FIT_DIR, FROZEN_RATE, FROZEN_TEMPERATURE, OUT, REPLICATE_SEED, SEED,
    fit_period, load_extended, period_prediction, sampler_progress,
)
from src.util import utcnow, write_json


ONLINE_PERIODS = [85, 86, 87, 88, 89]
DERIVED_COLUMNS = {
    "ordinary_iid", "ordinary_burst", "ordinary_temperature",
    "ordinary_burst_seed2", "stratum",
}


def paired_gain(candidate: np.ndarray, reference: np.ndarray, rows: pd.DataFrame) -> dict:
    values = np.log(np.clip(candidate, 1e-10, 1)) - np.log(np.clip(reference, 1e-10, 1))
    return {"tournament": ci(values, rows),
            "player_tournament": ci(values, rows, "player_tournament")}


def early_predictions() -> tuple[pd.DataFrame, np.ndarray]:
    """Reuse sealed forecasts through February, including the second seed."""
    frozen = pd.read_csv(OUT / "predictions_2025.csv")
    early = frozen[frozen.period <= 84].copy()
    second_mixture = early.ordinary_burst_seed2.to_numpy()
    keep = [column for column in early.columns if column not in DERIVED_COLUMNS]
    return early[keep].copy(), second_mixture


def predictions(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    test = data[data.source_year == 2025].copy()
    chunks = []
    if seed == SEED:
        early, _ = early_predictions()
        chunks.append(early)
    for period in ONLINE_PERIODS:
        rows = test[test.period == period]
        if rows.empty:
            continue
        chunk = period_prediction(data, rows, period, seed)
        chunks.append(chunk)
    out = pd.concat(chunks, ignore_index=True)
    # The sealed early block comes from CSV (date strings), while new period
    # predictions carry Timestamps. Normalize before chronological sorting.
    out["date"] = pd.to_datetime(out.date)
    return out.sort_values(["date", "tourney_id", "round", "match_id"]).reset_index(drop=True)


def diagnostics() -> dict:
    items = []
    for period in [84] + ONLINE_PERIODS:
        fit = json.loads((FIT_DIR / f"ordinary_period_{period}.json").read_text())
        items.append({"period": period, "n_train": fit["n_train"], "seconds": fit["seconds"],
                      **fit["diagnostics"]})
    return {"max_rhat": max(x["max_rhat"] for x in items),
            "min_ess_bulk": min(x["min_ess_bulk"] for x in items),
            "divergences": sum(x["divergences"] for x in items),
            "periods_rhat_over_1p01": [x["period"] for x in items if x["max_rhat"] > 1.01],
            "periods_rhat_over_1p05": [x["period"] for x in items if x["max_rhat"] > 1.05],
            "periods": items}


def run() -> dict:
    protocol = OUT / "PROTOCOL_ONLINE_POSTHOC.md"
    frozen_path = OUT / "FROZEN_BEFORE_2025.json"
    frozen_result_path = OUT / "results.json"
    frozen_predictions_path = OUT / "predictions_2025.csv"
    for path in [protocol, frozen_path, frozen_result_path, frozen_predictions_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    provenance = {
        "utc": utcnow(),
        "status": "exploratory_post_test",
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_settings_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "frozen_results_sha256": hashlib.sha256(frozen_result_path.read_bytes()).hexdigest(),
        "frozen_predictions_sha256": hashlib.sha256(frozen_predictions_path.read_bytes()).hexdigest(),
        "periods": ONLINE_PERIODS,
        "burst_rate": FROZEN_RATE,
        "temperature": FROZEN_TEMPERATURE,
    }
    provenance_path = OUT / "ONLINE_FROZEN.json"
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text())
        if {k: v for k, v in old.items() if k != "utc"} != {k: v for k, v in provenance.items() if k != "utc"}:
            raise ValueError("online protocol or source changed after the run was frozen")
        provenance = old
    else:
        write_json(provenance_path, provenance)

    data, audit = load_extended(2025)
    write_json(OUT / "online_data_audit.json", audit)
    primary = predictions(data, SEED)
    frozen = pd.read_csv(frozen_predictions_path).sort_values(
        ["date", "tourney_id", "round", "match_id"]).reset_index(drop=True)
    if not primary.match_id.equals(frozen.match_id):
        raise ValueError("online and frozen cohorts differ")
    if len(primary) != 2670:
        raise ValueError(f"unexpected online cohort size: {len(primary)}")

    online = {
        "online_iid": primary.iid.to_numpy(),
        "online_burst": ((1 - FROZEN_RATE) * primary.iid.to_numpy() +
                         FROZEN_RATE * primary.conditional_burst.to_numpy()),
        "online_temperature": expit(logit(np.clip(primary.iid.to_numpy(), 1e-8, 1 - 1e-8)) /
                                    FROZEN_TEMPERATURE),
    }
    frozen_values = {
        "frozen_iid": frozen.ordinary_iid.to_numpy(),
        "frozen_burst": frozen.ordinary_burst.to_numpy(),
        "frozen_temperature": frozen.ordinary_temperature.to_numpy(),
    }
    metrics = {name: score(probability, primary)
               for name, probability in {**online, **frozen_values}.items()}
    comparisons = {
        "online_burst_vs_frozen_burst": paired_gain(online["online_burst"],
                                                     frozen_values["frozen_burst"], primary),
        "online_iid_vs_frozen_iid": paired_gain(online["online_iid"],
                                                 frozen_values["frozen_iid"], primary),
        "online_temperature_vs_frozen_temperature": paired_gain(
            online["online_temperature"], frozen_values["frozen_temperature"], primary),
        "online_burst_vs_online_iid": paired_gain(online["online_burst"],
                                                   online["online_iid"], primary),
        "online_burst_vs_online_temperature": paired_gain(online["online_burst"],
                                                           online["online_temperature"], primary),
    }

    labels = strata(primary)
    stratified = {}
    for name, probability in online.items():
        stratified[name] = {}
        for label in sorted(set(labels)):
            mask = labels == label
            stratified[name][label] = score(probability[mask], primary[mask])

    later_second = predictions(data, REPLICATE_SEED)
    early, early_second = early_predictions()
    second_ids = list(early.match_id)
    second_values = list(early_second)
    later_second_mixture = ((1 - FROZEN_RATE) * later_second.iid.to_numpy() +
                            FROZEN_RATE * later_second.conditional_burst.to_numpy())
    second_ids.extend(later_second.match_id.tolist())
    second_values.extend(later_second_mixture.tolist())
    second = pd.Series(second_values, index=second_ids).loc[primary.match_id].to_numpy()
    simulation_check = {
        "primary_seed": SEED,
        "replicate_seed": REPLICATE_SEED,
        "primary_log_loss": float(-np.log(online["online_burst"]).mean()),
        "replicate_log_loss": float(-np.log(second).mean()),
        "mean_absolute_probability_difference": float(np.abs(online["online_burst"] - second).mean()),
    }

    for name, probability in {**online, **frozen_values}.items():
        primary[name] = probability
    primary["online_burst_seed2"] = second
    primary["stratum"] = labels
    primary.to_csv(OUT / "online_predictions_2025.csv", index=False)
    result = {"utc": utcnow(), "status": "exploratory_post_test", "n": len(primary),
              "provenance": provenance, "metrics": metrics, "comparisons": comparisons,
              "stratified_metrics": stratified, "simulation_check": simulation_check,
              "fit_diagnostics": diagnostics(),
              "note": "Positive paired log-loss gain favors the first named method."}
    write_json(OUT / "online_results.json", result)
    trials.log(stage="independent-ordinary-2025-online", kind="score_2025_exploratory",
               params={"periods": ONLINE_PERIODS, "burst_rate": FROZEN_RATE,
                       "temperature": FROZEN_TEMPERATURE},
               metrics={name: value["log_loss"]["tournament"]["mean"]
                        for name, value in metrics.items()},
               note="Exploratory: 2025 had already been scored by the frozen analysis.")
    report(result)
    return result


def report(result: dict) -> None:
    lines = ["# Exploratory online updates: 2025", "",
             "The 2025 holdout had already been opened before this online extension was registered. "
             "These results are exploratory.", "",
             "| Forecast | Log loss | Accuracy | Brier | Calibration slope |",
             "|---|---:|---:|---:|---:|"]
    order = ["frozen_iid", "online_iid", "frozen_burst", "online_burst",
             "frozen_temperature", "online_temperature"]
    for name in order:
        metric = result["metrics"][name]
        lines.append(f"| {name} | {metric['log_loss']['tournament']['mean']:.5f} | "
                     f"{metric['accuracy']['tournament']['mean']:.4f} | "
                     f"{metric['brier']['tournament']['mean']:.5f} | "
                     f"{metric['calibration']['slope']:.3f} |")
    lines += ["", "Paired log-loss gains (positive favors the first method):", ""]
    for name, item in result["comparisons"].items():
        tournament = item["tournament"]
        player = item["player_tournament"]
        lines.append(f"- {name}: {tournament['mean']:+.5f}; tournament CI "
                     f"[{tournament['lo']:+.5f}, {tournament['hi']:+.5f}]; "
                     f"player-tournament CI [{player['lo']:+.5f}, {player['hi']:+.5f}].")
    diag = result["fit_diagnostics"]
    sim = result["simulation_check"]
    lines += ["", "## Diagnostics", "",
              f"Across six posterior fits: max R-hat {diag['max_rhat']:.3f}, minimum bulk ESS "
              f"{diag['min_ess_bulk']:.0f}, {diag['divergences']} divergences, and periods above "
              f"R-hat 1.01 {diag['periods_rhat_over_1p01']}.",
              f"The second simulation seed changed mean probabilities by "
              f"{sim['mean_absolute_probability_difference']:.5f} and online burst log loss from "
              f"{sim['primary_log_loss']:.5f} to {sim['replicate_log_loss']:.5f}.", "",
              "See `PROTOCOL_ONLINE_POSTHOC.md`, `ONLINE_FROZEN.json`, `online_results.json`, "
              "and `online_predictions_2025.csv`."]
    (OUT / "online_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    sampler_progress()
    run()
