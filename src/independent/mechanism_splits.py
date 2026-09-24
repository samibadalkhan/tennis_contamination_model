"""Test where burst forecasts differ from temperature scaling on ATP 2025."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from src import trials
from src.independent.run import ci
from src.util import RESULTS, utcnow, write_json


SOURCE = RESULTS / "independent_2025" / "online_predictions_2025.csv"
DEVELOPMENT = RESULTS / "independent" / "ordinary_2013_predictions_4001.csv"
OUT = RESULTS / "independent_2025" / "mechanism_splits"
EDGE_BREAKS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.5000001]


def paired(candidate: np.ndarray, reference: np.ndarray, rows: pd.DataFrame) -> dict:
    gain = np.log(np.clip(candidate, 1e-10, 1)) - np.log(np.clip(reference, 1e-10, 1))
    return {
        "tournament": ci(gain, rows),
        "player_tournament": ci(gain, rows, "player_tournament"),
    }


def fit_temperature(probability: np.ndarray) -> dict:
    z = logit(np.clip(probability, 1e-8, 1 - 1e-8))
    objective = lambda temperature: float(-np.log(expit(z / temperature)).mean())
    result = minimize_scalar(objective, bounds=(0.5, 3.0), method="bounded",
                             options={"xatol": 1e-10})
    return {"temperature": float(result.x), "log_loss": float(result.fun),
            "success": bool(result.success)}


def run() -> dict:
    rows = pd.read_csv(SOURCE)
    development = pd.read_csv(DEVELOPMENT)
    required = {"best_of", "online_iid", "online_burst", "online_temperature",
                "frozen_burst", "frozen_temperature", "tourney_id", "p1", "p2"}
    if missing := required - set(rows):
        raise ValueError(f"missing columns: {sorted(missing)}")
    if rows.match_id.duplicated().any():
        raise ValueError("duplicate 2025 match IDs")

    by_format = {}
    temperatures = {}
    for best_of in sorted(rows.best_of.unique()):
        test = rows[rows.best_of == best_of].copy()
        dev = development[development.best_of == best_of].copy()
        if test.empty or dev.empty:
            continue
        fitted = fit_temperature(dev.iid.to_numpy())
        temperatures[str(int(best_of))] = {**fitted, "development_n": len(dev)}
        calibrated = expit(logit(np.clip(test.online_iid.to_numpy(), 1e-8, 1 - 1e-8)) /
                            fitted["temperature"])
        by_format[str(int(best_of))] = {
            "n": len(test),
            "burst_vs_global_temperature": paired(
                test.online_burst.to_numpy(), test.online_temperature.to_numpy(), test),
            "burst_vs_format_temperature": paired(
                test.online_burst.to_numpy(), calibrated, test),
            "log_loss": {
                "burst": float(-np.log(np.clip(test.online_burst, 1e-10, 1)).mean()),
                "global_temperature": float(-np.log(np.clip(test.online_temperature, 1e-10, 1)).mean()),
                "format_temperature": float(-np.log(np.clip(calibrated, 1e-10, 1)).mean()),
            },
        }

    edge = np.abs(rows.online_iid.to_numpy() - 0.5)
    labels = [f"{EDGE_BREAKS[i]:.1f}–{EDGE_BREAKS[i + 1]:.1f}"
              for i in range(len(EDGE_BREAKS) - 1)]
    rows["edge_band"] = pd.cut(edge, EDGE_BREAKS, labels=labels, right=False,
                               include_lowest=True)
    by_edge = {}
    for label in labels:
        part = rows[rows.edge_band == label].copy()
        if part.empty:
            continue
        by_edge[label] = {
            "n": len(part),
            "mean_abs_edge": float(np.abs(part.online_iid - 0.5).mean()),
            "online_burst_vs_temperature": paired(
                part.online_burst.to_numpy(), part.online_temperature.to_numpy(), part),
            "frozen_burst_vs_temperature": paired(
                part.frozen_burst.to_numpy(), part.frozen_temperature.to_numpy(), part),
        }

    result = {
        "utc": utcnow(),
        "status": "post_hoc_mechanism_analysis",
        "source": str(SOURCE),
        "development": str(DEVELOPMENT),
        "n": len(rows),
        "edge_definition": "absolute online iid probability minus 0.5",
        "format_temperatures_selected_on_2013": temperatures,
        "by_format": by_format,
        "by_edge": by_edge,
        "note": "Positive paired log-score gain favors burst.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "results.json", result)
    report(result)
    plot(result)
    trials.log(
        stage="independent-ordinary-2025-online",
        kind="posthoc-mechanism-splits",
        params={"edge_breaks": EDGE_BREAKS,
                "format_temperatures": temperatures},
        metrics={f"bo{key}_burst_vs_format_temperature":
                 value["burst_vs_format_temperature"]["tournament"]["mean"]
                 for key, value in by_format.items()},
        note="Post-hoc on opened ATP 2025 outcomes; no burst edge band was selected.",
    )
    return result


def report(result: dict) -> None:
    lines = [
        "# Where does burst differ from temperature?", "",
        "Post-hoc mechanism analysis of the already-opened ATP 2025 online forecasts. "
        "Positive paired log-score gain favors burst.", "",
        "## Match format", "",
        "| Format | n | 2013 format T | Burst vs global T | Burst vs format T |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, item in result["by_format"].items():
        global_ci = item["burst_vs_global_temperature"]["tournament"]
        format_ci = item["burst_vs_format_temperature"]["tournament"]
        temp = result["format_temperatures_selected_on_2013"][key]["temperature"]
        lines.append(
            f"| Best of {key} | {item['n']} | {temp:.3f} | "
            f"{global_ci['mean']:+.5f} [{global_ci['lo']:+.5f}, {global_ci['hi']:+.5f}] | "
            f"{format_ci['mean']:+.5f} [{format_ci['lo']:+.5f}, {format_ci['hi']:+.5f}] |"
        )
    lines += ["", "## Edge size", "",
              "| abs(iid - 0.5) | n | Online burst vs temperature |",
              "|---|---:|---:|"]
    for label, item in result["by_edge"].items():
        value = item["online_burst_vs_temperature"]["tournament"]
        lines.append(f"| {label} | {item['n']} | {value['mean']:+.5f} "
                     f"[{value['lo']:+.5f}, {value['hi']:+.5f}] |")
    lines += ["", "Intervals resample tournaments. The machine-readable result also includes "
              "player-tournament intervals and the frozen-model edge split."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def plot(result: dict) -> None:
    labels = list(result["by_edge"])
    values = [result["by_edge"][x]["online_burst_vs_temperature"]["tournament"] for x in labels]
    means = np.array([x["mean"] for x in values])
    lo = np.array([x["lo"] for x in values])
    hi = np.array([x["hi"] for x in values])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(np.arange(len(labels)), means, yerr=[means - lo, hi - means],
                fmt="o", capsize=4, color="#2457A6")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Absolute online iid edge")
    ax.set_ylabel("Paired log-score gain: burst − temperature")
    ax.set_title("ATP 2025: burst versus temperature by edge")
    fig.tight_layout()
    fig.savefig(OUT / "burst_vs_temperature_by_edge.png", dpi=180)
    plt.close(fig)

    formats = list(result["by_format"])
    global_values = [result["by_format"][x]["burst_vs_global_temperature"]["tournament"]
                     for x in formats]
    fitted_values = [result["by_format"][x]["burst_vs_format_temperature"]["tournament"]
                     for x in formats]
    x = np.arange(len(formats))
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for offset, values, label, color in [
        (-0.08, global_values, "Global T=1.5", "#2457A6"),
        (0.08, fitted_values, "Format-specific T", "#B44A3A"),
    ]:
        means = np.array([value["mean"] for value in values])
        lo = np.array([value["lo"] for value in values])
        hi = np.array([value["hi"] for value in values])
        ax.errorbar(x + offset, means, yerr=[means - lo, hi - means], fmt="o",
                    capsize=4, label=label, color=color)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, [f"Best of {value}" for value in formats])
    ax.set_ylabel("Paired log-score gain: burst − temperature")
    ax.set_title("ATP 2025: forecast mechanism by match format")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "burst_vs_temperature_by_format.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
