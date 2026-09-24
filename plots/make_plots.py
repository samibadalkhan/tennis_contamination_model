"""Build the curated gallery of final experiment figures."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tennis_plot_gallery")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/tennis_font_cache")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots"

COLORS = {
    "iid": "#6E7781",
    "burst": "#1565C0",
    "temperature": "#E07A1F",
    "atp": "#2457A6",
    "seed": "#B44A3A",
    "wta": "#2E7D32",
}

COPIES = {
    "atp_frozen_holdout.png": "results/independent_2025/ordinary_2025_holdout.png",
    "atp_online_updates.png": "results/independent_2025/online_2025_updates.png",
    "mcmc_seed_replication.png":
        "results/independent_2025/mcmc_seed_1729/mcmc_seed_summary.png",
    "burst_vs_temperature_by_format.png":
        "results/independent_2025/mechanism_splits/burst_vs_temperature_by_format.png",
    "burst_vs_temperature_by_edge.png":
        "results/independent_2025/mechanism_splits/burst_vs_temperature_by_edge.png",
    "wta_corrected_holdout.png": "results/wta_2025/corrected/wta_2025_summary.png",
    "point_dependence.png": "results/point_dependence/point_dependence.png",
    "point_structure_followup.png": "analysis/point_structure_followup/lag_structure.png",
}


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_final_figures() -> list[dict]:
    records = []
    for destination, source_name in COPIES.items():
        source = ROOT / source_name
        destination_path = OUT / destination
        shutil.copy2(source, destination_path)
        records.append({
            "output": destination,
            "source": source_name,
            "source_sha256": sha256(source),
            "output_sha256": sha256(destination_path),
        })
    return records


def forecast_log_loss() -> tuple[Path, list[str]]:
    frozen_path = "results/independent_2025/results.json"
    online_path = "results/independent_2025/online_results.json"
    wta_path = "results/wta_2025/corrected/results.json"
    frozen = read_json(frozen_path)
    online = read_json(online_path)
    wta = read_json(wta_path)

    cohorts = ["ATP frozen", "ATP online", "WTA corrected"]
    methods = ["iid", "burst", "temperature"]
    values = {
        "iid": [
            frozen["metrics"]["ordinary_iid"]["log_loss"]["tournament"]["mean"],
            online["metrics"]["online_iid"]["log_loss"]["tournament"]["mean"],
            wta["metrics"]["iid"]["log_loss"]["tournament"]["mean"],
        ],
        "burst": [
            frozen["metrics"]["ordinary_burst"]["log_loss"]["tournament"]["mean"],
            online["metrics"]["online_burst"]["log_loss"]["tournament"]["mean"],
            wta["metrics"]["burst"]["log_loss"]["tournament"]["mean"],
        ],
        "temperature": [
            frozen["metrics"]["ordinary_temperature"]["log_loss"]["tournament"]["mean"],
            online["metrics"]["online_temperature"]["log_loss"]["tournament"]["mean"],
            wta["metrics"]["temperature"]["log_loss"]["tournament"]["mean"],
        ],
    }

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    x = np.arange(len(cohorts))
    offsets = {"iid": -0.22, "burst": 0.0, "temperature": 0.22}
    labels = {"iid": "IID", "burst": "Burst", "temperature": "Temperature"}
    markers = {"iid": "o", "burst": "s", "temperature": "D"}
    for method in methods:
        xs = x + offsets[method]
        ax.scatter(xs, values[method], s=70, color=COLORS[method],
                   marker=markers[method], label=labels[method], zorder=3)
        for px, value in zip(xs, values[method]):
            ax.text(px, value + 0.0014, f"{value:.4f}", ha="center", va="bottom",
                    fontsize=8.5, color=COLORS[method])
    ax.set_xticks(x, cohorts)
    ax.set_ylabel("Mean log loss (lower is better)")
    ax.set_ylim(0.615, 0.670)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.set_title("Final 2025 forecast scores", fontsize=14, pad=14)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.84)
    fig.text(0.5, 0.035,
             "Descriptive means; use the experiment-specific paired intervals for inference.",
             ha="center", fontsize=9, color="#555555")
    path = OUT / "forecast_log_loss.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, [frozen_path, online_path, wta_path]


def fit_diagnostics() -> tuple[Path, list[str]]:
    groups = [
        ("ATP primary", ROOT / "results/independent_2025/fits", COLORS["atp"], "o"),
        ("ATP seed 1729", ROOT / "results/independent_2025/mcmc_seed_1729/fits",
         COLORS["seed"], "s"),
        ("WTA corrected", ROOT / "results/wta_2025/corrected/fits", COLORS["wta"], "D"),
    ]
    inputs: list[str] = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    total_divergences = 0
    series = []
    all_dates: set[str] = set()
    group_handles = []
    for label, folder, color, marker in groups:
        dates, rhats, ess = [], [], []
        for path in sorted(folder.glob("ordinary_period_*.json")):
            item = json.loads(path.read_text())
            dates.append(item["period_start"][:7])
            rhats.append(item["diagnostics"]["max_rhat"])
            ess.append(item["diagnostics"]["min_ess_bulk"])
            total_divergences += item["diagnostics"]["divergences"]
            inputs.append(str(path.relative_to(ROOT)))
        all_dates.update(dates)
        series.append((label, color, marker, dates, rhats, ess))

    ordered_dates = sorted(all_dates)
    date_positions = {date: index for index, date in enumerate(ordered_dates)}
    for label, color, marker, dates, rhats, ess in series:
        x = np.array([date_positions[date] for date in dates])
        handle, = axes[0].plot(x, rhats, marker=marker, color=color,
                               label=label, linewidth=1.8)
        axes[1].plot(x, ess, marker=marker, color=color, label=label, linewidth=1.8)
        group_handles.append(handle)

    rhat_101 = axes[0].axhline(1.01, color="#999999", linestyle="--", linewidth=1)
    rhat_105 = axes[0].axhline(1.05, color="#C62828", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Maximum R-hat")
    axes[0].set_title("Worst chain-mixing diagnostic per fit")
    axes[0].set_ylim(0.998, 1.055)

    ess_400 = axes[1].axhline(400, color="#999999", linestyle="--", linewidth=1)
    ess_100 = axes[1].axhline(100, color="#C62828", linestyle=":", linewidth=1)
    axes[1].set_ylabel("Minimum bulk effective sample size")
    axes[1].set_title("Weakest bulk ESS per fit")
    axes[1].set_ylim(0, 450)

    axes[0].set_xticks(np.arange(len(ordered_dates)), ordered_dates, rotation=35, ha="right")
    axes[1].set_xticks(np.arange(len(ordered_dates)), ordered_dates, rotation=35, ha="right")
    for ax in axes:
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.legend(group_handles, [item[0] for item in series], frameon=False,
               ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.90), fontsize=8.5)
    axes[0].legend([rhat_101, rhat_105], ["1.01 reference", "1.05 reference"],
                   frameon=False, fontsize=8, loc="upper right")
    axes[1].legend([ess_400, ess_100], ["400 reference", "100 reference"],
                   frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(
        f"Bayesian fit diagnostics across training cutoffs ({total_divergences} divergences)",
        fontsize=14, y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.76, wspace=0.17)
    fig.text(0.5, 0.035,
             "These MCMC fits have no training-loss curve; R-hat and ESS are the relevant fit diagnostics.",
             ha="center", fontsize=9, color="#555555")
    path = OUT / "fit_diagnostics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, inputs


def main() -> None:
    OUT.mkdir(exist_ok=True)
    copied = copy_final_figures()
    forecast_path, forecast_inputs = forecast_log_loss()
    diagnostics_path, diagnostic_inputs = fit_diagnostics()
    generated = []
    for path, inputs in [
        (forecast_path, forecast_inputs),
        (diagnostics_path, diagnostic_inputs),
    ]:
        generated.append({
            "output": path.name,
            "output_sha256": sha256(path),
            "inputs": {name: sha256(ROOT / name) for name in inputs},
        })
    manifest = {
        "description": "Curated copies and summaries of the final experiment figures",
        "copied": copied,
        "generated": generated,
        "script_sha256": sha256(Path(__file__)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
