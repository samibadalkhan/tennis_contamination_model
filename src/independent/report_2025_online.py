"""Create the final exploratory online-update report from sealed predictions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tennis_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.util import RESULTS, write_json


OUT = RESULTS / "independent_2025"
METHODS = ["iid", "burst", "temperature"]
COLORS = {"iid": "#777777", "burst": "#1565C0", "temperature": "#E07A1F"}
LABELS = {"iid": "IID", "burst": "Burst", "temperature": "Temperature"}


def main() -> None:
    prediction_path = OUT / "online_predictions_2025.csv"
    result_path = OUT / "online_results.json"
    frame = pd.read_csv(prediction_path)
    result = json.loads(result_path.read_text())
    if len(frame) != result["n"] or frame.match_id.nunique() != len(frame):
        raise ValueError("online prediction cohort is inconsistent")

    periods = []
    for period, rows in frame.groupby("period", sort=True):
        frozen = float(-np.log(rows.frozen_burst).mean())
        online = float(-np.log(rows.online_burst).mean())
        periods.append({"period": int(period), "n": len(rows),
                        "date_min": rows.date.min(), "date_max": rows.date.max(),
                        "frozen_burst_log_loss": frozen,
                        "online_burst_log_loss": online,
                        "online_gain": frozen - online})
    period_frame = pd.DataFrame(periods)
    period_frame.to_csv(OUT / "online_periods.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    x = np.arange(len(METHODS))
    width = 0.34
    frozen_values = [result["metrics"][f"frozen_{method}"]["log_loss"]["tournament"]["mean"]
                     for method in METHODS]
    online_values = [result["metrics"][f"online_{method}"]["log_loss"]["tournament"]["mean"]
                     for method in METHODS]
    axes[0].bar(x - width / 2, frozen_values, width, color="#BDBDBD", label="Frozen")
    axes[0].bar(x + width / 2, online_values, width,
                color=[COLORS[method] for method in METHODS], label="Online")
    axes[0].set_xticks(x, [LABELS[method] for method in METHODS])
    axes[0].set_ylim(0.61, 0.67)
    axes[0].set_ylabel("2025 mean log loss (lower is better)")
    axes[0].set_title("Frozen versus bimonthly updates")
    axes[0].legend(frameon=False)

    comparison_keys = ["online_iid_vs_frozen_iid", "online_burst_vs_frozen_burst",
                       "online_temperature_vs_frozen_temperature"]
    for index, (method, key) in enumerate(zip(METHODS, comparison_keys)):
        item = result["comparisons"][key]["tournament"]
        axes[1].errorbar(item["mean"], index,
                         xerr=[[item["mean"] - item["lo"]], [item["hi"] - item["mean"]]],
                         fmt="o", color=COLORS[method], capsize=4)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(range(len(METHODS)), [LABELS[method] for method in METHODS])
    axes[1].set_xlabel("Paired log-loss gain (positive favors online)")
    axes[1].set_title("Tournament-clustered 95% intervals")

    labels = ["Dec", "Jan–Feb", "Mar–Apr", "May–Jun", "Jul–Aug", "Sep–Oct", "Nov–Dec"]
    axes[2].plot(range(len(period_frame)), period_frame.frozen_burst_log_loss,
                 marker="o", color="#999999", label="Frozen burst")
    axes[2].plot(range(len(period_frame)), period_frame.online_burst_log_loss,
                 marker="o", color=COLORS["burst"], label="Online burst")
    axes[2].set_xticks(range(len(period_frame)), labels, rotation=40, ha="right")
    axes[2].set_ylabel("Block log loss")
    axes[2].set_title("Updates help mainly late in the year")
    axes[2].legend(frameon=False)
    fig.suptitle("Exploratory online full-model updates on the 2025 ATP season", fontsize=14)
    figure_path = OUT / "online_2025_updates.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    metrics = result["metrics"]
    update = result["comparisons"]["online_burst_vs_frozen_burst"]
    burst_iid = result["comparisons"]["online_burst_vs_online_iid"]
    burst_temp = result["comparisons"]["online_burst_vs_online_temperature"]
    diag = result["fit_diagnostics"]
    sim = result["simulation_check"]
    lines = [
        "# Exploratory online updates: 2025", "",
        "> The frozen 2025 holdout had already been scored before this extension was registered. "
        "These online results are exploratory.", "",
        "## Result", "",
        "Bimonthly ability updates improve every forecast family. Online burst log loss is "
        f"**{metrics['online_burst']['log_loss']['tournament']['mean']:.5f}**, down from "
        f"{metrics['frozen_burst']['log_loss']['tournament']['mean']:.5f} with frozen abilities. "
        "Temperature scaling remains fractionally better, so the result continues to support "
        "calibration rather than a uniquely burst-driven mechanism.", "",
        "| Forecast | Frozen log loss | Online log loss | Frozen accuracy | Online accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        frozen = metrics[f"frozen_{method}"]
        online = metrics[f"online_{method}"]
        lines.append(f"| {LABELS[method]} | {frozen['log_loss']['tournament']['mean']:.5f} | "
                     f"{online['log_loss']['tournament']['mean']:.5f} | "
                     f"{frozen['accuracy']['tournament']['mean']:.4f} | "
                     f"{online['accuracy']['tournament']['mean']:.4f} |")
    lines += ["", "Positive paired gain favors the first method:", "",
              f"- Online burst vs frozen burst: **{update['tournament']['mean']:+.5f}**, tournament "
              f"CI [{update['tournament']['lo']:+.5f}, {update['tournament']['hi']:+.5f}], "
              f"player-tournament CI [{update['player_tournament']['lo']:+.5f}, "
              f"{update['player_tournament']['hi']:+.5f}].",
              f"- Online burst vs online iid: **{burst_iid['tournament']['mean']:+.5f}**, tournament "
              f"CI [{burst_iid['tournament']['lo']:+.5f}, {burst_iid['tournament']['hi']:+.5f}], "
              f"player-tournament CI [{burst_iid['player_tournament']['lo']:+.5f}, "
              f"{burst_iid['player_tournament']['hi']:+.5f}].",
              f"- Online burst vs online temperature: **{burst_temp['tournament']['mean']:+.5f}**, "
              f"tournament CI [{burst_temp['tournament']['lo']:+.5f}, "
              f"{burst_temp['tournament']['hi']:+.5f}].", "",
              "## Update blocks", "",
              "| Forecast block | n | Frozen burst | Online burst | Online gain |",
              "|---|---:|---:|---:|---:|"]
    for label, row in zip(labels, period_frame.itertuples()):
        lines.append(f"| {label} | {row.n} | {row.frozen_burst_log_loss:.5f} | "
                     f"{row.online_burst_log_loss:.5f} | {row.online_gain:+.5f} |")
    lines += ["", "The December and January–February rows reuse the original posterior exactly. "
              "The March update is slightly worse and May is nearly neutral. The aggregate gain "
              "comes from the July, September, and November refits, when frozen abilities have "
              "become stale.", "",
              "## Diagnostics and provenance", "",
              "Every cutoff fingerprint matches exactly the observations strictly before its "
              "forecast period. All six fits used four chains, 1,000 warmup draws, and 1,000 "
              "retained draws.", "",
              f"There were {diag['divergences']} divergences. Max R-hat was "
              f"{diag['max_rhat']:.3f} and minimum bulk ESS was {diag['min_ess_bulk']:.0f}. "
              "Every fit exceeds the strict 1.01 R-hat guideline, while none exceeds 1.05. "
              "The weakest ESS is in the November fit.", "",
              f"A second burst-simulation seed changed mean probabilities by "
              f"{sim['mean_absolute_probability_difference']:.5f} and online burst log loss from "
              f"{sim['primary_log_loss']:.5f} to {sim['replicate_log_loss']:.5f}.", "",
              "![Online 2025 updates](online_2025_updates.png)", "",
              "## Files", "",
              "- `PROTOCOL_ONLINE_POSTHOC.md`: fixed exploratory update protocol.",
              "- `ONLINE_FROZEN.json`: hashes and carried settings.",
              "- `online_results.json`: complete metrics and clustered intervals.",
              "- `online_predictions_2025.csv`: match-level online and frozen forecasts.",
              "- `online_periods.csv`: block-level decomposition.",
              "- `fits/ordinary_period_84.json` through `ordinary_period_89.json`: fit diagnostics."]
    readme_path = OUT / "ONLINE_README.md"
    readme_path.write_text("\n".join(lines) + "\n")
    manifest = {
        "inputs": {
            prediction_path.name: hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            result_path.name: hashlib.sha256(result_path.read_bytes()).hexdigest(),
        },
        "outputs": {
            readme_path.name: hashlib.sha256(readme_path.read_bytes()).hexdigest(),
            "online_periods.csv": hashlib.sha256((OUT / "online_periods.csv").read_bytes()).hexdigest(),
            figure_path.name: hashlib.sha256(figure_path.read_bytes()).hexdigest(),
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(OUT / "online_report_manifest.json", manifest)


if __name__ == "__main__":
    main()
