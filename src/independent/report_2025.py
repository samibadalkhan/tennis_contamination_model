"""Post-process the sealed 2025 predictions into a compact report and figure."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tennis_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.independent.run import ci
from src.util import RESULTS, write_json


OUT = RESULTS / "independent_2025"
METHODS = ["ordinary_iid", "ordinary_burst", "ordinary_temperature"]
LABELS = {"ordinary_iid": "IID", "ordinary_burst": "Burst",
          "ordinary_temperature": "Temperature"}
COLORS = {"ordinary_iid": "#777777", "ordinary_burst": "#1565C0",
          "ordinary_temperature": "#E07A1F"}


def gain(candidate: pd.Series, reference: pd.Series, rows: pd.DataFrame) -> dict:
    values = np.log(np.clip(candidate.to_numpy(), 1e-10, 1)) - np.log(
        np.clip(reference.to_numpy(), 1e-10, 1))
    return {"tournament": ci(values, rows),
            "player_tournament": ci(values, rows, "player_tournament")}


def main() -> None:
    predictions_path = OUT / "predictions_2025.csv"
    results_path = OUT / "results.json"
    frame = pd.read_csv(predictions_path)
    result = json.loads(results_path.read_text())
    if len(frame) != result["n"] or frame.match_id.nunique() != len(frame):
        raise ValueError("saved prediction cohort is inconsistent")

    strata = {}
    for label, rows in frame.groupby("stratum"):
        strata[label] = {
            "n": len(rows),
            "log_loss": {method: float(-np.log(rows[method]).mean()) for method in METHODS},
            "burst_vs_iid": gain(rows.ordinary_burst, rows.ordinary_iid, rows),
            "burst_vs_temperature": gain(rows.ordinary_burst, rows.ordinary_temperature, rows),
        }
    write_json(OUT / "strata.json", strata)

    dated = frame.assign(month=pd.to_datetime(frame.date).dt.to_period("M").astype(str))
    monthly = dated.groupby("month").agg(
        n=("match_id", "size"),
        ordinary_iid=("ordinary_iid", lambda x: float(-np.log(x).mean())),
        ordinary_burst=("ordinary_burst", lambda x: float(-np.log(x).mean())),
        ordinary_temperature=("ordinary_temperature", lambda x: float(-np.log(x).mean())),
    ).reset_index()
    monthly.to_csv(OUT / "monthly.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    values = [result["metrics"][m]["log_loss"]["tournament"]["mean"] for m in METHODS]
    axes[0].bar([LABELS[m] for m in METHODS], values,
                color=[COLORS[m] for m in METHODS], width=0.68)
    axes[0].set_ylim(0.60, 0.675)
    axes[0].set_ylabel("2025 mean log loss (lower is better)")
    axes[0].set_title("Frozen season holdout")
    for index, value in enumerate(values):
        axes[0].text(index, value + 0.0015, f"{value:.4f}", ha="center", fontsize=10)

    comparisons = ["ordinary_burst_vs_ordinary_iid",
                   "ordinary_burst_vs_ordinary_temperature"]
    comparison_labels = ["Burst vs IID", "Burst vs temperature"]
    for index, (key, label) in enumerate(zip(comparisons, comparison_labels)):
        item = result["comparisons"][key]["tournament"]
        axes[1].errorbar(item["mean"], index,
                         xerr=[[item["mean"] - item["lo"]], [item["hi"] - item["mean"]]],
                         fmt="o", color=COLORS["ordinary_burst"], capsize=4)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(range(len(comparison_labels)), comparison_labels)
    axes[1].set_xlabel("Paired log-loss gain (positive favors burst)")
    axes[1].set_title("Tournament-clustered 95% intervals")

    x = np.arange(len(monthly))
    for method in METHODS:
        axes[2].plot(x, monthly[method], marker="o", markersize=3,
                     color=COLORS[method], label=LABELS[method])
    axes[2].set_xticks(x[::2], monthly.month.iloc[::2], rotation=45, ha="right")
    axes[2].set_ylabel("Monthly log loss")
    axes[2].set_title("Descriptive drift through the frozen year")
    axes[2].legend(frameon=False)
    fig.suptitle("Full ordinary Ingram model: 2025 ATP source-season holdout", fontsize=14)
    figure_path = OUT / "ordinary_2025_holdout.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    diag = result["fit_diagnostics"]
    sim = result["simulation_check"]
    report = [
        "# Ordinary full model: frozen 2025 holdout", "",
        "## Result", "",
        "The burst forecast clearly improves the uncalibrated iid forecast, but it does not beat "
        "the simpler temperature control. This supports probability shrinkage and does not isolate "
        "within-match bursts as the source of the gain.", "",
        "| Frozen forecast | 2025 log loss | Accuracy | Brier | Calibration slope |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        metric = result["metrics"][method]
        report.append(f"| {LABELS[method]} | {metric['log_loss']['tournament']['mean']:.5f} | "
                      f"{metric['accuracy']['tournament']['mean']:.4f} | "
                      f"{metric['brier']['tournament']['mean']:.5f} | "
                      f"{metric['calibration']['slope']:.3f} |")
    burst_iid = result["comparisons"]["ordinary_burst_vs_ordinary_iid"]
    burst_temp = result["comparisons"]["ordinary_burst_vs_ordinary_temperature"]
    report += ["", "Positive paired gain favors burst:", "",
               f"- Burst vs iid: **{burst_iid['tournament']['mean']:+.5f}**, tournament CI "
               f"[{burst_iid['tournament']['lo']:+.5f}, {burst_iid['tournament']['hi']:+.5f}], "
               f"player-tournament CI [{burst_iid['player_tournament']['lo']:+.5f}, "
               f"{burst_iid['player_tournament']['hi']:+.5f}].",
               f"- Burst vs temperature: **{burst_temp['tournament']['mean']:+.5f}**, tournament CI "
               f"[{burst_temp['tournament']['lo']:+.5f}, {burst_temp['tournament']['hi']:+.5f}], "
               f"player-tournament CI [{burst_temp['player_tournament']['lo']:+.5f}, "
               f"{burst_temp['player_tournament']['hi']:+.5f}].", "",
               "## Tournament appearances", "",
               "| Appearance stratum | n | IID | Burst | Temperature | Burst gain vs iid |",
               "|---|---:|---:|---:|---:|---:|"]
    for label in ["both_first", "mixed_first_later", "both_later", "unknown"]:
        item = strata[label]
        gains = item["burst_vs_iid"]["tournament"]
        report.append(f"| {label} | {item['n']} | {item['log_loss']['ordinary_iid']:.5f} | "
                      f"{item['log_loss']['ordinary_burst']:.5f} | "
                      f"{item['log_loss']['ordinary_temperature']:.5f} | "
                      f"{gains['mean']:+.5f} [{gains['lo']:+.5f}, {gains['hi']:+.5f}] |")
    report += ["", "## Scope and diagnostics", "",
               "This is a one-shot season holdout. One full posterior was trained on 35,732 eligible "
               "2011–2024 matches, then held fixed for all 2,670 matches in the 2025 source season. "
               "The burst rate 0.5 and temperature 1.5 were carried from the independently tuned "
               "2013 configuration. There was no 2025 tuning.", "",
               f"The fit had zero divergences, max R-hat {diag['max_rhat']:.3f}, and minimum bulk "
               f"ESS {diag['min_ess_bulk']:.0f}. The R-hat exceeds the strict 1.01 guideline but "
               "remains below 1.05. A second simulation seed changed mean probabilities by "
               f"{sim['mean_absolute_probability_difference']:.5f} and burst log loss from "
               f"{sim['primary_log_loss']:.5f} to {sim['replicate_log_loss']:.5f}.", "",
               "Because abilities are not refreshed during 2025, the 0.63521 burst score is not "
               "directly comparable to the earlier bimonthly-updated 2014 score of 0.58087. The "
               "monthly panel is descriptive and shows the cost of stale abilities later in the year.", "",
               "## Exploratory online extension", "",
               "After this frozen holdout was scored, five additional full posterior fits updated "
               "abilities at the March, May, July, September, and November cutoffs. Online burst "
               "log loss fell to **0.62656** and accuracy rose to **0.6431**. The paired "
               "online-versus-frozen burst gain was +0.00865, with tournament CI [+0.00128, "
               "+0.01721] and player-tournament CI [-0.00230, +0.02057]. Online temperature "
               "remained nominally best at 0.62544. The online analysis is exploratory because "
               "the 2025 outcomes had already been opened. See [`ONLINE_README.md`](ONLINE_README.md) "
               "for its fixed protocol, period-level results, diagnostics, and figure.", "",
               "![2025 holdout summary](ordinary_2025_holdout.png)", "",
               "## Files", "",
               "- `PROTOCOL.md`: frozen design and pre-test amendment.",
               "- `FROZEN_BEFORE_2025.json`: sealed forecast settings and hashes.",
               "- `results.json`: complete metrics and clustered intervals.",
               "- `predictions_2025.csv`: match-level forecasts.",
               "- `strata.json` and `monthly.csv`: report-derived summaries.",
               "- `fits/ordinary_period_84.json`: posterior configuration and diagnostics."]
    (OUT / "README.md").write_text("\n".join(report) + "\n")

    manifest = {
        "inputs": {
            "predictions_2025.csv": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
            "results.json": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        },
        "outputs": {
            "README.md": hashlib.sha256((OUT / "README.md").read_bytes()).hexdigest(),
            "strata.json": hashlib.sha256((OUT / "strata.json").read_bytes()).hexdigest(),
            "monthly.csv": hashlib.sha256((OUT / "monthly.csv").read_bytes()).hexdigest(),
            "ordinary_2025_holdout.png": hashlib.sha256(figure_path.read_bytes()).hexdigest(),
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(OUT / "report_manifest.json", manifest)


if __name__ == "__main__":
    main()
