"""Render the completed ATP MCMC-seed replication summary."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from src.util import RESULTS


OUT = RESULTS / "independent_2025" / "mcmc_seed_1729"


def run() -> None:
    replicate = json.loads((OUT / "results.json").read_text())
    original = json.loads((RESULTS / "independent_2025" / "online_results.json").read_text())
    names = ["iid", "burst", "temperature"]
    original_loss = [original["metrics"][f"online_{name}"]["log_loss"]["tournament"]["mean"]
                     for name in names]
    replicate_loss = [replicate["metrics"][f"replicate_{name}"]["log_loss"]["tournament"]["mean"]
                      for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    x = np.arange(3)
    axes[0].plot(x, original_loss, "o-", label="Original seed", color="#2457A6")
    axes[0].plot(x, replicate_loss, "o--", label="Seed 1729", color="#B44A3A")
    axes[0].set_xticks(x, [name.title() for name in names])
    axes[0].set_ylabel("Log loss")
    axes[0].set_title("Season scores")
    axes[0].legend(frameon=False)
    comparisons = [
        ("Burst − IID", original["comparisons"]["online_burst_vs_online_iid"]["tournament"],
         replicate["comparisons"]["replicate_burst_vs_replicate_iid"]["tournament"]),
        ("Burst − temp.", original["comparisons"]["online_burst_vs_online_temperature"]["tournament"],
         replicate["comparisons"]["replicate_burst_vs_replicate_temperature"]["tournament"]),
    ]
    for offset, index, label, color in [(-0.08, 1, "Original seed", "#2457A6"),
                                        (0.08, 2, "Seed 1729", "#B44A3A")]:
        values = [item[index] for item in comparisons]
        mean = np.array([value["mean"] for value in values])
        lo = np.array([value["lo"] for value in values])
        hi = np.array([value["hi"] for value in values])
        axes[1].errorbar(np.arange(2) + offset, mean, yerr=[mean - lo, hi - mean],
                         fmt="o", capsize=4, label=label, color=color)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(np.arange(2), [item[0] for item in comparisons])
    axes[1].set_ylabel("Paired log-score gain")
    axes[1].set_title("Tournament-bootstrap comparisons")
    fig.suptitle("ATP 2025 MCMC seed replication")
    fig.tight_layout()
    fig.savefig(OUT / "mcmc_seed_summary.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    run()
