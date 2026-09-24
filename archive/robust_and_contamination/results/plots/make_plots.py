"""One-off plotting script for the status-summary report. Not part of the
pipeline (no gate depends on these figures) -- reads already-computed
results/ingram/*.json plus a fresh filter pass for skill trajectories.

    python -m results.plots.make_plots
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import load
from src.models.ability import OnlineAbility, alpha_by_surface

OUT = __file__.rsplit("/", 1)[0]

# a fixed, colorblind-safe qualitative order (tab10-derived), assigned by
# entity identity and never re-cycled within a figure
C = {"blue": "#1f77b4", "orange": "#ff7f0e", "green": "#2ca02c", "red": "#d62728",
     "purple": "#9467bd", "brown": "#8c564b", "gray": "#7f7f7f", "ink": "#222222",
     "muted": "#8a8a8a"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc", "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6, "font.size": 10.5, "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig1_tuning():
    r2013 = json.load(open("results/ingram/reproduce.json"))["tuning"]
    r2018 = json.load(open("results/ingram/score_2019.json"))["tuning"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=False)
    for ax, tuning, title, chosen in [
        (axes[0], r2013, "2013 tuning -> scored on 2014 (Gate 1)", (0.03, 1.0)),
        (axes[1], r2018, "2018 tuning -> scored on 2019", (0.05, 0.25)),
    ]:
        priors = sorted({t["prior_var"] for t in tuning})
        colors = [C["blue"], C["orange"], C["green"]]
        for pv, col in zip(priors, colors):
            pts = sorted([t for t in tuning if t["prior_var"] == pv], key=lambda t: t["sigma"])
            ax.plot([t["sigma"] for t in pts], [t["log_loss"] for t in pts],
                     color=col, lw=2, marker="o", ms=4, label=f"prior_var={pv}")
        ax.axvline(chosen[0], color=C["ink"], lw=1, ls=":", alpha=0.6)
        ax.set_xlabel("sigma (skill drift SD / year)")
        ax.set_title(title, fontsize=10.5)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("validation log loss")
    fig.suptitle("Hyperparameter tuning: log loss vs. drift rate", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/tuning_grids.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig2_season_scores():
    d14 = json.load(open("results/ingram/reproduce.json"))
    d19 = json.load(open("results/ingram/score_2019.json"))
    labels = ["Ingram\n(reference)", "2014\n(Gate 1)", "2019\n(checkpoint)"]
    ll = [d14["ingram_target"]["log_loss"], d14["result"]["log_loss"], d19["result"]["log_loss"]]
    acc = [d14["ingram_target"]["accuracy"], d14["result"]["accuracy"], d19["result"]["accuracy"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    colors = [C["gray"], C["blue"], C["orange"]]
    axes[0].bar(x, ll, color=colors, width=0.55)
    for xi, v in zip(x, ll):
        axes[0].text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, color=C["ink"])
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("log loss"); axes[0].set_title("Log loss (lower is better)", fontsize=10.5)
    axes[0].set_ylim(0, max(ll) * 1.25)
    axes[1].bar(x, acc, color=colors, width=0.55)
    for xi, v in zip(x, acc):
        axes[1].text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, color=C["ink"])
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("accuracy"); axes[1].set_title("Winner-pick accuracy", fontsize=10.5)
    axes[1].set_ylim(0, 1.0)
    fig.suptitle("Ordinary-arm forecast quality: reproduction (2014) vs. next season (2019)",
                 y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/season_scores.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig3_alpha_by_surface():
    matches = load.load_matches("M")
    alpha = alpha_by_surface(matches[matches.year < 2025])
    from scipy.special import expit
    order = ["Clay", "Hard", "Carpet", "Grass"]
    order = [s for s in order if s in alpha] + [s for s in alpha if s not in order]
    vals = [expit(alpha[s]) for s in order]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(order, vals, color=C["blue"], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9.5, color=C["ink"])
    ax.set_ylim(0.55, 0.70)
    ax.set_ylabel("baseline serve-win probability")
    ax.set_title("Surface serve dominance (pre-2025 matches, avg. server vs. avg. returner)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f"{OUT}/alpha_by_surface.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig4_skill_trajectories():
    matches = load.load_matches("M")
    alpha = alpha_by_surface(matches[matches.year < 2014])
    f = OnlineAbility(alpha, sigma=0.03, prior_var=1.0)
    players = ["r federer", "n djokovic", "r nadal", "i karlovic"]
    hist = {p: {"t": [], "s": [], "r": []} for p in players}
    for row in matches.itertuples():
        if row.year > 2019:
            break
        f.process(row, forecast_first=False)
        for p in (row.p1, row.p2):
            if p in players:
                hist[p]["t"].append(row.date)
                hist[p]["s"].append(f.sm[p])
                hist[p]["r"].append(f.rm[p])
    colors = {"r federer": C["blue"], "n djokovic": C["orange"],
              "r nadal": C["green"], "i karlovic": C["purple"]}
    names = {"r federer": "Federer", "n djokovic": "Djokovic",
              "r nadal": "Nadal", "i karlovic": "Karlović"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for p in players:
        axes[0].plot(hist[p]["t"], hist[p]["s"], color=colors[p], lw=1.6, label=names[p])
        axes[1].plot(hist[p]["t"], hist[p]["r"], color=colors[p], lw=1.6, label=names[p])
    axes[0].axhline(0, color=C["muted"], lw=0.8, ls="--")
    axes[1].axhline(0, color=C["muted"], lw=0.8, ls="--")
    axes[0].set_title("Serve skill s_i (logit)", fontsize=10.5)
    axes[1].set_title("Return skill r_j (logit)", fontsize=10.5)
    axes[0].set_ylabel("skill (log-odds, relative to tour average)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.autofmt_xdate()
    fig.suptitle("Online random-walk ability filter: skill trajectories, 2000–2019 "
                 "(sigma=0.03/yr)", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/skill_trajectories.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_tuning()
    fig2_season_scores()
    fig3_alpha_by_surface()
    fig4_skill_trajectories()
    print("wrote 4 figures to", OUT)
