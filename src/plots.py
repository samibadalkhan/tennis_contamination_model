"""Stage 0 figures. Reads the checkpoints in results/stage0/ + the block cache,
writes PNGs to results/stage0/figures/.

Palette: Okabe-Ito (colorblind-safe categorical). Thin marks, one axis, legend
present for >=2 series, threshold lines labeled, a source line on each figure.

    python -m src.plots
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.util import DATA, STAGE0, read_json

CACHE = DATA / "cache"
FIG = STAGE0 / "figures"

# Okabe-Ito (CVD-safe)
BLUE, ORANGE, GREEN, VERM, SKY, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#999999"
INK = "#222222"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": "#cccccc", "axes.grid": True, "grid.color": "#eeeeee",
    "axes.axisbelow": True, "axes.titleweight": "bold", "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _blocks_z():
    d = pd.read_parquet(CACHE / "train_pred.parquet")
    g = (d.groupby(["match_id", "set_no", "server"], observed=True)
         .agg(n=("server_wins", "count"), wins=("server_wins", "sum"), p=("p", "mean"))
         .reset_index())
    g = g[g.n >= 10]
    z = (g.wins - g.n * g.p) / np.sqrt(g.n * g.p * (1 - g.p))
    return z.to_numpy()


def fig_residuals():
    """Overdispersion (spread) + asymmetry (skew) in one block-residual histogram."""
    od = read_json(STAGE0 / "overdispersion.json"); asym = read_json(STAGE0 / "asymmetry.json")
    z = _blocks_z()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(-4, 4, 49)
    ax.hist(z, bins=bins, density=True, color=BLUE, alpha=0.75, label="set-block residuals", edgecolor="white", linewidth=0.3)
    xs = np.linspace(-4, 4, 400)
    ax.plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=2, label="N(0,1): i.i.d. model expects this")
    ax.axvline(0, color=GREY, lw=1)
    ax.set_xlabel("standardized block residual  z = (wins − n·p̂) / √(n·p̂(1−p̂))")
    ax.set_ylabel("density")
    ax.set_title("Stage 0 — service blocks are overdispersed and left-skewed")
    ax.annotate(f"overdispersion  φ = {od['phi']}  (CI {od['ci95']})\n"
                f"→ residuals wider than binomial",
                xy=(0.02, 0.97), xycoords="axes fraction", va="top", ha="left", color=INK,
                bbox=dict(boxstyle="round", fc="#f4f8fb", ec="#cccccc"))
    ax.annotate(f"skew = {asym['skewness']}  (CI {asym['ci95']})\n→ excess of worse-than-expected\nblocks (left tail): directional",
                xy=(0.02, 0.55), xycoords="axes fraction", va="top", ha="left", color=VERM,
                bbox=dict(boxstyle="round", fc="#fdf3ee", ec="#e7c3ad"))
    ax.legend(loc="upper right", frameon=False)
    fig.text(0.01, 0.005, "Train fold (2011–2018), set-level service blocks n≥10. Source: results/stage0/{overdispersion,asymmetry}.json",
             fontsize=7, color=GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "1_block_residuals.png"); plt.close(fig)


def fig_momentum():
    """The headline: observed vs i.i.d.-under-scoring null, by game length."""
    mom = read_json(STAGE0 / "momentum_investigation.json")
    obs = mom["by_game_length_observed"]; nul = mom["by_game_length_iid_null"]
    labels = [k for k in obs if obs[k]["ratio"] is not None and nul.get(k, {}).get("ratio") is not None]
    o = [obs[k]["ratio"] for k in labels]; s = [nul[k]["ratio"] for k in labels]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x - w/2, o, w, color=BLUE, label="observed")
    ax.bar(x + w/2, s, w, color=ORANGE, label="i.i.d. points under tennis rules (no momentum)")
    ax.axhline(1.0, color=INK, lw=1.2, ls="--")
    ax.annotate("ratio = 1: no excess alternation", xy=(len(labels)-1, 1.0), xytext=(len(labels)-1, 0.86),
                ha="right", color=INK, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("within-game transition ratio\n(observed sign-changes / permutation-expected)")
    ax.set_xlabel("game length (points)")
    ax.set_title("Momentum? No — the 'anti-momentum' is a scoring-structure artifact")
    ax.annotate(f"overall: observed {mom['observed_transition_ratio']}  vs  null {mom['iid_under_scoring_null_ratio']}\n"
                f"real effect = {mom['real_effect_obs_over_null']} (≈ none). Bars match at every length.",
                xy=(0.02, 0.97), xycoords="axes fraction", va="top", ha="left", color=INK,
                bbox=dict(boxstyle="round", fc="#f4f8fb", ec="#cccccc"), fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.80), frameon=False)
    fig.text(0.01, 0.005, "Deuce forces alternation; momentum-free simulation reproduces the same ratios. Source: results/stage0/momentum_investigation.json",
             fontsize=7, color=GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "2_momentum_artifact.png"); plt.close(fig)


def fig_noise_floor():
    """Oracle contamination effect per (eps,delta) vs the held-out CI half-width."""
    nf = read_json(STAGE0 / "noise_floor.json")
    grid = nf["oracle_effect_grid"]; hw = nf["ci_half_width"]
    labels = [f"ε={g['eps']}\nδ={g['delta']}" for g in grid]
    vals = [g["oracle_logloss_gain"] for g in grid]
    det = [g["detectable"] for g in grid]
    colors = [GREEN if d else GREY for d in det]
    x = np.arange(len(grid))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x, vals, 0.62, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(hw, color=VERM, lw=1.8, ls="--")
    ax.annotate(f"noise floor: held-out log-loss CI half-width = {hw}",
                xy=(len(grid)-0.5, hw), xytext=(len(grid)-0.5, hw*2.0),
                ha="right", color=VERM, fontsize=8,
                arrowprops=dict(arrowstyle="->", color=VERM, lw=1))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("oracle log-loss gain from resolving contamination")
    ax.set_xlabel("contamination scenario (ε = share of blocks, δ = serve-win-prob drop)")
    ax.set_title(f"Noise floor — design is powered ({nf['n_detectable_cells']}/{len(grid)} scenarios detectable)")
    # legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN, label="above noise floor (detectable)"),
                       Patch(color=GREY, label="below noise floor")],
              loc="upper left", frameon=False)
    fig.text(0.01, 0.005, "Oracle = upper bound (knows contamination labels). Validation fold, clustered over matches. Source: results/stage0/noise_floor.json",
             fontsize=7, color=GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "3_noise_floor.png"); plt.close(fig)


def fig_retirement():
    """Stage 0.5: retiree serve-win degradation vs ordinary-loser control."""
    deg = read_json(STAGE0.parent / "stage05" / "degradation.json")
    if not deg:
        return
    figdir = STAGE0.parent / "stage05" / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    for ax, win, title in zip(axes, ("last_set", "last_k"),
                              ("last set played", "last 15 service points")):
        w = deg["windows"][win]
        groups = ["retirees\n(contaminated)", "ordinary\nlosers (control)"]
        r, c = w["retirement"], w["control_losers"]
        deltas = [r["mean_degradation"], c["mean_degradation"]]
        errs = [[r["mean_degradation"] - r["ci95"][0], c["mean_degradation"] - c["ci95"][0]],
                [r["ci95"][1] - r["mean_degradation"], c["ci95"][1] - c["mean_degradation"]]]
        ax.bar(groups, deltas, 0.6, color=[VERM, GREY], yerr=errs, capsize=6,
               edgecolor="white", error_kw=dict(ecolor=INK, lw=1.2))
        ax.axhline(0, color=INK, lw=1)
        ax.set_title(title)
        ax.set_ylabel("serve-win rate drop (baseline − lead-up)")
        ax.annotate(f"excess = {w['excess_over_control']}\n(n={r['n_matches']} retirements)",
                    xy=(0.5, 0.93), xycoords="axes fraction", ha="center", va="top",
                    color=INK, fontsize=9, bbox=dict(boxstyle="round", fc="#fdf3ee", ec="#e7c3ad"))
    fig.suptitle("Stage 0.5 — retirees' serve degrades ~3× more than ordinary losers (detector fires)",
                 fontweight="bold")
    fig.text(0.01, 0.005, "Match-clustered 95% CI. Train+val years, test untouched. Source: results/stage05/degradation.json",
             fontsize=7, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(figdir / "1_retirement_degradation.png"); plt.close(fig)


def run():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_residuals(); fig_momentum(); fig_noise_floor(); fig_retirement()
    print(f"wrote figures to {FIG.relative_to(DATA.parent)}/ and stage05/figures/")


if __name__ == "__main__":
    run()
