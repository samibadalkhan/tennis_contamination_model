"""Preregistered WTA 2025 online replication of the ordinary forecast study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.independent import run_2025 as base
from src.independent.bayes import serve_draws
from src.independent.run import ci, score
from src.independent.scoring import posterior_burst, posterior_iid_vectorized, rules
from src.util import DATA, RESULTS, utcnow, write_json


OUT = RESULTS / "wta_2025"
FIT_DIR = OUT / "fits"
CACHE_DIR = DATA / "wta_2025_cache"
LEVELS = {"G", "PM", "P", "I", "F"}
COUNT_COLUMNS = ["w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"]
PERIODS = [83, 84, 85, 86, 87, 88, 89]
MCMC_SEED = 1729
BURST_SEED = 4001
REPLICATE_BURST_SEED = 9029
SIMS = 16_384


def manifest_entries() -> dict[str, dict]:
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    return {entry["file"]: entry for entry in manifest["files"]}


def verify_source(year: int) -> str:
    path = DATA / "wta" / f"wta_matches_{year}.csv"
    relative = f"wta/{path.name}"
    entry = manifest_entries().get(relative)
    if entry is None:
        raise ValueError(f"{relative} absent from manifest")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"hash mismatch for {relative}")
    return digest


def load_wta(max_source_year: int) -> tuple[pd.DataFrame, dict]:
    frames, audits, hashes = [], [], {}
    for source_year in range(2011, max_source_year + 1):
        hashes[str(source_year)] = verify_source(source_year)
        raw = pd.read_csv(DATA / "wta" / f"wta_matches_{source_year}.csv", dtype=str)
        n_raw = len(raw)
        raw = raw[raw.tourney_level.isin(LEVELS)].copy()
        n_level = len(raw)
        for column in COUNT_COLUMNS + ["best_of"]:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
        raw["date"] = pd.to_datetime(raw.tourney_date, format="%Y%m%d", errors="coerce")
        required = COUNT_COLUMNS + ["date", "surface", "best_of", "winner_id", "loser_id",
                                    "tourney_id", "match_num", "tourney_name", "round"]
        raw = raw.dropna(subset=required)
        n_complete = len(raw)
        k1 = raw.w_1stWon + raw.w_2ndWon
        k2 = raw.l_1stWon + raw.l_2ndWon
        valid = ((raw.w_svpt > 0) & (raw.l_svpt > 0) &
                 (k1 >= 0) & (k1 <= raw.w_svpt) &
                 (k2 >= 0) & (k2 <= raw.l_svpt) & raw.best_of.isin([3, 5]))
        raw, k1, k2 = raw.loc[valid].copy(), k1.loc[valid], k2.loc[valid]
        year = raw.date.dt.year
        period = ((year - 2011) * 12 + raw.date.dt.month - 1) // 2
        frames.append(pd.DataFrame({
            "p1": "wta:" + raw.winner_id.str.replace(r"\.0$", "", regex=True),
            "p2": "wta:" + raw.loser_id.str.replace(r"\.0$", "", regex=True),
            "date": raw.date, "source_year": source_year, "year": year,
            "surface": raw.surface.str.lower(), "tournament": raw.tourney_name,
            "round": raw["round"], "score": raw.score.fillna(""),
            "n1": raw.w_svpt.astype(int), "k1": k1.astype(int),
            "n2": raw.l_svpt.astype(int), "k2": k2.astype(int),
            "period": period.astype(int),
            "match_id": "WTA-" + raw.tourney_id + "-" + raw.match_num,
            "tourney_id": raw.tourney_id, "best_of": raw.best_of.astype(int),
        }))
        audits.append({"source_year": source_year, "raw": n_raw, "eligible_level": n_level,
                       "complete_required": n_complete, "valid": int(valid.sum())})
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["date", "tourney_id", "round", "match_id"]).reset_index(drop=True)
    if data.match_id.duplicated().any() or (data.period < 0).any():
        raise ValueError("invalid WTA model table")
    audit = {"max_source_year": max_source_year, "hashes": hashes, "years": audits,
             "n": len(data), "date_min": str(data.date.min().date()),
             "date_max": str(data.date.max().date()),
             "period_min": int(data.period.min()), "period_max": int(data.period.max())}
    return data, audit


def fit(data: pd.DataFrame, period: int) -> dict:
    base.FIT_DIR = FIT_DIR
    universe = data[data.source_year <= 2024].copy() if period in [83, 84] else data
    return base.fit_period(universe, period, seed=MCMC_SEED)


def predict_period(data: pd.DataFrame, period: int, burst_seed: int) -> pd.DataFrame:
    rows = data[(data.source_year == 2025) & (data.period == period)].copy()
    rows = rows.sort_values(["date", "tourney_id", "round", "match_id"]).reset_index(drop=True)
    if rows.empty:
        return rows
    metadata = fit(data, period)
    stem = FIT_DIR / f"ordinary_period_{period}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"ordinary_period_{period}.npz"
    if cache.exists():
        with np.load(cache) as archive:
            if not np.array_equal(archive["match_ids"].astype(str), rows.match_id.astype(str)):
                raise ValueError(f"period {period} cache cohort mismatch")
            pa, pb, iid = archive["pa"], archive["pb"], archive["iid"]
    else:
        with np.load(stem.with_suffix(".npz")) as archive:
            arrays = {name: archive[name] for name in archive.files}
        pa, pb = serve_draws(rows, arrays, metadata["encoding"], seed=8000 + period)
        formats = np.array([rules(r.tournament, int(r.year), int(r.best_of))
                            for r in rows.itertuples()])
        iid = posterior_iid_vectorized(pa, pb, rows.best_of.to_numpy(),
                                       formats[:, 0], formats[:, 1])
        np.savez_compressed(cache, pa=pa, pb=pb, iid=iid,
                            match_ids=rows.match_id.to_numpy().astype(str))
    burst_cache = CACHE_DIR / f"ordinary_period_{period}_burst_{SIMS}_{burst_seed}.npy"
    if burst_cache.exists():
        conditional = np.load(burst_cache)
    else:
        formats = np.array([rules(r.tournament, int(r.year), int(r.best_of))
                            for r in rows.itertuples()])
        conditional = posterior_burst(pa, pb, rows.best_of.to_numpy(), formats[:, 0],
                                      formats[:, 1], SIMS, burst_seed + period * 1_000_003)
        np.save(burst_cache, conditional)
    rows["iid"] = iid
    rows["conditional_burst"] = conditional
    return rows


def paired(candidate: np.ndarray, reference: np.ndarray, rows: pd.DataFrame) -> dict:
    gain = np.log(np.clip(candidate, 1e-10, 1)) - np.log(np.clip(reference, 1e-10, 1))
    return {"tournament": ci(gain, rows),
            "player_tournament": ci(gain, rows, "player_tournament")}


def diagnostics(periods: list[int]) -> dict:
    items = []
    for period in periods:
        metadata = json.loads((FIT_DIR / f"ordinary_period_{period}.json").read_text())
        items.append({"period": period, "n_train": metadata["n_train"],
                      **metadata["diagnostics"]})
    return {"periods": items, "max_rhat": max(x["max_rhat"] for x in items),
            "min_ess_bulk": min(x["min_ess_bulk"] for x in items),
            "divergences": sum(x["divergences"] for x in items)}


def run() -> dict:
    protocol = OUT / "PROTOCOL.md"
    if not protocol.exists():
        raise FileNotFoundError(protocol)
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = {"utc": utcnow(), "status": "frozen_before_wta_2025_read",
              "periods": PERIODS, "levels": sorted(LEVELS), "mcmc_seed": MCMC_SEED,
              "burst_seed": BURST_SEED, "burst_rate": 0.5, "temperature": 1.5,
              "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "test_source_sha256": manifest_entries()["wta/wta_matches_2025.csv"]["sha256"]}
    freeze_path = OUT / "FROZEN_BEFORE_TEST.json"
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text())
        if {k: v for k, v in previous.items() if k != "utc"} != {k: v for k, v in freeze.items() if k != "utc"}:
            raise ValueError("WTA code or protocol changed after freeze")
        freeze = previous
    else:
        write_json(freeze_path, freeze)

    data, audit = load_wta(2025)
    write_json(OUT / "data_audit.json", audit)
    used_periods = [p for p in PERIODS if ((data.source_year == 2025) & (data.period == p)).any()]
    primary = pd.concat([predict_period(data, p, BURST_SEED) for p in used_periods],
                        ignore_index=True)
    primary = primary.sort_values(["date", "tourney_id", "round", "match_id"]).reset_index(drop=True)
    predictions = {
        "iid": primary.iid.to_numpy(),
        "burst": 0.5 * primary.iid.to_numpy() + 0.5 * primary.conditional_burst.to_numpy(),
        "temperature": expit(logit(np.clip(primary.iid.to_numpy(), 1e-8, 1 - 1e-8)) / 1.5),
    }
    metrics = {name: score(value, primary) for name, value in predictions.items()}
    comparisons = {
        "burst_vs_iid": paired(predictions["burst"], predictions["iid"], primary),
        "burst_vs_temperature": paired(predictions["burst"], predictions["temperature"], primary),
        "temperature_vs_iid": paired(predictions["temperature"], predictions["iid"], primary),
    }
    replicate = pd.concat([predict_period(data, p, REPLICATE_BURST_SEED) for p in used_periods],
                          ignore_index=True)
    replicate = replicate.set_index("match_id").loc[primary.match_id]
    replicate_burst = 0.5 * primary.iid.to_numpy() + 0.5 * replicate.conditional_burst.to_numpy()
    simulation_check = {"primary_seed": BURST_SEED, "replicate_seed": REPLICATE_BURST_SEED,
                        "primary_log_loss": float(-np.log(predictions["burst"]).mean()),
                        "replicate_log_loss": float(-np.log(replicate_burst).mean()),
                        "mean_absolute_probability_difference": float(
                            np.abs(predictions["burst"] - replicate_burst).mean())}
    for name, values in predictions.items():
        primary[name] = values
    primary["burst_seed2"] = replicate_burst
    primary.to_csv(OUT / "predictions_2025.csv", index=False)
    result = {"utc": utcnow(), "status": "preregistered_confirmatory", "n": len(primary),
              "freeze": freeze, "metrics": metrics, "comparisons": comparisons,
              "simulation_check": simulation_check, "fit_diagnostics": diagnostics(used_periods),
              "note": "Positive paired log-score gain favors the first method."}
    write_json(OUT / "results.json", result)
    report(result)
    plot(result)
    return result


def report(result: dict) -> None:
    lines = ["# Preregistered WTA 2025 replication", "",
             f"Scored {result['n']} eligible WTA matches with chronological bimonthly updates.", "",
             "| Forecast | Log loss | Accuracy | Brier | Calibration slope |",
             "|---|---:|---:|---:|---:|"]
    for name, item in result["metrics"].items():
        lines.append(f"| {name} | {item['log_loss']['tournament']['mean']:.5f} | "
                     f"{item['accuracy']['tournament']['mean']:.4f} | "
                     f"{item['brier']['tournament']['mean']:.5f} | "
                     f"{item['calibration']['slope']:.3f} |")
    lines += ["", "Paired log-score gains (positive favors first):", ""]
    for name, item in result["comparisons"].items():
        t, p = item["tournament"], item["player_tournament"]
        lines.append(f"- {name}: {t['mean']:+.5f}; tournament CI [{t['lo']:+.5f}, {t['hi']:+.5f}]; "
                     f"player-tournament CI [{p['lo']:+.5f}, {p['hi']:+.5f}].")
    d, s = result["fit_diagnostics"], result["simulation_check"]
    lines += ["", f"Across fits: max R-hat {d['max_rhat']:.3f}, minimum bulk ESS "
              f"{d['min_ess_bulk']:.0f}, {d['divergences']} divergences.",
              f"The burst simulation replicate changed probabilities by {s['mean_absolute_probability_difference']:.5f} "
              f"on average and log loss from {s['primary_log_loss']:.5f} to {s['replicate_log_loss']:.5f}."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def plot(result: dict) -> None:
    import matplotlib.pyplot as plt

    names = ["iid", "burst", "temperature"]
    labels = ["IID", "Burst", "Temperature"]
    losses = [result["metrics"][name]["log_loss"]["tournament"] for name in names]
    comparisons = ["burst_vs_iid", "burst_vs_temperature", "temperature_vs_iid"]
    comparison_labels = ["Burst − IID", "Burst − temp.", "Temp. − IID"]
    gains = [result["comparisons"][name]["tournament"] for name in comparisons]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    mean = np.array([x["mean"] for x in losses])
    lo = np.array([x["lo"] for x in losses])
    hi = np.array([x["hi"] for x in losses])
    axes[0].errorbar(np.arange(3), mean, yerr=[mean - lo, hi - mean], fmt="o",
                     capsize=4, color="#2457A6")
    axes[0].set_xticks(np.arange(3), labels)
    axes[0].set_ylabel("Log loss")
    axes[0].set_title("Absolute season scores")
    mean = np.array([x["mean"] for x in gains])
    lo = np.array([x["lo"] for x in gains])
    hi = np.array([x["hi"] for x in gains])
    axes[1].errorbar(np.arange(3), mean, yerr=[mean - lo, hi - mean], fmt="o",
                     capsize=4, color="#B44A3A")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(np.arange(3), comparison_labels, rotation=15)
    axes[1].set_ylabel("Paired log-score gain")
    axes[1].set_title("Matched forecast comparisons")
    fig.suptitle("Preregistered WTA 2025 replication")
    fig.tight_layout()
    fig.savefig(OUT / "wta_2025_summary.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    base.sampler_progress()
    print(json.dumps(run(), indent=2))
