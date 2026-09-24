"""Independent MCMC-seed replication of all ATP 2025 online posteriors."""
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


OUT = RESULTS / "independent_2025" / "mcmc_seed_1729"
FIT_DIR = OUT / "fits"
CACHE_DIR = DATA / "independent_2025_seed_1729_cache"
PERIODS = [84, 85, 86, 87, 88, 89]
MCMC_SEED = 1729
BURST_SEED = 4001


def paired(candidate: np.ndarray, reference: np.ndarray, rows: pd.DataFrame) -> dict:
    gain = np.log(np.clip(candidate, 1e-10, 1)) - np.log(np.clip(reference, 1e-10, 1))
    return {"tournament": ci(gain, rows),
            "player_tournament": ci(gain, rows, "player_tournament")}


def fit(data: pd.DataFrame, period: int) -> dict:
    base.FIT_DIR = FIT_DIR
    universe = data[data.source_year <= 2024].copy() if period == 84 else data
    return base.fit_period(universe, period, seed=MCMC_SEED)


def predict_block(data: pd.DataFrame, period: int) -> pd.DataFrame:
    if period == 84:
        rows = data[(data.source_year == 2025) & (data.period <= 84)].copy()
    else:
        rows = data[(data.source_year == 2025) & (data.period == period)].copy()
    rows = rows.sort_values(["date", "tourney_id", "round", "match_id"]).reset_index(drop=True)
    metadata = fit(data, period)
    stem = FIT_DIR / f"ordinary_period_{period}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"ordinary_period_{period}.npz"
    if cache.exists():
        with np.load(cache) as archive:
            if not np.array_equal(archive["match_ids"].astype(str), rows.match_id.astype(str)):
                raise ValueError(f"cache cohort mismatch in period {period}")
            pa, pb, iid = archive["pa"], archive["pb"], archive["iid"]
    else:
        with np.load(stem.with_suffix(".npz")) as archive:
            arrays = {name: archive[name] for name in archive.files}
        model_rows = rows.copy()
        if period == 84:
            model_rows["period"] = 84
        pa, pb = serve_draws(model_rows, arrays, metadata["encoding"], seed=8000 + period)
        formats = np.array([rules(r.tournament, int(r.year), int(r.best_of))
                            for r in rows.itertuples()])
        iid = posterior_iid_vectorized(pa, pb, rows.best_of.to_numpy(),
                                       formats[:, 0], formats[:, 1])
        np.savez_compressed(cache, pa=pa, pb=pb, iid=iid,
                            match_ids=rows.match_id.to_numpy().astype(str))
    burst_cache = CACHE_DIR / f"ordinary_period_{period}_burst_{base.SIMS}_{BURST_SEED}.npy"
    if burst_cache.exists():
        conditional = np.load(burst_cache)
    else:
        formats = np.array([rules(r.tournament, int(r.year), int(r.best_of))
                            for r in rows.itertuples()])
        conditional = posterior_burst(pa, pb, rows.best_of.to_numpy(), formats[:, 0],
                                      formats[:, 1], base.SIMS,
                                      BURST_SEED + period * 1_000_003)
        np.save(burst_cache, conditional)
    rows["replicate_iid"] = iid
    rows["replicate_burst"] = 0.5 * iid + 0.5 * conditional
    rows["replicate_temperature"] = expit(logit(np.clip(iid, 1e-8, 1 - 1e-8)) / 1.5)
    return rows


def diagnostics() -> dict:
    items = []
    for period in PERIODS:
        metadata = json.loads((FIT_DIR / f"ordinary_period_{period}.json").read_text())
        items.append({"period": period, "n_train": metadata["n_train"],
                      **metadata["diagnostics"]})
    return {"periods": items,
            "max_rhat": max(x["max_rhat"] for x in items),
            "min_ess_bulk": min(x["min_ess_bulk"] for x in items),
            "divergences": sum(x["divergences"] for x in items)}


def run() -> dict:
    protocol = RESULTS / "independent_2025" / "MCMC_REPLICATE_PROTOCOL.md"
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = {
        "utc": utcnow(), "mcmc_seed": MCMC_SEED, "burst_seed": BURST_SEED,
        "periods": PERIODS,
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_code_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
    }
    freeze_path = OUT / "FROZEN.json"
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text())
        if {k: v for k, v in previous.items() if k != "utc"} != {k: v for k, v in freeze.items() if k != "utc"}:
            raise ValueError("replication code or protocol changed after freeze")
        freeze = previous
    else:
        write_json(freeze_path, freeze)

    data, audit = base.load_extended(2025)
    write_json(OUT / "data_audit.json", audit)
    rows = pd.concat([predict_block(data, p) for p in PERIODS], ignore_index=True)
    rows = rows.sort_values(["date", "tourney_id", "round", "match_id"]).reset_index(drop=True)
    original = pd.read_csv(RESULTS / "independent_2025" / "online_predictions_2025.csv")
    original = original.set_index("match_id").loc[rows.match_id].reset_index()
    if len(rows) != len(original) or len(rows) != 2670:
        raise ValueError("replicate cohort differs from original")
    metrics = {name: score(rows[name].to_numpy(), rows)
               for name in ["replicate_iid", "replicate_burst", "replicate_temperature"]}
    comparisons = {
        "replicate_burst_vs_replicate_iid": paired(rows.replicate_burst, rows.replicate_iid, rows),
        "replicate_burst_vs_replicate_temperature": paired(
            rows.replicate_burst, rows.replicate_temperature, rows),
        "replicate_burst_vs_original_burst": paired(
            rows.replicate_burst, original.online_burst.to_numpy(), rows),
    }
    rows["original_online_burst"] = original.online_burst.to_numpy()
    rows.to_csv(OUT / "predictions.csv", index=False)
    result = {"utc": utcnow(), "status": "post_hoc_mcmc_seed_replication", "n": len(rows),
              "freeze": freeze, "metrics": metrics, "comparisons": comparisons,
              "mean_absolute_probability_difference": float(
                  np.abs(rows.replicate_burst - rows.original_online_burst).mean()),
              "diagnostics": diagnostics(),
              "note": "Positive paired log-score gain favors the first method."}
    write_json(OUT / "results.json", result)
    report(result)
    return result


def report(result: dict) -> None:
    lines = ["# ATP 2025 MCMC seed replication", "",
             f"Repeated all six online posteriors with NUTS seed {MCMC_SEED}.", "",
             "| Forecast | Log loss |", "|---|---:|"]
    for name, item in result["metrics"].items():
        lines.append(f"| {name} | {item['log_loss']['tournament']['mean']:.5f} |")
    lines += ["", "Paired log-score gains (positive favors first):", ""]
    for name, item in result["comparisons"].items():
        t, p = item["tournament"], item["player_tournament"]
        lines.append(f"- {name}: {t['mean']:+.5f}; tournament CI [{t['lo']:+.5f}, {t['hi']:+.5f}]; "
                     f"player-tournament CI [{p['lo']:+.5f}, {p['hi']:+.5f}].")
    d = result["diagnostics"]
    lines += ["", f"Mean absolute burst-probability change: "
              f"{result['mean_absolute_probability_difference']:.5f}.",
              f"Across fits: max R-hat {d['max_rhat']:.3f}, minimum bulk ESS "
              f"{d['min_ess_bulk']:.0f}, {d['divergences']} divergences."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    base.sampler_progress()
    print(json.dumps(run(), indent=2))
