"""Frozen 2025 ordinary full-model evaluation.

The prepare stage reads source seasons through 2024 only, freezes the forecast
settings carried from the verified 2014 reproduction, and fits one full
ordinary Ingram posterior through 2024.  The test stage refuses to run without
that frozen record, then reads and scores 2025 once.

Usage:
    .venv-independent/bin/python -m src.independent.run_2025 --stage prepare
    .venv-independent/bin/python -m src.independent.run_2025 --stage test
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=/private/tmp/tennis_pytensor_2025")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tennis_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/tennis_cache")

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src import trials
from src.independent.bayes import HYPERS, build_model, serve_draws
from src.independent.run import ci, score, strata
from src.independent.scoring import posterior_burst, posterior_iid_vectorized, rules
from src.util import DATA, RESULTS, utcnow, write_json


OUT = RESULTS / "independent_2025"
FIT_DIR = OUT / "fits"
CACHE_DIR = DATA / "independent_2025_cache"
RATES = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
TEMPERATURES = [0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
SIMS = 16_384
SEED = 4001
REPLICATE_SEED = 9029
SNAPSHOT_PERIOD = 84
FROZEN_RATE = 0.5
FROZEN_TEMPERATURE = 1.5
LEVELS = {"G", "M", "A", "F"}
COUNT_COLUMNS = ["w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"]


def _manifest_entries() -> dict[str, dict]:
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    return {x["file"]: x for x in manifest["files"]}


def _source_file(year: int) -> Path:
    return DATA / "atp" / f"atp_matches_{year}.csv"


def verify_source(year: int) -> str:
    path = _source_file(year)
    rel = f"atp/{path.name}"
    entry = _manifest_entries().get(rel)
    if entry is None:
        raise ValueError(f"{rel} is absent from data/MANIFEST.json")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != entry["sha256"]:
        raise ValueError(f"hash mismatch for {rel}: {observed} != {entry['sha256']}")
    return observed


def load_extended(max_source_year: int) -> tuple[pd.DataFrame, dict]:
    """Build the full-model table without reading any later source season."""
    frames = []
    audits = []
    hashes = {}
    for source_year in range(2011, max_source_year + 1):
        hashes[str(source_year)] = verify_source(source_year)
        raw = pd.read_csv(_source_file(source_year), dtype=str)
        n_raw = len(raw)
        raw = raw[raw["tourney_level"].isin(LEVELS)].copy()
        n_level = len(raw)
        for column in COUNT_COLUMNS + ["best_of"]:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
        raw["date"] = pd.to_datetime(raw["tourney_date"], format="%Y%m%d", errors="coerce")
        required = COUNT_COLUMNS + ["date", "surface", "best_of", "winner_id", "loser_id",
                                    "tourney_id", "match_num", "tourney_name", "round"]
        raw = raw.dropna(subset=required)
        n_complete = len(raw)
        k1 = raw.w_1stWon + raw.w_2ndWon
        k2 = raw.l_1stWon + raw.l_2ndWon
        valid = ((raw.w_svpt > 0) & (raw.l_svpt > 0) &
                 (k1 >= 0) & (k1 <= raw.w_svpt) &
                 (k2 >= 0) & (k2 <= raw.l_svpt) &
                 raw.best_of.isin([3, 5]))
        raw = raw.loc[valid].copy()
        k1 = k1.loc[valid]
        k2 = k2.loc[valid]
        date_year = raw.date.dt.year
        period = ((date_year - 2011) * 12 + raw.date.dt.month - 1) // 2
        frame = pd.DataFrame({
            "p1": "atp:" + raw.winner_id.str.replace(r"\.0$", "", regex=True),
            "p2": "atp:" + raw.loser_id.str.replace(r"\.0$", "", regex=True),
            "date": raw.date,
            "source_year": source_year,
            "year": date_year,
            "surface": raw.surface.str.lower(),
            "tournament": raw.tourney_name,
            "round": raw["round"],
            "score": raw.score.fillna(""),
            "n1": raw.w_svpt.astype(int),
            "k1": k1.astype(int),
            "n2": raw.l_svpt.astype(int),
            "k2": k2.astype(int),
            "period": period.astype(int),
            "match_id": ("ATP-" + raw.tourney_id + "-" + raw.match_num),
            "tourney_id": raw.tourney_id,
            "best_of": raw.best_of.astype(int),
        })
        frames.append(frame)
        audits.append({"source_year": source_year, "raw": n_raw, "eligible_level": n_level,
                       "complete_required": n_complete, "valid": len(frame),
                       "excluded": n_raw - len(frame)})
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["date", "tourney_id", "round", "match_id"]).reset_index(drop=True)
    if data.match_id.duplicated().any():
        dupes = data.loc[data.match_id.duplicated(False), "match_id"].head().tolist()
        raise ValueError(f"duplicate match IDs: {dupes}")
    if (data.period < 0).any():
        raise ValueError("pre-2011 observation entered the model")
    audit = {"max_source_year": max_source_year, "hashes": hashes, "years": audits,
             "n": len(data), "date_min": str(data.date.min().date()),
             "date_max": str(data.date.max().date()),
             "period_min": int(data.period.min()), "period_max": int(data.period.max())}
    return data, audit


def _fingerprint(frame: pd.DataFrame) -> str:
    columns = ["p1", "p2", "date", "surface", "tournament", "n1", "k1", "n2", "k2", "period"]
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def sampler_progress() -> None:
    """Print periodic sampler progress without changing the probability model."""
    import nutpie
    original = nutpie.sample

    def wrapped(*args, **kwargs):
        previous = kwargs.get("progress_callback")

        def progress(chains):
            if previous is not None:
                previous(chains)
            print("SAMPLING", [(c.finished_draws, c.total_draws, c.tuning) for c in chains], flush=True)

        kwargs.update(progress_callback=progress, progress_rate=30_000)
        return original(*args, **kwargs)

    nutpie.sample = wrapped


def fit_period(data: pd.DataFrame, period: int, draws: int = 1000, tune: int = 1000,
               chains: int = 4, seed: int = 521) -> dict:
    import arviz as az
    import pymc as pm

    train = data[data.period < period]
    if train.empty:
        raise ValueError(f"no training observations before period {period}")
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIT_DIR / f"ordinary_period_{period}"
    config = {"period": period, "draws": draws, "tune": tune, "chains": chains,
              "seed": seed, "train_fingerprint": _fingerprint(train),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if stem.with_suffix(".json").exists():
        old = json.loads(stem.with_suffix(".json").read_text())
        if old["config"] == config and stem.with_suffix(".npz").exists():
            print("CACHED", stem, flush=True)
            return old
        raise ValueError(f"existing fit has a different frozen config: {stem}")

    print("FIT ordinary", period, "matches", len(train), "draws", draws, "chains", chains, flush=True)
    started = time.monotonic()
    model, encoding = build_model(train, cap=None)
    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=min(chains, 4),
                          random_seed=seed + period, target_accept=0.9,
                          nuts_sampler="nutpie", progressbar=False, blas_cores=1)
    posterior = trace["posterior"]
    arrays = {}
    for name in HYPERS + ["last_s", "last_r", "surf", "tour"]:
        array = posterior[name].values
        arrays[name] = array.reshape((-1,) + array.shape[2:])
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    summary = az.summary(trace, var_names=HYPERS + ["last_s", "last_r", "surf", "tour"])
    summary.to_csv(str(stem) + "_diagnostics.csv")
    diagnostics = {
        "max_rhat": float(summary.r_hat.max()),
        "min_ess_bulk": float(summary.ess_bulk.min()),
        "divergences": int(trace["sample_stats"]["diverging"].values.sum()),
        "hyperparameters": summary.loc[HYPERS].reset_index().to_dict("records"),
    }
    encoding["last_period"] = [int(x) for x in encoding["last_period"]]
    result = {"utc": utcnow(), "config": config, "encoding": encoding,
              "n_train": len(train), "diagnostics": diagnostics,
              "seconds": time.monotonic() - started,
              "period_start": str(pd.Timestamp(2011 + period // 6, 1 + (period % 6) * 2, 1).date())}
    write_json(stem.with_suffix(".json"), result)
    trials.log(stage="independent-ordinary-2025", kind="fit", params=config,
               metrics={"max_rhat": diagnostics["max_rhat"],
                        "min_ess_bulk": diagnostics["min_ess_bulk"],
                        "divergences": diagnostics["divergences"],
                        "seconds": result["seconds"]})
    print("DONE ordinary", period, diagnostics["max_rhat"], diagnostics["min_ess_bulk"],
          "seconds", round(result["seconds"]), flush=True)
    del trace, posterior, model, arrays
    gc.collect()
    return result


def period_prediction(data: pd.DataFrame, rows: pd.DataFrame, period: int,
                      burst_seed: int = SEED) -> pd.DataFrame:
    rows = rows[rows.period == period].copy().reset_index(drop=True)
    if rows.empty:
        return rows
    metadata = fit_period(data, period)
    stem = FIT_DIR / f"ordinary_period_{period}"
    cache = CACHE_DIR / f"ordinary_period_{period}.npz"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with np.load(cache) as z:
            if not np.array_equal(z["match_ids"].astype(str), rows.match_id.to_numpy().astype(str)):
                raise ValueError(f"prediction cache cohort mismatch for period {period}")
            pa, pb, iid = z["pa"], z["pb"], z["iid"]
    else:
        with np.load(stem.with_suffix(".npz")) as archive:
            arrays = {name: archive[name] for name in archive.files}
        pa, pb = serve_draws(rows, arrays, metadata["encoding"], seed=8000 + period)
        formats = np.array([rules(r.tournament, int(r.year), int(r.best_of)) for r in rows.itertuples()])
        iid = posterior_iid_vectorized(pa, pb, rows.best_of.to_numpy(), formats[:, 0], formats[:, 1])
        np.savez_compressed(cache, pa=pa, pb=pb, iid=iid,
                            match_ids=rows.match_id.to_numpy().astype(str))
    rows["iid"] = iid
    burst_path = CACHE_DIR / f"ordinary_period_{period}_burst_{SIMS}_{burst_seed}.npy"
    if burst_path.exists():
        conditional = np.load(burst_path)
    else:
        formats = np.array([rules(r.tournament, int(r.year), int(r.best_of)) for r in rows.itertuples()])
        print("SIMULATE ordinary", period, "matches", len(rows), "S", SIMS,
              "seed", burst_seed, flush=True)
        conditional = posterior_burst(pa, pb, rows.best_of.to_numpy(), formats[:, 0], formats[:, 1],
                                      SIMS, burst_seed + period * 1_000_003)
        np.save(burst_path, conditional)
        trials.log(stage="independent-ordinary-2025", kind="conditional-burst-simulation",
                   params={"period": period, "S": SIMS, "seed": burst_seed,
                           "severity": 1.5, "duration": 40, "start_uniform": [0, 80]})
    if len(conditional) != len(rows):
        raise ValueError(f"burst cache length mismatch for period {period}")
    rows["conditional_burst"] = conditional
    return rows


def season_predictions(data: pd.DataFrame, source_year: int, burst_seed: int = SEED) -> pd.DataFrame:
    rows = data[data.source_year == source_year].copy()
    periods = sorted(rows.period.unique())
    predictions = pd.concat([period_prediction(data, rows, int(p), burst_seed) for p in periods],
                            ignore_index=True)
    predictions = predictions.sort_values(["date", "tourney_id", "round", "match_id"]).reset_index(drop=True)
    predictions.to_csv(OUT / f"ordinary_{source_year}_predictions_{burst_seed}.csv", index=False)
    return predictions


def tune_2024() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    data, audit = load_extended(2024)
    write_json(OUT / "development_data_audit.json", audit)
    rows = season_predictions(data, 2024, SEED)
    curve = []
    for rate in RATES:
        probability = (1 - rate) * rows.iid.to_numpy() + rate * rows.conditional_burst.to_numpy()
        loss = float(-np.log(np.clip(probability, 1e-10, 1)).mean())
        curve.append({"rate": rate, "log_loss": loss})
        trials.log(stage="independent-ordinary-2025", kind="tune_burst_2024",
                   params={"rate": rate}, metrics={"log_loss": loss, "n": len(rows)})
    rate = min(curve, key=lambda x: (x["log_loss"], x["rate"]))["rate"]
    temp_curve = []
    for temperature in TEMPERATURES:
        probability = expit(logit(np.clip(rows.iid.to_numpy(), 1e-8, 1 - 1e-8)) / temperature)
        loss = float(-np.log(np.clip(probability, 1e-10, 1)).mean())
        temp_curve.append({"temperature": temperature, "log_loss": loss})
        trials.log(stage="independent-ordinary-2025", kind="tune_temperature_2024",
                   params={"temperature": temperature}, metrics={"log_loss": loss, "n": len(rows)})
    temperature = min(temp_curve, key=lambda x: (x["log_loss"], x["temperature"]))["temperature"]
    frozen = {"utc": utcnow(), "tune_source_year": 2024, "n": len(rows),
              "burst": {"chosen_rate": rate, "curve": curve, "severity": 1.5,
                        "duration": 40, "start_uniform": [0, 80], "simulations": SIMS,
                        "seed": SEED},
              "temperature": {"chosen": temperature, "curve": temp_curve},
              "protocol_sha256": hashlib.sha256((OUT / "PROTOCOL.md").read_bytes()).hexdigest(),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target = OUT / "FROZEN_BEFORE_2025.json"
    if target.exists():
        old = json.loads(target.read_text())
        comparable = {k: v for k, v in old.items() if k != "utc"}
        current = {k: v for k, v in frozen.items() if k != "utc"}
        if comparable != current:
            raise ValueError("existing frozen settings differ; do not overwrite")
        frozen = old
    else:
        write_json(target, frozen)
    print("FROZEN rate", rate, "temperature", temperature, "n", len(rows), flush=True)
    return frozen


def prepare_2025() -> dict:
    """Freeze carried settings and fit through 2024 without reading 2025."""
    OUT.mkdir(parents=True, exist_ok=True)
    data, audit = load_extended(2024)
    write_json(OUT / "training_data_audit.json", audit)
    frozen = {
        "utc": utcnow(),
        "training_source_years": [2011, 2024],
        "test_source_year": 2025,
        "snapshot_period": SNAPSHOT_PERIOD,
        "burst": {"chosen_rate": FROZEN_RATE, "selected_on": 2013,
                  "source": "results/independent/ordinary_frozen_forecast.json",
                  "severity": 1.5, "duration": 40, "start_uniform": [0, 80],
                  "simulations": SIMS, "seed": SEED},
        "temperature": {"chosen": FROZEN_TEMPERATURE, "selected_on": 2013,
                        "source": "results/independent/comparison.json"},
        "protocol_sha256": hashlib.sha256((OUT / "PROTOCOL.md").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_hashes": audit["hashes"],
    }
    target = OUT / "FROZEN_BEFORE_2025.json"
    if target.exists():
        old = json.loads(target.read_text())
        comparable = {k: v for k, v in old.items() if k != "utc"}
        current = {k: v for k, v in frozen.items() if k != "utc"}
        if comparable != current:
            raise ValueError("existing frozen settings differ; do not overwrite")
        frozen = old
    else:
        write_json(target, frozen)
    fit_period(data, SNAPSHOT_PERIOD)
    print("PREPARED snapshot period", SNAPSHOT_PERIOD, "rate", FROZEN_RATE,
          "temperature", FROZEN_TEMPERATURE, flush=True)
    return frozen


def snapshot_predictions(train_data: pd.DataFrame, rows: pd.DataFrame,
                         burst_seed: int = SEED) -> pd.DataFrame:
    """Forecast every row from the single posterior frozen through 2024."""
    rows = rows.copy().sort_values(["date", "tourney_id", "round", "match_id"]).reset_index(drop=True)
    metadata = fit_period(train_data, SNAPSHOT_PERIOD)
    stem = FIT_DIR / f"ordinary_period_{SNAPSHOT_PERIOD}"
    cache = CACHE_DIR / "ordinary_snapshot_2025.npz"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with np.load(cache) as archive:
            if not np.array_equal(archive["match_ids"].astype(str), rows.match_id.to_numpy().astype(str)):
                raise ValueError("2025 snapshot cache cohort mismatch")
            pa, pb, iid = archive["pa"], archive["pb"], archive["iid"]
    else:
        with np.load(stem.with_suffix(".npz")) as archive:
            arrays = {name: archive[name] for name in archive.files}
        model_rows = rows.copy()
        model_rows["period"] = SNAPSHOT_PERIOD
        pa, pb = serve_draws(model_rows, arrays, metadata["encoding"], seed=8000 + SNAPSHOT_PERIOD)
        formats = np.array([rules(r.tournament, int(r.year), int(r.best_of)) for r in rows.itertuples()])
        iid = posterior_iid_vectorized(pa, pb, rows.best_of.to_numpy(), formats[:, 0], formats[:, 1])
        np.savez_compressed(cache, pa=pa, pb=pb, iid=iid,
                            match_ids=rows.match_id.to_numpy().astype(str))
    rows["iid"] = iid
    burst_path = CACHE_DIR / f"ordinary_snapshot_2025_burst_{SIMS}_{burst_seed}.npy"
    if burst_path.exists():
        conditional = np.load(burst_path)
    else:
        formats = np.array([rules(r.tournament, int(r.year), int(r.best_of)) for r in rows.itertuples()])
        print("SIMULATE ordinary snapshot matches", len(rows), "S", SIMS,
              "seed", burst_seed, flush=True)
        conditional = posterior_burst(pa, pb, rows.best_of.to_numpy(), formats[:, 0], formats[:, 1],
                                      SIMS, burst_seed + SNAPSHOT_PERIOD * 1_000_003)
        np.save(burst_path, conditional)
        trials.log(stage="independent-ordinary-2025", kind="conditional-burst-simulation",
                   params={"snapshot_period": SNAPSHOT_PERIOD, "S": SIMS, "seed": burst_seed,
                           "severity": 1.5, "duration": 40, "start_uniform": [0, 80]})
    if len(conditional) != len(rows):
        raise ValueError("2025 snapshot burst cache length mismatch")
    rows["conditional_burst"] = conditional
    return rows


def _paired_gain(candidate: np.ndarray, reference: np.ndarray, rows: pd.DataFrame) -> dict:
    values = np.log(np.clip(candidate, 1e-10, 1)) - np.log(np.clip(reference, 1e-10, 1))
    return {"tournament": ci(values, rows),
            "player_tournament": ci(values, rows, "player_tournament")}


def _diagnostics(periods: list[int]) -> dict:
    items = []
    for period in periods:
        fit = json.loads((FIT_DIR / f"ordinary_period_{period}.json").read_text())
        items.append({"period": period, "n_train": fit["n_train"], "seconds": fit["seconds"],
                      **fit["diagnostics"]})
    return {"max_rhat": max(x["max_rhat"] for x in items),
            "min_ess_bulk": min(x["min_ess_bulk"] for x in items),
            "divergences": sum(x["divergences"] for x in items),
            "periods_rhat_over_1p01": [x["period"] for x in items if x["max_rhat"] > 1.01],
            "periods_rhat_over_1p05": [x["period"] for x in items if x["max_rhat"] > 1.05],
            "periods": items}


def test_2025() -> dict:
    frozen_path = OUT / "FROZEN_BEFORE_2025.json"
    if not frozen_path.exists():
        raise RuntimeError("run --stage tune and freeze settings before reading 2025")
    frozen = json.loads(frozen_path.read_text())
    if frozen["protocol_sha256"] != hashlib.sha256((OUT / "PROTOCOL.md").read_bytes()).hexdigest():
        raise RuntimeError("protocol changed after settings were frozen")
    if frozen["code_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("evaluation code changed after settings were frozen")

    data, audit = load_extended(2025)
    write_json(OUT / "test_data_audit.json", audit)
    train_data = data[data.source_year <= 2024].copy()
    test_rows = data[data.source_year == 2025].copy()
    rows = snapshot_predictions(train_data, test_rows, SEED)
    rate = frozen["burst"]["chosen_rate"]
    temperature = frozen["temperature"]["chosen"]
    predictions = {
        "ordinary_iid": rows.iid.to_numpy(),
        "ordinary_burst": ((1 - rate) * rows.iid.to_numpy() +
                           rate * rows.conditional_burst.to_numpy()),
        "ordinary_temperature": expit(logit(np.clip(rows.iid.to_numpy(), 1e-8, 1 - 1e-8)) /
                                      temperature),
    }
    metrics = {name: score(probability, rows) for name, probability in predictions.items()}
    comparisons = {
        "ordinary_burst_vs_ordinary_iid": _paired_gain(predictions["ordinary_burst"],
                                                        predictions["ordinary_iid"], rows),
        "ordinary_burst_vs_ordinary_temperature": _paired_gain(predictions["ordinary_burst"],
                                                                predictions["ordinary_temperature"], rows),
        "ordinary_temperature_vs_ordinary_iid": _paired_gain(predictions["ordinary_temperature"],
                                                              predictions["ordinary_iid"], rows),
    }
    labels = strata(rows)
    stratified = {}
    for name, probability in predictions.items():
        stratified[name] = {}
        for label in sorted(set(labels)):
            mask = labels == label
            stratified[name][label] = score(probability[mask], rows[mask])

    replicate = snapshot_predictions(train_data, test_rows, REPLICATE_SEED)
    replicate_probability = ((1 - rate) * replicate.iid.to_numpy() +
                             rate * replicate.conditional_burst.to_numpy())
    simulation_check = {
        "primary_seed": SEED, "replicate_seed": REPLICATE_SEED,
        "primary_log_loss": metrics["ordinary_burst"]["log_loss"]["tournament"]["mean"],
        "replicate_log_loss": float(-np.log(np.clip(replicate_probability, 1e-10, 1)).mean()),
        "mean_absolute_probability_difference":
            float(np.abs(predictions["ordinary_burst"] - replicate_probability).mean()),
    }
    for name, probability in predictions.items():
        rows[name] = probability
    rows["ordinary_burst_seed2"] = replicate_probability
    rows["stratum"] = labels
    rows.to_csv(OUT / "predictions_2025.csv", index=False)
    periods = [SNAPSHOT_PERIOD]
    result = {"utc": utcnow(), "n": len(rows), "source_year": 2025,
              "frozen": frozen, "metrics": metrics, "comparisons": comparisons,
              "stratified_metrics": stratified, "simulation_check": simulation_check,
              "fit_diagnostics": _diagnostics(periods),
              "note": "Positive paired log-loss gain favors the first named method."}
    write_json(OUT / "results.json", result)
    trials.log(stage="independent-ordinary-2025", kind="score_2025_frozen",
               params={"burst_rate": rate, "temperature": temperature,
                       "simulations": SIMS, "seed": SEED},
               metrics={name: value["log_loss"]["tournament"]["mean"]
                        for name, value in metrics.items()})
    write_report(result)
    return result


def write_report(result: dict) -> None:
    lines = ["# Full ordinary model with burst forecasts: 2025 holdout", "",
             f"Generated {result['utc']}; {result['n']} eligible matches from the 2025 ATP source season.", "",
             f"Burst rate {result['frozen']['burst']['chosen_rate']} and temperature "
             f"{result['frozen']['temperature']['chosen']} were selected on 2013 for the prior independent "
             "reproduction and carried forward before 2025 was read.", "",
             "| Forecast | Log loss (tournament 95% CI) | Accuracy | Brier | Calibration slope | ECE-10 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in result["metrics"].items():
        ll = metrics["log_loss"]["tournament"]
        acc = metrics["accuracy"]["tournament"]["mean"]
        brier = metrics["brier"]["tournament"]["mean"]
        cal = metrics["calibration"]
        lines.append(f"| {name} | {ll['mean']:.5f} [{ll['lo']:.5f}, {ll['hi']:.5f}] | "
                     f"{acc:.4f} | {brier:.5f} | {cal['slope']:.3f} | {cal['ece_10']:.4f} |")
    lines += ["", "Paired log-loss gains (positive favors the first method):", ""]
    for name, comparison in result["comparisons"].items():
        t = comparison["tournament"]
        p = comparison["player_tournament"]
        lines.append(f"- {name}: {t['mean']:+.5f}; tournament CI [{t['lo']:+.5f}, {t['hi']:+.5f}]; "
                     f"player-tournament CI [{p['lo']:+.5f}, {p['hi']:+.5f}].")
    diagnostics = result["fit_diagnostics"]
    check = result["simulation_check"]
    lines += ["", "## Diagnostics", "",
              f"For the frozen posterior fit: max R-hat "
              f"{diagnostics['max_rhat']:.3f}, minimum bulk ESS {diagnostics['min_ess_bulk']:.0f}, "
              f"and {diagnostics['divergences']} divergences. Periods above R-hat 1.01: "
              f"{diagnostics['periods_rhat_over_1p01']}.",
              f"The second simulation seed changed mean absolute match probability by "
              f"{check['mean_absolute_probability_difference']:.5f} and log loss from "
              f"{check['primary_log_loss']:.5f} to {check['replicate_log_loss']:.5f}.", "",
              "The absolute tournament-bootstrap intervals describe variability across events in one season. "
              "The paired intervals are the relevant uncertainty for method comparisons.", "",
              "See `PROTOCOL.md`, `FROZEN_BEFORE_2025.json`, `results.json`, and "
              "`predictions_2025.csv` for the complete record."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["audit", "prepare", "test"], required=True)
    args = parser.parse_args()
    sampler_progress()
    if args.stage == "audit":
        data, audit = load_extended(2024)
        print(json.dumps(audit, indent=2))
        print("2024 periods", sorted(data.loc[data.source_year == 2024, "period"].unique().tolist()))
    elif args.stage == "prepare":
        prepare_2025()
    else:
        test_2025()
