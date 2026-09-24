"""Read saved online-2025 progress without importing or running the model.

Uses only the standard library. Cache presence is checkpoint evidence, not
proof that a process is alive or a percentage of sampler computation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def snapshot(root: Path = ROOT) -> dict:
    out = root / "results/independent_2025"
    cache = root / "data/independent_2025_cache"
    warnings = []

    def read_json(path):
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            warnings.append(f"Cannot read {path.relative_to(root)}: {exc}")
            return None

    def present(path):
        return path.is_file() and path.stat().st_size > 0

    seal = read_json(out / "ONLINE_FROZEN.json")
    checks = {}
    if seal:
        targets = {
            "protocol_sha256": out / "PROTOCOL_ONLINE_POSTHOC.md",
            "source_sha256": root / "src/independent/run_2025_online.py",
            "frozen_settings_sha256": out / "FROZEN_BEFORE_2025.json",
            "frozen_results_sha256": out / "results.json",
            "frozen_predictions_sha256": out / "predictions_2025.csv",
        }
        for key, path in targets.items():
            checks[key] = (path.is_file() and
                           hashlib.sha256(path.read_bytes()).hexdigest() == seal.get(key))
            if not checks[key]:
                warnings.append(f"Online seal mismatch or missing file: {path.relative_to(root)}")

    counts = Counter()
    predictions = out / "predictions_2025.csv"
    if predictions.exists():
        try:
            with predictions.open(newline="") as handle:
                counts.update(int(row["period"]) for row in csv.DictReader(handle))
        except (OSError, ValueError, KeyError) as exc:
            counts.clear()
            warnings.append(f"Cannot count sealed forecast cohort: {exc}")

    periods = seal["periods"] if seal else []
    fits = []
    for period in ([84] + periods if seal else []):
        stem = out / "fits" / f"ordinary_period_{period}"
        meta = read_json(stem.with_suffix(".json"))
        complete = meta is not None and present(stem.with_suffix(".npz"))
        reused = period == 84
        n = sum(v for k, v in counts.items() if k <= 84) if reused else counts[period]
        components = complete and (bool(counts) if reused else
                                  present(cache / f"ordinary_period_{period}.npz"))
        row = {"period": period, "reused": reused, "matches": n,
               "fit_saved": complete,
               "primary_saved": components and (reused or present(
                   cache / f"ordinary_period_{period}_burst_16384_4001.npy")),
               "replicate_saved": components and (reused or present(
                   cache / f"ordinary_period_{period}_burst_16384_9029.npy"))}
        if meta:
            row.update(utc=meta["utc"], seconds=meta["seconds"], n_train=meta["n_train"],
                       diagnostics={k: meta["diagnostics"][k] for k in
                                    ["max_rhat", "min_ess_bulk", "divergences"]})
        fits.append(row)

    result = read_json(out / "online_results.json")
    final_files = {name: present(out / name) for name in
                   ["online_results.json", "online_predictions_2025.csv", "online_report.md"]}
    state = ("complete" if result and all(final_files.values()) else
             "finalizing" if result else "incomplete" if seal else "not_started")
    if warnings:
        state = "needs_review"
    return {"observed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "state": state, "process_status": "not_checked",
            "protocol_status": seal.get("status") if seal else None,
            "started_utc": seal.get("utc") if seal else None,
            "online_seal_checks": checks, "fits": fits,
            "new_refits_saved": sum(x["fit_saved"] for x in fits if not x["reused"]),
            "new_refits_planned": len(periods), "cohort_matches": sum(counts.values()),
            "primary_matches_saved": sum(x["matches"] for x in fits if x["primary_saved"]),
            "final_files": final_files, "warnings": warnings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print a JSON snapshot")
    args = parser.parse_args()
    report = snapshot()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Online 2025: {report['state']} as of {report['observed_utc']}")
        print("Read-only checkpoints; process liveness and in-flight draws are not checked.")
        print(f"New refits saved: {report['new_refits_saved']}/{report['new_refits_planned']}")
        print(f"Primary forecast components: {report['primary_matches_saved']}/"
              f"{report['cohort_matches']} matches (coverage, not compute completion)")
        for row in report["fits"]:
            label = "reused" if row["reused"] else "saved" if row["fit_saved"] else "pending"
            print(f"  Period {row['period']}: {label}; {row['matches']} matches; "
                  f"primary={'yes' if row['primary_saved'] else 'no'}, "
                  f"second seed={'yes' if row['replicate_saved'] else 'no'}")
            if "diagnostics" in row:
                d = row["diagnostics"]
                print(f"    {row['seconds'] / 60:.1f} min; R-hat {d['max_rhat']:.4f}; "
                      f"min ESS {d['min_ess_bulk']:.1f}; divergences {d['divergences']}")
        if report["online_seal_checks"]:
            print("Online sealed inputs: " + ("all match" if all(
                report["online_seal_checks"].values()) else "MISMATCH — see warnings"))
        print("R-hat above 1.01 warrants review; no new season score is inferred from partial fits.")
        print("Final artifacts: " + ", ".join(
            f"{name}={'saved' if exists else 'pending'}"
            for name, exists in report["final_files"].items()))
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
    return 1 if report["warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
