# Tennis point dependence and match forecasting

This repository asks whether departures from independent tennis points improve
match forecasts from a full Ingram-style Bayesian ability model. The active
work is deliberately narrow: reproduce the published baseline, evaluate the
ordinary model on ATP 2025, compare a fixed burst forecast with temperature
calibration, replicate the MCMC seed, test WTA 2025 under a frozen protocol,
audit its team-event correction, and measure within-match point dependence
directly.

The current ATP result is clear. Online ability updates improve the frozen
forecast, and both burst and temperature calibration improve the raw iid
forecast. Burst does not improve on temperature overall, in any edge band, or
in best-of-five matches. The Slam points show about 19% excess set-level
dispersion. An initial small lag-one association shrinks substantially after
set centering, and its remaining interval spans zero; a scoring-only iid
control also reproduces an apparent across-game reversal. The point data show
extra heterogeneity, but they do not identify a physical burst mechanism. The
corrected WTA run confirms that burst improves raw iid probabilities, while
again failing to distinguish burst from simple temperature calibration.

- [Current status](STATUS.md)
- [Results index](results/README.md)
- [Important plots](plots/README.md)
- [Matched Ingram comparison](docs/MATCHED_INGRAM_COMPARISON.md)
- [Point-structure follow-up](analysis/point_structure_followup/README.md)
- [Data pipeline](docs/DATA_PIPELINE.md)
- [Archived robust and contamination work](archive/robust_and_contamination/README.md)

## Active layout

| Path | Purpose |
|---|---|
| `src/independent/bayes.py` | Full bimonthly Bayesian serve/return hierarchy |
| `src/independent/scoring.py` | Exact iid recursion and coherent burst simulator |
| `src/independent/run.py` | 2014 author-fixture reproduction and development comparison |
| `src/independent/run_2025.py` | Frozen ATP 2025 evaluation |
| `src/independent/run_2025_online.py` | Exploratory bimonthly ATP 2025 updates |
| `src/independent/mcmc_seed_replicate.py` | Independent posterior-seed rerun |
| `src/independent/run_wta_2025_corrected.py` | Protocol correction enforcing the team-event exclusion |
| `src/independent/mechanism_splits.py` | Format, edge, and per-format-temperature comparisons |
| `src/independent/point_dependence.py` | Direct serial and between-set dependence measurements |
| `results/independent_2025/` | ATP forecasts, protocols, diagnostics, and mechanism splits |
| `results/wta_2025/` | WTA protocol, correction audit, and final results |
| `results/point_dependence/` | Direct dependence estimates and plot |
| `analysis/point_structure_followup/` | Set-centering and scoring-only controls |
| `archive/` | Superseded robust, contamination, synthetic, and regime work |

The superseded `src/independent/run_wta_2025.py` and
`results/independent_2025/ONLINE_FROZEN_FAILED_ORDERING.json` remain at their
original paths only because the final point-structure freeze records their
hashes. They are provenance inputs, not supported experiment entry points.
The noncompliant WTA outputs and cache are indexed in the
[final cleanup archive](archive/final_experiment_cleanup_2026-09-23/README.md).

## Environment and checks

Run from the repository root:

```bash
python3.13 -m venv .venv-independent
.venv-independent/bin/python -m pip install -r requirements-independent.txt
python -m pytest -q
```

The Bayesian reruns are expensive. Existing fit metadata and prediction caches
are reused only when their frozen configuration and cohort fingerprints match.

## Data and provenance

```bash
python -m src.fetch --list
python -m src.fetch --verify
python -m src.build_processed
.venv-independent/bin/python -m src.independent.fetch_reference
```

Source pins, hashes, attribution, and licensing are recorded in
[`ATTRIBUTION.md`](ATTRIBUTION.md) and `data/MANIFEST.json`. Raw data, derived
caches, and large posterior arrays remain local and ignored. The 2025 ATP
season has already been opened, so its online and split analyses are
exploratory. The WTA protocol was frozen before its 2025 file was read; a
subsequent audit found that the first implementation omitted the written
team-event exclusion, and the constrained correction is recorded separately.
The complete raw source bundle remains under `data/` because the frozen
manifest and verification command cover it as a unit, even where a particular
final model uses only a subset of years or sources.

`results/trials.jsonl` is the append-only historical run ledger. It still
contains archived experiments so the research record remains complete.

## License

Code is released under the [MIT License](LICENSE). Tennis data and the
match-level result tables derived from it are
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/),
compiled by Jeff Sackmann; see [`ATTRIBUTION.md`](ATTRIBUTION.md).
