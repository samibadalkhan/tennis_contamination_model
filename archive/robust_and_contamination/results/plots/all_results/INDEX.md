# Consolidated result figures

Generated from saved development artifacts; no model refits and no 2025 outcomes.

The later frozen 2025 ordinary-model holdout is reported separately because it
was run after these development-only figures were generated:

- [`independent_2025/README.md`](../../independent_2025/README.md)
- [`independent_2025/ordinary_2025_holdout.png`](../../independent_2025/ordinary_2025_holdout.png)
- [`independent_2025/ONLINE_README.md`](../../independent_2025/ONLINE_README.md)
- [`independent_2025/online_2025_updates.png`](../../independent_2025/online_2025_updates.png)

## Current evidence

![Executive summary](01_executive_summary.png)

![Full-model scores and calibration](02_full_model_scores_and_calibration.png)

![Full-model paired gains](03_full_model_paired_gains.png)

![Tuning and simulation](04_tuning_and_simulation.png)

![Sampling diagnostics](05_sampling_diagnostics.png)

![Early real point and match results](06_early_real_point_and_match.png)

![Synthetic point-pilot recovery](07_point_pilot_recovery.png)

![Synthetic point filtering](08_point_pilot_filtering.png)

## Provenance and failed checks

The older simplified online-filter results are retained for provenance and are
superseded by the full Bayesian comparison.

![Legacy simplified pipeline](09_legacy_simplified_pipeline.png)

The separate two-seed Bayesian synthetic recovery run failed its sampler
diagnostics. Its RMSE differences are deliberately not plotted as evidence.

![Invalid synthetic diagnostics](10_invalid_synthetic_diagnostics.png)

![Early real tuning and events](11_early_real_tuning_and_events.png)

![Synthetic permanent-change adaptation](12_point_pilot_adaptation.png)

Existing detailed figures remain available:

- [`point_robust_pilot/trajectories.png`](../../point_robust_pilot/trajectories.png)
- [`point_robust_pilot/block_weights.png`](../../point_robust_pilot/block_weights.png)
- [`early_real_2013/point_gains.png`](../../early_real_2013/point_gains.png)
- [`plots/skill_trajectories.png`](../skill_trajectories.png)
- [`plots/alpha_by_surface.png`](../alpha_by_surface.png)
- [`plots/tuning_grids.png`](../tuning_grids.png)
- [`plots/season_scores.png`](../season_scores.png)

Primary numerical sources: [`independent/comparison.json`](../../independent/comparison.json),
[`early_real_2013/results.json`](../../early_real_2013/results.json), and
[`point_robust_pilot/results.json`](../../point_robust_pilot/results.json).
