# Active results

| Directory | Status | Entry point |
|---|---|---|
| [independent](independent/README.md) | Full 2014 Ingram reproduction and development settings | `src.independent.run` |
| [independent_2025](independent_2025/README.md) | Frozen and online ATP 2025 forecasts; mechanism and seed checks | `src.independent.run_2025_online`, `src.independent.mechanism_splits`, `src.independent.mcmc_seed_replicate` |
| [point_dependence](point_dependence/README.md) | Direct Slam-point serial association and per-set dispersion | `src.independent.point_dependence` |
| [point_structure_followup](../analysis/point_structure_followup/README.md) | Set-centered association and scoring-only iid control | isolated analysis scripts |
| [wta_2025](wta_2025/README.md) | WTA 2025 constrained protocol correction complete | `src.independent.run_wta_2025_corrected` |

The ATP 2025 outcome set is already open; its online and subgroup analyses are
exploratory. The WTA protocol was written before loading its 2025 source file,
but the first completed implementation failed to exclude United Cup. The
constrained correction was frozen after that error was found and is labeled
accordingly. Burst improves iid by 0.02646, while its 0.00128 point advantage
over temperature has paired intervals spanning zero.
Use paired tournament and player-tournament intervals for model comparisons.
Absolute single-season error bars answer a different question.

The point-structure follow-up qualifies the raw lag-one result: most of the
men's estimate disappears after set centering, its remaining interval spans
zero, and an iid tennis-scoring simulation creates an apparent across-game
reversal. The set-dispersion result remains descriptive evidence of
heterogeneity, not evidence for the fitted burst mechanism.

`trials.jsonl` is append-only and includes both active and archived work.
`processed_build.json` and `splits.json` record the point-data build. Large
posterior arrays and raw data remain ignored locally.

Superseded robust, synthetic, approximate-filter, contamination, and regime
artifacts are indexed in
[`archive/robust_and_contamination/README.md`](../archive/robust_and_contamination/README.md).

## License

Result tables here contain match-level rows derived from tennis data compiled
by [Jeff Sackmann](https://github.com/JeffSackmann) and are licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); see
[`ATTRIBUTION.md`](../ATTRIBUTION.md). Per-match predictions for the 2013–2014
Ingram author fixture are kept local because that fixture carries no license.
