# WTA 2025 protocol deviation: team-event filter

Recorded 2026-09-23 after the initial result was produced and before the
corrected implementation was run.

The frozen protocol excludes team events. The initial implementation filtered
the permitted level labels but did not separately filter tournament names.
Jeff Sackmann's WTA files encode United Cup as level `I`, so this admitted 48
eligible United Cup matches from 2023 and 25 from 2024 into model training, and
14 eligible United Cup matches from the 2025 source into the test set.

The initial 2,499-match result is therefore an implementation-deviating run and
must not be presented as the protocol-compliant confirmatory result. Its files
are retained as an audit trail under
[`archive/final_experiment_cleanup_2026-09-23/results/wta_2025/initial_noncompliant/`](../../archive/final_experiment_cleanup_2026-09-23/results/wta_2025/initial_noncompliant/README.md).

The corrective implementation makes one cohort change: after applying the
frozen level rule, exclude tournament names matching `United Cup`, `Fed Cup`,
`Billie Jean King Cup`, or `Hopman Cup`, case-insensitively. In the included
2011--2025 level strata, only United Cup matches are affected. No model,
forecast, seed, simulation, metric, interval, or other cohort rule changes.
All seven bimonthly posteriors will be refitted because the 2023--2024 training
rows change.

Because the 2025 outcomes and initial result were already open when this bug
was discovered, the corrected run is labeled a constrained protocol
correction rather than a clean preregistered replication. The correction is
determined by the written frozen exclusion and does not use outcomes to choose
which rows to remove.
