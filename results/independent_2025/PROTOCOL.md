# Frozen protocol: ordinary full model with burst forecasts on 2025

Registered 2026-09-22 before reading or scoring the 2025 ATP outcome rows for
this experiment. The user requested this ordinary-only follow-up after the
robust estimators failed to show a useful real-data advantage.

Amended before any 2025 access: an attempted 2024 tuning fit was stopped during
warmup after direct timing showed that six tuning fits plus seven test fits
would require roughly half a day. No posterior from that interrupted fit is
valid or used. The amended design uses one full posterior trained through 2024
and carries forward the already frozen forecast settings from the independent
2013-to-2014 reproduction. This is a one-shot season holdout.

## Question

Does the full ordinary Ingram model, combined with the previously specified
non-iid burst forecast, achieve good held-out log loss on the 2025 ATP season?
The comparison against an iid forecast asks whether the burst forecast adds
predictive value. A temperature control tests whether any gain is simply
probability shrinkage.

## Data and cohort

- Source: hashed local `atp_matches_2011.csv` through
  `atp_matches_2025.csv`; every file must match `data/MANIFEST.json`.
- ATP men's singles at levels Grand Slam, Masters, tour-level A, and Tour
  Finals (`G`, `M`, `A`, `F`). Davis Cup and lower levels are excluded.
- A match is usable when date, surface, best-of, both ATP player IDs, and both
  players' serve totals and first/second-serve points won are present and
  internally valid. Retirements remain when those fields are valid. Walkovers
  without played serve points cannot contribute.
- Player identity is the ATP player ID. Tournament effects use tournament
  names, matching the published model's construction.
- Training cohort: all eligible rows in the 2011 through 2024 source-season
  files. Test cohort: rows in the 2025 source-season file. This keeps
  late-December tour events in their official source season.

## Ability model and chronology

- Ordinary likelihood only: the full published hierarchy with serve and
  return Gaussian random walks, player-by-surface effects, tournament effects,
  and the published half-normal/normal priors as implemented and independently
  checked in `src/independent/bayes.py`.
- The history begins in 2011. Time is divided into the same two-month periods
  as the successful 2014 reproduction.
- One posterior is fit at the start-of-2025 cutoff, using all eligible rows in
  the 2011 through 2024 source files and no 2025-source rows. Its end-of-2024
  player states are propagated one two-month step, as in the verified full
  model, and held fixed for the 2025-source-season forecasts.
- Four NUTS chains, 1,000 warmup and 1,000 retained draws per chain,
  `target_accept=0.9`. Report divergences, maximum R-hat, and minimum bulk ESS.
- No updating occurs during the test season. This is a stricter, harder
  one-shot holdout than the verified bimonthly reproduction and its forecasts
  will become stale later in 2025.

## Forecasts frozen before test

All methods use historical match scoring rules and posterior averaging.

1. **Ordinary iid**: exact iid tennis recursion.
2. **Ordinary burst**: the existing coherent point simulator. Conditional on a
   burst, one randomly chosen player is impaired by 1.5 logit units for 40
   points, starting uniformly at match point 0 through 80. Match result and
   scoring path share the same simulation. Use 16,384 simulations per match.
   Mix the conditional-burst probability with the iid probability at rate
   **0.5**, the value selected on 2013 and frozen before the successful 2014
   reproduction. No new tuning is performed.
3. **Ordinary temperature**: transform the iid match probability by
   `logit(p) / T`, with **T = 1.5**, selected on 2013 and frozen before the
   successful 2014 reproduction. No new tuning is performed.

Simulation seed 4001 is primary. Seed 9029 is a Monte Carlo sensitivity check
using the already selected burst rate; it cannot alter tuning.

## Outcomes and reporting

- Primary: 2025 mean match log loss for ordinary burst.
- Required comparisons: paired log-loss gain of burst versus iid and burst
  versus temperature. Positive gain favors burst.
- Secondary: accuracy, Brier score, calibration slope, and ECE-10.
- Report first appearances and later appearances within tournaments, while
  noting that ability is fixed within each two-month block.
- Give paired 95% bootstrap intervals by tournament and by player-tournament.
  With one season, these express event-to-event sampling variability and may be
  wide. Absolute-score intervals are secondary and will not be used to select
  a method.
- Store match-level predictions, machine-readable results, a readable report,
  fit diagnostics, source/code hashes, and append every tuning/evaluation
  variant to `results/trials.jsonl`.

No burst mechanism, carried forecast setting, cohort rule, model prior, or
scoring rule may change after the 2025 rows are loaded. Any later variant is
exploratory and must use a new result path.
