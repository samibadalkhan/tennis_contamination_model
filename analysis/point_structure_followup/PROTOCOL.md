# Direct point structure: isolated exploratory follow-up

Frozen before executing this follow-up on 2026-09-23. Existing ATP forecasts
and the direct-dependence summary are already known. This is a descriptive
analysis within the current ordinary-model/point-dependence scope. It does not
reopen the archived robust estimators or burst-regime grid.

## Questions and fixed measurements

Measure residual association at lags **1, 2, 4, 8, 16 service points**. A lag
counts a player's own observed service opportunities, not consecutive match
points or elapsed time. Compare six fixed views:

1. Match-server centering, with pairs allowed across sets.
2. Match-server centering restricted to eligible same-set pairs.
3. Set-server centering on exactly those same pairs.
4. Set-server centering restricted to pairs within one game.
5. Set-server centering restricted to pairs in different games of the same set.
6. Set-server centering excluding tiebreak endpoints.

Views 2 and 3 must have identical eligible pair counts. Their paired difference
shows how the measured statistic changes after accounting for set-level rate
variation; it is not a fraction of variance explained or a causal mediation
estimate. Views 4 and 5 partition view 3. Report effects separately for men and
women, including all five lags and all six views.

## Cohort and ordering

Read only `data/processed/points.parquet`, with its existing build provenance.
Verify the relevant raw Slam point and match file hashes against the manifest,
and record hashes of the processed table and build manifest. This does not
rebuild the processed table or prove every transformation is correct.

Use years through 2024; never open ATP/WTA 2025 outcomes. Keep unambiguous
identities, known tours, binary outcomes, positive set/game numbers and valid
point numbers. Verify that original point order strictly increases per match.
Split a match into contiguous observed segments wherever point numbers skip;
never form a pair across such a gap. Require two consistent player identities
per match. At least 20 service points and nonconstant outcomes are required
per match-server. For same-set views, require at least 12 service points and
nonconstant outcomes in that set-server. Log all exclusions.

## Statistic and uncertainty

For a centering group of size n, success fraction p, and an eligible pair (i,j),
the numerator contribution is

    (x_i - p)(x_j - p) + p(1-p)/(n-1).

The denominator contribution is p(1-p). Sum numerators and denominators before
taking their ratio. The added term corrects the negative covariance from
conditioning on the group's observed total successes under unrestricted random
permutation. This is a variance-weighted association statistic, not a fitted
autoregressive coefficient.

Report 1,000 bootstrap replicates (seed 73019), resampling whole matches,
whole tour/year/Slam events, and using product Poisson weights for both
canonical-player/event memberships. Preserve the same replicate weights for
paired changes between views. The player/event analysis uses the processed
canonical identities, not separately verified ATP/WTA IDs. Cross-event player
dependence can remain. All 95% intervals are exploratory and unadjusted for
multiple comparisons. Do not choose a preferred lag using the results.

## Interpretation limits

**Unrestricted permutation is not a scoring-aware tennis null.** Game/set/match
termination and selected pair categories depend on outcomes. Even iid tennis
points can therefore give nonzero values in these views. Bootstrap intervals
describe variability across the observed clusters, not a valid causal test of
injury, fatigue or momentum. Set means are estimated using the entire set,
so this is a descriptive decomposition, not a prospective prediction score.

Signals mainly absorbed by set centering suggest that slow changes or
set-level selection deserve further investigation. Signals localized within
games are especially vulnerable to scoring constraints. Signals persisting
between games after set centering motivate a later scoring-aware control,
but do not establish a physical mechanism. No physiological measurements are
available in this analysis; do not diagnose players or infer burst frequency.

## Isolation and resource budget

All new code, tests, manifests, reports and a local append-only trial ledger
live in `analysis/point_structure_followup/`. Do not edit active source,
protocols, fit caches, reports, the Git index, or the shared trial log. The local
ledger records this diagnostic without racing the live sampler's logger.

Run single-threaded, at lower process priority where supported, using cached
processed data; no posterior fitting, match simulations or parameter tuning.
Save source/input hashes before analysis, then verify the protected active
source and sealed inputs afterward. Existing live runs may append new results;
that external activity is not modified by this follow-up.
