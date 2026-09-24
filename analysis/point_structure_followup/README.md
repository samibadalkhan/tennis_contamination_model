# Physical interpretation follow-up: completed

Two inexpensive, isolated diagnostics were completed on September 23, 2026.
**The results do not establish temporary physical impairment as the reason
burst forecasts helped.** They narrow down which apparent signals need better
controls before receiving a physical interpretation.

## Observed point structure

The [descriptive analysis](report.md) used **1,867,212 recorded Slam points**,
examining five service-point lags and six fixed views. Its eligible baseline
lag-one pairs come from 5,271 men's and 4,941 women's matches. The retained
table also includes 26 matches with no eligible baseline pair.

For men, on exactly the same within-set point pairs, adjusted lag-one
association decreases from **0.00648 to 0.00264** when centering outcomes on
the set's serve rate rather than the match's serve rate. The paired change is
**0.00384 [0.00296, 0.00467]** under the canonical-player/event bootstrap.
The remaining association has interval **[−0.00062, +0.00590]**. Women's
remaining association is **−0.00118 [−0.00543, +0.00285]**.

This motivates studying set-level changes and selection. It does not identify
what caused them: the set averages use the whole set, and tennis scoring
constrains which points and pairs are observed. These are descriptive statistics,
not prospective forecasting gains or estimates of impairment duration.

## Scoring alone creates a misleading pattern

The [scoring-only control](SCORING_CONTROL.md) generated **2,048 matches** in
four illustrative, fixed iid matchups. Player abilities never change, and no
burst, fatigue, injury, or momentum mechanism is present.

Despite that, all four cells produce negative across-game lag-one association:
point estimates range from **−0.0301 to −0.0171**, and each Monte Carlo interval
excludes zero. The corresponding real-data estimates are −0.0271 for men and
−0.0447 for women. Thus an apparent between-game reversal can be manufactured
by scoring and pair selection alone.

These illustrative simulations are not matched to the real cohort, coverage,
or historical scoring rules. Their numerical similarity cannot establish how
much real dependence is an artifact, and their estimates must not be subtracted
from the real statistics as a physical-effect correction. Positive within-game
estimates in the control are imprecise; their intervals include zero.

## What this changes

The direct-dependence statistics should remain descriptive. The unrestricted
permutation reference does not by itself establish a physical non-iid mechanism
under tennis stopping rules. A defensible next mechanism study would first use
a cohort-matched, historically scored, coverage-aware iid control, then ask
whether residual temporal structure improves prediction over calibration and
ordinary matchup uncertainty. This follow-up did not retune or reopen 2025,
restart the archived regime grid, or launch additional Bayesian fits.

## Verification and isolation

- Ten semantic tests passed, including exact finite-population permutation
  expectations, gap handling, matched pair counts, game partitions and scoring.
- Saved associations were independently recomputed from the saved component
  table. Simulated win fractions agree with the analytic iid recursion within
  three Monte Carlo standard errors in all four cells.
- Ninety-eight raw source files passed their manifest hashes. The processed
  table and build provenance were hashed, without rebuilding them.
- Protected scientific source, frozen protocols, and input hashes remained
  unchanged. The repository-scope note `CLAUDE.md` was rewritten after this
  analysis was frozen, so its historical hash in `FROZEN.json` no longer
  matches; all other 35 protected path entries match. No shared ledger writes,
  model changes, sampler restarts or Git staging occurred during the analysis.
- All new artifacts and the local append-only trial log stay in this directory.
  Both diagnostics used one numerical thread at reduced process priority.

See [the descriptive protocol](PROTOCOL.md), [the control protocol](SCORING_CONTROL.md),
`FROZEN.json`, `SCORING_FROZEN.json`, `results.json` and `scoring_control.json`.
The saved aggregate components derive from Jeff Sackmann's data and retain
the attribution and CC BY-NC-SA terms documented in [ATTRIBUTION.md](../../ATTRIBUTION.md).
