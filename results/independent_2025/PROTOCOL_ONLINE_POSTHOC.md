# Exploratory protocol: online 2025 ordinary full model

Registered after the frozen one-shot 2025 holdout was scored. This analysis is
therefore exploratory. It answers whether updating abilities through 2025
improves the already evaluated frozen model; it is not a fresh confirmatory
use of the 2025 test season.

## Fixed choices

- Keep the exact cohort, player IDs, hierarchy, priors, historical scoring
  rules, posterior settings, and source hashes from `PROTOCOL.md`.
- Keep burst severity 1.5 logits, duration 40 points, onset uniform from point
  0 through 80, 16,384 simulations, and mixture rate 0.5.
- Keep temperature 1.5.
- Do not tune anything on 2025.

## Online update rule

- Reuse the start-of-2025 period-84 posterior for the 2025-source matches dated
  from 2024-12-27 through 2025-02-28.
- Refit the full ordinary model at the start of periods 85 through 89: March,
  May, July, September, and November 2025.
- Each fit uses all eligible matches whose two-month period is strictly earlier
  than the forecast period. Thus a forecast never uses its own match, a later
  match, or another outcome from its forecast block.
- Each refit uses four NUTS chains, 1,000 warmup draws, 1,000 retained draws,
  `target_accept=0.9`, and the same seeds as the frozen fit.

## Evaluation

- Score online iid, online burst, and online temperature forecasts on the same
  2,670-match source-season cohort as the frozen analysis.
- Primary exploratory comparison: online burst versus frozen burst, using
  paired log-loss gains clustered by tournament and by player-tournament.
- Also report online iid versus frozen iid, online burst versus online iid, and
  online burst versus online temperature.
- Report accuracy, Brier score, calibration, monthly log loss, tournament
  appearance strata, posterior diagnostics, and a second burst-simulation seed.
- Positive paired gain favors the first named method.

All artifacts use distinct `online_` names. The sealed frozen predictions and
results remain unchanged.

## Implementation correction before scoring

All five posterior fits and primary forecast simulations completed, but the
first scoring attempt stopped at its cohort assertion. The match-ID sets and
period counts were identical; concatenating the sealed early CSV rows (date
strings) with new rows (timestamps) had produced a nonchronological mixed-type
sort. Dates are now explicitly parsed before sorting. The failed provenance
record is retained as `ONLINE_FROZEN_FAILED_ORDERING.json`; no probability,
model, hyperparameter, cohort member, or outcome changed.
