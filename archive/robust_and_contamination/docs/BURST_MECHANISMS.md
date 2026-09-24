# Exploratory physical interpretation of burst forecasts

Specified 2026-09-22, after the existing development and frozen-2025 results
were known, at the user's request. This is an exploratory follow-up, not a new
confirmatory test or passage of the original synthetic gate. Execution waits
for the current online-2025 run to finish. Its scores do not choose any settings
below. No new 2025 raw data or posterior fits are needed.

## Questions

1. Does temporary, persistent impairment help predict observed next points
   beyond probability calibration and a match-long ability mismatch?
2. Do earlier or later onset, shorter or longer duration, and mild or severe
   impairment produce distinguishable development predictions?
3. Are predictive gains concentrated late in matches or in matches ending in
   retirement? These are descriptive signatures, not injury diagnoses.

Match-outcome improvement alone cannot distinguish these mechanisms. This
batch therefore uses ordered point sequences and predict-before-observe point
log loss. It does not run another match-forecast parameter search on 2025.

## Data and fixed ability backbone

Reuse the audited men's Slam cohort in `src.experiments.early_real`: 2011–2013
only, stable ATP IDs, verified file hashes, observed points only. ATP supplies
identities and metadata, not extra ability observations. No 2025 files are
opened by the experiment; the queue only checks the completed online artifacts.

- For 2012 validation, train the existing ordinary approximate filter on 2011,
  predict each 2012 match before updating it, using the pre-existing drift
  setting 0.03 and prior variance 1.0. Intercepts use 2011 only.
- For 2013 evaluation, reuse the saved ordinary **online** pre-match service
  probabilities from `results/early_real_2013/point_predictions.csv`. Check
  match IDs, server IDs, point counts and wins against the verified sequences.
- These are the point-corpus approximate ability estimates, not the full
  Bayesian ATP model. The analysis tests sequence structure on an available,
  audited development cohort; it cannot by itself explain the full model's
  2025 gains. Do not substitute one model's scores for the other's.

## Candidate mechanisms

All candidates receive the same pre-match probabilities and the actual server
of the upcoming recorded point. Each candidate predicts that point, then
conditions its latent-state probabilities on its observed outcome. Latent
states reset at the beginning of every match. No point is dropped for looking
like contamination. Outcomes after a prediction cannot change it.

| Family | Meaning | Fixed grid |
|---|---|---|
| `iid` | Original pre-match service probabilities | One baseline |
| `temperature` | Point-probability calibration | T = 0.75, 1, 1.25, 1.5, 2 |
| `independent_shocks` | Symmetric impairment redrawn independently at each point, with no temporal persistence | severity = 0.5, 1.5; rate = 0.1, 0.5, 1 |
| `match_offset` | One randomly chosen player has an unobserved mismatch lasting the entire match | severity = 0.5, 1.5; rate = 0.1, 0.5, 1 |
| `early_burst` | At most one temporary impairment of one randomly chosen player | severity = 0.5, 1.5; length = 10, 40, 80 match points; rate = 0.1, 0.5, 1; onset uniform on 0–80 |
| `late_burst` | Same, with onset shifted later as a crude fatigue-like hypothesis | Same grid; onset uniform on 80–160 |
| `forecast_default` | Fixed mechanism from the match simulator, now tested on point sequences | severity = 1.5, length = 40, rate = 0.5, onset 0–80 |

Severity subtracts the specified amount from the affected player's point-win
log odds on both serve and return; equivalently it adds that amount to the
opponent's serve-win log odds while the affected player returns. The rate is
prior probability of a potential event, not measured prevalence. Both possible
affected players have equal prior weight. Enumerate all allowed onsets exactly;
there is no Monte Carlo approximation in these sequence predictions. Match
termination can truncate or precede a scheduled burst.

Point indices are recorded `PointNumber - 1`, not compressed array positions.
Across missing-point gaps the scheduled clock advances, but missing outcomes
never update latent weights. Partial matches remain in evaluation.
The match-offset control addresses errors in baseline matchup strength;
independent shocks and temperature address probability shrinkage without
persistent episodes. None is a physiological measurement.

## Selection and evaluation

Score all **55** candidates on 2012 next-point log loss, weighted by observed
points. Select one candidate within each family using 2012 only; ties retain
the first listed candidate. Save this selection before scoring any 2013
sequence. Report the complete validation grid, including boundary optima.

Evaluate the seven fixed family representatives on 2013. Report point loss
and paired gains versus iid; for both transient families also compare against
temperature, independent shocks and match offset. Include paired early-versus-
late comparison. Positive gain means the candidate has lower loss.

Use 2,000 bootstrap draws: whole-tournament resampling and player-tournament
Poisson product weights involving both players, applied to paired per-match
loss sums and point counts. Report 95% intervals. These are exploratory,
unadjusted comparisons conditional on fitted abilities and selected parameters;
there are only four evaluation tournaments and dependencies across events can
remain. Show results by event, and descriptive completed/retired and point-index
0–79/80+ strata. Late-point strata condition on the match having survived that
long; they do not identify a fatigue effect. No causal injury or fatigue labels
will be assigned to individual players.

Interpretation requires transient models to beat the calibration **and**
match-offset controls, with stable direction across events. Even then the
finding is temporal predictive structure compatible with temporary impairment;
pressure, tactics, unmodeled conditions and recording artifacts remain possible.
If several durations/severities perform similarly, report weak identification.
If controls match the transient models, report that a physical burst mechanism
has not been established. Never select another experiment to force a positive
conclusion.

## Execution and outputs

`scripts/queue_mechanism_followup.py --queue` seals the protocol, code and
existing inputs, then starts a local waiting worker. It requires the online
report, full prediction table, valid results and matching sealed provenance
before launching the experiment. Failed hash checks stop execution.
The worker waits at most eight hours and writes visible status/error records.
An exclusive worker lock prevents duplicate execution.

The follow-up uses one numerical thread, no NUTS fits, and no match simulation.
The experiment has a 30-minute elapsed-time budget checked between candidates
and evaluation families. Semantic tests use artificial sequences before the
queue is launched; real-data scoring begins only after the prerequisite run.

Outputs in `results/burst_mechanisms/`: `FROZEN.json`, `status.json`,
`validation_grid.csv`, `selection_before_2013.json`, `match_losses.csv`,
`results.json`, and `report.md`. The ignored `worker.log` records execution.
Each attempted real-data variant is appended to `results/trials.jsonl` before
scoring. Existing protocols, sources and results remain intact.
