# Trained through 2012: the block filter does not improve 2013 predictions

**Decision: STOP this block-filter replacement branch.** The user requested a
full train/validation and Ingram-2014 benchmark comparison only if the early
results looked promising. This candidate worsens both point and match log loss
relative to ordinary fitting on the same 2013 cohort, so that condition is not
met. No further full training or 2014 benchmark run will be started for this
candidate. The 2013 scores are not directly comparable to
Ingram's 2014 results because the season, cohort and fitted model differ.
The existing Huber arm's improvement is a separate finding; it does not validate
the proposed block-filter replacement. This is a decision about this candidate,
not a negative conclusion about robust learning generally.

Fitted on **671 recorded men's Grand Slam matches / 145,321 points from
2011–2012**, then evaluated **346 matches / 75,723 points from 2013**.
These are matches available in the point corpus, not the full ATP schedule.
Annual drift was selected separately for each arm using 2012 only; all selected
0.03 from the fixed grid. No parameter was selected using 2013 scores.

With abilities frozen at the end of 2012, the new block filter performs slightly
worse than ordinary fitting. The existing Huber update performs better here.

| Estimator | 2013 point log loss, 95% CI | Paired loss reduction vs ordinary, 95% CI |
|---|---|---|
| Ordinary | 0.651994 [0.647706, 0.656654] | Reference |
| Existing Huber, cap 2.5 | 0.651549 [0.647302, 0.656153] | +0.000445 [+0.000192, +0.000738] |
| Block-gradient filter, 16 points | 0.652124 [0.647826, 0.656763] | −0.000130 [−0.000233, −0.000047] |

Lower log loss is better; positive paired reduction favors the candidate.
Point intervals resample server-player/tournament clusters. The block filter's
point estimate is worse in each of the four evaluation events. Event-level
and tournament-bootstrap sensitivity results are saved in the full report and
JSON. There are only four events, and player/event resampling does not account
for all shared-opponent or across-event dependence, so do not extrapolate the
apparent precision into a general claim about robust learning.

The secondary **predict, then update online during 2013** comparison agrees:
block-filter paired reduction −0.000108 [−0.000231, −0.000012]; Huber reduction
+0.000507 [+0.000175, +0.000924]. No held-out match updates its own prediction,
and no held-out point is removed from scoring.

Match forecasting also fails to favor the block filter. Using the same iid
scoring model and frozen abilities, its paired match log-loss reduction is
−0.0097 [−0.0138, −0.0059]; Huber's is +0.0235 [+0.0100, +0.0356]. These
intervals use only four tournament clusters. Completed-match sensitivity and
all match accuracy/loss intervals are in the [full report](report.md).

## What the filter actually changed

It rejected **248 of 145,321 training points in 14 matches**. In the secondary
online evaluation, it rejected 224 of 75,723 points in 11 matches, only after
those matches had been scored. These are rejection counts, not measured true
contamination. The mechanism is conservative and alters relatively little of
this real dataset. The retained data feed the same ability updater as ordinary
fitting; both the learning signal and information change when points are removed.

Real blocks have unequal numbers of serves from each player. The implementation
therefore calibrates the covariance statistic against each block's actual
service exposure, retains short final blocks, and does not bridge gaps in point
numbers. This is a necessary adaptation of the balanced synthetic pilot;
it is still an experimental SEVER-inspired prefilter, not published SEVER/MMW.

There is **no observed ground truth for underlying skill or contamination**.
The synthetic result showed partial recovery under a specified generator; this
real-data result shows no forecasting benefit from the current block filter.
Neither establishes whether an individual rejected stretch was injury, ordinary
variation, genuine change, or a recording artifact. Huber's gain may reflect
regularization; it is not proof of successful decontamination.

## Data checks and limitations

- All 27 consumed input files passed their manifest SHA-256 checks. Only
  2011, 2012 and 2013 raw files were opened by this experiment; 2025 was not read.
- Player identities use stable ATP IDs, with three explicit full-name alias
  mappings. Every point match joined unambiguously, and available point-file
  winner labels agreed with ATP. No initial-plus-surname identity merging.
- There were 195 training players and 27 previously unseen evaluation players.
  New players receive the common prior. The both-players-seen subset also fails
  to favor the block filter.
- Round order is verified to have disjoint participants within a round. Exact
  match dates are unavailable; ability variance diffuses between tournament
  starts, not within an event.
- Coverage is incomplete. Recorded serve counts differ from ATP totals in
  37 training matches and 21 evaluation matches. Some records are very partial:
  Djokovic–Haas at Roland Garros 2013 contains only four recorded played points,
  all with the same server. Those four are retained, not imputed into a full
  match. Gaps split blocks; missing observations are not labeled contamination.
- This is a small development comparison. 2013 already occurs in prior project
  development, so it is not represented as a never-before-seen final test.

## Files and reproduction

- [Protocol fixed before scoring](../../docs/POINT_ROBUST_EARLY_REAL.md)
- [Full report](report.md) and [point-gain plot](point_gains.png)
- [Evaluation runner](../../src/experiments/early_real.py)
- [Real-point block filter](../../src/models/block_filter.py)
- `audit.json`: source verification, joins, missing-point and count discrepancies
- `results.json`: tuning, selection, paired intervals and source hashes
- `point_predictions.csv`, `match_predictions.csv`: all paired predictions
- `abilities_2012.csv`: per-player fitted serve/return means and variances
- `block_diagnostics.json`: every rejection and its match/point location

Twenty initial semantic checks passed, followed by an additional check for
single-server partial matches. Saved source hashes, all 18 append-only trial
records, cohort alignment, losses and paired gains were verified independently.
The generated plot was visually inspected.

```bash
python -m unittest src.experiments.test_early_real src.experiments.test_point_robustness
python -m src.experiments.early_real --out results/early_real_2013_rerun
```

Data attribution: Jeff Sackmann, CC BY-NC-SA 4.0. Derived observations retain
that attribution and license; see the repository's `ATTRIBUTION.md`.
