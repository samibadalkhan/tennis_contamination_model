# Point-block robustness pilot

Target: underlying serve/return skill, discounting brief disruptions while
learning sustained changes. This is a small mechanism experiment, not a pass
of the project's full synthetic gate or a result on real tennis data.

## Fixed design (written before the first run)

- Eight players; 64 observation sessions; all players play once per session.
  Random pairings are independent of outcomes. Every encounter has 192 ordered
  points, alternating four service points per player. These are fixed-length
  observation sequences, **not scored tennis matches or a tournament bracket**.
- Two truth generators: random walks (0.04 logit SD per session) and smooth
  sine/cosine trajectories. Estimator drift is fixed at 0.04 per session for
  both generators. No synthetic search selects drift or robustness settings.
- Twenty paired seeds per generator and condition. Initial skill SD 0.35,
  serve intercept logit(0.64), prior variance 1.0.
- Conditions: clean; a contiguous 48-point disruption of players 0 and 1 in
  sessions 24 through 35; a permanent -0.6 logit change in both abilities at
  session 24; and a permanent +0.6 change at the same time. Temporary impairment
  is -2.0 logits in both abilities. Burst positions need not align with blocks.
  Players, trajectories, schedules and uniform outcome draws are shared across
  conditions. A player affected throughout an encounter is a boundary case
  that this within-encounter detector cannot identify.
- Six arms: ordinary; the actual existing OnlineAbility Huber update (cap 2.5);
  block-gradient filtering at 8, 16 and 32 points; and an oracle excluding only
  points whose probabilities were changed by planted temporary disruptions.
  Oracle knowledge is unavailable to the other arms. Permanent changes are
  never marked as corruption. The oracle is a reference, not a guaranteed
  finite-sample lower bound on error.

## Candidate estimator

Use the two gradients for an encounter's serve-logit contrasts. Equal-size
blocks contain equal service opportunities for each player. Fit the two
binomial rates on retained blocks, standardize the block scores by their
binomial standard deviations, center them, and compute their leading covariance
eigenvector. If its eigenvalue exceeds a clean conditional-randomization
threshold, discard the block with the largest squared projection and refit.
Stop when the threshold is met or 25% of the encounter's blocks have been
discarded. Apply each block's weight to both players' updates.

The threshold is the fixed 99th percentile of 255 random partitions of the
observed service totals under a homogeneous, independent Bernoulli model. This
is a clean-model calibration, not fitting a threshold to recovery results.
The initial threshold is held fixed during removal. It is **not** a nominal
99% guarantee for the adaptively selected final subset. No actual contamination
labels, truth, future sessions or corrupted-world results enter the filter.

This is a **SEVER-inspired block-score spectral filter**, not an implementation
of published SEVER or MMW and not a proof of arbitrary-point-contamination
robustness. Reference: [Diakonikolas et al., ICML 2019](https://proceedings.mlr.press/v97/diakonikolas19a.html).
Centering scores within the encounter deliberately makes a persistent level
shift different from heterogeneous blocks. A uniform disruption of an entire
encounter is consequently invisible to this detector. A point-contamination
fraction is not the same as a block-contamination fraction: a burst crossing
block boundaries can touch more blocks than the rejection budget permits.

The final retained counts feed the same existing online ability filter as the
ordinary arm. The candidate reduces information as well as score by removing
observations before updating. It inherits the diagonal Gaussian approximation
and one-step update of that backbone. This isolates whether block information
helps; a full dynamic robust-gradient optimizer is a later step.

## Evaluation and boundaries

All ability estimates use only observations through the reported session.
Abilities are constant inside each encounter. Every arm sees the same points.
Forecasting and forecast-time contamination play no part in the experiment.

Report RMSE from session 24 onward, separately for affected players and others,
plus all-player and per-player RMSE. Align only the single unidentified common
serve/return offset; never center serve and return separately. Compare paired
seed differences with 95% bootstrap intervals separately for each generator.
The simulation seed is the independent cluster, not the individual point.
These intervals quantify Monte Carlo uncertainty in this fixed synthetic design.

For permanent changes, recovery requires both serve and return errors to stay
within 0.2 logits for three consecutive sessions. Report recovery fraction and
lag among recoveries; non-recoveries remain explicitly censored at the horizon.
This is an absolute-error recovery diagnostic and need not reflect the instant
at which a method first recognizes the jump.

Report removed clean-point fraction, removed disrupted-point fraction, and
per-player errors. A figure uses seed 0 and block size 16, specified in advance;
it is illustrative rather than selected for a favorable result. All three
block sizes stay visible in the tables. No automatic gate verdict and no
winning hyperparameter selection. Save configuration, source hashes, seed-level
metrics and trajectories; append every executed configuration to trials.jsonl.

Run: `python -m src.experiments.point_robustness`

Checks: `python -m unittest src.experiments.test_point_robustness`
