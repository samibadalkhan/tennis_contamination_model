# First point-block experiment: partial recovery of underlying skill

The block-gradient filter reduces contamination-induced ability error in this
fixed synthetic experiment. It leaves substantial contamination behind. This
supports investigating the mechanism further, not replacing the production
estimator yet or claiming a general robustness guarantee.

The preselected 16-point illustration gives the following affected-player RMSE
under temporary bursts. Values are in ability logits; brackets are 95% bootstrap
intervals resampling whole simulation seeds.

| Truth generator | Ordinary | Existing Huber (2.5) | Block filter (16) | Known-mask oracle |
|---|---|---|---|---|
| Random walk | 0.1717 [0.1613, 0.1812] | 0.1702 [0.1598, 0.1797] | 0.1486 [0.1394, 0.1576] | 0.0915 [0.0864, 0.0968] |
| Smooth trajectories | 0.1581 [0.1507, 0.1652] | 0.1568 [0.1495, 0.1638] | 0.1340 [0.1242, 0.1435] | 0.0786 [0.0729, 0.0850] |

The **paired RMSE reduction versus ordinary** is 0.0230 [0.0177, 0.0292]
for random walks and 0.0241 [0.0186, 0.0303] for smooth trajectories. Other
players also benefit: their reductions are 0.0039 [0.0026, 0.0052] and
0.0044 [0.0032, 0.0058], respectively. These are experimental Monte Carlo
intervals, not estimates of an effect on real tennis players.

On clean data, the paired changes are small: +0.0003 [-0.0001, +0.0008] and
-0.0000 [-0.0006, +0.0005]. Permanent skill increases and decreases remain
learnable; see the full report for recovery fractions, conditional lags and
their uncertainty. For example, after permanent declines in the random-walk
worlds, both ordinary and block-16 recover in 7.1 [5.6, 8.7] sessions on average,
among recovered cases. All 40 player trajectories recover in each arm in that
cell; the observed bootstrap recovery interval [1, 1] does not imply certain
recovery in unseen worlds.

There is substantial room left: block-16 removes only 22.7% [20.7%, 25.1%]
of disrupted points in random-walk worlds and 22.7% [19.4%, 26.2%] in smooth
worlds. It also removes 0.32% [0.28%, 0.38%] and 0.27% [0.22%, 0.33%] of clean
points in the burst conditions, respectively. These are block rejection
decisions, not reliable individual-point contamination labels.

All fixed block sizes (8, 16 and 32) remain visible in the [full report](report.md).
The 32-point setting has lower observed burst RMSE in this particular setup;
that is not a validation-based choice of block size. Burst duration and severity
were fixed, so optimizing the choice here would risk tailoring the estimator to
the generator.

## What was built

The [experiment](../../src/experiments/point_robustness.py) generates ordered
Bernoulli points with known time-varying serve/return skills. A spectral
prefilter inspects centered, standardized gradients across contiguous blocks,
rejects a limited number of blocks with unusual joint structure, and feeds the
retained observations to the existing ability updater. Rejected observations
reduce both the learning signal and information supplied by that encounter.

This is a local SEVER-inspired prefilter plus the existing online updater,
**not a complete dynamic robust-gradient optimizer or published SEVER/MMW**.
It can distinguish within-encounter heterogeneity from a uniform level change,
but cannot distinguish an entirely impaired encounter from a genuine shift.
The point sequences have fixed lengths and balanced service opportunities;
they are not tennis matches generated under scoring rules.

## Review and reproduce

- [Settings fixed before execution](../../docs/POINT_ROBUST_PILOT.md)
- [Ability trajectories](trajectories.png) and [block decisions](block_weights.png)
- [All metrics and source hashes](results.json); fitted trajectories in `trajectories.npz`
- [Semantic tests](../../src/experiments/test_point_robustness.py): 10 passed, including
  label isolation, chronology, shared observation weights, oracle masking and
  explicit non-recovery censoring.

The run contains 20 paired seeds per generator/condition, two generators, four
conditions, six arms. All 960 arm configurations were appended to
`results/trials.jsonl`; no real-data file was read. Saved source hashes and
oracle/ordinary equality in uncorrupted conditions were verified. Both plots
were rendered and visually inspected.

```bash
python -m unittest src.experiments.test_point_robustness
python -m src.experiments.point_robustness --out results/point_robust_pilot_rerun
```

The next experiment should vary burst strength and duration, include benign
within-match dependence, and test whole-encounter disruptions. Those are the
main boundaries of the current result. Freeze that sweep before running it;
keep synthetic checks separate from choosing settings on real validation data.
