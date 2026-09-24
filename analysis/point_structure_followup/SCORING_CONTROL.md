# Can scoring alone create the apparent pattern?

These simulations contain **no bursts, fatigue, injury, momentum or ability changes**. Each point is independent conditional on its server. Tennis scoring and stopping remain in place.

| Serve probabilities | Format | Within-game lag 1 [Monte Carlo interval] | Across-game lag 1 [Monte Carlo interval] | Same-set lag 1 [Monte Carlo interval] |
|---|---|---:|---:|---:|
| 0.65, 0.65 | Best of 3 | +0.00123 [-0.00637, +0.00908] | -0.03007 [-0.04599, -0.01388] | -0.00299 [-0.00994, +0.00379] |
| 0.65, 0.65 | Best of 5 | +0.00254 [-0.00322, +0.00829] | -0.01714 [-0.02876, -0.00439] | -0.00010 [-0.00507, +0.00512] |
| 0.65, 0.60 | Best of 3 | +0.00312 [-0.00449, +0.01014] | -0.02212 [-0.03901, -0.00651] | -0.00015 [-0.00712, +0.00650] |
| 0.65, 0.60 | Best of 5 | +0.00463 [-0.00153, +0.01046] | -0.02741 [-0.04020, -0.01391] | +0.00049 [-0.00487, +0.00593] |

The simulations demonstrate which qualitative patterns can be produced by the measurement and scoring structure alone. They are illustrative matchups, not a matched null for the real cohort. Subtracting their estimates from the real data would not give a valid physical-effect estimate.

## Simulator check

| Serve probabilities / format | Simulated A win fraction | Exact iid probability | Monte Carlo SE |
|---|---:|---:|---:|
| 0.65/0.65, bo3 | 0.5098 | 0.5000 | 0.0221 |
| 0.65/0.65, bo5 | 0.5195 | 0.5000 | 0.0221 |
| 0.65/0.60, bo3 | 0.7344 | 0.7365 | 0.0195 |
| 0.65/0.60, bo5 | 0.7695 | 0.7854 | 0.0181 |

All five lags, six views, seeds, and source hashes are in scoring_control.json. See SCORING_CONTROL.md for the protocol and report.md for the observed-data decomposition.
