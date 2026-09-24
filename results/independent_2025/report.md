# Full ordinary model with burst forecasts: 2025 holdout

Generated 2026-09-22T20:07:12+00:00; 2670 eligible matches from the 2025 ATP source season.

Burst rate 0.5 and temperature 1.5 were selected on 2013 for the prior independent reproduction and carried forward before 2025 was read.

| Forecast | Log loss (tournament 95% CI) | Accuracy | Brier | Calibration slope | ECE-10 |
|---|---:|---:|---:|---:|---:|
| ordinary_iid | 0.66348 [0.63922, 0.68871] | 0.6337 | 0.23067 | 0.525 | 0.0873 |
| ordinary_burst | 0.63521 [0.61755, 0.65373] | 0.6337 | 0.22257 | 0.767 | 0.0347 |
| ordinary_temperature | 0.63396 [0.61631, 0.65297] | 0.6337 | 0.22241 | 0.788 | 0.0323 |

Paired log-loss gains (positive favors the first method):

- ordinary_burst_vs_ordinary_iid: +0.02827; tournament CI [+0.02036, +0.03660]; player-tournament CI [+0.01550, +0.04191].
- ordinary_burst_vs_ordinary_temperature: -0.00125; tournament CI [-0.00390, +0.00093]; player-tournament CI [-0.00491, +0.00182].
- ordinary_temperature_vs_ordinary_iid: +0.02952; tournament CI [+0.02138, +0.03768]; player-tournament CI [+0.01636, +0.04352].

## Diagnostics

For the frozen posterior fit: max R-hat 1.020, minimum bulk ESS 231, and 0 divergences. Periods above R-hat 1.01: [84].
The second simulation seed changed mean absolute match probability by 0.00208 and log loss from 0.63521 to 0.63547.

The absolute tournament-bootstrap intervals describe variability across events in one season. The paired intervals are the relevant uncertainty for method comparisons.

See `PROTOCOL.md`, `FROZEN_BEFORE_2025.json`, `results.json`, and `predictions_2025.csv` for the complete record.
