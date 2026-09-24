# Initial noncompliant WTA 2025 run

This audit-only run omitted the frozen team-event exclusion. It contains 73
United Cup training rows and 14 United Cup test rows; use the
[corrected report](../../../../../results/wta_2025/README.md) for conclusions.

Scored 2499 eligible WTA matches with chronological bimonthly updates.

| Forecast | Log loss | Accuracy | Brier | Calibration slope |
|---|---:|---:|---:|---:|
| iid | 0.64970 | 0.6507 | 0.22313 | 0.571 |
| burst | 0.62327 | 0.6503 | 0.21660 | 0.917 |
| temperature | 0.62472 | 0.6507 | 0.21688 | 0.856 |

Paired log-score gains (positive favors first):

- burst_vs_iid: +0.02643; tournament CI [+0.01493, +0.03926]; player-tournament CI [+0.01030, +0.04570].
- burst_vs_temperature: +0.00145; tournament CI [-0.00043, +0.00392]; player-tournament CI [-0.00103, +0.00562].
- temperature_vs_iid: +0.02498; tournament CI [+0.01511, +0.03555]; player-tournament CI [+0.01099, +0.04096].

Across fits: max R-hat 1.026, minimum bulk ESS 273, 0 divergences.
The burst simulation replicate changed probabilities by 0.00213 on average and log loss from 0.62327 to 0.62328.
