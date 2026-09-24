# WTA 2025 replication

The constrained protocol correction scored 2,485 eligible WTA matches with
chronological bimonthly updates and the frozen team-event exclusion enforced.

| Forecast | Log loss | Accuracy | Brier | Calibration slope |
|---|---:|---:|---:|---:|
| iid | 0.65162 | 0.6491 | 0.22396 | 0.567 |
| burst | 0.62516 | 0.6487 | 0.21744 | 0.909 |
| temperature | 0.62644 | 0.6491 | 0.21763 | 0.851 |

Paired log-score gains, where positive favors the first method:

- Burst over iid: +0.02646; tournament CI [+0.01526, +0.03887];
  player-tournament CI [+0.00997, +0.04620].
- Burst over temperature: +0.00128; tournament CI [-0.00061, +0.00376];
  player-tournament CI [-0.00141, +0.00536].
- Temperature over iid: +0.02518; tournament CI [+0.01550, +0.03566];
  player-tournament CI [+0.01090, +0.04134].

The primary burst-versus-iid improvement survives on WTA. Temperature also
improves iid, and the paired comparison cannot distinguish burst from the much
simpler calibration correction. Accuracy barely changes; the gain is almost
entirely from correcting overconfident probabilities.

The first 2,499-match implementation had admitted 73 United Cup training rows
and 14 United Cup test rows despite the frozen protocol's exclusion of team
events. The error was recorded in
[`PROTOCOL_DEVIATION.md`](PROTOCOL_DEVIATION.md) before the correction began.
Because outcomes were open by then, this result is labeled a constrained
protocol correction rather than a clean preregistered replication. The first
run is retained in the
[final cleanup archive](../../archive/final_experiment_cleanup_2026-09-23/results/wta_2025/initial_noncompliant/README.md).

Across the seven corrected fits there were no divergences, maximum R-hat was
1.032, and minimum bulk ESS was 253. The R-hat maximum is the period-86 global
intercept; nearly all other parameters are at or below 1.010. A second burst
simulation seed changed probabilities by 0.00205 on average and log loss from
0.62516 to 0.62498.

- [Corrected numerical report](corrected/README.md)
- [Corrected result JSON](corrected/results.json)
- [Frozen protocol](PROTOCOL.md)
- [Correction freeze](corrected/FROZEN_CORRECTION.json)

![Protocol-corrected WTA 2025 results](corrected/wta_2025_summary.png)
