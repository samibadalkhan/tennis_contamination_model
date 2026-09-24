# Exploratory online updates: 2025

The 2025 holdout had already been opened before this online extension was registered. These results are exploratory.

| Forecast | Log loss | Accuracy | Brier | Calibration slope |
|---|---:|---:|---:|---:|
| frozen_iid | 0.66348 | 0.6337 | 0.23067 | 0.525 |
| online_iid | 0.64844 | 0.6442 | 0.22602 | 0.574 |
| frozen_burst | 0.63521 | 0.6337 | 0.22257 | 0.767 |
| online_burst | 0.62656 | 0.6431 | 0.21933 | 0.841 |
| frozen_temperature | 0.63396 | 0.6337 | 0.22241 | 0.788 |
| online_temperature | 0.62544 | 0.6442 | 0.21908 | 0.860 |

Paired log-loss gains (positive favors the first method):

- online_burst_vs_frozen_burst: +0.00865; tournament CI [+0.00128, +0.01721]; player-tournament CI [-0.00230, +0.02057].
- online_iid_vs_frozen_iid: +0.01505; tournament CI [+0.00419, +0.02784]; player-tournament CI [-0.00142, +0.03249].
- online_temperature_vs_frozen_temperature: +0.00852; tournament CI [+0.00135, +0.01681]; player-tournament CI [-0.00187, +0.01969].
- online_burst_vs_online_iid: +0.02188; tournament CI [+0.01508, +0.02895]; player-tournament CI [+0.00942, +0.03437].
- online_burst_vs_online_temperature: -0.00112; tournament CI [-0.00341, +0.00078]; player-tournament CI [-0.00451, +0.00176].

## Diagnostics

Across six posterior fits: max R-hat 1.025, minimum bulk ESS 124, 0 divergences, and periods above R-hat 1.01 [84, 85, 86, 87, 88, 89].
The second simulation seed changed mean probabilities by 0.00210 and online burst log loss from 0.62656 to 0.62660.

See `PROTOCOL_ONLINE_POSTHOC.md`, `ONLINE_FROZEN.json`, `online_results.json`, and `online_predictions_2025.csv`.
