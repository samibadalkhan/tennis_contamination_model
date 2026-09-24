# Matched Ingram comparison

The `iid`, `burst`, and `temperature` forecasts all start from the same full
ordinary Ingram-style posterior. They use identical training observations,
player IDs, information cutoffs, posterior serve/return draws, evaluation
matches, and tennis rules. Only the final forecast transform differs:

| Forecast | Calculation |
|---|---|
| iid | Exact posterior-averaged match probability under independent points |
| burst | Equal mixture of iid and a simulated 40-point impairment forecast |
| temperature | `logit(iid) / 1.5` |

The burst forecast is not a separately trained ability model. This matched
construction isolates the forecast mechanism from ability estimation.

## Completed ATP 2025 comparison

The frozen posterior was trained on 35,732 eligible matches from 2011--2024
and scored 2,670 matches in the 2025 source season. Its log losses were 0.66348
for iid, 0.63521 for burst, and 0.63396 for temperature.

The exploratory online extension refit at two-month cutoffs using only earlier
periods. It produced log losses of 0.64844 for iid, 0.62656 for burst, and
0.62544 for temperature. Burst gained 0.02188 over matched iid, but its gain
over matched temperature was -0.00112 with an interval spanning zero. Online
updating improved burst over the frozen burst forecast by 0.00865.

The cohort, transformations, and cached posterior components were checked in
[`matched_comparison_audit.json`](../results/independent_2025/matched_comparison_audit.json).
The completed files are
[`online_report.md`](../results/independent_2025/online_report.md),
`online_results.json`, and `online_predictions_2025.csv`.

## WTA 2025 correction

The frozen WTA protocol used the same three forecast transforms with
chronological bimonthly updates. Its constrained team-event correction scored
2,485 matches: iid log loss was 0.65162, burst 0.62516, and temperature
0.62644. Burst improved iid by 0.02646 with both paired intervals above zero.
Its +0.00128 gain over temperature had tournament CI
[-0.00061, +0.00376] and player-tournament CI [-0.00141, +0.00536].

The first implementation omitted the written team-event exclusion. That bug
was found after outcomes were open, recorded before correction, and fixed by
refitting every posterior without 73 United Cup training rows or 14 United Cup
test rows. See the [WTA report](../results/wta_2025/README.md).

## Scope

The 2014 author-fixture run separately verifies the literature benchmark. The
ATP 2025 cohort uses current source files and historical deciding-set rules, so
its absolute loss is not directly comparable with the paper's 2014 loss. The
online extension is exploratory because ATP 2025 had already been scored. The
WTA protocol was frozen before its test source was opened, but its corrected
implementation is labeled a constrained protocol correction because the
team-event bug was found after the first result.
