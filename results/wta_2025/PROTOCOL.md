# Preregistered WTA 2025 replication

Status: frozen before loading `data/wta/wta_matches_2025.csv` or computing any
WTA 2025 prediction or outcome statistic.

## Question

Does the ordinary Ingram point model with the previously selected burst
forecast improve 2025 WTA match log loss, and does it improve on an equally
simple global temperature correction?

## Cohort

- Training source seasons: WTA 2011--2024.
- Test source season: WTA 2025, whose manifest SHA-256 is
  `e4eb002bc6d828b0bd21e04ed286e68722ddc8ccbbb27ea980bf62e938377f96`.
- Include Grand Slam, Premier Mandatory/1000, Premier/500, International/250,
  and Tour Finals levels (`G`, `PM`, `P`, `I`, `F`) with complete player IDs,
  surface, format, and valid service-point totals. These historical WTA level
  labels were verified on training seasons through 2024 before the test file
  was opened. Exclude team events, Olympics, and lower-tier events.
- Preserve every eligible row in the 2025 source file, including tournaments
  dated in late December 2024. Sort by date, tournament ID, round, and match ID.

The source file will not be opened until this protocol and its evaluation code
have been written and hashed into a freeze record.

## Model and forecast timing

Use the full published Ingram-style Bayesian point model already reproduced in
`src/independent/bayes.py`: bimonthly player serve and return random walks,
player-by-surface effects, tournament effects, and a binomial point likelihood.
Use four NUTS chains, 1,000 warmup draws and 1,000 retained draws per chain,
`target_accept=0.9`, and seed 1729. No robust likelihood is included.

Forecasts are made in chronological bimonthly blocks. A match in period `p`
uses only source rows with model period less than `p`. The period-84 fit is
restricted to source seasons through 2024. Thereafter period 85 uses results
through February 2025, period 86 through April, and so on. Late-December rows
in the 2025 source season are forecast from the period-83 model trained through
period 82. Each saved fit records a fingerprint of its training rows.

## Locked forecasts

For each match, posterior serve probabilities are converted to:

1. `iid`: exact match probability under independent points.
2. `burst`: an equal mixture of `iid` and a conditional 40-point impairment
   simulation. The impaired player is uniform, onset is uniform over points
   0--80, logit severity is 1.5, and the simulation uses 16,384 draws with
   seed 4001. These settings were selected before this WTA test on the 2013 ATP
   development season.
3. `temperature`: `logit(iid) / 1.5`, using the temperature selected before
   this WTA test on the same ATP development work.

No parameter will be selected using WTA 2025 outcomes. A second burst
simulation seed (9029) is a numerical check only.

## Outcomes and inference

The primary estimand is the mean paired log-score gain of `burst` over `iid`,
where positive values favor burst. The main secondary estimand is burst over
temperature. Also report temperature over iid, log loss, Brier score, accuracy,
calibration slope, and ECE-10.

Report two-sided 95% percentile bootstrap intervals clustered by tournament
and, as a dependence-sensitive check, by the connected player--tournament
blocks used in the ATP analysis. The result supports a forecast improvement
only if the corresponding paired interval excludes zero in the favorable
direction. Absolute single-season intervals are descriptive; paired intervals
answer the model-comparison questions.

## Diagnostics and deviations

Report divergences, maximum rank-normalized R-hat, and minimum bulk ESS for
every fit. Any code, cohort, sampler, or analysis change after the freeze is a
protocol deviation and must be recorded before reporting its result. Failed
runs remain in the record.
