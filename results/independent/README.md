# Full-model 2014 reproduction

The full Bayesian hierarchy reproduces the Ingram benchmark on the separately
pinned author fixture. It uses bimonthly serve and return random walks,
player-by-surface effects, tournament effects, and posterior-averaged match
probabilities.

| Forecast | Log loss | Accuracy |
|---|---:|---:|
| Published benchmark | 0.59200 | 0.68800 |
| Reproduction, publication rules | 0.59293 | 0.68841 |
| Reproduction, historical rules | 0.59322 | 0.68841 |
| Ordinary + burst | 0.58087 | 0.68841 |
| Ordinary + temperature | 0.58027 | 0.68841 |

The development comparison selected burst mixture rate 0.5 and temperature
1.5 on 2013. Those settings were carried to ATP 2025 before its source season
was opened. The ordinary 2013 predictions remain active because the
per-format temperature analysis uses them.

The production ordinary fits had zero divergences, with maximum R-hat 1.029.
See `baseline.json`, `comparison.json`, `comparison_2014_predictions.csv` (local only), and
`PROTOCOL.md` for the sealed record. The robust arm and failed synthetic
exercises formerly stored beside these files are archived under
`archive/robust_and_contamination/results/independent/`.
