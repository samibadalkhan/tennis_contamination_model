# Independent full-model comparison

Generated 2026-09-22T10:36:10+00:00; 2208 author-fixture matches in 2014. Tune on 2013.

Same full Ingram ability model and priors; ordinary binomial objective versus Huber-deviance robust objective (cap 2.5).
Historical deciding-set rules; posterior-averaged forecasts. Positive paired gain favors candidate.

| Method | Log loss (tournament 95% CI) | Accuracy | Brier | Cal. slope | ECE-10 |
|---|---|---|---|---|---|
| ordinary_iid | 0.59322 [0.56661, 0.62524] | 0.688 | 0.2024 | 0.693 | 0.0502 |
| ordinary_burst | 0.58087 [0.56057, 0.60511] | 0.688 | 0.1993 | 1.017 | 0.0151 |
| ordinary_temperature | 0.58027 [0.55966, 0.60522] | 0.688 | 0.1992 | 1.040 | 0.0173 |
| robust_2p5_iid | 0.59251 [0.56616, 0.62373] | 0.687 | 0.2022 | 0.695 | 0.0518 |
| robust_2p5_burst | 0.58044 [0.56031, 0.60400] | 0.689 | 0.1992 | 1.020 | 0.0131 |

Paired comparisons:

- robust_2p5_burst_vs_ordinary_iid: +0.01278; tournament CI [+0.00438, +0.02316]; player-tournament CI [-0.00015, +0.02815].
- robust_2p5_burst_vs_ordinary_burst: +0.00043; tournament CI [-0.00025, +0.00120]; player-tournament CI [-0.00066, +0.00162].
- robust_2p5_iid_vs_ordinary_iid: +0.00071; tournament CI [-0.00028, +0.00191]; player-tournament CI [-0.00079, +0.00239].
- ordinary_burst_vs_ordinary_iid: +0.01236; tournament CI [+0.00403, +0.02248]; player-tournament CI [-0.00054, +0.02766].
- ordinary_burst_vs_ordinary_temperature: -0.00060; tournament CI [-0.00183, +0.00056]; player-tournament CI [-0.00432, +0.00253].

Robust-estimation gain within the same forecast rule by appearance stratum:

| Stratum | Robust burst vs ordinary burst | Robust iid vs ordinary iid | n |
|---|---:|---:|---:|
| both_first | +0.00020 [-0.00097, +0.00147] | +0.00072 [-0.00081, +0.00240] | 982 |
| mixed_first_later | +0.00136 [-0.00016, +0.00305] | +0.00162 [-0.00078, +0.00453] | 349 |
| both_later | +0.00035 [-0.00036, +0.00112] | +0.00037 [-0.00074, +0.00154] | 866 |

Unknown appearance order (11 round-robin rows) is omitted from stratum inference.

Frozen burst rates: {'ordinary': 0.5, 'robust_2p5': 0.5}. Temperature control: 1.5.

Fit diagnostics: ordinary: max R-hat 1.029, min bulk ESS 245, 0 divergences, R-hat >1.01 in periods [16, 17, 18, 19, 20, 21, 22, 23]; robust_2p5: max R-hat 1.017, min bulk ESS 259, 0 divergences, R-hat >1.01 in periods [15, 16, 17, 18, 20, 21, 22, 23].

Player-tournament intervals use pigeonhole resampling of both player memberships; they are a sensitivity analysis.
The author cohort excludes retirements. This tests a specified robust objective and burst forecast, not arbitrary contamination guarantees.
Fits use bimonthly information cutoffs, so first/later-round differences do not measure within-tournament updating.
See PROTOCOL.md, fit diagnostics, and comparison.json for the simulation replicate and full results.
