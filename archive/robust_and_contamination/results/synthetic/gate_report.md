# Stage 2 — synthetic burst gate

_generated 2026-09-22T03:48:58+00:00 · 6 seeds × 26 weeks · caps [1.345, 2.5] · trials 292_

Robust method: **Huber bounded-influence filter** (per-match innovation clipped at `robust_c·√H`). MMW / gradient-covariance filtering is a documented upgrade, not what this gate certifies.

## Gate verdict

- **Recovery** (robust beats ordinary under bursts): **FAIL**
- **Efficiency** (robust costs <10% with nothing planted): **PASS**
- **Lag** (bounded tracking of permanent jumps): **PASS**
- **Overall: NOT passed**

## Recovery RMSE by cell (combined serve+return, mean ± seed sd)

| generator / condition | ordinary | robust 1.345 | robust 2.5 |
|---|---|---|---|
| rw / none | 0.206 ± 0.016 | 0.198 ± 0.017 | 0.203 ± 0.014 |
| rw / point_burst | 0.242 ± 0.018 | 0.220 ± 0.017 | 0.231 ± 0.014 |
| rw / tournament | 0.509 ± 0.064 | 0.248 ± 0.034 | 0.290 ± 0.035 |
| rw / permanent_jump | 0.297 ± 0.016 | 0.315 ± 0.015 | 0.298 ± 0.015 |
| arc / none | 0.251 ± 0.010 | 0.262 ± 0.012 | 0.249 ± 0.009 |
| arc / point_burst | 0.270 ± 0.018 | 0.277 ± 0.021 | 0.265 ± 0.016 |
| arc / tournament | 0.636 ± 0.053 | 0.345 ± 0.034 | 0.379 ± 0.032 |
| arc / permanent_jump | 0.324 ± 0.017 | 0.360 ± 0.016 | 0.327 ± 0.018 |

## Realism check (sim should give Ingram ~0.59 log loss)

- rw: ordinary iid forecast log loss **0.529 ± 0.043**
- arc: ordinary iid forecast log loss **0.543 ± 0.052**

## Permanent-jump lag (weeks to recover; asymptotic error)

- rw:  ordinary lag=9.6w asymp=0.208  robust_1.345 lag=11.9w asymp=0.270  robust_2.5 lag=10.2w asymp=0.220
- arc:  ordinary lag=8.0w asymp=0.202  robust_1.345 lag=10.2w asymp=0.264  robust_2.5 lag=8.7w asymp=0.216

## Timescale map (robust advantage; + = robust better)

| filter σ | burst len frac | robust advantage |
|---|---|---|
| 0.03 | 0.1 | +0.006 |
| 0.03 | 0.25 | +0.022 |
| 0.03 | 0.5 | +0.068 |
| 0.06 | 0.1 | +0.006 |
| 0.06 | 0.25 | +0.022 |
| 0.06 | 0.5 | +0.068 |
| 0.12 | 0.1 | +0.006 |
| 0.12 | 0.25 | +0.022 |
| 0.12 | 0.5 | +0.069 |

## Severity × length sweep (robust advantage, tight cap 1.345)

| severity | burst len frac | advantage |
|---|---|---|
| 1.0 | 0.15 | +0.007 |
| 1.0 | 0.3 | +0.018 |
| 2.0 | 0.15 | +0.016 |
| 2.0 | 0.3 | +0.032 |
| 3.0 | 0.15 | +0.016 |
| 3.0 | 0.3 | +0.041 |
