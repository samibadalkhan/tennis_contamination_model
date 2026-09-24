# Stage 2 — synthetic burst gate (SMOKE)

_generated 2026-09-22T03:48:03+00:00 · 3 seeds × 12 weeks · caps [1.345, 2.5] · trials 148_

Robust method: **Huber bounded-influence filter** (per-match innovation clipped at `robust_c·√H`). MMW / gradient-covariance filtering is a documented upgrade, not what this gate certifies.

## Gate verdict

- **Recovery** (robust beats ordinary under bursts): **FAIL**
- **Efficiency** (robust costs <10% with nothing planted): **PASS**
- **Lag** (bounded tracking of permanent jumps): **PASS**
- **Overall: NOT passed**

## Recovery RMSE by cell (combined serve+return, mean ± seed sd)

| generator / condition | ordinary | robust 1.345 | robust 2.5 |
|---|---|---|---|
| rw / none | 0.279 ± 0.024 | 0.256 ± 0.007 | 0.264 ± 0.017 |
| rw / point_burst | 0.298 ± 0.038 | 0.273 ± 0.030 | 0.280 ± 0.033 |
| rw / tournament | 0.724 ± 0.138 | 0.304 ± 0.007 | 0.374 ± 0.011 |
| rw / permanent_jump | 0.364 ± 0.010 | 0.380 ± 0.009 | 0.361 ± 0.011 |
| arc / none | 0.272 ± 0.012 | 0.282 ± 0.028 | 0.271 ± 0.017 |
| arc / point_burst | 0.310 ± 0.024 | 0.297 ± 0.029 | 0.293 ± 0.022 |
| arc / tournament | 0.760 ± 0.127 | 0.348 ± 0.044 | 0.402 ± 0.041 |
| arc / permanent_jump | 0.357 ± 0.005 | 0.391 ± 0.016 | 0.358 ± 0.005 |

## Realism check (sim should give Ingram ~0.59 log loss)

- rw: ordinary iid forecast log loss **0.557 ± 0.049**
- arc: ordinary iid forecast log loss **0.583 ± 0.136**

## Permanent-jump lag (weeks to recover; asymptotic error)

- rw:  ordinary lag=4.4w asymp=0.262  robust_1.345 lag=5.3w asymp=0.333  robust_2.5 lag=4.9w asymp=0.278
- arc:  ordinary lag=3.5w asymp=0.239  robust_1.345 lag=5.0w asymp=0.331  robust_2.5 lag=4.3w asymp=0.263
