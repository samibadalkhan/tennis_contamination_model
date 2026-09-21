# Momentum investigation — is the Stage 0 anti-momentum real?

_generated 2026-09-20T18:37:54+00:00_

**Finding.** ARTIFACT (essentially). Scoring structure alone produces a transition ratio of 1.0499; the observed 1.0527 leaves a residual real effect of just 0.26% (anti-persistence). The Stage 0 'anti-momentum' was ~95% an artifact of the permutation null. There is NO evidence of positive momentum, and the tiny residual is within the simplifications of this null (constant within-game p; real serve prob varies by score state, adding still more structural alternation), so it is an upper bound, not a finding.

- observed within-game transition ratio: **1.0527**
- i.i.d.-under-scoring null (momentum-free by construction): **1.0499** (95% over reps [1.0492, 1.0515])
- real point-to-point effect (observed / null): **1.0026** (1.00 = no effect, <1 = momentum, >1 = anti-persistence)

## By game length (transition ratio)
| game length (points) | observed | i.i.d. null |
|---|---|---|
| 4 (4-0) | 1.0265 | None |
| 5 (4-1) | 1.0899 | 1.0927 |
| 6 (4-2) | 1.0516 | 1.0494 |
| 7-9 (short deuce) | 0.9391 | 0.9332 |
| 10+ (long deuce) | 1.1137 | 1.1086 |

The ratio climbs with game length in BOTH observed and simulated data: short decisive games (4-0/4-1) sit near 1, long deuce games are pushed well above 1 — because reaching and holding deuce mechanically requires trading points. That length dependence is present with zero momentum in the model, which is the tell that the permutation null, not player psychology, produced the original 'anti-momentum'.

## Consequence for the pipeline
- The Stage 0 serial statistic should be read against the i.i.d.-under-scoring null, not the permutation null. Interpreted correctly, there is little-to-no point-to-point dependence beyond what tennis scoring imposes.
- This does not change the overdispersion or asymmetry findings (those are block-level, not sequence-order). It corrects the momentum read only.
- A proper Stage 1 momentum test (K&M-style) must condition on score state, which absorbs exactly this structural alternation.
