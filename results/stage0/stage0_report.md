# Stage 0 report — i.i.d.-given-server sanity check + noise floor

_generated 2026-09-20T18:27:33+00:00 · trials logged so far: 13_

**Diagnostic only.** Gates whether Stage 0.5 (retirement positive control) is worth running. Fits on TRAIN, measures noise floor on VALIDATION, never touches TEST.

## Split (declared once, frozen)
- train [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018], val [2019, 2020], test [2021, 2022, 2023, 2024] (untouched)
- 1,444,001 train+val points; serve-win rate by tour {'M': 0.6382, 'W': 0.5676}

## Model
- i.i.d.-given-server: L2 logistic on serve(player) + return(player) + slam|year|tour
- C selected on val = 0.1 (grid {'0.1': 0.667913, '0.3': 0.668847, '1.0': 0.669095, '3.0': 0.669487})
- val log-loss 0.667913 vs constant-rate baseline 0.668662 (lower is better)
- strongest serve effects: [['R. Federer', 0.65], ['J. Isner', 0.588], ['John Isner', 0.559], ['Serena Williams', 0.554], ['Ivo Karlovic', 0.55]]

## Diagnostics (train, in-sample; match-clustered 95% CI)
- **Overdispersion** φ = 1.2413 CI [1.2245, 1.2589] (overdispersed). 36,831 set-blocks, median n=29.0.
- **Serial dependence** within-game transition ratio = 1.0527 CI [1.051, 1.0546] (runs z = 57.863) — looks like anti-persistence, but see the momentum investigation below: ~95% of it is a scoring-structure artifact of the permutation null. Against an i.i.d.-under-scoring null, the real effect is 1.0026 (≈none).
- **Asymmetry** block-residual skewness = -0.1768 CI [-0.1985, -0.1548] (left-skewed (directional)).

## Noise floor (validation, held-out)
- mean val log-loss 0.667913 CI [0.666501, 0.669275], half-width 0.001387.
- oracle-detectable (eps,delta) cells: 7 / 9. **Verdict: adequate.**
  - grid: (0.02,0.1)→0.0004; (0.02,0.2)→0.00159*; (0.02,0.3)→0.0036*; (0.05,0.1)→0.00097; (0.05,0.2)→0.00385*; (0.05,0.3)→0.00872*; (0.1,0.1)→0.00183*; (0.1,0.2)→0.00728*; (0.1,0.3)→0.01652*  (* = above noise floor)

## Momentum investigation (why anti-momentum?)
- Observed transition ratio 1.0527 vs an i.i.d.-under-scoring null of 1.05 (95% [1.0491, 1.052]). Real effect 1.0026.
- **ARTIFACT (essentially).** Winning points does NOT build momentum here; the apparent anti-momentum is tennis scoring (deuce forces alternation), not psychology. See momentum_investigation.md. A real momentum test belongs in Stage 1, conditioned on score state.

## Read-out
- If overdispersion CI sits above 1 AND some plausible (eps,delta) clears the noise floor → the detector has resolution; **proceed to Stage 0.5**.
- If no plausible cell clears the noise floor → **INCONCLUSIVE by design** (a fact about the design, not about tennis); redesign before continuing.
- Asymmetry sign hints strategic (symmetric) vs involuntary (left) H, but does not by itself separate D4 from D5 — that is Stage 2.

## Limitations (Stage 0)
- Diagnostics are in-sample on train (residual structure); only the noise floor is held-out. Overdispersion/asymmetry describe the fitted model's residuals, not out-of-sample generalization.
- Player identity is the raw slam-corpus name string; variant spellings (e.g. 'J. Isner' vs 'John Isner') fragment a player's serve/return effect across two levels. A D6 name-normalization issue to resolve before the ATP retirement join at Stage 0.5; it slightly weakens the fit here.
- Serve-number (1st/2nd) is absent pre-2019 (all-missing in early years); the base i.i.d.-given-server model does not use it, so this is neutral for Stage 0 but is a reduced-covariate issue for later stages.

## Figures (results/stage0/figures/, `python -m src.plots`)
- `1_block_residuals.png` — overdispersion + left-skew vs the i.i.d. N(0,1) expectation.
- `2_momentum_artifact.png` — observed vs momentum-free simulation; the anti-momentum is scoring structure.
- `3_noise_floor.png` — oracle contamination effect per (ε,δ) vs the held-out noise floor.

_Negative ≠ inconclusive. This report states which one Stage 0 produced._
