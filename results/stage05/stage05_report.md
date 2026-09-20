# Stage 0.5 report — retirement positive control

_generated 2026-09-20T18:39:42+00:00 · trials logged: 26_

**Does the detector see degradation on known-contaminated data?** Points preceding a retirement are the one near-certain contaminated sample. If service-win rate does not visibly degrade there, the detector does not work and we stop. Labels are external (RET in ATP/WTA scores); test fold untouched.

## Labels & join
- 263 retirements (by tour {'M': 201, 'W': 62}) and 9235 completed-match losers (control) in 2011–2020.
- Joined to point corpus by canonical name: **166 retirements matched**, 97 unmatched (coverage: Hawkeye-court only, first-rounders missing — biases the available retirement sample, a known D6 limitation).

## Degradation (retiree serve-win rate, lead-up vs earlier; clustered CI)
- **last set played**: retiree serve-win 0.547 → 0.4297 (Δ=0.1173, CI [0.089, 0.1454], n=126); control losers Δ=0.0383 CI [0.0354, 0.0412] (n=6732); **excess = 0.079**.
- **last 15 service points**: retiree serve-win 0.5551 → 0.4356 (Δ=0.1195, CI [0.0946, 0.1432], n=146); control losers Δ=0.0601 CI [0.0567, 0.0637] (n=6736); **excess = 0.0594**.

## Verdict
**FIRES: retirees degrade in the lead-up beyond ordinary losers -> the detector works on known-contaminated data; proceed to Stage 1.**

## Caveats
- Within-match baseline (earlier serve points) vs lead-up; the loser control absorbs the generic 'losers serve worse late' effect, so the excess is the contamination-specific signal.
- Retirement coverage is biased by Hawkeye-only recording (first-round retirements missing). The positive control still validates the detector; it does not estimate population ε.
- This is detector validation, not a test of the contamination hypothesis (that is Stages 2+).
