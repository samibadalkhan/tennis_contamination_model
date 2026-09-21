# Stage 0 — audit & split (Ingram forecasting design)

_generated 2026-09-21T21:31:08+00:00 · trials logged: 32_

Gate 0: coverage, artifacts, chain of custody, power, pre-registration. No model fit here; 2025 test outcomes untouched.

## Split
- develop ≤2024, **test = 2025 ATP season (once)**; validation rolling-origin; cluster by player-tournament. (Old slam-year split retired — see archive/.)

## Coverage
- 2025 test season: **2944 ATP matches** in 141 tournaments.
- Slam point data: all in development, 97 slam-year-tour cells; years 2011–2024.
- Full grid: results/audit/match_coverage_grid.csv, slam_point_coverage.csv.

## Artifacts
- Slam→ATP resolution: **0.794** (8346 matched, 2167 unmatched — Hawkeye coverage + name variants; investigate, don't drop).
- Retirements kept & flagged (the key diagnostic). Singles-only enforced upstream.
- Reduced-covariate: slam serve-number absent pre-2019; ATP serve stats 88–99% present by year. Absence is never a signal.

## Chain of custody
- SHA pin resolved: recorded == independent == live HEAD (True); the spec's `83733358…` is the bad value, never recorded.
- Per-file SHA-256 verified (python -m src.fetch --verify); hash failure is a hard stop. ATP/WTA provenance is WEAK (headline data → hashes carry the weight); keep a cold copy.

## Power check (2024 proxy; 2025 outcomes untouched)
- The comparison is **paired** (both arms score the same matches), so the noise floor is the CI on the per-match cross-entropy **difference**: half-width **0.0155** (clustered by tournament, 153 tournaments).
- For context, the *absolute* mean-CE CI half-width is 0.0221 — NOT the right floor here; using it would wrongly read as underpowered.
- 2025 test size: 2944 matches / 141 tournaments (comparable floor).
- Plausible gains vs paired floor: 0.005✗, 0.01✗, 0.02✓. **Verdict: conditionally adequate — gains ≥ 0.020 clear the CONSERVATIVE paired floor (0.0155, an upper bound); smaller gains need confirmation once the real, more-correlated arms exist.**
  Two real arms are more correlated than these proxies, so the true paired floor is likely smaller still. Re-check once both arms exist.

## Pre-registration
- results/audit/PRE_REGISTRATION.md locks the hypothesis, model, both data combinations (match-only headline + hybrid), the 2×2 ablation, the split, the metric, the first-match/later-round report, and the gates — before any test contact.

## Gate status → next
- Audit complete. **Next gate: reproduce Ingram 2014 (~0.592 log loss).** Then the synthetic gate, then develop both arms.
