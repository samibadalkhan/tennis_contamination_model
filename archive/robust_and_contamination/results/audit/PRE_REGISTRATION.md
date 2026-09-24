# Pre-registration — robust vs. ordinary tennis ability estimation

_Registered 2026-09-22T03:14:26+00:00. Locked BEFORE any contact with the 2025 ATP test season.
The test set is single-use; this document fixes the analysis so results cannot
be searched for after the fact._

## Hypothesis
Robustly estimating players' time-varying serve/return abilities improves
held-out match prediction over ordinary estimation of the same model, when both
forecast through the same point-by-point simulator with its contamination rule
tuned separately per arm. Not attempted: learning the contamination
distribution, labeling corrupted points, or recovering condition labels.

## Model (shared by both arms)
p_t = σ(α + s_i(t) − r_j(t) + βᵀx_t), with s_i, r_j Gaussian random walks
(variance σ²). Context x: surface, serve number, tournament, round. Ingram (2019)
specification (per-player surface effects, tournament intercepts). **Only the
estimator differs between arms** (ordinary likelihood fit vs. robust MMW/filtering);
identical likelihood and data in both.

## Data combinations (both pre-registered)
1. **Match-only (headline).** Both arms on ATP match terms. Directly comparable
   to Ingram; tests match-scale robustness.
2. **Hybrid.** ATP match terms as the random-walk backbone; slam point terms for
   slam matches, **replacing** (never augmenting) the match term for those
   matches. Both arms use the identical likelihood. Tests point-level burst
   robustness — the mechanism the hypothesis is about.

## Forecasting
Every forecast is a point-by-point match simulation under the actual match
format, contamination injected by a random burst rule. **The rule is tuned per
arm on validation** (robust is expected to tune a higher rate than ordinary; the
gap ≈ what robust estimation removed). Never the analytic iid formula.

## Ablation (2×2)
estimation {ordinary, robust} × forecast {iid, tuned burst}. The claim is
ordinary-vs-robust each at its best forecast; the iid row isolates the forecast
rule.

## Split & evaluation
- Develop ≤2024; **test = 2025 ATP season, once**. Validation by rolling-origin
  within development; hyperparameters (ε, σ, per-arm forecast rule) frozen before
  test.
- Primary metric: **match cross-entropy** (as Ingram); accuracy and calibration
  secondary.
- **Report first-match-of-tournament separately from later rounds** (robust is
  predicted to win the first, possibly lose the second).
- Uncertainty: bootstrap clustered by **player-tournament**; tournament-level as
  a conservative check.

## Gates (must pass in order, before test)
1. Reproduce Ingram 2014 ≈ 0.592 log loss (small drift allowed).
2. Synthetic gate: robust beats ordinary on planted-burst recovery, costs little
   with none planted, lags boundedly on permanent jumps.

## Pre-registered comparisons (the only ones that count as confirmatory)
- C1 (headline): ordinary vs robust, match-only, tuned forecasts, on 2025.
- C2: ordinary vs robust, hybrid, tuned forecasts, on 2025.
- C3: the 2×2 ablation, on 2025.
- C4: first-match-of-tournament vs later-rounds split, for C1 and C2.
Anything else is exploratory and labeled as such.

## Discipline
Every model variant is appended to results/trials.jsonl (never reset) — with a
frozen test set the trial count is the only defense against overfitting by
search. Negative (resolution present, effect absent) and inconclusive (noise
floor exceeded the effect) are reported as distinct outcomes.
