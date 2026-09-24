# Independent full-model reproduction and non-iid comparison

This is an exploratory, separately implemented reproduction requested after the
repository review. Existing development results were already known. This is not
a new confirmatory pre-registration, and no 2025 outcomes enter this workflow.

- Source: Martin Ingram's published 2011–2014 fixture, pinned and hashed under
  `data/independent_ingram/MANIFEST.json`. Exactly 2,208 matches in 2014. The fixture
  excludes retirements and walkovers, so this comparison cannot establish results
  specifically for injury retirements or the full modern ATP cohort.
- Ordinary fit: full published hierarchical binomial model, including player
  surface skills, tournament intercepts, independent serve/return Gaussian random
  walks, and the author's normal/half-normal priors. Four NUTS chains, 1,000 warmup
  and 1,000 retained draws per chain. Analytically marginalize unobserved walk
  states; this changes computation, not the probability model.
- Time: two-calendar-month periods from January 2011. Predict every match in a
  period from earlier periods only. Calendar boundaries implement the paper's
  intended timing rather than the old Python code's approximate month arithmetic.
  This schedule avoids ambiguous within-tournament ordering but does not update
  abilities between rounds of the same tournament.
- Robust fit: same latent model, priors, and observations, replacing the sum of
  binomial deviance losses by Huber losses of the deviance residuals. Cap 2.5 is
  fixed before these results, from the existing project's predeclared caps. This
  is a robust generalized posterior, not the original capped online filter, not
  MMW, and not an epsilon-contamination guarantee. Posterior uncertainty under
  this objective is not automatically frequentist calibrated.
- Forecasts: posterior-average exact iid match probabilities and a point-by-point
  non-iid simulator. A potential burst impairs one randomly selected player's
  serve AND return, begins uniformly at point indices 0..80, lasts 40 points, and
  changes log odds by 1.5. A burst starting after match completion has no effect.
  This makes potential-burst probability different from a fraction of corrupted
  points. All points, counts, and the match winner are generated together.
- Rate grid: 0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0. Select separately for each estimator
  using 2013, then freeze before evaluating 2014. The rate mixture combines exact
  posterior iid probability with simulated conditional-burst probability.
- Simulation: initially 16,384 conditional-burst matches per fixture match, with
  a second independent seed on 2014 for a numerical sensitivity check. Use the
  same sample count and burst mechanism in both arms. Jeffreys smoothing applies
  only to the Monte Carlo component.
- Scoring: natural-log loss primary; accuracy, Brier score, and calibration
  secondary. Keep per-match IDs, predictions, serve posterior draws, and source
  hashes so comparisons can be reconstructed without refitting.
- Intervals: paired tournament bootstrap, plus a player-tournament cluster
  multiplier sensitivity analysis with both players contributing symmetrically.
  These are conditional on the fitted models; they do not include uncertainty
  from repeating the entire development/tuning exercise.
- Controls: an ordinary forecast with a validation-tuned temperature parameter
  distinguishes simple calibration from any need for burst structure. Compare
  all four estimator/forecast cells on the identical 2014 matches. For literature
  reproduction, also retain the author's all-sets-tiebreak iid convention; the
  main comparison uses event-appropriate deciding-set rules.
- Diagnostics: report NUTS divergences, R-hat, effective sample sizes, run time,
  and Monte Carlo sensitivity. Do not label a poorly converged fit a successful
  replication. Full Bayesian reproduction is assessed independently of whether
  robust estimation wins.

Reference: https://martiningram.github.io/papers/bayes_point_based.pdf
Reference implementation: https://github.com/martiningram/tennis_bayes_point_based
Scoring-rule sources: AO's "Final set tiebreaks at Australian Open 2019" and
Wimbledon's 2019 annual report / 2021 compendium, consulted during implementation.

The previous project's synthetic gate remains failed; no result here silently
changes that verdict. This independent experiment answers the requested
comparison, rather than claiming completion of every gate in the older plan.
