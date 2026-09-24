# Project status

Updated 2026-09-23.

## What the completed results say

The full model reproduces Ingram's 2014 result: publication-rule log loss
0.59293 and accuracy 0.68841 versus 0.592 and 0.688 reported. On the frozen ATP
2025 season, iid log loss is 0.66348, burst is 0.63521, and temperature is
0.63396. With bimonthly online updates, iid is 0.64844, burst is 0.62656, and
temperature is 0.62544. Online burst improves over frozen burst by 0.00865 in
paired log score, but trails online temperature by 0.00112; the latter interval
includes zero.

The post-hoc mechanism split gives the tested burst model nowhere distinctive
to win. Its paired gain over global temperature is +0.00048 in best-of-three
and -0.00779 in best-of-five. Temperatures fitted separately on 2013 are 1.479
for best-of-three and 1.287 for best-of-five; burst still trails the
format-specific best-of-five temperature by 0.00354. None of five prespecified
absolute-edge bands shows a burst advantage with an interval excluding zero.

The WTA 2025 constrained protocol correction gives iid log loss 0.65162,
burst 0.62516, and temperature 0.62644 on 2,485 matches. Burst improves iid by
0.02646 with tournament CI [+0.01526, +0.03887] and player-tournament CI
[+0.00997, +0.04620]. Temperature also improves iid. Burst's +0.00128 gain
over temperature remains unresolved: its tournament CI is
[-0.00061, +0.00376] and player-tournament CI [-0.00141, +0.00536]. This
replicates the calibration benefit on WTA without identifying a burst-specific
forecast advantage.

Direct measurements show extra set-level heterogeneity: service-rate
dispersion is 1.192 [1.172, 1.211] relative to the conditional-binomial
reference for men and 1.191 [1.163, 1.218] for women. The initial men's
permutation-adjusted lag-one association is 0.00655 [0.00473, 0.00841], but on
identical within-set pairs it falls from 0.00648 to 0.00264 after centering on
set rather than match serve rate; the remaining player-event interval is
[-0.00062, +0.00590]. A scoring-only iid control also creates negative
across-game lag-one association. These observations reject a simple reading
of the raw point sequence, rather than identifying a physical impairment
process. The tested burst forecast still does not improve match probabilities
over calibration.

## Stability and protocol checks

- The ATP 2025 six-posterior seed-1729 replication is complete. Burst log loss
  is 0.62662 and temperature is 0.62554; burst-minus-temperature paired gain is
  -0.00109 with tournament CI [-0.00322, +0.00072]. Replicated burst differs
  from the original burst by only -0.00007 in mean log score and 0.00242 in
  mean absolute probability. The score is numerically stable, while sampling
  remains weak: maximum R-hat 1.047 and minimum bulk ESS 100.
- The initial WTA implementation admitted United Cup despite the frozen
  exclusion of team events: 73 training rows and 14 test rows. The deviation
  was recorded before correction, and all seven posteriors were refitted with
  only that exclusion changed. The corrected run had no divergences, maximum
  R-hat 1.032, and minimum bulk ESS 253. Because outcomes were open when the
  bug was found, it is a constrained protocol correction rather than a clean
  preregistered replication.

## Interpretation

The ordinary time-varying ability model works, especially when updated during
the season. Its raw match probabilities are too extreme and benefit from
calibration on both ATP and WTA. The evidence does not support the specific
40-point impairment story over a much simpler temperature correction. The
point data establish extra set-level variation, while the lag pattern is
sensitive to conditioning and tennis scoring. A physical interpretation needs
a cohort-matched, historically scored, coverage-aware iid control before any
new mechanism is fitted.

The robust likelihood, point-block filter, hybrid likelihood, synthetic gates,
and queued regime grid are closed. Their code, reports, protocols, and local
caches were moved to [`archive/robust_and_contamination/`](archive/robust_and_contamination/).
