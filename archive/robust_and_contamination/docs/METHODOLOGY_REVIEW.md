# Methodology and results review

Reviewed 2026-09-21 (America/Los_Angeles), including the simulated-forecast results that finished during this review.

The research question is coherent and worth testing. The saved baseline scores reproduce exactly. However, the implementation does not yet establish the stated point-burst robustness claim, and several bugs invalidate particular diagnostics. Treat these as exploratory results from a simplified online filter, not as a validated reproduction of Ingram or a completed test of the registered method.

**Follow-up completed 2026-09-22.** A separate implementation now reproduces
the full published Bayesian model on Ingram's exact author fixture: 2014 log
loss 0.59293 and accuracy 0.68841 versus the paper's 0.592 and 0.688. On the
same 2,208 matches, Huber-deviance robust estimation is tied with ordinary
estimation (iid gain +0.00071, tournament CI [-0.00028, +0.00191]; separately
tuned burst-forecast gain +0.00043 [-0.00025, +0.00120]). Probability
calibration supplies the large improvement: ordinary burst loss is 0.58087 and
ordinary temperature-scaled loss is 0.58027. The latter shows that the gain
does not identify real point bursts. See
[`results/independent/README.md`](../results/independent/README.md) and the
fixed [`PROTOCOL.md`](../results/independent/PROTOCOL.md). All 24 fits have zero
divergences, though worst R-hat is 1.029 ordinary and 1.017 robust, which is a
reported convergence caveat. No 2025 outcomes enter the follow-up.

The new ordered-point synthetic pilot is internally reproducible across all
160 worlds and shows that block structure can improve latent-skill recovery in
its planted-burst setting. Its preselected block-16 gain does not transfer to
the independently rebuilt 2011–2013 real-point development run: block-16 is
slightly worse than ordinary on held-out point loss, while aggregate Huber is
about 0.0005 better. A post-hoc 2012-tuned temperature control more than
explains Huber's larger match-log-loss advantage at the point-estimate level;
the chosen value is the top of its pre-existing grid, and only four evaluation
tournaments make that comparison imprecise. These newer
results address the original review with separate, corrected workflows; they
do not retroactively validate the older broken artifacts described below.

Read the project design, instructions, status, pipeline documentation, current result reports, and archived research notes; traced the active model, simulation, synthetic, loading, audit, and evaluation code. ChatGPT conversation history was not accessible through the available tools. No model implementation or existing result artifact was changed for this review. Numeric checks used development seasons only; the raw-data integrity check hashed files without analyzing test outcomes.

**What the results currently say**

| Development comparison | Ordinary log loss | Huber 1.345 | Huber 2.5 |
|---|---:|---:|---:|
| 2014 iid, sigma 0.03 / prior 1.0 | 0.6128 | 0.6211 | 0.6162 |
| 2019 iid, sigma 0.03 / prior 1.0 | 0.6563 | 0.6683 | 0.6589 |
| 2014 separately tuned burst simulation, sigma 0.03 / prior 0.25 | 0.59844 | 0.60412 | 0.59930 |

Lower is better. The independently tuned baseline checkpoint is 0.6100498762 on 2014 and 0.6529620337 on 2019; it uses different hyperparameters from the fixed-parameter iid comparison. These numbers must not be interchanged.

The current Huber variants do not improve these development forecasts. The synthetic experiment suggests protection against large match/tournament distortions, but its implementation needs repair before that result can certify the method. The simulated forecast's improvement over the ordinary iid checkpoint is promising as a calibration result; neither a contamination mechanism nor statistical significance has been established.

**1. The simulated comparison reports false zero effects and confidence intervals. High priority.**

In `src/sim_forecast.py:48`, `tourneys` is initialized but never appended to. `forecast_season` returns predictions alongside an empty tournament array. Lines 126–132 aggregate over zero clusters, and the bootstrap silently returns `(0, 0, 0, 0)`.

A one-match reproduction returned one prediction and zero tournament IDs. The saved 2014 report consequently prints zero paired gains and zero-width intervals for both robust arms. From the reported aggregate losses, the actual mean gains, defined as ordinary minus robust loss, are **-0.0056738149** and **-0.0008615242**. Their uncertainty cannot be recovered from aggregate losses alone.

Append tournament IDs, preserve match IDs and per-match predictions, and assert equal nonzero array lengths before aggregation. Recompute the paired bootstrap. The saved zero intervals are not evidence of equivalence.

**2. The synthetic tournament generator can produce invalid probabilities. High priority.**

`src/synthetic.py:136–172` represents self-impairment and opponent-impairment as disjoint segments. Tournament impairment can affect both opponents simultaneously (`:198–212`). Then both segment fractions equal one, so they sum to two. `_counts` allocates all points to the first segment, while its returned mean adds both segments.

Reproduction with the default baseline and severity 2: segment probabilities 0.1939358563 and 0.9292591295, each with weight 1, produce a claimed probability **1.1231949859**. That value is passed into the analytic match forecaster. Under the intended equal serve-and-return impairment, the opposing logit shifts should cancel in this example.

Apply simultaneous effects to a common timeline or use properly normalized joint states. Assert all probabilities and segment weights are valid. Rerun the synthetic results after repair; the frequency of these cases and their aggregate impact have not been quantified here.

**3. The synthetic experiment does not implement the claimed point-burst world, and its lag criterion is weak.**

`sim_match` draws binomial service counts and independently draws the winner using an iid match probability computed from average point probabilities. The bracket therefore responds to latent impairment, but the winner is not generated by the same realized points supplied to the estimator. No ordered point sequence or burst position enters the match outcome. This can serve as a simplified match-observation experiment; it does not certify robustness to contiguous bursts under tennis scoring.

Generate one coherent point sequence, derive score, winner, counts, and bracket advancement from it, and test random and fatigue-related bursts. The default full gate does not exercise `structured=True`.

For permanent jumps, `jump_lag` censors non-recovery at the remaining horizon (`:313`), whereas the gate accepts average lag below 90% of the full horizon (`:417`). Jumps occur around the middle half of the season, so even players who never recover will satisfy the lag-time part of that criterion. The asymptotic-error check remains active, but the combined PASS does not establish recovery within a meaningful time limit. Report recovery proportion, censoring, and sustained recovery for both serve and return.

`results/synthetic/gate.json` explicitly records `gate_passed: false`. Continuing development can be reasonable, but should be documented as exploratory continuation or a prospectively revised match-scale study, not as passage of the original gate. The recovery threshold also uses marginal seed standard deviations rather than paired uncertainty in the difference.

**4. ATP identities are silently merged. High priority.**

`src/load.py:229` keys players by first initial plus surname and discards the existing ATP IDs. The ability dictionaries use those keys directly. The collision flags in the separate processed point/slam layer do not protect the headline ATP filter.

Confirmed development-data collisions include Alex Kuznetsov (104864) and Andrey Kuznetsov (105723), Eduardo Nava (124013) and Emilio Nava (207182), and Ze Zhang (105585) and Zhizhen Zhang (111190). These pairs currently share ability states. Use the stable ATP IDs for the ATP model and an explicit crosswalk for point-data joins; names should be display labels. Recompute results afterward.

**5. The first-match diagnostic measures opening round, not each player's first match.**

`src/robust_check.py:81–85` labels every match in a tournament's minimum observed round as first. This misses byes and labels later round-robin appearances as first.

Counting prior player appearances within each tournament in the current evaluation order found **330 matches in 2014 and 317 in 2019** labeled later despite at least one participant making their first appearance. Conversely, **11 and 22** matches labeled first had a participant who had already played. These counts describe errors relative to the current input sequence; chronology itself also needs repair.

Track appearances per player and tournament. Predefine how to handle matches where only one player has already played, rather than forcing them into an ambiguous binary split. The present split does not establish the hypothesized across-tournament versus within-tournament mechanism.

**6. Chronology is not guaranteed by the available sorting keys.**

`src/load.py:225–238` uses tournament start date plus tournament ID and round. All round-robin matches tie on these keys, so file order becomes update order. The 2014 Finals file lists Djokovic's Wawrinka and Berdych matches before his Cilic match. His official match report identifies Wawrinka as the second group match and Berdych as the subsequent match ([Djokovic's official report](https://novakdjokovic.com/en/news/tennis/nole-blasts-past-wawrinka-plays-berdych-on-friday/)). The current sequence can therefore update on later results before predicting earlier matches.

Use verified match order/dates. Where those cannot be established, batch forecasts at a safe information cutoff and update afterward. Sorting by match number alone is not automatically a solution. Tournament start dates also suppress within-event drift and need explicit treatment for overlapping events and real-time forecasting claims.

**7. This is an Ingram-inspired baseline, not a reproduction of the published model.**

The original paper uses player-specific surface adjustments, tournament intercepts, hierarchical priors, NUTS inference, and posterior-averaged match predictions. Its 2014 evaluation contains 2,208 matches after excluding retirements, walkovers, and missing service statistics. This repository uses a scalar online Gaussian approximation with a pooled surface intercept, plug-in skill means, and 2,573 evaluation matches, including retirements. See [Ingram's paper, sections 2.1–2.2.3](https://martiningram.github.io/papers/bayes_point_based.pdf).

The shared random-walk idea is legitimate, but being within an implementation-defined 0.02 tolerance does not validate equivalence of model or protocol. Either implement a separate faithful benchmark or rename the gate as a sanity checkpoint. Keeping retirements in the contamination study is reasonable; use an explicitly separate comparable cohort for the literature reproduction.

The model also discards posterior correlations and evaluates predictions at estimated means. These approximations could contribute to miscalibration. Improvement from adding simulated noise does not establish that real injury bursts caused the original miscalibration.

**8. The implemented robust estimator does not justify the design's contamination guarantees.**

`src/models/ability.py:74–84` clips the score of each aggregated service observation at `c * sqrt(H)` while retaining the ordinary information/variance update. This is a useful bounded-score heuristic, but it is not the specified MMW/covariance-filtering procedure, does not implement an epsilon budget per player-tournament, and does not detect collective point-gradient structure.

The code can shrink uncertainty substantially even when it rejects most of an innovation. The threshold uses binomial observation variance without accounting for uncertainty in the predicted skill. Both choices merit explicit justification and checks for newcomers, prolonged slumps, and posterior coverage. Huber's 1.345 constant does not automatically confer 95% efficiency in this dynamic binomial setting.

The clean inference available now is about this particular Huber-style filter. It is not a negative result for all robust estimators or a demonstrated arbitrary-contamination guarantee.

**9. Forecast simulation needs numerical and mechanistic controls.**

`src/sim_forecast.py:35` uses only 200 simulations per match. Near probability 0.5, binomial Monte Carlo standard error is about 0.035. Season averaging helps, but nonlinear log loss, rare upsets, smoothing, and tuning make a convergence check necessary, particularly for a reported arm gap of 0.00086. Equal simulation counts do not guarantee equal log-loss bias across differently calibrated arms.

The simulator applies impairment only when the impaired player serves; the synthetic generator impairs both serve and return. These are different mechanisms. Both simulation and analytic forecasting also use a seven-point tiebreak at 6–6 in every set, rather than match-specific historical deciding-set rules. Bursts are scheduled against approximate match length and may not occur before a match ends.

Validate scores and arm differences over increased simulation counts and independent seeds. Include a validation-tuned probability calibration baseline to determine whether apparent burst benefits require burst structure. For the current rule, probability is a mixture of the no-burst and burst-conditioned probabilities, so those components could be estimated accurately once and mixed across rate candidates.

The statement that equal tuned burst rates mean robustness removed nothing is not identified by this experiment. Rates also reflect severity, duration, model misspecification, uncertainty, and the discrete tuning grid. Two arms currently select the largest rate in the grid; their optima have not been bracketed.

**10. Audit and test safeguards are weaker than the prose claims.**

- The ATP loader reads all available years before callers filter them. Existing pre-2025 model loops stop before test updates, so this alone does not demonstrate test leakage. But claims that 2025 is never read are false. Add a file-selection guard and a deliberate final-test entry point. `reproduce_ingram` also accepts arbitrary evaluation/tuning years without validating their order or enforcing the test boundary.
- The audit's 2,944-match test count includes event levels outside the headline model's eligibility. It is not the final scoring sample size. No test eligibility counts were recalculated in this review.
- Missing service statistics remove matches from scoring as well as updating, and the headline rows lack retirement flags. In 2014, 27 of 2,600 G/M/A/F matches are excluded for service-stat availability, including one of 87 retirements. In 2019, 31 of 2,634 are excluded; in 2024, 19 of 2,752. Retain otherwise eligible matches for forecasts even when their ability update must be skipped, and define walkover/retirement scoring rules explicitly.
- The paired-power correction is conceptually right, but a proxy CI half-width is not a power calculation or a proved upper bound. Approximate 80% power at two-sided 5% significance requires an effect around 2.8 standard errors, not 1.96. Reassess with real paired development losses and the intended eligible cohort. Tournament bootstrap is a reasonable sensitivity analysis, but the promised player-tournament analysis is absent and cross-event temporal dependence deserves consideration.
- The slam-to-ATP match rate is only 79.4%. Hawkeye undercoverage explains ATP matches missing from the point corpus, not why an observed slam match cannot be found in the ATP/WTA results. Resolve this before a hybrid comparison, and include tour and stable identities in join keys.
- The pre-registration can be overwritten and does not yet specify enough operational details to lock tuning, cohort, simulation precision, and primary robust-model selection. Preserve a versioned final analysis specification before test.

**Documentation and interpretation**

`STATUS.md` contains stale 2014 gap, accuracy, and prior-variance statements; it mixes the corrected 0.6100 checkpoint with older claims of a 0.012 gap and prior variance 1.0. The README still marks implemented components as unbuilt. Prefer regenerating numerical summaries from the result artifacts.

The archived momentum investigation correctly demonstrates that the permutation null can create apparent anti-persistence under ordinary tennis scoring. It does not prove that momentum is absent. Similarly, poorer late-match service among retirees is a useful positive control, not proof that all those observations represent temporary contamination or that the robust estimator detects it.

**Verification performed**

- `python -m src.fetch --verify`: all 169 files match recorded hashes. This verifies local integrity, not the historical authenticity of the mirror.
- `python -m src.forecast`: all self-tests pass, including the corrected tiebreak orientation checks.
- `python -m src.simulate`: all existing checks against the iid recursion pass. Those tests cover the simplified scoring format and do not certify historical formats or burst mechanisms.
- Recomputed the saved 2014 and 2019 ordinary scores with recorded hyperparameters using a file-selection restriction to development years: exact agreement with both JSON artifacts.
- Reproduced empty-cluster zero intervals, invalid dual-impairment probability, distinct ATP IDs merged under names, and first-appearance misclassification.

Repair identities, chronology, cluster aggregation, and synthetic generation first. Then rerun the development comparisons with explicit model scope, valid uncertainty, and coherent simulation. The 2025 season should remain unused for model selection while those issues are resolved.
