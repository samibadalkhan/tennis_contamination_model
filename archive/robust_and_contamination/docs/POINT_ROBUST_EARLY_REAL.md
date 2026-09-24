# Early real-data read: train through 2012, evaluate 2013

Exploratory continuation explicitly requested by the user after the synthetic
pilot. This is not passage of the full synthetic gate, a reproduction of Ingram,
or the single-use 2025 evaluation. Freeze this protocol before scoring 2013.

## Cohort and time boundary

Use men's singles Grand Slam point sequences from 2011–2012 for fitting;
evaluate the available 2013 sequences. Only these years' raw point/match files
and ATP match files may be opened. Verify every input against `data/MANIFEST.json`
before reading observations. ATP supplies stable player IDs, round, tournament
start date, surface and outcome, **not additional training observations**.

Join by the two full normalized player names within the same event, using only
explicit spelling aliases for Stanislas/Stan Wawrinka, Richard/Ricardas Berankis,
and Rogerio Dutra Da Silva/Rogerio Dutra Silva. Refuse ambiguous or unmatched
point matches rather than silently merging initial-plus-surname keys.

Order tournaments by their start date and matches by bracket round. Matches in
the same round have disjoint players, which must be asserted. Actual match dates
are unavailable here: diffuse the ability prior between tournament starts,
with no within-tournament diffusion. File order breaks ties only for matches
with disjoint participants. Preserve point order; fail on duplicated or
non-increasing played point numbers. Do not create a block across a missing-point
gap. Keep retirements and their observed points. No unobserved point is imputed.

## Estimators and selection

Three arms share OnlineAbility's latent model and diagonal Gaussian updater:
ordinary, existing Huber cap 2.5, and spectral block filtering at the preselected
16-point size. The latter retains the pilot's 99th-percentile conditional-null
threshold (255 randomizations) and maximum 25% rejection budget. For real
sequences, generalize the null to each block's actual service counts, including
zero-exposure columns and short final blocks. Budget removal by point count,
not number of blocks. Require at least four blocks before filtering.

For each arm, select annual drift SD from **[0.03, 0.1, 0.3, 1.0]** by pre-match
point log loss on 2012, with 2011 initialization and online updates after each
2012 match. Prior variance is fixed at 1.0. Surface intercepts for this selection
are computed only from 2011. After selection, refit 2011–2012 from scratch using
intercepts estimated from those two training years. Nothing is selected on 2013.
All candidates and final fits are logged append-only. No synthetic result
selects the block size, budget, null level, or drift.

## Evaluation

Primary: **frozen end-of-2012 abilities**, predicting all observed 2013 points
using their server/opponent/surface. New players receive the shared zero-mean
prior. Secondary: pre-match predictions followed by online updates in 2013,
with all hyperparameters frozen. No held-out match contributes to its own
prediction. The frozen arm never learns from any 2013 outcome.

Primary metric is per-point log loss; also report point Brier score, point
accuracy and a both-players-seen-in-training subset. Points in a held-out match
are all scored; **never remove evaluation points to improve the score**.
Report opening-round and later-round diagnostics and per-event differences.
True contamination and true abilities are unknown: these are predictive checks,
not proof that a cleaner skill state has been recovered.

Use paired player-tournament bootstrap intervals for point metrics, assigning
each point to its server's tournament cluster. This accounts for repeated
service observations, but not all shared-opponent or cross-event dependencies.
Also report tournament-cluster bootstrap sensitivity; there are only four
evaluation events, so these intervals cannot establish broad generalization.

Secondary match metrics use the same iid analytic forecast in all arms to
isolate estimation, with the historical deciding-set rules implemented in
`src/independent/scoring.py`: advantage deciding sets outside the US Open and
a seven-point tiebreak at 6–6 at the US Open for this period. They do not include
an injury/retirement model; keep and flag retirements and show completed-match
metrics separately. Match uncertainty is tournament-clustered (four clusters).
This is an iid estimation diagnostic, not the project's final tuned simulated
forecast comparison. Historical changes are documented by
[USTA](https://www.usta.com/en/home/improve/tips-and-instruction/national/tennis-scoring-rules.html)
and the [US Open](https://www.usopen.org/amp/en_US/news/articles/2022-03-16/us_open_to_join_all_grand_slams_in_playing_10point_final_set_tiebreak.html).

Save input/source hashes, join/coverage/gap audit, tuning results, predictions,
training ability snapshots and block rejection diagnostics. Label the findings
as development results: 2013 already appears in earlier project development.
