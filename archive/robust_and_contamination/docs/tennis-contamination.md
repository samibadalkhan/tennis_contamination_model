# Robust estimation of evolving serve and return ability under contamination

## Hypothesis

Temporary distortions — short bursts of injury or tilt — get absorbed into conventional ability estimates and carried into future predictions. Robust estimation should limit that while still tracking genuine career change.

**Claim.** If up to an ε fraction of points may be arbitrarily corrupted, concentrated in short bursts, then robustly estimating players' time-varying serve and return abilities improves held-out match prediction over ordinary estimation of the same model — when both forecast through the same match simulator, with its contamination rule tuned separately for each arm.

Not attempted: learning the contamination distribution, diagnosing which observations are corrupted, or recovering latent condition labels. The goal is better estimates without explaining the corruption.

## Model

Server i, returner j, point t:

```
p_t = σ( α + s_i(t) − r_j(t) + βᵀx_t )
```

- `s_i(t)`, `r_j(t)` — serve and return abilities, Gaussian random walks: `s(t+1) = s(t) + η`, `η ~ N(0, σ²)`
- `x_t` — observed context: surface, serve number, tournament, round
- `σ` — temporal regularization; sets how fast ability can drift

Match Ingram's specification — per-player surface effects, tournament intercepts — so the ordinary arm reproduces it. Same ability model in both arms; only the estimator differs.

## Contamination

**Unit: points, in short bursts.** A minority of points are corrupted, concentrated in time — a few games of tilt, an injured stretch. Persistent changes aren't contamination: they're genuine ability change, absorbed by the random walk, or they end in dropout.

**Identifying assumption: bursts are faster than ability drifts.** σ sets the boundary. Smooth enough, and a burst can't move the walk, so it registers as corruption. Too loose, and the walk tracks the burst as ability. σ is tuned on validation, but the timescale gap has to exist in the data or nothing separates them.

**Where the gain comes from.** A robust estimator's guarantee holds wherever corruption lands. The gain *over ordinary fitting* appears when corruption is concentrated — bursts, particular player-tournaments. Diffuse corruption shifts everyone slightly, which ordinary fitting handles about as well.

**Budget unit.** Fix it: per tournament or per player-tournament. Aggregate guarantees don't imply per-player protection — each player's ability rests on relatively little data, so a burst can badly bias one player while staying inside a global budget.

## Estimation

| Arm | Fit |
|---|---|
| **Ordinary** | standard likelihood fit of the random-walk model — Ingram (2019) |
| **Robust** | same likelihood, fit with MMW or filtering |

**Robustness operates within the dynamic model.** A player whose ability genuinely declines produces a run of poor results; judged against a *static* ability, they look like corruption and get discarded. Fitting robustly within the random-walk model means only departures the drift can't explain are downweighted.

**Why point-level data helps.** A single corrupted point carries almost no information — a Bernoulli outcome is bounded. MMW and filtering detect corruption through its *collective* structure: a burst of corrupted points pushes the same ability parameters the same way, which shows up in the covariance of per-point gradients. Points get weighted individually; corruption gets detected from bursts.

**Hyperparameters, validation only**: robustness budget ε, temporal regularization σ, and forecast-contamination parameters per arm.

## Forecasting

Every forecast is a **point-by-point match simulation** — estimated abilities, matchup, service order, scoring rules — with contamination injected by a specified random rule (burst rate and severity). Repeat to estimate match-win probability. Never use the analytic iid formula; it discards burst structure.

**The contamination rule must be tuned per arm.** The two arms handle contamination oppositely, so a shared rule favors one:

| | Forecast adds contamination | Forecast adds none |
|---|---|---|
| **Ordinary fit** | double-counts — already averaged into the abilities | roughly right |
| **Robust fit** | right — restores what estimation removed | overconfident |

Tune separately on validation and compare each arm at its optimum. Diagnostic: the robust arm should tune a *higher* rate than the ordinary arm; the gap is roughly what robust estimation removed. Equal rates mean it removed nothing.

### Ablation

| | iid forecast | Tuned burst forecast |
|---|---|---|
| **Ordinary fit** | Ingram — SOTA | isolates the forecast rule |
| **Robust fit** | expected overconfident — confirms the mechanism | full method |

The claim is the **right column**: ordinary vs. robust, each with its best forecast. The top row asks whether burst-structured forecasting helps on its own — within-match dependence pulls match-win probabilities toward 50%, which can improve calibration independent of estimation.

## Where robust estimation should help, and hurt

The partition predicts a split:

- **Across tournaments — robust should win.** An injury burst last month shouldn't drag down this month's forecast.
- **Within a tournament — robust may lose.** If a player is impaired all week, the round-1 dip *is* informative about round 2, and robust estimation discards it. The random forecast rule can't know *this* player is impaired now.

Report each player's **first match of a tournament** separately from **later rounds**. Robust winning the first and losing the second is the predicted pattern — and shows exactly what condition-aware forecasting would add.

**Lag.** At onset, a short burst and a lasting change look identical, so the robust arm downweights a new slump until it persists long enough for the walk to absorb it. Breakthroughs and permanent post-injury declines are tracked late.

**Dropout.** A lasting injury often ends in retirement — a corrupted final stretch, then the player vanishes. Robust weighting handles the stretch; a returning player's estimate reflects pre-injury ability, which is wrong if the injury lasted.

## Benchmark and baselines

**Ingram (2019) protocol**: hold out a full ATP season, predict match outcomes. Reported 0.592 log loss and 68.8% accuracy on 2014. The ordinary arm with an iid forecast *is* this model.

| Baseline | Role |
|---|---|
| Ingram (2019) | ordinary arm, iid forecast — structural SOTA |
| Elo (FiveThirtyEight-style) | performance — historically ~70% accuracy |
| Bookmakers | ceiling — out of scope |

Claims are relative, not near-optimal.

**Reproduce Ingram's 2014 result first.** If a reimplementation doesn't land near 0.592, the baseline is wrong and every comparison is void. Expect small drift — the data may have been revised.

## Data

| Source | Role | Provenance |
|---|---|---|
| ATP match stats | **headline** — match-scale robustness, Ingram protocol | mirror; June 2026 snapshot; upstream SHA **not recorded** |
| Slam point-by-point | burst-scale robustness | mirror; names upstream `6febb77` (Oct 2024) |
| WTA match stats | replication | mirror |
| MCP | not used — human-charted, different instrument | live upstream |

**Granularity limits what each source can see.** Short bursts are visible only in slam point data. ATP gives serve totals per match: a burst covering 20 of a player's ~75 service points shifts the match rate by about one standard deviation, lost in noise. ATP tests robustness to match-scale corruption; the point-level burst hypothesis needs slam points.

Two ways to combine, **pre-registered before testing**:
- **Match-only** — both arms on ATP match terms. Directly comparable to Ingram; tests match-scale robustness only.
- **Hybrid** — ATP match terms as the random-walk backbone, slam point terms for slam matches. Point terms *replace* the match term for those matches, or they're double-counted. **Both arms use the identical likelihood**, or robust estimation is confounded with extra data.

Recommended: match-only as the headline, hybrid as a pre-registered second comparison.

**Upstream is gone** — Sackmann's slam/atp/wta repos 404 as of Sept 2026. Data comes from the pinned archival mirror; new tournaments don't extend it.

**License**: CC BY-NC-SA 4.0, irrevocable. Attribute Jeff Sackmann, link the original URLs as provenance, non-commercial only, derivatives carry the license.

**Chain of custody**: full 40-char SHA — resolve the recorded mirror pin (`83733358…`) against an independent one (`83733587353df8a41f2fd4f516147d5aa83f5a8d`); they diverge at character six · per-file SHA-256 in `MANIFEST.json` · hash failure is a hard stop · archive, not fork — document as a trust risk · Hugging Face copy as fallback only if bytes match · keep a cold copy. The headline dataset has the weaker provenance, so the hashes carry more weight.

### Known artifacts

| Issue | Consequence |
|---|---|
| ATP stats missing or partial; retirements truncated | Retirements are the key diagnostic — keep and flag |
| Slam data: Hawkeye courts only; missing matches mostly first-rounders | Early exits under-covered |
| Serve number, speed, rally length absent at some events | Reduced-covariate model; absence is never signal |
| Same quantity, different column across events | Harmonize by meaning |
| Schema grows over time | Covariate availability correlates with year |
| Doubles/mixed in same directory | Filter to singles |
| Slam to Oct 2024, ATP to June 2026 | Assert every slam match resolves in ATP |

Verify columns against the archived `data_dictionary.txt`. Never hardcode a schema.

### Scoring structure — the simulator must follow it

- **Game**: first to 4, win by 2; one server throughout.
- **Set**: first to 6, win by 2; tiebreak at 6–6; serve alternates by game.
- **Tiebreak**: serve alternates within it.
- **Match**: best of 5 (men's slams), best of 3 (most tour events).
- **Final-set rules changed in-sample** (~2019, ~2022). Verify dates; simulate each match in its actual format.
- **First server**: if not recorded, randomize per simulation run.
- **First vs. second serve**: mechanical covariate, in `x`.
- **Serve dominance**: server wins ≈64–65% of points (men's), high-50s (women's).

## Synthetic data

**A pass/fail gate, not a tuning source.** Synthetic data never selects hyperparameters — that would encode your assumptions into the model. It runs *before* the real test: it doesn't consume the real fold, and if the method can't recover planted trajectories, the real result is uninterpretable.

| Planted | Tests |
|---|---|
| No contamination | efficiency cost of robustness with nothing to be robust to |
| Point-level bursts | recovery; sweep burst rate, severity, length |
| Tournament-scale corruption | recovery at match scale |
| Permanent jumps | lag in tracking genuine change |

Sweep burst length against random-walk speed to map where timescale separation fails.

- **Truth from multiple generators** — robust-fit abilities, ordinary-fit abilities, pure random-walk trajectories. Truth drawn only from the robust fit biases the gate toward passing.
- **Fresh outcomes**, never spliced into real data.
- **Simulated bracket** — winners advance, so impairment causes early exit and selection on outcome is reproduced.
- **Structured bursts** alongside random ones — tied to late rounds and accumulated fatigue.
- **Realism check** — the sim world should give Ingram roughly its real log loss (~0.59). Tuning the *sim* this way isn't tuning the model.
- **Per-player recovery error**, not just aggregate.

**Gate**: robust beats ordinary on recovery under planted bursts, costs little with none planted, and lags boundedly on permanent jumps.

## Frozen-data protocol

The corpus won't grow; the test set is single-use.
- **Develop on ≤2024, test on the 2025 ATP season** — mirrors Ingram's one-season holdout and keeps slam point data in development.
- Tune on validation with rolling-origin evaluation.
- **Pre-register** every comparison — including match-only vs. hybrid — before touching test. Touch it once.
- An underpowered result is permanent.

## Plan

Gated. Don't advance past a failed gate.

0. **Audit and split.** Coverage grid, artifact audit, chain of custody (resolve the SHA discrepancy), power check, pre-registration.
1. **Reproduce Ingram.** ~0.592 on 2014. **Gate.**
2. **Synthetic gate.** Recovery, efficiency cost, lag, timescale map. **Gate.**
3. **Develop.** Fit both arms; tune ε, σ and the forecast rule per arm on validation.
4. **Diagnose on development data.** What gets downweighted — pre-retirement stretches should rank high. The gap between the arms' tuned forecast rates.
5. **Test — once.** 2025 season, chronological, hyperparameters frozen.
6. **Optional.** WTA replication.

## Evaluation

- **Strictly chronological.** Each match forecast from information available beforehand. Abilities update online through the test season; hyperparameters stay frozen.
- **Metric**: match cross-entropy (primary, as Ingram); accuracy and calibration secondary.
- **Split** first match of a tournament vs. later rounds.
- **Ablation**: estimation × forecast rule.
- **Uncertainty**: bootstrap clustered by player-tournament; tournament-level as a conservative check.
- **Trial log**: every variant, append-only. With a frozen test set it's the only defense against overfitting by search.

## Priors

| Question | Prior |
|---|---|
| Contamination exists | ~95% — retirements guarantee some |
| Ingram reproduces within small drift | ~70% |
| Synthetic gate passes | ~70% |
| Robust beats ordinary on held-out cross-entropy | ~30% |
| If so, the gain concentrates on first matches of tournaments | ~60% |
| Burst forecasting alone beats Ingram | ~35% |
| Robust beats Elo | ~15% |
| A defensible, writeable result | ~90% |

Most likely: a small robust gain concentrated on forecasts made after contaminated stretches, possibly within noise overall, with later rounds favoring ordinary.

## Outcomes

The question is *whether robust estimation improves forecasts, and where* — not *does robust win*. Every branch answers it.

- Robust beats ordinary, gap concentrated after known contamination → works for the stated reason
- Robust beats ordinary, gap not concentrated → the gain comes from something else, likely regularization
- Robust wins across tournaments, loses within → the predicted split; condition-aware forecasting is the natural extension
- No difference → contamination too rare or mild to matter, or the walk already absorbs it
- Ordinary wins → lag on genuine change costs more than robustness saves
- Burst forecasting helps both arms equally → the gain is calibration, not estimation
- Synthetic gate fails → the method can't recover on this data structure

**Negative ≠ inconclusive.** Negative: enough resolution, effect absent — a fact about tennis. Inconclusive: noise floor exceeded the effect — a fact about the design. Report which.

## References

Confirmed:
- Ingram (2019), "A point-based Bayesian hierarchical model to predict the outcome of tennis matches," *JQAS* 15(4):313–325
- Klaassen & Magnus (2001), "Are points in tennis independent and identically distributed?", *JASA* 96:500–509
- Klaassen & Magnus (2003), "Forecasting the winner of a tennis match," *EJOR* 148(2):257–267
- Kovalchik (2016), "Searching for the GOAT of tennis win prediction," *JQAS* 12(3):127–138
- Morris & Bialik (2015) — FiveThirtyEight Elo variant

Verify before citing:
- Huber (1964) — ε-contamination
- Diakonikolas et al. — filtering
- The specific MMW robust-estimation paper used
