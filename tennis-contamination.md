# Purifying player-ability estimates under ε-contamination

## Object of study

We observe point outcomes nested in games, sets, matches. The target is θ = player ability. The generating process is unknown and **is not assumed i.i.d.**

## Motivating framing

Klaassen & Magnus (2001, *JASA* 96:500–509) tested the i.i.d.-given-server null on ~90,000 Wimbledon points and found points are neither independent nor identically distributed — winning the previous point raises the chance of winning the current one, and servers do worse at "important" points, with both effects stronger for weaker players. They also concluded the deviations are *small*, so i.i.d. remains a decent approximation.

**The claim here is that "small on average" is compatible with "large where it occurs."** A small ε with a large local effect regresses down to a modest smooth coefficient. Fit outcome on importance across all points and you recover a small negative β even if the truth is that most important points are played normally and a minority collapse. The literature has only ever fit the smooth description. This project fits the other one and compares.

## The departure space

Five ways i.i.d.-given-server fails. Not mutually exclusive, and each can masquerade as the others.

| | Departure | Mechanism |
|---|---|---|
| **D1** | Non-identical across contexts | score state / importance, first vs. second serve, surface, round |
| **D2** | Within-match drift | fatigue, warm-up, tactical adjustment |
| **D3** | Serial dependence | momentum / hot hand, or negative dependence |
| **D4** | Overdispersion | ability is a random effect — good days and bad days |
| **D5** | **Contamination** | a distinct subpopulation from a different process |
| **D6** | **Instrumental contamination** | recording artifacts, coverage gaps, schema drift — see below |

**D1–D3 are partly candidates for redescription as D5**, not just nuisance structure to control away. That is the sharpened version of the hypothesis.

Scope it: **mechanical** covariates (first vs. second serve, surface, round) are genuine state changes and stay in F_θ — the mixture never touches them. Only the **psychological/strategic** residue (importance, clutch, momentum) is contamination-eligible.

## Claim

```
F = (1 - ε)·F_θ  +  ε·H
```

**D5 is real and separable**, and an estimator that downweights H recovers θ closer to true ability than one that does not — including on effects the literature currently models as smooth.

### Two flavors of H

They make opposite predictions and may need separate components.

- **Strategic** (intentional): cheap points at 40–0, coasting in a set already lost, going for broke when down a break. Predictable from score state. Likely **variance-increasing** — more aces *and* more double faults — rather than mean-decreasing.
- **Involuntary**: injury, tilt, choke. Directional (strictly worse), less predictable ex ante, and the flavor that should correlate with external labels.

If both are present, a single left-skewed H will not fit and ε should decompose.

## Discriminators

Overdispersion alone doesn't separate D4 from D5 — both produce excess variance. In increasing evidential weight:

1. **Clustering.** D4 is exchangeable; excess spreads evenly. D5 arrives in *runs*. Compare block-level service-win rates against beta-binomial vs. two-component mixture. Contamination ⇒ excess concentrates in a minority of blocks.

2. **Asymmetry.** Involuntary D5 is directional; D4 is symmetric. Cheapest test — run first. (Note: strategic D5 may be symmetric-but-fat, so a symmetric result doesn't kill D5 outright, it argues against the involuntary flavor.)

3. **Bimodality at high-importance points.** Aimed at a specific documented effect rather than generic residuals. Smooth model ⇒ uniform shift across all high-importance points. Contamination ⇒ normal bulk plus a minority of collapsed blocks.

4. **Weak-player scaling — the K&M test.** They found the effect is stronger for weaker players. Smooth model: weaker players have larger **β** (pressure shifts everyone, more so the weak). Contamination: weaker players have larger **ε** (they fall apart more *often*), but conditional on the clean state their ability is unbiased. Estimate both and see which parameterization the weak-player pattern loads onto.

5. **External correlation.** Decisive in principle, but see the label-scarcity caveat below. D4 has no reason to align with retirements or medical timeouts. Involuntary D5 does.

## Scoring structure — what the code needs to respect

Strictly nested: **point → game → set → match.**

- **Game**: first to 4 points, win by 2 (deuce/advantage ⇒ unbounded length). **One player serves the whole game** — which is why the game is the natural contamination block: serve ability is constant across it, so a game-level anomaly can't be explained by the server changing.
- **Set**: first to 6 games win by 2; 6–6 ⇒ tiebreak. Serve alternates by game.
- **Tiebreak**: first to 7, win by 2. Serving alternates *within* the tiebreak (one point, then two at a time), **breaking the one-server-per-game invariant**. Handle separately or exclude from game-level blocking.
- **Match**: best of 5 (men's slams), best of 3 (women's slams, most tour).
- **Sub-point**: two serve attempts per point. First vs. second serve is a large, mechanical D1 covariate — use it, don't let the mixture explain it.
- **Serve dominance**: server wins ≈64–65% of points (men's tour), high-50s (women's). Ability decomposes into **serve + return**, not one scalar.
- **Match-win probability** recurses up the scoring tree from point probabilities (Klaassen & Magnus 2003). This is what turns the point model into held-out *match* predictions.
- **Structural break in-sample**: final-set rules changed during 2011–2024 (advantage sets at some slams; Wimbledon 12–12 tiebreak and AO super-tiebreak ~2019; slams standardized on a 10-point final-set tiebreak ~2022). **Verify exact dates against the data.** It changes the length and composition of deciding sets mid-sample.

## Data

All compiled by Jeff Sackmann. **License: CC BY-NC-SA 4.0** — attribute Sackmann as compiler, non-commercial only (this excludes commercial research), any redistributed or derived dataset carries the same license. The grant is **irrevocable**: the upstream repos being gone does not revoke the license on already-distributed copies, so our use is fine. He enforces this; respect it.

| Repo (provenance origin) | What | Use for |
|---|---|---|
| `tennis_slam_pointbypoint` **(404)** | Grand Slam point-by-point, 2011–2024, `*-points.csv` + `*-matches.csv` per slam-year (~166 CSVs) | **Primary** point-level dataset |
| `tennis_atp` **(404)** | Match results, rankings, players; `atp_matches_YYYY.csv` | External labels (retirements), ranking covariates for the weak-player test |
| [`tennis_MatchChartingProject`](https://github.com/JeffSackmann/tennis_MatchChartingProject) (live) | Shot-by-shot, 5,000+ matches: `charting-m-points.csv`, `charting-m-matches.csv` | Rally length, error type — separates strategic from involuntary H |
| `tennis_wta` **(404)** | Women's equivalent | Replication / generalization |

### Sourcing & provenance (changed — the upstream repos are gone)

As of Sept 2026 `tennis_slam_pointbypoint`, `tennis_atp`, and `tennis_wta` all **404**; only `tennis_MatchChartingProject` is still live upstream. **Do not re-point the fetch at a `JeffSackmann/...` URL for slam/atp/wta** — it will fail, and silently substituting a source breaks provenance.

- `slam`/`atp`/`wta` come from the archival mirror [`Aneeshers/tennis-sackmann-archive`](https://github.com/Aneeshers/tennis-sackmann-archive) (also on HuggingFace), pinned by **full 40-char SHA**. `mcp` comes from live upstream, pinned.
- Provenance is **uneven**: `slam` (primary) names its upstream commit `6febb77`, Oct 2024 — **good**. `atp`/`wta` are a June 2026 snapshot with **no recorded upstream SHA** — **weak; a known trust risk to flag in any writeup**.
- The mirror is an **archive, not a fork** — no shared git history, so it **cannot be diffed against upstream**. Integrity rests on **content hashes, not lineage**: `src/fetch.py` records a **SHA-256 per file** in `data/MANIFEST.json`, verifies before use (`--verify`; a mismatch is a hard stop), keeps the HuggingFace copy as a byte-verified second fallback, and the mirror pin is verified against the live HEAD. **Keep an independent cold copy; do not assume the mirror persists.**

**The dataset is frozen and cannot be extended.** Slam point-by-point ends Oct 2024; there will be **no 2025–26 slam data** to validate on later. The **train-early / test-late split declared at Stage 0 is the only out-of-sample period this project will ever have** — declare it once, carefully, and never re-split.

**Snapshot-date mismatch.** Slam is frozen at Oct 2024 while ATP/WTA run to June 2026. Benign for the retirement-label join (slam → atp, ATP is the superset) but **assert it**: every slam `match_id` must resolve to an ATP match. Unmatched rows are a join bug or a name-normalization failure — **investigate, do not drop**.

**Fields needed**: match id, set/game/point numbering, server, point winner, running score, serve number, elapsed time. Verify column names against the `data_dictionary.txt` and `UPSTREAM_README.md` the mirror preserves in each directory — do not hardcode from memory.

**External labels — thinner than they look.** Genuinely external: **retirements** (from `tennis_atp` `score` strings, e.g. `RET`, joined by player + tournament + year) and **medical timeouts** where charting data records them. That is close to the whole list.

Note there are **no dead rubbers in slam singles** — that is a Davis Cup / team-tie concept; every set in a slam match still matters. What exists instead is *effectively decided* states where a player has given up, and those can only be inferred from score, which is circular. This materially weakens discriminator 5: plan around retirements as the primary label and treat score-inferred contamination as suggestive, not confirmatory.

Label *before* fitting — validation, never features.

## D6 — data artifacts

Recording artifacts produce anomalous block rates a mixture model can't tell from injury or tilt. Untreated, ε absorbs data quality. Most likely route to a false positive.

### Known issues

| Issue | Consequence |
|---|---|
| Coverage is Hawkeye-court only; missing matches are mostly first-rounders | **Cuts against the hypothesis.** First rounds are where blowouts and giving-up live — i.e. strategic D5. ε biased low. Declare as a limitation. |
| Serve speed, **first/second serve**, rally length absent at some tournaments | Your cleanest D1 control isn't universal. Use a reduced-covariate F_θ where it's missing, or its absence reads as contamination. |
| Same quantity, different column across events | Harmonize by meaning, not column name. |
| Schema grows over time (distance run, serve/return depth added later) | Covariate availability correlates with year. |
| Doubles and mixed share the directory (`-doubles`, `-mixed`) | Filter to singles. |
| MCP skews to popular players, high-leverage, televised matches | ε from MCP doesn't transfer to the population. |

### Check before pooling

Per slam-year: column presence, missingness, value ranges, points-per-match and points-per-game distributions.

Also: sentinel/zero encodings for unknown server or winner; `match_id` parse stability across years; truncation for retirements and walkovers — retirement truncation **is** the Stage 0.5 control, don't drop it.

`tennis_atp` joins are name-based. Accents, hyphenation and name changes fail non-randomly (non-anglicized names cluster). Audit the unmatched set.

### Mitigations

- **Slam × year fixed effects** in F_θ; fit per-slam as a cross-check. Surface differs (grass highest serve dominance, clay lowest) — unmodeled, it loads onto the contamination component.
- **Key diagnostic**: does ε track data-quality proxies (missingness, schema version, slam) more than player identity? If yes, you're measuring the recorder.
- **Sensitivity run**: repeat on the cleanest single slam-year. Effect vanishes → it was D6.

## Stages

**Stage 0 — i.i.d. sanity check.** Diagnostic only, not a foundation. Fit i.i.d.-given-server (server + returner effects, regularized), then establish:
- Does overdispersion exist? Block-level variance vs. binomial.
- Serial dependence? Runs test, autocorrelation of outcomes and squared residuals.
- Symmetric or left-skewed excess?
- **Noise floor**: clustered-bootstrap interval on held-out log-loss. If it exceeds any plausible ε-effect, the design is underpowered — redesign before continuing.

**Stage 0.5 — positive control. Run before building anything else.** Points in the games preceding a retirement are the one near-certain contaminated sample available. Does service-win rate visibly degrade there relative to the same player's baseline? If a detector cannot recover degradation on known-contaminated data, it does not work — and you learn that for hours of effort rather than weeks. Pair with an importance-conditioned histogram of block rates (shift vs. bimodality). These two together decide whether Stages 2–4 are worth building.

**Stage 1 — K&M baseline.** Rebuild the smooth specification: importance / score state, lagged outcome, within-match time, random effects for player quality. This is the competitor, and it must be fit well — a weak version makes the whole comparison meaningless.

**Stage 2 — D4 vs D5.** Beta-binomial / random-effect against two-component mixture on block-level rates, both on top of F_θ. Run discriminators 1–4.

**Stage 3 — mixture.** Block-level latent contamination indicator by EM. Points remain the unit of observation.

**Block size is a power decision, not just a conceptual one.** A service game is ~6–8 points: at p≈0.65 the SD of the block rate is ~0.18, so a 65%→45% collapse is barely 1 SD — conceptually clean (constant server) but statistically hopeless for per-block classification. Per-player service points *within a set* (~30–35) brings SD to ~0.08 and puts a 20pp drop at ~2.5 SD. Use set-level blocks as the default; games only for diagnostics. Either way, **do not expect confident per-block labels at any size** — EM recovers ε from the aggregate *shape* of the block-rate distribution, not by classifying blocks individually. Report it that way.

Then discriminator 5: do low-weight blocks land on external labels?

**Stage 4 (optional) — latent HMM.** Two-state chain over points; clean state emits Bernoulli(p_θ), contaminated state its own process, persistent transitions. EM responsibilities become learned per-point weights. Yields ε as the stationary distribution and episode length as the relaxation time.

## Guardrails

- **Split by match and by time.** Never by point. Ability drifts ⇒ train early, test late.
- **Metric — read the target-mismatch note below before choosing one.** Primary: log-loss / Brier of the **full mixture** (clean + ε·H) on all held-out points. Confirmatory: the **clean component** scored on externally-labeled-clean points only. Secondary: held-out match outcomes via the scoring tree. Disagreement between point-level and match-level is informative.
- **Clustered bootstrap over matches.** Under dependence, effective n ≈ match count, not point count.
- **Do not tune robustness constants or ε on test.** Validation only; report sensitivity.
- **Baseline is Stage 1, not Stage 0.** Beating i.i.d. proves almost nothing.

## Failure modes

- **Target mismatch — the one most likely to sink a true hypothesis.** The test set is contaminated at the same rate as training. So the estimate that best predicts *raw* future outcomes is the contaminated one — it has the right marginal. Purifying θ̂ makes it a better estimate of *ability when playing normally* and a **worse** predictor of unconditional outcomes. A naive "does robust θ̂ beat MLE θ̂ on held-out log-loss" test therefore penalizes the hypothesis even when it is true. Fixes, in order: (a) score the **full mixture** against a single-component model — if D5 is real the mixture should win, because it models contamination instead of averaging it in; (b) score the **clean component on labeled-clean points only**; (c) treat θ̂ as a rating and validate externally — does the purified rating forecast *future matches* better than contaminated θ̂ or Elo? Do (a) as primary, (b) as confirmation.
- **Identifiability — the central risk, and worse under the sharpened hypothesis.** You are claiming a mixture better explains data the field has fit with smooth coefficients for two decades. Those models mimic each other by construction. Discriminator 4 and the external labels carry the argument; likelihood comparison will not.
- **D6 leakage.** Instrumental artifacts land in the contaminated component and inflate ε. The data-quality-proxy diagnostic and the clean-subset sensitivity run are the defenses.
- **Circularity.** Define contamination as "points the model fits badly" and you will always find it. Labels come from outside the model.
- **Mimicry.** A two-component mixture *is* one way to build an overdispersed distribution.

## Prior calibration

Rough expectations, set before running anything, so results aren't graded against optimism:

| Question | Prior |
|---|---|
| Contamination exists at all | ~95% — a player who retires was hurt beforehand. Definitionally a second process. Not the interesting question. |
| It beats D4 as a *description* | ~25–30% — beta-binomial absorbs this kind of excess well, and mixture-vs-overdispersion likelihood surfaces are notoriously flat on short binomial blocks. |
| Purification improves held-out prediction | ~15% naively — but see **target mismatch** above; the naive metric is mis-specified against the hypothesis. |
| Retirement positive control fires | ~75% — and if it doesn't, stop. |
| Some defensible, writeable result | ~90% |

Most likely single outcome: Stage 1 covariates and random effects absorb most of the excess. Plan for that; it is still a good project if the power analysis shows the effect could have been detected.

**Data coverage**: the slam point-by-point corpus is **not uniformly complete** across 2011–2024 in the pinned snapshot — 2011–2019 and 2021 carry all four slams, 2020 is missing Wimbledon (cancelled), and 2022–2024 carry only two slams each. Confirm the coverage grid in the per-slam-year audit before assuming sample size. Even so, total points still exceed K&M's ~90,000 by well over an order of magnitude, so power is not the binding constraint — identifiability and labels are.

## Outcomes

The question is *which departure dominates*, not *does D5 win*. Every branch answers it. None is a failure condition.

This matters mechanically, not morally: a design that treats one branch as the goal creates pressure to reach it, which is the incentive that produces overfit results. The discriminators only work if you're indifferent to how they come out.

- Mixture beats Stage 1 and beats D4 out-of-sample, **and** low-weight blocks track external labels → D5 separable; contamination dominates.
- Mixture wins on likelihood but low-weight blocks don't track labels → the component is absorbing misspecification, not isolating mechanism. The more interesting version of the identifiability problem.
- Weak-player pattern loads on β rather than ε → the smooth description is right.
- Beta-binomial wins → the departure is a random effect. D4 over D5.
- Stage 1 absorbs most of the excess → it was D1–D3 as ordinary structure. Plausibly the single most likely outcome.

**Negative is not the same as inconclusive.**
- **Negative** — the design had resolution to detect a plausible effect and it isn't there. A fact about tennis.
- **Inconclusive** — noise floor exceeded effect size. A fact about the design.

Stage 0's power check exists to tell these apart *before* the work; which one you got goes in the writeup either way.

Record every model variant tried. Trial count is required for any honest significance claim — and it's what lets you report a negative without anyone asking whether you stopped searching early.

## References

- Klaassen & Magnus (2001), "Are Points in Tennis Independent and Identically Distributed?", *JASA* 96(454):500–509 — the null, and the smooth competitor.
- Klaassen & Magnus (2003), "Forecasting the winner of a tennis match", *EJOR* 148:257–267 — match-win probability from point probabilities.
- Morris (1977) — formal definition of point importance.
- Jackson & Mosurski (1997) — cross-set psychological momentum.
- Barnett & Clarke (2005) — estimating serve probabilities from historical data.
