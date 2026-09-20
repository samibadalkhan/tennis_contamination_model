# Tennis ability estimation under ε-contamination

Estimating tennis player ability θ when the point-generating process is a
mixture `F = (1-ε)·F_θ + ε·H` — a clean ability process contaminated by a
distinct subpopulation (strategic coasting, injury, tilt, choke) at rate ε.

The full hypothesis, the departure space (D1–D6), the discriminators, the
staged plan, and the prior calibration are in
[`tennis-contamination.md`](tennis-contamination.md). Project-specific traps and
non-negotiables are in [`CLAUDE.md`](CLAUDE.md). **Read both before writing
modeling code.**

## Layout

```
data/          # gitignored raw corpus; populated by src/fetch.py
src/
  fetch.py     # download + pin + record provenance  (the only way data enters)
  load.py      # ingest + schema harmonization
  blocks.py    # blocking, per-slam-year audit
  models/      # baseline (K&M) -> beta_binomial (D4) -> mixture (D5) -> hmm
  evaluate.py  # splits, clustered bootstrap, metrics
  trials.py    # append-only variant log
notebooks/     # exploration only; never load-bearing
results/
```

Stages are **gated, not parallel** (see `tennis-contamination.md` "Stages").
Everything under `src/models/` past the K&M baseline is intentionally a stub
until the Stage 0.5 positive control passes. Keep `load → blocks → models →
evaluate` independently runnable.

## Getting the data

The corpus is **not committed** (CC BY-NC-SA 4.0, and it is large). Fetch it:

```bash
python -m src.fetch            # all groups: slam, atp, wta, mcp  (~450 MB)
python -m src.fetch slam atp   # just the primaries needed for Stages 0–1
python -m src.fetch --list     # show the file plan without downloading
python -m src.fetch --verify   # re-hash local files against MANIFEST (hard stop on mismatch)
python -m src.fetch --source hf  # fetch via the Hugging Face fallback
```

`src/fetch.py` pins exact commit SHAs, records a **SHA-256 per file** in
`data/MANIFEST.json`, and writes provenance/license notes (`ATTRIBUTION.txt`,
`DATA_LICENSE.txt`) alongside the data.

Data layout after a full fetch:

```
data/
  slam_pointbypoint/   # PRIMARY: *-points.csv + *-matches.csv per slam-year (singles only)
  atp/                 # atp_matches_2011..2024, players, rankings (10s/20s)
  wta/                 # women's equivalent
  match_charting/      # charting-m-{matches,points-2010s,points-2020s}.csv
  MANIFEST.json  ATTRIBUTION.txt  DATA_LICENSE.txt
```

### Provenance (read `ATTRIBUTION.md` in full)

The original `JeffSackmann/tennis_{slam_pointbypoint,atp,wta}` repos are **404
as of Sept 2026**. `slam`/`atp`/`wta` come from the archival mirror
`Aneeshers/tennis-sackmann-archive` (pinned by full 40-char SHA, verified
against the live `main` HEAD); MCP comes from live upstream. Provenance is
**uneven**: `slam` names its upstream commit (good), but `atp`/`wta` are a June
2026 snapshot with **no recorded upstream SHA** (weak — a known trust risk). The
mirror is an **archive, not a fork** — no shared git history, so it cannot be
diffed against upstream; integrity rests on the per-file SHA-256, not lineage.
Keep an independent cold copy; do not assume the mirror persists.

### The dataset is frozen

Slam point-by-point ends **Oct 2024** — there will be no 2025–26 slam data to
validate on. The **train-early / test-late split declared at Stage 0 is the only
out-of-sample period this project will ever have.** Declare it once, carefully,
and never re-split.

### Snapshot-date mismatch

Slam is frozen at Oct 2024 while ATP/WTA run to June 2026. Benign for the
retirement-label join (slam → atp), since ATP is a superset — but **assert it**:
every slam `match_id` must resolve to an ATP match. Unmatched rows are a join
bug or a name-normalization failure (accents/hyphenation cluster non-randomly);
**investigate, do not drop.**

### Known coverage gap

The slam point-by-point corpus is **not uniformly complete** across 2011–2024:
2011–2019 and 2021 carry all four slams, 2020 is missing Wimbledon (cancelled),
and 2022–2024 carry only two slams each in this snapshot. Verify coverage in the
per-slam-year audit (`blocks.py`) before assuming sample size — the spec calls
this out explicitly, and missingness here is non-random.

### Schema

Verify column names against the `data_dictionary.txt` and `UPSTREAM_README.md`
that the mirror preserves in each directory. **Never hardcode a schema** — it
varies by slam and year.

## Attribution & license

Data compiled by **Jeff Sackmann**, used under **CC BY-NC-SA 4.0**
(non-commercial, attribution required, share-alike). See
[`ATTRIBUTION.md`](ATTRIBUTION.md). This governs the data only; project code is
licensed separately.
