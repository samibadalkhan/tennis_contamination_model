# Attribution, provenance & data license

## Data

All tennis data used in this project was **compiled by [Jeff Sackmann](https://github.com/JeffSackmann)**
and is used under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/))**.

Binding conditions, all honored here:

- **Attribution** — credit Jeff Sackmann **as compiler** and link the original
  source repositories (below) as provenance origins.
- **NonCommercial** — no commercial use. This **excludes carrying the data into
  commercial research**. This is a non-commercial research project; keep it that way.
- **ShareAlike** — any redistributed data **or derived dataset** must carry the
  same CC BY-NC-SA 4.0 license.

**The grant is irrevocable.** The upstream repositories have been removed (see
below), but license withdrawal is not possible under CC 4.0: the grant on copies
already distributed stands. Our use of the archived copies is therefore fine.
Jeff Sackmann enforces this license — respect it.

### Sources (provenance origins)

The original repositories are **now dead** (404 as of Sept 2026) but are linked
here as the provenance origins the license attribution requires:

| Dataset | Original repository | Status | Role |
|---|---|---|---|
| Grand Slam point-by-point (2011–2024) | `JeffSackmann/tennis_slam_pointbypoint` | **404** | Primary point-level dataset |
| ATP results / rankings / players | `JeffSackmann/tennis_atp` | **404** | Retirement labels (Stage 0.5), ranking covariates (weak-player test) |
| WTA results / rankings / players | `JeffSackmann/tennis_wta` | **404** | Replication / generalization |
| Match Charting Project (shot-by-shot) | `JeffSackmann/tennis_MatchChartingProject` | live | Rally length / error type — strategic vs. involuntary |

### How it was retrieved, and its provenance

Because `tennis_slam_pointbypoint`, `tennis_atp`, and `tennis_wta` are gone,
`slam`/`atp`/`wta` are pulled from the archival mirror
[`Aneeshers/tennis-sackmann-archive`](https://github.com/Aneeshers/tennis-sackmann-archive)
(also on the [Hugging Face Hub](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive)),
pinned by full 40-character commit SHA. The Match Charting Project is pulled
directly from `JeffSackmann`, also pinned. **Do not re-point the fetch script at
a `JeffSackmann/...` URL for slam/atp/wta** — it will 404, and silently
substituting a source breaks provenance. `src/fetch.py` asserts this.

**Provenance is uneven, and the manifest records it honestly:**

| Group | Snapshot | Upstream commit | Provenance |
|---|---|---|---|
| `slam` (primary) | Oct 2024 | `6febb77` (abbrev.; full SHA unrecoverable — upstream gone) | **Good** — mirror names the upstream commit |
| `atp` | June 2026 | **not recorded** | **Weak — known trust risk** |
| `wta` | June 2026 | **not recorded** | **Weak — known trust risk** |
| `mcp` | Sept 2026 | `1813a13…` (full, live) | Good — pinned to a live upstream commit |

**The mirror is an archive, not a fork.** It shares no git history with the
original repositories, so its contents **cannot be diffed against upstream**.
Integrity therefore rests on **content hashes, not lineage**.

### Integrity — how trust is anchored

A commit pin to a third-party repo is not self-verifying if that repo also
disappears. So:

- **Full 40-char SHAs only**, never abbreviations. The mirror pin
  `83733587353df8a41f2fd4f516147d5aa83f5a8d` was **verified on 2026-09-20
  against the live `main` HEAD** (byte-identical).
- **Per-file SHA-256** is recorded in `data/MANIFEST.json`.
- **Verify before use:** `python -m src.fetch --verify` re-hashes every local
  file against the manifest. **A failed check is a hard stop, not a warning.**
- **Second fallback:** the Hugging Face copy of the same archive
  (`hf_url` per file in the manifest; `--source hf`), usable **only if its bytes
  reproduce the recorded hashes**.
- **Keep an independent cold copy.** Do not assume the mirror persists — it may
  vanish the way upstream did.

## Code

Project source code is licensed separately from the data (see repository
`LICENSE` / `README.md`). The CC BY-NC-SA 4.0 license above governs the **data
only**.

## References

- Klaassen & Magnus (2001), "Are Points in Tennis Independent and Identically
  Distributed? Evidence from a Dynamic Binary Panel Data Model," *JASA*
  96(454):500–509.
- Klaassen & Magnus (2003), "Forecasting the winner of a tennis match," *EJOR*
  148:257–267.
