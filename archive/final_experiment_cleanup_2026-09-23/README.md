# Final experiment cleanup archive

This directory records artifacts removed from the active experiment tree on
September 23, 2026. The cleanup kept the dependency closure of the final 2014
Ingram reproduction, ATP 2025 frozen and online forecasts, MCMC seed rerun,
mechanism splits, corrected WTA 2025 run, direct point-dependence analysis, and
point-structure follow-up.

## Archived experiment artifacts

| Original path | Archived path | Reason |
|---|---|---|
| `results/wta_2025/initial_noncompliant/` | `results/wta_2025/initial_noncompliant/` | First WTA run omitted the frozen team-event exclusion and was superseded by `results/wta_2025/corrected/`. |
| `data/wta_2025_cache/` | `data/wta_2025_cache/` | Posterior forecast cache used only by the noncompliant WTA run. |
| `results/wta_2025/FAILED_PARALLEL_ATTEMPT.json` | `results/wta_2025/FAILED_PARALLEL_ATTEMPT.json` | Failed parallel launch record with no scored result. |
| `results/.gitkeep` | `placeholders/results.gitkeep` | Obsolete placeholder after the results tree became populated. |

The superseded implementation at `src/independent/run_wta_2025.py` remains in
the active tree solely because `analysis/point_structure_followup/FROZEN.json`
hash-locks that exact path. The failed ATP online ordering freeze similarly
remains at its recorded path because it is a protected provenance input. The
active README marks both as provenance-only.

## Runtime debris

`runtime/` contains Python and pytest caches, Matplotlib's font cache, worker
logs, and a completed-run status file. These are not scientific inputs or
results and are retained only to make this cleanup reversible.

The raw source corpus remains under `data/`. It is retained as one frozen
provenance bundle because `data/MANIFEST.json` and `python -m src.fetch
--verify` cover the full set. The processed bundle also remains complete
because its frozen build record names all four derived tables.

The final point-structure freeze contains 36 protected path entries. Thirty-five
still match. `CLAUDE.md` was rewritten afterward to describe the final project
scope, so the immutable freeze and result JSON retain its earlier hash as an
explicit historical record.
