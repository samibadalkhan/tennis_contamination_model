# Archived robust and contamination work

This directory contains the closed branches that no longer belong to the
active forecasting program:

- robust and approximate ability estimators;
- ordered-point block filtering and early real-data tests;
- synthetic contamination and recovery experiments;
- simplified Ingram checkpoints and simulated forecasts;
- the original contamination audit and design documents;
- the queued burst-regime grid, which was not run;
- stale status and helper scripts tied to those branches.

The branches were archived after the full model showed no robust-estimation
advantage and the real point-block candidate failed its decision rule. The
files remain for provenance and to keep the append-only trial ledger honest.
They are historical snapshots, not supported entry points, and their relative
imports may no longer resolve from inside the archive.

Mixed full-model artifacts needed to trace the selected ATP temperature remain
under `results/independent/`. Large archived posterior arrays are ignored by
the local `.gitignore`; their JSON metadata and diagnostics remain reviewable.
The branch's retired caches and compiled historical source were moved from
`data/archive/` to `data/` here during the final cleanup.
