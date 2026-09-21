# archive/ — superseded contamination-detection design

Frozen snapshots from the project's **first design**, which aimed to *detect and
separate* ε-contamination in tennis point data (mixture model `F=(1−ε)F_θ+εH`,
overdispersion / asymmetry / serial diagnostics, EM, a retirement positive
control). The project has since pivoted (see the current `CLAUDE.md` /
`tennis-contamination.md`) to **robust vs. ordinary estimation of time-varying
serve/return ability, judged on held-out match forecasting** (Ingram protocol).

These files are kept for provenance and to keep the append-only trial log
(`results/trials.jsonl`, still live at the repo root) honest. **They are not
part of the current pipeline and are not meant to run** — they reference the old
slam-year split (`train`/`val`/`test`) that the new `splits.py` retired, and
some import modules that also moved here.

- `src/stage0.py` — i.i.d.-given-server diagnostic (overdispersion, serial
  dependence, asymmetry, noise floor). The static model is replaced by the new
  random-walk `ability` model.
- `src/stage05.py` — retirement positive control. Retirements are still used in
  the new design, but as a *diagnostic of what the robust arm downweights*, not
  a detector test. The label-extraction/join logic lives on in `build_processed`.
- `src/blocks.py` — set-level service blocking for the EM mixture.
- `src/investigate_momentum.py`, `src/plots.py` — the momentum artifact
  investigation and Stage 0 figures.
- `src/models/{baseline,beta_binomial,mixture,hmm}.py` — old model stubs.
- `results/stage0/`, `results/stage05/` — the results those produced.

What survived into the new design (still in `src/`): `fetch.py`, `load.py`
(incl. `canonical_name`), `build_processed.py`, `splits.py` (re-declared),
`evaluate.py`, `trials.py`, `util.py`.
