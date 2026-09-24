# Full method: simulated forecast, per-arm tuned — 2014 (development, NOT test)

_generated 2026-09-22T04:18:44+00:00 · tuned on 2013 · S=200 sims · sev=1.5 len_frac=0.35 · sigma=0.03 prior=0.25 · trials 629_

Reference — ordinary + iid analytic forecast: log loss **0.6100**

Diagnostic (design): the robust arm should tune a HIGHER contamination rate than ordinary; equal rates mean robustness removed nothing.

| arm | tuned rate | log loss | acc | mean pred | paired gain vs ordinary-sim |
|---|---|---|---|---|---|
| ordinary | 0.35 | 0.5984 | 0.668 | 0.590 | — |
| robust_1.345 | 0.2 | 0.6041 | 0.672 | 0.598 | -0.0057 [-0.0113, -0.0002] |
| robust_2.5 | 0.35 | 0.5993 | 0.677 | 0.591 | -0.0009 [-0.0045, +0.0028] |

## Tuning curves (log loss vs contamination rate, on 2013)

- **ordinary**: 0.0:0.6040  0.1:0.5977  0.2:0.5946  0.35:0.5919
- **robust_1.345**: 0.0:0.6135  0.1:0.6051  0.2:0.5990  0.35:0.5994
- **robust_2.5**: 0.0:0.6075  0.1:0.5997  0.2:0.5950  0.35:0.5924
