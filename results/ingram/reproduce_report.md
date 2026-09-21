# Gate 1 — reproduce Ingram (2019)

_generated 2026-09-21T21:53:07+00:00 · trials logged: 82_

Target (2014): log loss **0.592**, accuracy **0.688**.
Chosen on 2013: sigma=0.03, prior_var=1.0.

## 2014 reproduction
- log loss **0.6044** (gap +0.0124), accuracy **0.6790**, on 2573 matches.
- **Gate PASSED** (within 0.02 of Ingram).

## 2013 tuning (log loss)
- sigma=0.01, prior_var=0.25: 0.6020 (acc 0.676)
- sigma=0.01, prior_var=0.5: 0.6016 (acc 0.678)
- sigma=0.01, prior_var=1.0: 0.6012 (acc 0.679)
- sigma=0.02, prior_var=0.25: 0.5980 (acc 0.677)
- sigma=0.02, prior_var=0.5: 0.5977 (acc 0.679)
- sigma=0.02, prior_var=1.0: 0.5974 (acc 0.679)
- sigma=0.03, prior_var=0.25: 0.5971 (acc 0.674)
- sigma=0.03, prior_var=0.5: 0.5970 (acc 0.676)
- sigma=0.03, prior_var=1.0: 0.5970 (acc 0.676)
- sigma=0.05, prior_var=0.25: 0.5984 (acc 0.674)
- sigma=0.05, prior_var=0.5: 0.5987 (acc 0.674)
- sigma=0.05, prior_var=1.0: 0.5990 (acc 0.674)
- sigma=0.08, prior_var=0.25: 0.6033 (acc 0.674)
- sigma=0.08, prior_var=0.5: 0.6040 (acc 0.674)
- sigma=0.08, prior_var=1.0: 0.6049 (acc 0.674)
- sigma=0.12, prior_var=0.25: 0.6119 (acc 0.674)
- sigma=0.12, prior_var=0.5: 0.6132 (acc 0.673)
- sigma=0.12, prior_var=1.0: 0.6148 (acc 0.672)
- sigma=0.18, prior_var=0.25: 0.6276 (acc 0.669)
- sigma=0.18, prior_var=0.5: 0.6297 (acc 0.667)
- sigma=0.18, prior_var=1.0: 0.6323 (acc 0.667)
