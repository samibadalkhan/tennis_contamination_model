# Gate 1 — reproduce Ingram (2019)

_generated 2026-09-22T04:10:24+00:00 · trials logged: 588_

Ingram 2014 reference: log loss **0.592**, accuracy **0.688**.
Hyperparameters tuned on 2013: sigma=0.03, prior_var=0.25.

## 2014 result
- log loss **0.6100** (gap +0.0180 vs Ingram), accuracy **0.6763**, mean forecast 0.614, on 2573 matches.
- **Gate PASSED** (within 0.02 of Ingram).

## 2013 tuning (log loss)
- sigma=0.03, prior_var=0.25: 0.6051 (acc 0.674)
- sigma=0.02, prior_var=0.5: 0.6052 (acc 0.679)
- sigma=0.02, prior_var=0.25: 0.6052 (acc 0.677)
- sigma=0.02, prior_var=1.0: 0.6052 (acc 0.679)
- sigma=0.03, prior_var=0.5: 0.6053 (acc 0.676)
- sigma=0.03, prior_var=1.0: 0.6055 (acc 0.676)
- sigma=0.05, prior_var=0.25: 0.6078 (acc 0.674)
- sigma=0.01, prior_var=1.0: 0.6082 (acc 0.679)
- sigma=0.01, prior_var=0.5: 0.6083 (acc 0.678)
- sigma=0.05, prior_var=0.5: 0.6084 (acc 0.674)
- sigma=0.01, prior_var=0.25: 0.6085 (acc 0.676)
- sigma=0.05, prior_var=1.0: 0.6091 (acc 0.674)
- sigma=0.08, prior_var=0.25: 0.6147 (acc 0.674)
- sigma=0.08, prior_var=0.5: 0.6158 (acc 0.674)
- sigma=0.08, prior_var=1.0: 0.6171 (acc 0.674)
- sigma=0.12, prior_var=0.25: 0.6262 (acc 0.674)
- sigma=0.12, prior_var=0.5: 0.6279 (acc 0.673)
- sigma=0.12, prior_var=1.0: 0.6300 (acc 0.672)
- sigma=0.18, prior_var=0.25: 0.6464 (acc 0.669)
- sigma=0.18, prior_var=0.5: 0.6490 (acc 0.667)
- sigma=0.18, prior_var=1.0: 0.6521 (acc 0.667)
