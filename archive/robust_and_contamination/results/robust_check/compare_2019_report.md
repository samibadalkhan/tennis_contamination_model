# Ordinary vs. robust on 2019 (development read — NOT test)

_generated 2026-09-22T04:13:06+00:00 · 2603 matches · sigma=0.03 prior_var=1.0 (frozen, shared) · iid forecast both arms · trials 614_

Same likelihood, same data, same forecast; only the Huber cap differs. Positive gain = robust better. Bootstrap clustered by tournament.

- **ordinary**: log loss **0.6563**, acc 0.633
- **robust_1.345** (cap 1.345): log loss **0.6683**, acc 0.623 — paired gain vs ordinary **-0.0120** [-0.0187, -0.0053] (significant)
- **robust_2.5** (cap 2.5): log loss **0.6589**, acc 0.632 — paired gain vs ordinary **-0.0026** [-0.0048, -0.0003] (significant)

## First match of tournament vs. later rounds (paired gain)

- **robust_1.345**: first (1195) -0.0110 [-0.0219, +0.0004]  ·  later (1408) -0.0129 [-0.0187, -0.0068]
- **robust_2.5**: first (1195) -0.0048 [-0.0090, -0.0008]  ·  later (1408) -0.0007 [-0.0031, +0.0017]
