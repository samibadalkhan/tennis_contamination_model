# Ordinary vs. robust on 2014 (development read — NOT test)

_generated 2026-09-22T04:12:56+00:00 · 2573 matches · sigma=0.03 prior_var=1.0 (frozen, shared) · iid forecast both arms · trials 612_

Same likelihood, same data, same forecast; only the Huber cap differs. Positive gain = robust better. Bootstrap clustered by tournament.

- **ordinary**: log loss **0.6128**, acc 0.679
- **robust_1.345** (cap 1.345): log loss **0.6211**, acc 0.679 — paired gain vs ordinary **-0.0083** [-0.0167, +0.0004] (within noise)
- **robust_2.5** (cap 2.5): log loss **0.6162**, acc 0.682 — paired gain vs ordinary **-0.0034** [-0.0086, -0.0000] (significant)

## First match of tournament vs. later rounds (paired gain)

- **robust_1.345**: first (1170) -0.0035 [-0.0190, +0.0137]  ·  later (1403) -0.0123 [-0.0219, -0.0055]
- **robust_2.5**: first (1170) -0.0036 [-0.0083, +0.0009]  ·  later (1403) -0.0032 [-0.0097, +0.0009]
