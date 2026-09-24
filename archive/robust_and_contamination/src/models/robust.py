"""Robust arm: the shared random-walk ability filter with the Huber-style cap on.

The ONLY divergence from the ordinary arm (``ordinary.py``) is ``robust_c``: each
per-match innovation is saturated at ``robust_c * sqrt(H)`` (H = observed
information), so a short burst of corrupted points cannot yank a player's skill.
Same likelihood, same data, same random-walk dynamics — see CLAUDE.md.

``robust_c`` is the Huber tuning constant (the clip threshold), NOT the
contamination fraction epsilon; smaller = more robust, less efficient. It is a
HYPERPARAMETER: tuned per arm on validation (Stage 4), NEVER on synthetic data
and NEVER on test. The synthetic gate (Stage 2) therefore uses a small set of
PRE-DECLARED fixed caps purely to show the mechanism works across a plausible
range — it does not select one.

    DEFAULT_CAPS = (1.345, 2.5)
      1.345 — Huber's classic 95%-Gaussian-efficiency constant (tight)
      2.5   — a looser cap (more efficient, less aggressive downweighting)
"""

from __future__ import annotations

from src.models.ability import OnlineAbility

DEFAULT_CAPS = (1.345, 2.5)


def make(alpha_by_surface: dict, sigma: float, robust_c: float,
         prior_var: float = 1.0) -> OnlineAbility:
    if robust_c is None:
        raise ValueError("robust arm needs a finite robust_c; use ordinary.make for the uncapped fit")
    return OnlineAbility(alpha_by_surface, sigma=sigma, prior_var=prior_var,
                         robust_c=float(robust_c))
