"""Ordinary arm: the shared random-walk ability filter with NO innovation cap.

This is the standard online assumed-density fit of the random-walk model —
Ingram (2019). It is literally ``OnlineAbility(robust_c=None)``; the robust arm
(``robust.py``) is the *same* filter with the cap on. Keeping the only divergence
in these two thin wrappers is what makes the arm comparison clean (CLAUDE.md:
"Same likelihood, same data, both arms").
"""

from __future__ import annotations

from src.models.ability import OnlineAbility


def make(alpha_by_surface: dict, sigma: float, prior_var: float = 1.0) -> OnlineAbility:
    return OnlineAbility(alpha_by_surface, sigma=sigma, prior_var=prior_var,
                         robust_c=None)
