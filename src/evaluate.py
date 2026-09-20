"""Splits, clustered bootstrap, and metrics.

Non-negotiables (CLAUDE.md):
  * Split by MATCH and by TIME, never by point. Train early years, test late.
    The dataset is FROZEN (slam ends Oct 2024; no 2025-26 data will ever come),
    so this train-early/test-late split is the ONLY out-of-sample period the
    project will ever have. Declare it once, up front, and never re-split.
  * Cluster ALL uncertainty by match; bootstrap over matches (effective
    n ~ match count under serial dependence). A naive point-level interval
    would call a trivial difference significant.
  * Never tune epsilon / robustness constants / block size on test.
  * Score the FULL MIXTURE vs a single-component model -- not robust theta-hat
    vs MLE on held-out log-loss (that target mismatch hands the win to the
    contaminated estimate by construction). (Stage 2+.)

The clustered bootstrap here reduces each statistic to per-match sufficient
statistics, then resamples matches with replacement and recombines. That keeps
B=thousands of resamples cheap and exact (no row-level reshuffling), and makes
"effective n = number of matches" literal.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-observation binary log-loss (natural log)."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def clustered_ci(per_match: dict[str, np.ndarray], combine,
                 B: int = 2000, ci: float = 0.95, seed: int = 0):
    """Clustered bootstrap over matches.

    ``per_match`` maps a name to an array of length M (one entry per match) of
    that match's contribution to a sufficient statistic. ``combine`` takes a
    dict of *summed* statistics and returns the scalar estimate. Returns
    (point_estimate, lo, hi, se).
    """
    keys = list(per_match)
    M = len(per_match[keys[0]])
    mat = np.vstack([per_match[k] for k in keys])           # (K, M)
    point = combine({k: mat[i].sum() for i, k in enumerate(keys)})
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, M, size=M)
        sums = {k: mat[i, idx].sum() for i, k in enumerate(keys)}
        vals[b] = combine(sums)
    lo, hi = np.percentile(vals, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(point), float(lo), float(hi), float(np.std(vals, ddof=1))


# --- combiners for specific statistics ----------------------------------------

def _mean(sums):                       # sum_x / n
    return sums["sum_x"] / max(sums["n"], EPS)


def _ratio(sums):                      # numer / denom  (e.g. Pearson dispersion)
    return sums["numer"] / max(sums["denom"], EPS)


def _corr(sums):                       # Pearson corr from co-moments
    n = sums["n"]
    if n < 2:
        return float("nan")
    cov = sums["sxy"] - sums["sx"] * sums["sy"] / n
    vx = sums["sxx"] - sums["sx"] ** 2 / n
    vy = sums["syy"] - sums["sy"] ** 2 / n
    denom = np.sqrt(max(vx, EPS) * max(vy, EPS))
    return cov / denom if denom > 0 else float("nan")


def _skew(sums):                       # sample skewness from raw moments
    n = sums["nb"]
    if n < 3:
        return float("nan")
    m1 = sums["s1"] / n
    m2 = sums["s2"] / n - m1 ** 2
    m3 = sums["s3"] / n - 3 * m1 * (sums["s2"] / n) + 2 * m1 ** 3
    return m3 / (m2 ** 1.5) if m2 > EPS else float("nan")
