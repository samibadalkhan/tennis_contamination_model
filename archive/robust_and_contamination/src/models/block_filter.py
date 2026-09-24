"""Experimental spectral block-score filtering with unequal service exposure.

Adapts the point pilot to ordered real matches. This is not published SEVER or
MMW, and does not identify proven contamination. Calibration assumes constant,
independent Bernoulli probabilities within each of two service streams.
"""
from __future__ import annotations

import numpy as np


def blocks_for(point_numbers: np.ndarray, block_size: int) -> list[np.ndarray]:
    numbers = np.asarray(point_numbers)
    if block_size < 1 or numbers.ndim != 1 or not np.isfinite(numbers).all():
        raise ValueError('invalid block size or point numbers')
    if not np.all(numbers == numbers.astype(int)) or np.any(np.diff(numbers) <= 0):
        raise ValueError('played point numbers must be unique increasing integers')
    if not len(numbers):
        return []
    segments = np.split(np.arange(len(numbers)), np.flatnonzero(np.diff(numbers) != 1) + 1)
    return [segment[i:i + block_size] for segment in segments
            for i in range(0, len(segment), block_size)]


def scores(counts: np.ndarray, exposures: np.ndarray) -> np.ndarray:
    """Standardize each observed score by its own binomial information."""
    p = (counts.sum(axis=0) + .5) / (exposures.sum(axis=0) + 1)
    scale = np.sqrt(exposures * p * (1 - p))
    g = np.divide(counts - exposures * p, scale,
                  out=np.zeros_like(counts, dtype=float), where=scale > 0)
    return g - g.mean(axis=0)


def conditional_threshold(counts, exposures, draws=255, quantile=.99):
    rng = np.random.default_rng(7193)
    samples = np.stack([
        rng.multivariate_hypergeometric(exposures[:, s], int(counts[:, s].sum()), size=draws)
        for s in (0, 1)], axis=2)
    p = (counts.sum(axis=0) + .5) / (exposures.sum(axis=0) + 1)
    scale = np.sqrt(exposures * p * (1 - p))
    g = np.divide(samples - exposures * p, scale,
                  out=np.zeros_like(samples, dtype=float), where=scale > 0)
    g -= g.mean(axis=1, keepdims=True)
    covariance = np.einsum('bmi,bmj->bij', g, g) / len(exposures)
    return float(np.quantile(np.linalg.eigvalsh(covariance)[:, -1], quantile, method='higher'))


def filter_points(serving, outcome, point_numbers=None, block_size=16,
                  max_removal=.25, null_draws=255, null_quantile=.99):
    serving, outcome = np.asarray(serving), np.asarray(outcome)
    if serving.ndim != 1 or serving.shape != outcome.shape or not np.isin(serving, [0, 1]).all():
        raise ValueError('serving and outcome must be aligned binary vectors')
    if not np.isin(outcome, [0, 1]).all() or not 0 <= max_removal < 1:
        raise ValueError('invalid outcomes or removal budget')
    if null_draws < 2 or not 0 < null_quantile < 1:
        raise ValueError('invalid null calibration')
    if point_numbers is None:
        point_numbers = np.arange(len(outcome))
    if len(point_numbers) != len(outcome):
        raise ValueError('point numbers must align with outcomes')
    blocks = blocks_for(point_numbers, block_size)
    keep = np.ones(len(outcome), bool)
    lengths = np.array([len(b) for b in blocks], dtype=int)
    info = {'block_size': block_size, 'blocks': len(blocks), 'removed_points': 0,
            'removed_blocks': [], 'threshold': None, 'eigenvalues': [],
            'lengths': lengths.tolist()}
    if len(blocks) < 4:
        return keep, info
    exposures = np.array([[(serving[b] == s).sum() for s in (0, 1)] for b in blocks])
    counts = np.array([[(outcome[b] * (serving[b] == s)).sum() for s in (0, 1)] for b in blocks])
    threshold = conditional_threshold(counts, exposures, null_draws, null_quantile)
    info['threshold'] = threshold
    active = np.ones(len(blocks), bool)
    budget = int(np.floor(max_removal * len(outcome)))
    while active.sum() >= 4:
        g = scores(counts[active], exposures[active])
        values, vectors = np.linalg.eigh(g.T @ g / len(g))
        top = float(values[-1])
        info['eigenvalues'].append(top)
        if top <= threshold + 1e-12:
            break
        indices = np.flatnonzero(active)
        projection = (g @ vectors[:, -1]) ** 2
        eligible = lengths[indices] <= budget - info['removed_points']
        if not eligible.any():
            break
        projection[~eligible] = -np.inf
        chosen = int(indices[np.argmax(projection)])
        active[chosen] = False
        keep[blocks[chosen]] = False
        info['removed_points'] += int(lengths[chosen])
        info['removed_blocks'].append(chosen)
    return keep, info
