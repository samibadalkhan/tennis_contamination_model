"""Semantic checks for the isolated point-block pilot."""
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.experiments import point_robustness as pilot


def observation(counts):
    n = 8
    serving = np.tile(np.repeat([0, 1], 4), len(counts) * 2)
    outcomes = np.zeros_like(serving)
    for block, row in enumerate(counts):
        for side, k in enumerate(row):
            idx = np.flatnonzero(serving[block * 16:(block + 1) * 16] == side)
            outcomes[block * 16 + idx[:k]] = 1
    return pilot.Observation(0, 0, 1, serving, outcomes)


class PointRobustnessTests(unittest.TestCase):
    def test_point_counts_preserve_outcomes(self):
        truth, observations, masks = pilot.world(pilot.Config(), 'rw', 'burst', 0)
        for obs in observations:
            for size in pilot.BLOCK_SIZES:
                counts, n = pilot.block_counts(obs, size)
                self.assertEqual(counts.sum(), obs.outcome.sum())
                self.assertEqual(2 * n * len(counts), len(obs.outcome))
                self.assertTrue(np.all((counts >= 0) & (counts <= n)))

    def test_bursts_change_only_marked_outcomes_and_not_skill(self):
        cfg = pilot.Config()
        clean_truth, clean, _ = pilot.world(cfg, 'rw', 'clean', 3)
        burst_truth, burst, masks = pilot.world(cfg, 'rw', 'burst', 3)
        np.testing.assert_array_equal(clean_truth, burst_truth)
        self.assertTrue(any(m.any() for m in masks))
        for a, b, mask in zip(clean, burst, masks):
            self.assertEqual((a.session, a.a, a.b), (b.session, b.a, b.b))
            np.testing.assert_array_equal(a.outcome[~mask], b.outcome[~mask])
            if b.session < cfg.onset or b.session >= cfg.burst_end:
                self.assertFalse(mask.any())

    def test_permanent_change_is_not_marked_corruption(self):
        cfg = pilot.Config()
        clean, _, _ = pilot.world(cfg, 'smooth', 'clean', 2)
        drop, _, masks = pilot.world(cfg, 'smooth', 'permanent_drop', 2)
        np.testing.assert_allclose(drop[:cfg.onset], clean[:cfg.onset])
        np.testing.assert_allclose(drop[cfg.onset:, :2] - clean[cfg.onset:, :2], -.6)
        self.assertFalse(any(m.any() for m in masks))

    def test_uniform_level_changes_are_not_rejected(self):
        for counts in ([6, 5], [2, 7], [8, 0]):
            keep, info = pilot.spectral_keep(observation([counts] * 12), 16)
            self.assertTrue(keep.all())

    def test_spectral_filter_responds_to_collective_structure(self):
        clustered = observation([[6, 5]] * 9 + [[0, 8]] * 3)
        keep, info = pilot.spectral_keep(clustered, 16)
        self.assertGreater(info['removed_blocks'], 0)
        self.assertLessEqual(info['removed_blocks'], 3)
        # One block decision applies to both service streams and all its points.
        self.assertTrue(np.all(keep.reshape(-1, 16) == keep.reshape(-1, 16)[:, :1]))
        # Same outcomes spread evenly have no between-block covariance signal.
        counts, _ = pilot.block_counts(clustered, 16)
        totals = counts.sum(axis=0)
        diffuse_counts = np.tile(totals // 12, (12, 1))
        for side in (0, 1):
            diffuse_counts[:totals[side] % 12, side] += 1
        diffuse_keep, _ = pilot.spectral_keep(observation(diffuse_counts), 16)
        self.assertTrue(diffuse_keep.all())

    def test_unweighted_update_is_existing_backbone(self):
        cfg = pilot.Config()
        obs = observation([[6, 5]] * 12)
        a = pilot.make_filter(cfg, 'ordinary')
        b = pilot.make_filter(cfg, 'ordinary')
        pilot.update(a, obs, np.ones(len(obs.outcome), bool))
        row = SimpleNamespace(date=date(2001, 1, 1), p1=0, p2=1,
                              p1_spw=72, p1_svpt=96, p2_spw=60, p2_svpt=96, surface='hard')
        b.process(row)
        for attr in ('sm', 'rm', 'sv', 'rv'):
            self.assertEqual(getattr(a, attr), getattr(b, attr))

    def test_removed_points_supply_no_information(self):
        cfg = pilot.Config()
        obs = observation([[6, 5]] * 12)
        f = pilot.make_filter(cfg, 'ordinary')
        pilot.update(f, obs, np.zeros(len(obs.outcome), bool))
        self.assertEqual(f.sm, {0: 0., 1: 0.})
        self.assertEqual(f.sv, {0: 1., 1: 1.})
        self.assertEqual(f.rv, {0: 1., 1: 1.})

    def test_common_gauge_preserves_serve_return_contrasts(self):
        x = np.random.default_rng(3).normal(size=(5, 8, 2))
        aligned = pilot.align(x)
        np.testing.assert_allclose(aligned[:, 0, 0] - aligned[:, 1, 1], x[:, 0, 0] - x[:, 1, 1])
        np.testing.assert_allclose(pilot.align(x + 4), aligned, atol=1e-14)

    def test_nonrecovery_stays_censored(self):
        truth = np.zeros((10, 8, 2))
        estimate = truth.copy()
        estimate[:, :2] = 1
        self.assertEqual(pilot.recovery_lags(estimate, truth, 2), [None, None])
        estimate[6:] = 0
        self.assertEqual(pilot.recovery_lags(estimate, truth, 2), [4, 4])
        estimate[8, :2] = 1
        self.assertEqual(pilot.recovery_lags(estimate, truth, 2), [None, None])

    def test_labels_only_reach_oracle_and_future_cannot_change_past(self):
        cfg = pilot.Config(sessions=8, onset=3, burst_end=6)
        truth, obs, masks = pilot.world(cfg, 'rw', 'burst', 0)
        with patch.object(pilot, 'world', return_value=(truth, obs, masks)):
            _, _, original = pilot.run_world(cfg, 'rw', 'burst', 0)
        other_masks = [~m for m in masks]
        with patch.object(pilot, 'world', return_value=(truth, obs, other_masks)):
            _, _, relabeled = pilot.run_world(cfg, 'rw', 'burst', 0)
        for arm in pilot.ARMS[:-1]:
            np.testing.assert_array_equal(original[arm], relabeled[arm])
        changed = [replace(o, outcome=1-o.outcome) if o.session >= 5 else o for o in obs]
        with patch.object(pilot, 'world', return_value=(truth, changed, masks)):
            _, _, future_changed = pilot.run_world(cfg, 'rw', 'burst', 0)
        for arm in pilot.ARMS:
            np.testing.assert_array_equal(original[arm][:5], future_changed[arm][:5])


if __name__ == '__main__':
    unittest.main()
