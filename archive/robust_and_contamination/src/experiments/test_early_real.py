"""Checks for real-point adaptation and temporal evaluation boundaries."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import copy
import unittest

import numpy as np
import pandas as pd

from src.experiments import early_real as real
from src.models import block_filter as block
from src.experiments.point_robustness import standardized_scores


def fixture(mid='a', year=2013):
    serving = np.tile(np.repeat([0, 1], [5, 7]), 16)
    y = (np.random.default_rng(3).random(len(serving)) < .64).astype(int)
    return real.Match(mid, year, f'{year}-580', 'Australian Open', pd.Timestamp(year, 1, 14),
                      'R128', 'Hard', 101, 102, 'Player A', 'Player B', True, False,
                      serving, y, np.arange(1, len(y) + 1))


class EarlyRealTests(unittest.TestCase):
    def test_gaps_never_share_blocks(self):
        numbers = np.r_[np.arange(1, 10), np.arange(20, 46)]
        blocks = block.blocks_for(numbers, 16)
        np.testing.assert_array_equal(np.concatenate(blocks), np.arange(len(numbers)))
        for b in blocks:
            self.assertTrue((np.diff(numbers[b]) == 1).all())
            self.assertLessEqual(len(b), 16)
        for invalid in ([1, 2, 2], [1, 3, 2], [1, 2.5]):
            with self.assertRaises(ValueError):
                block.blocks_for(np.array(invalid), 16)

    def test_balanced_score_matches_pilot(self):
        counts = np.array([[3, 7], [4, 6], [5, 4], [2, 8]])
        np.testing.assert_allclose(block.scores(counts, np.full_like(counts, 8)),
                                   standardized_scores(counts, 8))

    def test_zero_exposure_and_partial_block(self):
        for serving in (np.zeros(99, dtype=int), np.r_[np.zeros(65, int), np.ones(34, int)]):
            outcomes = np.random.default_rng(4).integers(0, 2, len(serving))
            keep, info = block.filter_points(serving, outcomes)
            self.assertEqual(len(keep), len(serving))
            self.assertTrue(np.isfinite(info['threshold']))
            self.assertLessEqual((~keep).sum(), len(serving)//4)
            self.assertEqual(sum(info['lengths']), len(serving))
        keep, _ = block.filter_points(np.zeros(12, int), np.ones(12, int))
        self.assertTrue(keep.all())

    def test_budget_counts_points_not_blocks(self):
        m = fixture()
        outcomes = m.outcome.copy()
        outcomes[-48:] = np.where(m.serving[-48:] == 0, 0, 1)
        keep, info = block.filter_points(m.serving[:-3], outcomes[:-3])
        self.assertLessEqual((~keep).sum(), int(.25 * len(keep)))
        self.assertEqual(info['removed_points'], int((~keep).sum()))
        for b in block.blocks_for(np.arange(len(keep)), 16):
            self.assertTrue((keep[b] == keep[b[0]]).all())

    def test_frozen_predictions_do_not_read_outcomes_or_mutate_fit(self):
        f = real.make_filter({'Hard': .6}, .1, 'block_16')
        real.learn(f, fixture('train', 2012), 'ordinary', {})
        before = copy.deepcopy(f.__dict__)
        m = fixture()
        changed = replace(m, outcome=1-m.outcome, p1_won=not m.p1_won)
        with patch.object(real, 'learn', side_effect=AssertionError('frozen update')):
            p, _ = real.evaluate(f, [m], 'block_16', 'frozen', {101, 102}, {})
            q, _ = real.evaluate(f, [changed], 'block_16', 'frozen', {101, 102}, {})
        np.testing.assert_array_equal(p.p, q.p)
        self.assertEqual(before, f.__dict__)
        self.assertEqual(p.n.sum(), len(m.outcome))

    def test_online_prediction_precedes_own_update(self):
        m = fixture()
        f = real.make_filter({'Hard': .6}, .1, 'block_16')
        expected = real.predict(f, m)
        p, _ = real.evaluate(f, [m], 'block_16', 'online', set(), {})
        np.testing.assert_array_equal(p.p, expected)
        self.assertNotEqual(real.predict(f, m), expected)

    def test_partial_match_with_only_one_observed_server_is_kept(self):
        m = replace(fixture(), serving=np.array([0]), outcome=np.array([1]),
                    point_numbers=np.array([200]))
        f = real.make_filter({'Hard': .6}, .1, 'block_16')
        p, matches = real.evaluate(f, [m], 'block_16', 'online', set(), {})
        self.assertEqual(len(p), 1)
        self.assertEqual(p.n.sum(), 1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(f.sm[102], 0.)
        self.assertEqual(f.sv[102], 1.)

    def test_future_observations_do_not_change_earlier_predictions(self):
        first = fixture('first')
        second = replace(fixture('second'), date=pd.Timestamp(2013, 5, 27), round='R64')
        for arm in real.ARMS:
            f = real.make_filter({'Hard': .6}, .1, arm)
            p, _ = real.evaluate(copy.deepcopy(f), [first, second], arm, 'online', set(), {})
            changed = replace(second, outcome=1-second.outcome)
            q, _ = real.evaluate(copy.deepcopy(f), [first, changed], arm, 'online', set(), {})
            np.testing.assert_array_equal(p.p, q.p)

    def test_initials_are_not_merged(self):
        self.assertNotEqual(real.name_key('Alex Kuznetsov'), real.name_key('Andrey Kuznetsov'))
        self.assertEqual(real.name_key('Stanislas Wawrinka'), real.name_key('Stan Wawrinka'))

    def test_forbidden_year_rejected_before_read(self):
        with patch.object(Path, 'read_text', side_effect=AssertionError('read before guard')):
            with self.assertRaises(ValueError):
                real.verify_inputs(years=(2011, 2012, 2025))

    def test_paired_alignment_is_enforced(self):
        base = pd.DataFrame({'match_id': ['x', 'y'], 'tourney_id': ['a', 'b'],
                             'server': [1, 2], 'n': [10, 20], 'loss_sum': [5., 10.]})
        candidate = base.copy()
        candidate['loss_sum'] -= candidate.n * .01
        d = real.paired_gain(base, candidate, ['tourney_id', 'server'],
                            ['match_id', 'tourney_id', 'server'])
        self.assertAlmostEqual(d['mean'], .01)
        self.assertAlmostEqual(d['lo'], .01)
        candidate.loc[0, 'n'] = 11
        with self.assertRaises(ValueError):
            real.paired_gain(base, candidate, ['tourney_id'], ['match_id', 'tourney_id', 'server'])


if __name__ == '__main__':
    unittest.main()
