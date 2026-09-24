import unittest
import numpy as np

from scoring_control import simulate


class ScoringControlTests(unittest.TestCase):
    def test_all_points_for_a_ends_in_straight_sets(self):
        for best_of in (3, 5):
            (servers, outcomes, sets, games, tb), winner = simulate(
                1., 0., best_of, np.random.default_rng(91))
            self.assertEqual(winner, 1)
            self.assertEqual(len(outcomes), 24 * (best_of // 2 + 1))
            self.assertFalse(tb.any())
            self.assertTrue(np.all(outcomes == (servers == 0)))
            self.assertEqual(sets.max(), best_of // 2 + 1)

    def test_recorded_point_and_game_structure(self):
        (servers, outcomes, sets, games, tb), _ = simulate(.65, .60, 5, np.random.default_rng(37))
        self.assertEqual(len({len(a) for a in (servers, outcomes, sets, games, tb)}), 1)
        self.assertTrue(np.isin(outcomes, [0, 1]).all())
        for set_no, game_no in set(zip(sets, games)):
            mask = (sets == set_no) & (games == game_no)
            if tb[mask].any():
                s = servers[mask]
                expected = np.array([s[0] ^ (((i + 1) // 2) % 2) for i in range(len(s))])
                np.testing.assert_array_equal(s, expected)
            else:
                self.assertEqual(len(np.unique(servers[mask])), 1)
                won = outcomes[mask].sum()
                lost = mask.sum() - won
                self.assertGreaterEqual(max(won, lost), 4)
                self.assertGreaterEqual(abs(won - lost), 2)


if __name__ == "__main__":
    unittest.main()
