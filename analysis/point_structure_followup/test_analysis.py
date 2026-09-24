"""Semantic checks on artificial sequences only; no project data is opened."""
from itertools import combinations, product
import unittest

import numpy as np
import pandas as pd

from analyze import components, build_components, LAGS, VIEWS


def evaluate(x, sets=None, games=None, segments=None, tiebreak=None):
    n = len(x)
    return components(np.asarray(x), np.ones(n) if sets is None else sets,
                      np.ones(n) if games is None else games,
                      np.zeros(n) if segments is None else segments,
                      np.zeros(n, dtype=bool) if tiebreak is None else tiebreak,
                      min_match=4, min_set=2)


class ComponentsTests(unittest.TestCase):
    def test_exact_fixed_total_permutation_expectation(self):
        samples = []
        for ones in combinations(range(8), 4):
            x = np.zeros(8)
            x[list(ones)] = 1
            samples.append(evaluate(x)[0, :, 0])
        np.testing.assert_allclose(np.mean(samples, axis=0), 0, atol=1e-14)

    def test_exact_set_conditional_permutation_expectation(self):
        samples = []
        for first, second in product(combinations(range(4), 2), repeat=2):
            x = np.zeros(8)
            x[list(first)] = 1
            x[np.array(second) + 4] = 1
            samples.append(evaluate(x, sets=np.repeat([1, 2], 4))[2, :, 0])
        np.testing.assert_allclose(np.mean(samples, axis=0), 0, atol=1e-14)

    def test_matched_pair_counts_and_game_partition(self):
        x = [0, 1, 1, 0, 1, 0, 0, 1] * 4
        value = evaluate(x, sets=np.repeat([1, 2], 16), games=np.tile(np.repeat([1, 3], 8), 2))
        np.testing.assert_array_equal(value[1, :, 2], value[2, :, 2])
        np.testing.assert_allclose(value[3] + value[4], value[2], atol=1e-14)

    def test_outcome_complement_invariance(self):
        x = np.array([0, 1, 1, 0, 1, 1, 0, 1])
        np.testing.assert_allclose(evaluate(x), evaluate(1 - x), atol=1e-14)

    def test_pairs_do_not_bridge_missing_points(self):
        x = [0, 1, 0, 1, 0, 1, 0, 1]
        value = evaluate(x, segments=np.repeat([0, 1], 4))
        self.assertEqual(value[0, 0, 2], 6)  # Three lag-one pairs in each segment.
        self.assertEqual(value[0, LAGS.index(4), 2], 0)

    def test_tiebreak_exclusion(self):
        tb = np.array([False] * 6 + [True] * 2)
        value = evaluate([0, 1, 0, 1, 0, 1, 0, 1], tiebreak=tb)
        self.assertEqual(value[VIEWS.index("within_set_no_tiebreak"), 0, 2], 5)

    def test_constant_sequence_has_no_false_information(self):
        self.assertFalse(evaluate([1] * 20).any())

    def test_opponent_service_game_is_not_a_recording_gap(self):
        # A service game, B service game, then A service game, with all points recorded.
        n = 24
        server = np.array(["a"] * 8 + ["b"] * 8 + ["a"] * 8)
        frame = pd.DataFrame({"match_id": ["m"] * n, "tour": ["M"] * n,
                              "year": [2014] * n, "slam": ["wimbledon"] * n,
                              "point_no": np.arange(1, n + 1), "server": server,
                              "returner": np.where(server == "a", "b", "a"),
                              "server_wins": np.tile([0, 1], n // 2),
                              "set_no": np.ones(n), "game_no": np.repeat([1, 2, 3], 8),
                              "ambiguous_identity": [False] * n, "is_tiebreak": [False] * n})
        _, _, audit = build_components(frame)
        self.assertEqual(audit["point_number_gaps"], 0)
        bad = frame.copy()
        bad.loc[12, "point_no"] = 12
        with self.assertRaisesRegex(ValueError, "ordered"):
            build_components(bad)


if __name__ == "__main__":
    unittest.main()
