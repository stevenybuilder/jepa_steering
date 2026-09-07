import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/geometry_map"))
from diagnose_horizon_readouts import decomposition, persistence_error, grouped_summary


class ReadoutTests(unittest.TestCase):
    def test_cross_term_required_even_when_total_zero(self):
        result = decomposition([[0., 0, 0]], [[1., 0, 0]], [[0., 0, 0]])
        self.assertEqual(result["total_squared_norm"].item(), 0)
        self.assertEqual(result["observer_squared_norm"].item(), 1)
        self.assertEqual(result["projected_squared_norm"].item(), 1)
        self.assertEqual(result["cross_term"].item(), -2)

    def test_random_float64_identity(self):
        rng = np.random.default_rng(1)
        result = decomposition(*(rng.normal(size=(6, 3)) for _ in range(3)))
        self.assertLess(result["vector_identity_max_abs"], 1e-12)
        self.assertLess(result["squared_identity_max_abs"], 1e-12)

    def test_persistence_does_not_use_oracle_previous_state(self):
        truth = np.ones((6, 3)); start = np.zeros((6, 3))
        self.assertTrue(np.array_equal(persistence_error(start, truth), -truth))
        start[-1, 0] = 1
        with self.assertRaises(RuntimeError):
            persistence_error(start, truth)

    def test_grouped_bootstrap_resamples_episode_values(self):
        value = grouped_summary(np.array([1., 3.]), np.array([[0, 0], [1, 1], [0, 1]]))
        self.assertEqual(value["mean"], 2.)
        self.assertEqual(value["episode_range"], [1., 3.])


if __name__ == "__main__":
    unittest.main()
