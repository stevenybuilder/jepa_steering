import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/geometry_map"))
from fit_residual_search_v1 import action_probes, orthonormal_rows


class BasisTests(unittest.TestCase):
    def test_symmetric15_independent_action_directions(self):
        raw = torch.randn(5, 4)
        values = action_probes(raw)
        self.assertEqual(values.shape, (32, 5, 4))
        self.assertTrue(torch.equal(values[-1], raw))
        self.assertTrue(torch.equal(values[1:31:2], -values[2:31:2]))
        self.assertEqual(values[:31, :, 3].count_nonzero(), 0)
        self.assertEqual(torch.linalg.matrix_rank(values[1:31:2].reshape(15, 20)), 15)

    def test_basis_is_raw_orthonormal_and_capped_by_rank(self):
        values = np.random.default_rng(2).normal(size=(12, 100))
        basis, _ = orthonormal_rows(values, 16)
        self.assertEqual(len(basis), 12)
        np.testing.assert_allclose(basis@basis.T, np.eye(12), atol=2e-6)


if __name__ == "__main__":
    unittest.main()
