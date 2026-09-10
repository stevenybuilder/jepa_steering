import unittest
import numpy as np
from decompose_chart_reconstruction import error_parts


class DecompositionTests(unittest.TestCase):
    def test_exact_orthogonal_parts(self):
        result = error_parts(np.array([[3., 4., 12.]]), np.array([[1., 0., 0.], [0., 1., 0.]]))
        self.assertEqual(result['total_squared_l2'], 169.)
        self.assertEqual(result['within_chart_squared_l2'], 25.)
        self.assertEqual(result['outside_chart_squared_l2'], 144.)

    def test_rotation_preserves_parts(self):
        rng = np.random.default_rng(4)
        q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        error = rng.normal(size=(12, 8))
        basis = np.eye(8)[:3]
        a = error_parts(error, basis)
        b = error_parts(error@q, basis@q)
        for key in a:
            self.assertAlmostEqual(a[key], b[key])


if __name__ == '__main__':
    unittest.main()
