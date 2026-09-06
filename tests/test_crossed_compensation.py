import importlib.util
from pathlib import Path
import unittest
import numpy as np

SPEC = importlib.util.spec_from_file_location("crossed", Path(__file__).parents[1] / "scripts/geometry_map/analyze_crossed_compensation.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class CrossedTests(unittest.TestCase):
    def test_exact_cancellation(self):
        bb = np.zeros((2, 3))
        bias = np.array([[1., 0, 0], [2., 0, 0]])
        result = M.crossed_decomposition(bb, bias, -bias, bb)
        for row in result["rows"]:
            self.assertEqual(row["cancellation_projection"], 1.)
            self.assertEqual(row["edit_action_cosine"], -1.)
            self.assertEqual(row["interaction_norm"], 0.)
            self.assertEqual(row["total_forecast_change_norm"], 0.)

    def test_orthogonal_action_is_not_compensation(self):
        row = M.crossed_decomposition([[0., 0]], [[1., 0]], [[0., 1]], [[1., 1]])["rows"][0]
        self.assertEqual(row["edit_action_cosine"], 0.)
        self.assertEqual(row["cancellation_projection"], 0.)

    def test_nonlinear_interaction_separated(self):
        row = M.crossed_decomposition([[1.]], [[2.]], [[3.]], [[9.]], goal=[0.])["rows"][0]
        self.assertEqual(row["interaction_norm"], 5.)
        self.assertEqual(row["total_forecast_change_norm"], 8.)
        self.assertEqual(row["total_cost_change"], 80.)
        self.assertAlmostEqual(row["total_cost_change"], row["fixed_action_cost_bias"] - row["base_model_selection_gain"] + row["cost_interaction"])

    def test_native_full_spatial_weighting(self):
        rng = np.random.default_rng(10)
        visual, proprio = rng.normal(size=(6, 4, 5)), rng.normal(size=(6, 3))
        packed = M.native_metric_vectors(visual, proprio, .1)
        np.testing.assert_allclose(np.sum(packed ** 2, axis=1), np.mean(visual ** 2, axis=(1, 2)) + .1 * np.mean(proprio ** 2, axis=1), rtol=1e-14)
        self.assertFalse(np.allclose(np.mean(visual ** 2, axis=(1, 2)), np.mean(visual.mean(1) ** 2, axis=1)))

    def test_zero_edit_undefined_cosine_not_false_evidence(self):
        row = M.crossed_decomposition([[1., 1]], [[1., 1]], [[2., 2]], [[2., 2]])["rows"][0]
        self.assertIsNone(row["edit_action_cosine"])
        self.assertIsNone(row["cancellation_projection"])

    def test_bad_shapes_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            M.crossed_decomposition([[1]], [[1, 2]], [[1]], [[1]])
        with self.assertRaises(ValueError):
            M.crossed_decomposition([[1]], [[np.nan]], [[1]], [[1]])
        with self.assertRaises(ValueError):
            M.native_metric_vectors([[1]], [[2]], -.1)


if __name__ == "__main__":
    unittest.main()
