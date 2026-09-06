import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("compensation", Path(__file__).resolve().parents[1] / "scripts/geometry_map/analyze_planner_compensation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompensationTests(unittest.TestCase):
    def test_raw_clipped_and_physical_are_not_conflated(self):
        base = np.zeros((2, 4))
        commands = np.array([[0, 0, 3, 0], [0, 0, -2, 0.]])
        result = module.command_metrics(commands, base)
        self.assertEqual(result["raw_z_sum_change_vs_unsteered"], 1.)
        self.assertEqual(result["unit_box_clipped_z_sum_change"], 0.)
        self.assertEqual(result["raw"]["z_l1"], 5.)
        self.assertAlmostEqual(result["raw"]["z_l2"], 13**.5)

    def test_net_norm_and_chunk_magnitude_differ(self):
        actions = np.array([[1., 0, 0, 0], [-1., 0, 0, 0]])
        result = module.command_metrics(actions, np.zeros_like(actions))
        self.assertEqual(result["raw"]["translation_net_norm"], 0.)
        self.assertAlmostEqual(result["raw"]["translation_chunk_frobenius"], 2**.5)

    def test_signed_opposition_and_zero_are_explicit(self):
        self.assertTrue(module.opposition(1., -2.))
        self.assertTrue(module.opposition(-1., 2.))
        self.assertFalse(module.opposition(-1., -2.))
        self.assertIsNone(module.opposition(0., 1.))

    def test_frozen_readout_does_not_fit_to_new_inputs(self):
        readout = {"mean": np.array([1., 2.]), "scale": np.array([2., 4.]),
                   "coef": np.array([[1., 0.], [0., 1.], [1., 1.]]), "intercept": np.zeros(3)}
        np.testing.assert_array_equal(module.decode(readout, np.array([[3., 6.]])), [[1., 1., 2.]])

    def test_alignment_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            module.command_metrics(np.zeros((2, 4)), np.zeros((3, 4)))

    def test_displacements_use_observed_start_and_five_action_alignment(self):
        start = np.array([1., 2., 3.])
        predicted = np.tile(start, (6, 1)) + np.arange(1, 7)[:, None]
        actual = predicted[:3] + .25
        result = module.align_displacements(predicted, actual, start)
        self.assertEqual(result["first_five_action_predicted_z_displacement_m"], 1.)
        self.assertEqual(result["first_five_action_actual_z_displacement_m"], 1.25)
        self.assertEqual(result["actual_raw_action_indices"], [5, 10, 15])


if __name__ == "__main__":
    unittest.main()
