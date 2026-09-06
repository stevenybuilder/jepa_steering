import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"/"geometry_map"))
from screen_coordinate_time import (ALPHAS, FRAMES, eligible, fit_models, per_episode_ci,
                                    predict_models, residual_basis, targets_and_frames, to_world)


class CoordinateTimeTests(unittest.TestCase):
    def test_frames_round_trip_and_time_is_target(self):
        rng = np.random.default_rng(3)
        current, following, goal, obj = rng.normal(size=(4, 12, 3))
        time = np.arange(12)*5
        y, cols, origins, rotations, _ = targets_and_frames(current, following, goal, time, obj)
        for name in (*FRAMES, "object_relative"):
            np.testing.assert_allclose(to_world(y[:, cols[name]], rotations[name])+origins[name], following, atol=1e-12)
        np.testing.assert_array_equal(y[:, cols["observed_episode_raw_step"]].ravel(), time)

    def test_goal_degeneracy_not_silently_dropped(self):
        current = np.zeros((2, 3))
        goal = np.array([[0., 0., 0.], [0., 0., 1.]])
        y, _, _, _, valid = targets_and_frames(current, np.ones((2, 3)), goal, [0, 5])
        self.assertFalse(valid.any())
        self.assertTrue(np.isfinite(y).all())

    def test_confirmation_cannot_be_selected(self):
        self.assertFalse(eligible({"seed": 1, "episode": 90}, "discovery"))
        self.assertFalse(eligible({"seed": 1, "episode": 90}, "validation"))
        with self.assertRaises(ValueError):
            eligible({"seed": 1, "episode": 90}, "confirmation")

    def test_equal_full_inputs_and_candidate_count(self):
        rng = np.random.default_rng(4)
        x, y = rng.normal(size=(40, 8)), rng.normal(size=(40, 4))
        fit = fit_models(x, y)
        self.assertEqual(len(fit["models"]), 2*len(ALPHAS))
        self.assertEqual(fit["models"]["rbf:0.01"].X_fit_.shape[1], 8)
        self.assertEqual(fit["models"]["linear:0.01"].n_features_in_, 8)
        before = fit["mean"].copy()
        predict_models(fit, np.full((3, 8), 999.))
        np.testing.assert_array_equal(before, fit["mean"])

    def test_raw_residual_projection_preserves_complement(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(90, 12))*np.arange(1, 13)
        y = x[:, :3]+rng.normal(size=(90, 3))
        basis = residual_basis(x, y, .01, 8)
        np.testing.assert_allclose(basis.T@basis, np.eye(8), atol=1e-8)
        delta = basis@(basis.T@(x[1]-x[0]))
        np.testing.assert_allclose(delta-basis@(basis.T@delta), 0, atol=1e-10)

    def test_bootstrap_unit_is_episode_not_rows(self):
        values = np.array([1., 1., 3., 3.])
        report = per_episode_ci(values, np.array([0, 0, 1, 1]))
        self.assertEqual(report["episodes"], 2)
        self.assertEqual(report["mean"], 2.)


if __name__ == "__main__":
    unittest.main()
