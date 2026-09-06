"""Mathematical invariants behind cached discovery geometry diagnostics."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("complete_cached_geometry", ROOT / "scripts/geometry_map/complete_cached_geometry.py")
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


class CachedGeometryTests(unittest.TestCase):
    def test_common_metric_angles_match_known_geometry(self):
        a = np.array([[1.], [0.], [0.]])
        b = np.array([[1.], [1.], [0.]])
        angle = geometry.subspace_comparison(a, b)["principal_angles_deg"][0]
        self.assertAlmostEqual(angle, 45., places=8)
        same = geometry.subspace_comparison(np.column_stack([a, b]), np.column_stack([b, -a]))
        self.assertLess(max(same["principal_angles_deg"]), 2e-6)
        # Applying only one target's feature scaling changes the metric, so such a
        # comparison must never be substituted for the common-coordinate angle.
        wrong = geometry.subspace_comparison(a, np.diag([10., 1., 1.]) @ b)
        self.assertGreater(abs(angle - wrong["principal_angles_deg"][0]), 30.)

    def test_isotropic_ridge_target_rotation_equivalence(self):
        rng = np.random.default_rng(44)
        x, y, test = rng.normal(size=(160, 15)), rng.normal(size=(160, 3)), rng.normal(size=(30, 15))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        w, b = geometry.ridge_fit(x, y, 10.)
        rw, rb = geometry.ridge_fit(x, y @ rotation, 10.)
        np.testing.assert_allclose(test @ w + b, (test @ rw + rb) @ rotation.T, atol=1e-12)

    def test_step_and_cumulative_progress_are_distinct(self):
        distance = np.array([10., 8., 9.] + list(np.linspace(7., 0., 17)))
        labels = {"goal_distance": distance, "frame_index": np.arange(0, 100, 5),
                  "goal_vector": np.ones((20, 3)), "wall_signed_distance": np.zeros(20),
                  "height_above_wall_top": np.zeros(20), "time_fraction": np.arange(20) / 19,
                  "action_translation_sum": np.zeros((19, 3)), "action_translation_magnitude": np.zeros(19),
                  "realized_hand_delta": np.zeros((19, 3))}
        result = geometry.corrected_labels(labels)
        np.testing.assert_allclose(result["step_progress"][:2], [2., -1.])
        np.testing.assert_allclose(result["cumulative_progress"][:3], [0., 2., 1.])
        np.testing.assert_allclose(np.cumsum(result["step_progress"]), result["cumulative_progress_after_step"])

    def test_zero_net_action_direction_is_masked_and_finite(self):
        direction, zero, norm = geometry.normalized_direction([[0., 0., 0.], [3., 4., 0.]])
        np.testing.assert_array_equal(zero, [True, False])
        np.testing.assert_allclose(direction, [[0., 0., 0.], [.6, .8, 0.]])
        np.testing.assert_allclose(norm, [0., 5.])
        chunk = np.array([[1., 0., 0.], [-1., 0., 0.]])
        self.assertEqual(np.linalg.norm(chunk.sum(axis=0)), 0.)
        self.assertGreater(np.linalg.norm(chunk), 0.)

    def test_goal_frame_roundtrip_and_degeneracies(self):
        rotation, valid, counts = geometry.goal_basis(np.array([[1., 2., 3.], [0., 0., 1.], [0., 0., 0.]]))
        np.testing.assert_array_equal(valid, [True, False, False])
        motion = np.array([[.1, -.4, .2]])
        np.testing.assert_allclose(geometry.to_world(geometry.to_local(motion, rotation[:1]), rotation[:1]), motion, atol=1e-12)
        self.assertEqual(counts["zero_goal_distance_count"], 1)

    def test_receipt_filter_excludes_held_out_entries(self):
        done = {"complete": True, "outputs": [{"seed": 1, "episode": ep} for ep in [-1, 0, 49, 50, 74, 75, 99]]}
        self.assertEqual([item["episode"] for item in geometry.selected_entries(done)], [0, 49])


if __name__ == "__main__":
    unittest.main()
