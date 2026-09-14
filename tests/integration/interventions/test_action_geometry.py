import unittest

import torch

from offline_study.interventions.action_geometry import central_estimates, donor_actions, matched_deltas
from offline_study.fitting.geometry_fit import make_geometry_protocol


class ActionGeometryTests(unittest.TestCase):
    def test_protocol_freezes_fit_derived_margin_and_all_required_contrasts(self):
        protocol = make_geometry_protocol({"task": "pusht", "manifest_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64, "delivered_l2": .02,
            "fit_native_proprio_mse_h6": 2.}, "c" * 64, "2026-09-07T00:00:00Z")
        self.assertEqual(len(protocol["arms"]), 7)
        self.assertEqual(len(protocol["primary_contrasts"]), 11)
        self.assertEqual(protocol["development_analysis_plan"]["smallest_useful_effect"], .02)
        self.assertEqual(protocol["geometry"]["omitted_coordinate"], 0.)

    def test_four_donors_recover_polynomial_center_and_reflect_about_line(self):
        t = torch.tensor([-1., -.5, .5, 1.], dtype=torch.float64)
        donors = torch.stack((4 + t, 7 + t.square(), 2 + t.pow(3)), dim=1)[None]
        estimates, valid = central_estimates(donors)
        self.assertTrue(valid.item())
        torch.testing.assert_close(estimates["cubic"], torch.tensor([[4., 7., 2.]], dtype=torch.float64))
        self.assertGreater(float((estimates["equal_anchor_linear"] - estimates["cubic"]).norm()), .5)
        torch.testing.assert_close((estimates["cubic"] + estimates["reflected_curvature"]) / 2,
                                   estimates["projected_cubic"])
        chord = donors[:, 3] - donors[:, 0]
        self.assertAlmostEqual(float(((estimates["cubic"] - estimates["projected_cubic"]) * chord).sum()), 0.)

    def test_closed_chord_is_flagged_and_all_common_energy_arms_are_zero(self):
        donors = torch.tensor([[[1., 0.], [2., 2.], [2., -2.], [1., 0.]]])
        estimates, valid = central_estimates(donors)
        self.assertFalse(valid.item())
        deltas, eligible = matched_deltas(estimates, torch.zeros(1, 2), valid, torch.tensor([1., 0.]), .3)
        self.assertFalse(eligible.item())
        self.assertTrue(all(torch.count_nonzero(value) == 0 for value in deltas.values()))

    def test_delivered_energy_matches_across_every_active_arm(self):
        generator = torch.Generator().manual_seed(81)
        donors = torch.randn(3, 4, 2, 5, generator=generator)
        estimates, valid = central_estimates(donors)
        native = torch.randn(3, 2, 5, generator=generator)
        random = torch.randn(2, 5, generator=generator)
        random /= random.norm()
        deltas, eligible = matched_deltas(estimates, native, valid, random, .25)
        self.assertTrue(eligible.all())
        for name, delta in deltas.items():
            target = 0. if name in ("native", "zero_dose") else .25
            torch.testing.assert_close(delta.flatten(1).norm(dim=1), torch.full((3,), target))

    def test_donors_change_only_h3_and_recorded_actions_are_unchanged(self):
        actions = torch.randn(6, 2, 10)
        original = actions.clone()
        direction = torch.ones(10) / 10. ** .5
        expanded = donor_actions(actions, direction, .1)
        torch.testing.assert_close(actions, original, rtol=0, atol=0)
        torch.testing.assert_close(expanded[:, 4::5], original, rtol=0, atol=0)
        for horizon in (0, 1, 3, 4, 5):
            torch.testing.assert_close(expanded[horizon], original[horizon].repeat_interleave(5, 0))
        torch.testing.assert_close(expanded[2, 0] - original[2, 0], -.1 * direction)

    def test_estimator_rejects_wrong_donor_count_and_nonfinite_data(self):
        with self.assertRaises(ValueError):
            central_estimates(torch.zeros(2, 5, 3))
        with self.assertRaises(ValueError):
            central_estimates(torch.full((2, 4, 3), float("nan")))
