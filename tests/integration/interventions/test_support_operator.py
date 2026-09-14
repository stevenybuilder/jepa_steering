import unittest
import torch
from offline_study.interventions.support_operator import principal_basis, fit_target, predict_target, expand_basis, solve_correction, random_basis, compile_fields, spatial_positions, fit_bases, CATEGORIES
from offline_study.fitting.support_fit import make_support_protocol
from offline_study.interventions.interventions import CATEGORY_ARMS


class SupportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(44)

    def test_principal_nested_and_degenerate(self):
        x = torch.randn(40, 24)
        mean, basis = principal_basis(x, 8)
        torch.testing.assert_close(basis @ basis.T, torch.eye(8), atol=2e-6, rtol=2e-6)
        self.assertEqual(basis[:4].shape, (4, 24))
        with self.assertRaises(ValueError):
            principal_basis(torch.ones(20, 10), 8)

    def test_common_target_has_no_recipient_future_argument(self):
        x = torch.randn(40, 12)
        _, basis = principal_basis(x, 3)
        error = x @ basis.T @ torch.randn(3, 7) + 2
        model = fit_target(x, error)
        torch.testing.assert_close(predict_target(x, model), error, atol=1e-4, rtol=1e-4)

    def test_direct_sum_and_random_spectrum(self):
        spec = {"blocks": [2, 3], "positions": [1, 9], "basis": torch.randn(1, 2, 2, 4)}
        spec["basis"] /= spec["basis"].norm()
        dense = expand_basis(spec, 1)
        self.assertEqual(dense.shape, (1, 6, 256, 4))
        self.assertAlmostEqual(float(dense.norm()), 1., places=6)
        self.assertEqual(float(dense[:, [0, 1, 4, 5]].norm()), 0.)
        random = random_basis(spec, 17)
        self.assertAlmostEqual(float(random["basis"].norm()), 1., places=6)
        edits = compile_fields(dense[None].expand(2, -1, -1, -1, -1), [2, 3])
        self.assertAlmostEqual(sum(edit.delivered_l2[0]**2 for edit in edits), 1., places=6)

    def test_response_solve_sign_and_damping(self):
        response = torch.eye(3)[None].expand(2, -1, -1)
        target = torch.tensor([[1., 2., -3.], [0., 0., 0.]])
        result = solve_correction(response, target, torch.eye(3))
        torch.testing.assert_close(result, target / 1.01)

    def test_support_positions_and_registered_protocols(self):
        fields = torch.randn(24, 6, 256, 4)
        errors = torch.randn(24, 20)
        bases, maps, positions = fit_bases(fields, errors)
        self.assertEqual(len(positions["one_patch"]), 1)
        self.assertEqual(len(positions["contiguous_group"]), 16)
        self.assertEqual(len(set(positions["equal_size_scattered_group"])), 16)
        for name in ("equal_size_scattered_group", "random_position_equal_size_scattered_group"):
            self.assertEqual(len({(p // 16) % 4 for p in positions[name]}), 1)
            self.assertEqual(len({p % 4 for p in positions[name]}), 1)
        self.assertEqual(maps["distribution_layer"]["single_block3"], ("rank", 1))
        receipt = {"task": "pusht", "manifest_sha256": "a" * 64,
                   "checkpoint_sha256": "b" * 64, "fit_native_proprio_mse_h6": .2,
                   "delivered_l2": .1, "created_at": "2026-09-07T10:00:00Z"}
        fitted = {"bases": bases, "maps": maps, "positions": positions}
        for category in CATEGORIES:
            protocol = make_support_protocol(receipt, "c" * 64, category, fitted)
            self.assertEqual({arm["name"] for arm in protocol["arms"]}, CATEGORY_ARMS[category])
            self.assertEqual(protocol["development_analysis_plan"]["smallest_useful_effect"], .002)


if __name__ == "__main__":
    unittest.main()
