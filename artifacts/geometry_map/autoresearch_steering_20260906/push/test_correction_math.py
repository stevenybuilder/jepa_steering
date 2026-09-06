"""Synthetic checks of prediction contracts and physically fair edit controls."""
import inspect
import unittest

import torch

from correction_math import fit_predict, cap_edit, norm_match, corrected_cost


class CorrectionMathTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(20260906)

    def random(self, *shape):
        return torch.randn(*shape, generator=self.generator)

    def test_linear_learns_mapping_and_mean_without_future_labels(self):
        x = self.random(80, 3)
        coefficients = torch.tensor([[2., -1.], [.5, 3.], [-1., .2]])
        y = x @ coefficients + torch.tensor([7., -3.])
        test = self.random(12, 3)
        expected = test @ coefficients + torch.tensor([7., -3.])
        pred, _ = fit_predict(x, y, test, torch.arange(80) // 10, "linear")
        self.assertLess(float((pred - expected).square().mean()), .1)
        self.assertEqual(list(inspect.signature(fit_predict).parameters),
                         ["train_x", "train_residual", "test_x", "train_group", "family"])
        self.assertEqual(pred.dtype, torch.float32)

    def test_rbf_learns_nonlinear_curve(self):
        x = torch.linspace(-3, 3, 101)[:, None]
        y = torch.sin(x * 1.4) + 2.
        test = torch.linspace(-2.5, 2.5, 19)[:, None]
        pred, _ = fit_predict(x, y, test, torch.arange(101) // 10, "rbf")
        self.assertLess(float((pred - (torch.sin(test * 1.4) + 2.)).square().mean()), .04)

    def test_train_normalization_unchanged_by_other_test_rows(self):
        x = self.random(24, 4)
        x[:, 0] = 9.
        y = self.random(24, 2)
        test = self.random(3, 4)
        for family in ("linear", "rbf", "knn"):
            first, meta = fit_predict(x, y, test, torch.arange(24) // 4, family)
            second, meta2 = fit_predict(x, y, torch.cat([test, torch.full((1, 4), 1e4)]),
                                        torch.arange(24) // 4, family)
            torch.testing.assert_close(first, second[:3])
            self.assertEqual(meta["train_feature_scale"], meta2["train_feature_scale"])
            self.assertEqual(meta["train_feature_scale"][0], 1.)

    def test_zero_residual_and_degenerate_linear_intercept(self):
        x = self.random(20, 3)
        for family in ("linear", "rbf", "knn"):
            pred, _ = fit_predict(x, torch.zeros(20, 2), x[:2], torch.arange(20) // 4, family)
            self.assertTrue(torch.equal(pred, torch.zeros_like(pred)))
        pred, meta = fit_predict(torch.ones(8, 3), torch.full((8, 2), 4.),
                                 torch.full((2, 3), 7.), torch.arange(8), "linear")
        self.assertTrue(torch.equal(pred, torch.full((2, 2), 4.)))
        self.assertEqual(meta["ridge_regularization"], 0.)

    def test_knn_group_cap_and_small_group_fallback(self):
        x = torch.arange(16, dtype=torch.float32)[:, None]
        groups = torch.arange(16) // 4
        pred, meta = fit_predict(x, x, torch.tensor([[.1]]), groups, "knn")
        selected = meta["selected_train_indices"][0]
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(sum(int(groups[i]) == g for i in selected) == 2 for g in range(4)))
        self.assertTrue(torch.isfinite(pred).all())
        _, smaller = fit_predict(x[:8], x[:8], torch.tensor([[.1]]), groups[:8], "knn")
        self.assertEqual(smaller["selected_counts"], [4])

    def test_training_row_order_invariance_without_ties(self):
        x, y, test = self.random(32, 5), self.random(32, 3), self.random(4, 5)
        groups = torch.arange(32) // 4
        order = torch.randperm(32, generator=self.generator)
        for family in ("linear", "rbf", "knn"):
            pred, _ = fit_predict(x, y, test, groups, family)
            permuted, _ = fit_predict(x[order], y[order], test, groups[order], family)
            torch.testing.assert_close(pred, permuted, atol=1e-6, rtol=1e-5)

    def test_caps_shams_and_zero_control(self):
        visual, delta, sham = self.random(5, 11), self.random(5, 11), self.random(5, 11)
        for fraction in (0., .005, .02, .1):
            edit = cap_edit(delta, visual, fraction)
            self.assertTrue(torch.all(edit.norm(dim=1) <= fraction * visual.norm(dim=1) + 1e-7))
            matched = norm_match(sham, edit)
            torch.testing.assert_close(edit.norm(dim=1), matched.norm(dim=1))
            if fraction == 0:
                self.assertTrue(torch.equal(edit, torch.zeros_like(edit)))
        with self.assertRaises(ValueError):
            norm_match(torch.zeros_like(sham), delta)

    def test_corrected_cost_preserves_proprio_and_exact_identity(self):
        pred, goal, delta = self.random(5, 11), self.random(11), self.random(5, 11) * .1
        proprio = torch.tensor([.3, .6, .9, .2, .7])
        native = (pred - goal).square().mean(1) + proprio
        identity = corrected_cost(native, pred, goal, torch.zeros_like(delta))
        self.assertTrue(torch.equal(identity, native))
        native_double = native.double() + 1e-12
        identity_double = corrected_cost(native_double, pred, goal, torch.zeros_like(delta))
        self.assertTrue(torch.equal(identity_double, native_double))
        cost = corrected_cost(native, pred, goal, delta)
        expected = (pred + delta - goal).square().mean(1) + proprio
        torch.testing.assert_close(cost, expected, atol=4e-7, rtol=1e-6)

    def test_numpy_input_and_nonfinite_rejection(self):
        x, y = self.random(12, 2), self.random(12, 3)
        pred, _ = fit_predict(x.numpy(), y.numpy(), x[:2].numpy(), list(range(12)), "linear")
        self.assertEqual(pred.device.type, "cpu")
        x[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            fit_predict(x, y, x[:2], list(range(12)), "linear")


if __name__ == "__main__":
    unittest.main()
