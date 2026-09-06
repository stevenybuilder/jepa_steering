import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location("interface", Path(__file__).resolve().parents[1] / "scripts/geometry_map/capture_interface_response.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class InterfaceResponseTests(unittest.TestCase):
    def test_linear_response_has_zero_curvature(self):
        zero = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
        delta = torch.tensor([[2., 0.], [0., 3.]], dtype=torch.float64)
        result = module.finite_response(zero, zero + .25 * delta, zero - .25 * delta, .25)
        self.assertEqual(result["central_second_difference_l2"], [0., 0.])
        self.assertEqual(result["first_derivative_l2"], [2., 3.])

    def test_quadratic_response_known_derivatives(self):
        zero = torch.tensor([2., 3.], dtype=torch.float64)
        linear, quadratic, beta = torch.tensor([4., 5.]), torch.tensor([6., 7.]), .25
        result = module.finite_response(zero, zero + beta * linear + beta**2 * quadratic,
                                        zero - beta * linear + beta**2 * quadratic, beta)
        self.assertEqual(result["first_derivative_l2"], [4., 5.])
        self.assertEqual(result["second_derivative_l2"], [12., 14.])

    def test_zero_odd_response_is_explicit_not_infinite(self):
        value = torch.ones(2, 4)
        result = module.finite_response(value, value, value, .125)
        self.assertEqual(result["even_to_odd_norm_ratio"], [None, None])
        self.assertEqual(result["zero_odd_response_mask"], [True, True])

    def test_rejects_invalid_dose_shape_and_nonfinite(self):
        zero = torch.ones(2, 4)
        for beta in (0., -.5):
            with self.assertRaises(ValueError):
                module.finite_response(zero, zero, zero, beta)
        with self.assertRaises(ValueError):
            module.finite_response(zero, zero[:, :2], zero, .5)
        with self.assertRaises(ValueError):
            module.finite_response(zero, zero * float("nan"), zero, .5)

    def test_hooks_capture_first_only_and_cleanup(self):
        block, norm, projection = torch.nn.Identity(), torch.nn.LayerNorm(4), torch.nn.Linear(4, 3)
        value = torch.randn(3, 256, 4)
        capture = module.CaptureFirstInterfaces(block, norm, projection, 3, (0, 1))
        with capture:
            for _ in range(2):
                result = projection(norm(block(value)))
        self.assertEqual(capture.values["p3_edited_residual"].shape, (2, 256, 4))
        self.assertTrue(torch.equal(capture.values["predictor_proj_output"], result[:2]))
        self.assertTrue(all(count == 2 for count in capture.counts.values()))
        self.assertTrue(all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in (block, norm, projection)))

    def test_fixed_signed_doses_reverse_one_direction(self):
        direction = torch.tensor([[1., -2., 3.]])
        for beta in (.125, .25, .5):
            self.assertTrue(torch.equal(direction * (-beta), -(direction * beta)))
        self.assertEqual(module.INDICES, tuple(range(8)))
        self.assertEqual(set(module.DOSES), {0., .125, -.125, .25, -.25, .5, -.5})

    def test_adaln_projection_preserves_single_time_axis(self):
        identity = torch.nn.Identity()
        capture = module.CaptureFirstInterfaces(identity, identity, identity, 3, (0, 1))
        value = torch.randn(3, 1, 256, 384)
        capture.save("predictor_proj_input", value)
        self.assertEqual(capture.values["predictor_proj_input"].shape, (2, 1, 256, 384))
        with self.assertRaises(RuntimeError):
            capture.save("predictor_proj_output", torch.zeros(3, 2, 256, 384))
        with self.assertRaises(RuntimeError):
            capture.save("predictor_norm_input", value)

    def test_first_difference_variation_detects_odd_cubic(self):
        direction = torch.tensor([[1., 2.]], dtype=torch.float64)
        beta, reference = .25, .125
        result = module.first_difference_variation(direction * beta**3, -direction * beta**3, beta,
                                                   direction * reference**3, -direction * reference**3, reference)
        self.assertAlmostEqual(result["normalized_derivative_vector_change"][0], 3.)

    def test_derivative_direction_change_not_hidden_by_equal_norms(self):
        ref, changed = torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]])
        result = module.first_difference_variation(changed * .25, -changed * .25, .25,
                                                   ref * .125, -ref * .125)
        self.assertEqual(result["normalized_derivative_norm_change"], [0.])
        self.assertAlmostEqual(result["normalized_derivative_vector_change"][0], 2**.5)


if __name__ == "__main__":
    unittest.main()
