import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/geometry_map"))
from capture_horizon_coordinates import horizon_labels, CaptureP3Horizons, exact, decode_physical, tensor_hash


class HorizonTests(unittest.TestCase):
    def test_late_native_single_context_time_labels(self):
        labels = horizon_labels(31, .62, np.arange(1, 7)*.1+.62)
        np.testing.assert_array_equal(labels["real_raw_step"], np.arange(1, 7)*5+31)
        np.testing.assert_array_equal(labels["context_real_raw_step"], np.full(6, 31))

    def test_fresh_tensor_receipt_binds_dtype_shape_values(self):
        value = torch.arange(4).float()
        self.assertEqual(tensor_hash(value), tensor_hash(value.clone()))
        self.assertNotEqual(tensor_hash(value), tensor_hash(value.half()))
        self.assertNotEqual(tensor_hash(value), tensor_hash(value.reshape(2, 2)))

    def test_real_and_imagined_time_distinct(self):
        labels = horizon_labels(16, .32, np.arange(1, 7)*.1+.32)
        np.testing.assert_array_equal(labels["imagined_step"], np.arange(1, 7))
        np.testing.assert_array_equal(labels["real_raw_step"], np.arange(1, 7)*5+16)
        self.assertTrue((labels["context_real_raw_step"] == 16).all())

    def test_invalid_time_rejected(self):
        with self.assertRaises(ValueError):
            horizon_labels(1, 0., np.zeros(6))

    def test_capture_all_six_newest_tokens_no_edit(self):
        block = torch.nn.Identity()
        value = torch.randn(1, 512, 400)
        with CaptureP3Horizons(block) as capture:
            for _ in range(6):
                self.assertIs(block(value), value)
        self.assertEqual(len(capture.values), 6)
        exact(capture.values[0], value[:, -256:], "newest")
        self.assertFalse(block._forward_hooks)

    def test_wrong_horizon_count_cleans_hooks(self):
        block = torch.nn.Identity()
        with self.assertRaises(RuntimeError):
            with CaptureP3Horizons(block):
                block(torch.zeros(1, 256, 400))
        self.assertFalse(block._forward_hooks)

    def test_physical_readout_not_future_label(self):
        readout = {"mean": np.zeros(4), "scale": np.ones(4), "coef": np.eye(3, 4), "intercept": np.arange(3)}
        np.testing.assert_array_equal(decode_physical(np.ones((2, 4)), readout), np.tile(np.arange(3)+1, (2, 1)))

    def test_exact_rejects_small_difference(self):
        with self.assertRaises(RuntimeError):
            exact(np.array([1.]), np.array([1.+1e-12]), "strict")


if __name__ == "__main__":
    unittest.main()
