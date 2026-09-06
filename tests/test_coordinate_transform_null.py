import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"/"geometry_map"))
from coordinate_transform_null import transform_control


class TransformNullTests(unittest.TestCase):
    def test_state_dependent_rotations_preserve_error(self):
        rng = np.random.default_rng(18)
        rotations = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(30)])
        current, following, predicted = rng.normal(size=(3, 30, 3))
        converted, target, errors = transform_control(predicted, current, following, rotations)
        np.testing.assert_allclose(errors, np.square(predicted-following).mean(1), atol=1e-12)
        self.assertEqual(converted.shape, target.shape)

    def test_nonorthogonal_transforms_fail(self):
        with self.assertRaises(ValueError):
            transform_control(np.ones((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)), np.broadcast_to(2*np.eye(3), (2, 3, 3)))

    def test_float32_inputs_accumulate_in_same_metric(self):
        rng = np.random.default_rng(12)
        truth = rng.normal(size=(40, 3)).astype(np.float32)
        pred = truth+np.float32(.0001)
        current = rng.normal(size=(40, 3)).astype(np.float32)
        rotation = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(40)])
        transform_control(pred, current, truth, rotation)


if __name__ == "__main__":
    unittest.main()
