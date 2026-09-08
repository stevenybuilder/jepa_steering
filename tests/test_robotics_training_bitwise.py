"""Exact numerical-proof comparisons; CPU tests never establish GPU parity."""
from collections import OrderedDict
import copy
import unittest

import numpy as np
import torch

from offline_study.robotics_training_pilot import assert_bitwise


class BitwiseProofTests(unittest.TestCase):
    def test_signed_zero_not_accepted_as_bitwise_equal(self):
        for positive, negative in ((0.0, -0.0), (np.float32(0), np.float32(-0.0)),
                (np.array([0.]), np.array([-0.])),
                (torch.tensor([0.]), torch.tensor([-0.])),
                (torch.tensor(0., dtype=torch.bfloat16), torch.tensor(-0., dtype=torch.bfloat16))):
            with self.assertRaises(ValueError):
                assert_bitwise(positive, negative)

    def test_nested_optimizer_and_rng_roundtrip(self):
        state = OrderedDict(parameters=torch.arange(12.).reshape(3, 4),
            optimizer={0: {"step": torch.tensor(1.), "exp_avg": torch.ones(3)}},
            python=(3, (1, 2, 3), None), numpy=np.random.RandomState(234).get_state(),
            flags=[True, "native", 1e-7])
        assert_bitwise(state, copy.deepcopy(state))
        changed = copy.deepcopy(state)
        changed["optimizer"][0]["exp_avg"][1] += 1
        with self.assertRaises(ValueError):
            assert_bitwise(state, changed)

    def test_types_shapes_and_layout_are_not_coerced(self):
        for left, right in ((True, 1), ([1], (1,)), ({1: 0}, {True: 0}),
                (torch.ones(2), torch.ones(1, 2)),
                (torch.ones(2, 2), torch.ones(2, 2).T),
                (np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float64)),
                (np.ones((2, 2)), np.ones((2, 2)).T)):
            with self.assertRaises(ValueError):
                assert_bitwise(left, right)

    def test_empty_and_scalar_dense_tensors_supported(self):
        for tensor in (torch.empty(0), torch.tensor(1), torch.tensor(1., dtype=torch.bfloat16)):
            assert_bitwise(tensor, tensor.clone())

    def test_unsupported_object_and_sparse_state_refused(self):
        for item in (object(), np.array([object()], dtype=object),
                     torch.sparse_coo_tensor([[0]], [1.], (2,))):
            with self.assertRaises(ValueError):
                assert_bitwise(item, item)


if __name__ == "__main__":
    unittest.main()
