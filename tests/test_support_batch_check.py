import unittest

import torch

from offline_study.planning_support import TRANSFER_POLICY
from offline_study.support_batch_check import difference, response_chunk_size


class ResponseBatchTests(unittest.TestCase):
    def test_fixed_scheduling_is_restored_on_failure(self):
        original = dict(TRANSFER_POLICY)
        with self.assertRaises(RuntimeError):
            with response_chunk_size(16):
                self.assertEqual(TRANSFER_POLICY["candidate_chunk_size"], 16)
                raise RuntimeError("synthetic")
        self.assertEqual(TRANSFER_POLICY, original)
        with self.assertRaises(ValueError):
            with response_chunk_size(24):
                pass

    def test_numerical_difference_is_not_hidden(self):
        a = torch.ones(5)
        self.assertTrue(difference(a, a.clone())["bitwise_equal"])
        self.assertFalse(difference(a, a + 1e-5)["bitwise_equal"])


if __name__ == "__main__":
    unittest.main()
