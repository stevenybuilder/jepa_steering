import unittest

import numpy as np
import torch

from offline_study.tasks.droid.droid_native import array_hash, assert_same, observation_digest, peek_stimulus
from offline_study.tasks.droid.droid_replication import assigned_rows
from offline_study.planning.planning_contract import seed_schedule


class DroidNativeTests(unittest.TestCase):
    def test_trace_preserves_persistent_native_generator(self):
        generator = torch.Generator().manual_seed(3001)
        for _ in range(8):
            before = generator.get_state()
            index, offset, after = peek_stimulus(generator, 15)
            self.assertTrue(torch.equal(before, generator.get_state()))
            self.assertEqual(index, torch.randint(0, 15, (1,), generator=generator).item())
            self.assertEqual(offset, torch.randint(0, 2, (1,), generator=generator).item())
            self.assertTrue(torch.equal(after, generator.get_state()))

    def test_shards_keep_complete_logical_streams(self):
        rows = seed_schedule(1, 64, 8, 3)
        parts = [assigned_rows(rows, [rank]) for rank in range(8)]
        self.assertEqual([row for part in parts for row in part], rows)
        self.assertEqual([len(part) for part in parts], [8] * 8)
        for ranks in ([], [0, 0], [8]):
            with self.assertRaises(ValueError):
                assigned_rows(rows, ranks)

    def test_parity_includes_dtype_and_rng_state(self):
        value = ({"pixels": torch.zeros(2, 3)}, np.random.RandomState(234).get_state())
        assert_same(value, value)
        with self.assertRaises(ValueError):
            assert_same(torch.zeros(3), torch.zeros(3, dtype=torch.float64))
        with self.assertRaises(ValueError):
            assert_same(np.random.RandomState(234).get_state(), np.random.RandomState(235).get_state())

    def test_fingerprint_includes_shape_and_dtype(self):
        array = np.zeros((3, 7), dtype=np.float64)
        self.assertEqual(array_hash(array), array_hash(array.copy()))
        self.assertNotEqual(array_hash(array), array_hash(array.reshape(7, 3)))
        self.assertNotEqual(array_hash(array), array_hash(array.astype(np.float32)))

    def test_droid_uses_256px_not_other_tasks_224px(self):
        obs = {"visual": torch.arange(3 * 256 * 256).remainder(256).byte().reshape(1, 3, 256, 256),
               "proprio": torch.zeros(1, 7)}
        self.assertEqual(len(observation_digest(obs)), 64)
        obs["visual"] = obs["visual"][:, :, :224, :224]
        with self.assertRaises(ValueError):
            observation_digest(obs)


if __name__ == "__main__":
    unittest.main()
