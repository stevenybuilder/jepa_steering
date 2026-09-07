import unittest
from torch.utils.data import DistributedSampler

from offline_study.training_pilot import VirtualRankBatchSampler


class TrainingSamplerTests(unittest.TestCase):
    def test_virtual_batches_match_each_native_distributed_stream(self):
        for length in (53568, 145800, 129, 127):
            sampler = VirtualRankBatchSampler(length, epoch=3)
            actual = list(sampler)
            self.assertEqual(len(actual), len(sampler))
            for rank in range(16):
                native = DistributedSampler(range(length), num_replicas=16, rank=rank, shuffle=True)
                native.set_epoch(3)
                stream = list(native)
                used = [index for batch in actual for index in batch[rank * 8:(rank + 1) * 8]]
                self.assertEqual(used, stream[:len(stream) // 8 * 8])


if __name__ == "__main__":
    unittest.main()
