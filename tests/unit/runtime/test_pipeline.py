import tempfile
import unittest
from pathlib import Path

import torch

from offline_study.runtime.pipeline import AsyncCacheWriter, PrefetchIterator


class PipelineTests(unittest.TestCase):
    def test_prefetch_preserves_order_and_propagates_errors(self):
        iterator = PrefetchIterator(range(7), depth=2)
        self.assertEqual(list(iterator), list(range(7)))
        self.assertEqual(iterator.items_produced, 7)
        self.assertGreaterEqual(iterator.producer_seconds, 0.)
        self.assertGreaterEqual(iterator.consumer_wait_seconds, 0.)

        def broken():
            yield "first"
            raise RuntimeError("decode failed")

        iterator = PrefetchIterator(broken(), depth=1)
        self.assertEqual(next(iterator), "first")
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            next(iterator)

    def test_zero_depth_is_synchronous(self):
        iterator = PrefetchIterator([1, 2], depth=0)
        self.assertEqual(list(iterator), [1, 2])
        self.assertEqual(iterator.items_produced, 2)
        self.assertGreaterEqual(iterator.consumer_wait_seconds, 0.)

    def test_async_cache_writer_drains_before_close_returns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = AsyncCacheWriter(depth=2)
            for index in range(3):
                writer.submit(root / f"{index}.pt", {"value": torch.tensor([index])})
            writer.close()
            self.assertEqual(
                [int(torch.load(root / f"{index}.pt", weights_only=True)["value"][0]) for index in range(3)],
                [0, 1, 2],
            )
            self.assertGreaterEqual(writer.writer_seconds, 0.)


if __name__ == "__main__":
    unittest.main()
