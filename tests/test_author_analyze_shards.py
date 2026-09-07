import tempfile
import unittest
from pathlib import Path

from offline_study.author_analyze import expected_shards


class AnalyzeShardTests(unittest.TestCase):
    def test_accepts_complete_physical_partition_not_partial_or_extra(self):
        for count in (2, 8, 32):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for index in range(count - 1):
                    (root / f"shard-{index}").mkdir()
                with self.assertRaises(ValueError):
                    expected_shards(root, count)
                (root / f"shard-{count - 1}").mkdir()
                self.assertEqual(len(expected_shards(root, count)), count)
                (root / f"shard-{count}").mkdir()
                with self.assertRaises(ValueError):
                    expected_shards(root, count)

    def test_requires_directories_and_positive_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "shard-0").touch()
            with self.assertRaises(ValueError):
                expected_shards(root, 1)
            with self.assertRaises(ValueError):
                expected_shards(root, 0)
