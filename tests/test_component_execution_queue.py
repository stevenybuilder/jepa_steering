"""Allocation tests only: never contact a provider or start a GPU process."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/vast/component_extension_queue.py'
spec = importlib.util.spec_from_file_location('component_queue', path)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class ComponentQueueTests(unittest.TestCase):
    def test_single_host_preserves_all_16_streams(self):
        expected = {(t, r) for t in ('reach', 'reach-wall') for r in range(8)}
        for gpus in (1, 2, 4, 8):
            queues = queue.assignments(gpus)
            items = [x for work in queues.values() for x in work]
            self.assertEqual(len(items), 16)
            self.assertEqual(set(items), expected)
            self.assertEqual(12 * 4 * len(items), 768)

    def test_eight_independent_workers_not_eight_times_the_sample(self):
        slices = [queue.assignments(1, 8, slot)[0] for slot in range(8)]
        flattened = [x for work in slices for x in work]
        self.assertEqual(len(set(flattened)), 16)
        self.assertEqual(len(flattened), 16)
        for slot, work in enumerate(slices):
            self.assertEqual(work, [('reach', slot), ('reach-wall', slot)])
            self.assertEqual(len(work) * 4 * 12, 96)  # BOTH tasks, all FOUR arms.

    def test_invalid_and_overlapping_slot_range_rejected(self):
        for args in ((2, 8, 7), (1, 8, -1), (1, 8, 8), (3, 8, 0), (1, 17, 0)):
            with self.assertRaises(ValueError):
                queue.assignments(*args)

    def test_mixed_hosts_cover_each_original_stream_once(self):
        allocations = [queue.assignments(1, 8, 1), queue.assignments(1, 8, 2),
            queue.assignments(2, 8, logical_slots=[0, 3]),
            queue.assignments(4, 8, logical_slots=[4, 5, 6, 7])]
        streams = [s for allocation in allocations for work in allocation.values() for s in work]
        self.assertEqual(len(streams), 16)
        self.assertEqual(len(set(streams)), 16)
        self.assertEqual(len(streams) * 4 * 12, 768)
        for allocation in allocations:
            for work in allocation.values():
                self.assertEqual({rank for _, rank in work}, {work[0][1]})

    def test_explicit_slots_reject_duplicate_and_out_of_range(self):
        for slots in ([0, 0], [0, 8], [-1, 1], [0]):
            with self.assertRaises(ValueError):
                queue.assignments(2, 8, logical_slots=slots)


if __name__ == '__main__':
    unittest.main()
