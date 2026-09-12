import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
import sys

path = Path(__file__).resolve().parents[1] / 'scripts/vast/component_completion_tail.py'
spec = importlib.util.spec_from_file_location('tail', path)
tail = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tail)
sys.path.insert(0,str(path.parent))
from component_rental_fleet import claimed_slots


class CompletionTests(unittest.TestCase):
    def test_capacity_allocator_cannot_rent_already_reserved_tail_slots(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            for slot in range(4):
                root=base/'rentals'/f'slot-{slot:02d}';root.mkdir(parents=True)
                (root/'RESERVATION.json').write_text(json.dumps({'gpu_offset':slot,'gpus':1}))
                (root/'CONTINUATION.json').write_text(json.dumps({'extra':tail.EXTRA[slot]}))
            self.assertEqual(claimed_slots(base),set(range(8)))

    def test_four_existing_workers_cover_all_original_scenarios_once(self):
        old = {slot: [('reach', slot), ('reach-wall', slot)] for slot in range(4)}
        extras = tail.coverage(old)
        all_streams = [s for work in old.values() for s in work] + [s for work in extras.values() for s in work]
        self.assertEqual(len(all_streams), 16)
        self.assertEqual(len(set(all_streams)), 16)
        self.assertEqual(len(all_streams) * len(tail.ARMS) * 12, 768)
        self.assertEqual(extras[2], [])  # Slow worker already has the longest original allocation.

    def test_changed_or_duplicate_original_assignment_is_rejected(self):
        old = {slot: [('reach', slot), ('reach-wall', slot)] for slot in range(4)}
        old[3] = old[0]
        with self.assertRaises(ValueError):
            tail.coverage(old)


if __name__ == '__main__':
    unittest.main()
