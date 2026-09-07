import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts/geometry_map"))
SPEC = importlib.util.spec_from_file_location("monitor_eight_addendum", ROOT / "scripts/geometry_map/monitor_eight_addendum.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class MonitorTests(unittest.TestCase):
    def test_script_hash_is_path(self):
        self.assertEqual(len(M.sha(Path(M.__file__))), 64)

    def test_no_local_receipt_does_not_deserialize(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(M.verify_local({"local_directory": "not-created"}, Path(folder)))

    def test_lease_missing_prevents_remote_call(self):
        with tempfile.TemporaryDirectory() as folder:
            board = Path(folder) / "board.md"
            board.write_text("No leased worker\n")
            with self.assertRaisesRegex(RuntimeError, "Lease"):
                M.check({"local_directory": "none", "instance_id": 1}, Path(folder), {}, board)


if __name__ == "__main__": unittest.main()
