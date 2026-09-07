import importlib.util
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location("handoff", Path(__file__).parents[1] / "scripts/vast/run_author_extension_after_baseline.py")
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class HandoffTests(unittest.TestCase):
    def test_zombie_is_exited_but_sleeping_worker_is_live(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "123"
            directory.mkdir()
            for state, expected in (("S (sleeping)", True), ("R (running)", True), ("Z (zombie)", False)):
                (directory / "status").write_text("Name:\tpython\nState:\t" + state + "\n")
                self.assertEqual(handoff.process_is_running(123, root), expected)
            self.assertFalse(handoff.process_is_running(124, root))
