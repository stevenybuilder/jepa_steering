"""The readable history plot must retain its full measured cohort."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"scripts"))
spec = importlib.util.spec_from_file_location("action_history_figure", ROOT/"scripts/build_action_history_figure.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class HistoryFigureTests(unittest.TestCase):
    def test_all24_cells_and16_context_scope(self):
        frame, provenance = m.load()
        self.assertEqual(len(frame), 24)
        self.assertEqual(provenance["cohort"], 16)
        self.assertTrue((frame.n == 8).all())
        self.assertEqual(set(frame.horizon), {3, 4, 6})
        self.assertEqual(set(frame.bank), {"original", "fresh"})

    def test_unpreserved_or_incomplete_cannot_be_plotted(self):
        for key, value in (("all_cloud_verified", False), ("cohort", 15), ("physical_outcomes_measured", True)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = root/"paper/data"
                data.mkdir(parents=True)
                receipt = json.loads((ROOT/"paper/data/action_counterfactual_summary.json").read_text())
                receipt[key] = value
                (data/"action_counterfactual_summary.json").write_text(json.dumps(receipt))
                with patch.object(m, "ROOT", root), self.assertRaises(ValueError):
                    m.load()

    def test_changed_metric_table_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/"paper/data"
            data.mkdir(parents=True)
            shutil.copy(ROOT/"paper/data/action_counterfactual_summary.json", data)
            (data/"action_counterfactual_summary.csv").write_text("tampered")
            with patch.object(m, "ROOT", root), self.assertRaisesRegex(ValueError, "summary changed"):
                m.load()


if __name__ == "__main__":
    unittest.main()
