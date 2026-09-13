"""Presentation guards: source coverage and honest uncertainty definitions."""
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
spec = importlib.util.spec_from_file_location("publication_figures", ROOT/"scripts/build_publication_figures.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PublicationFigureTests(unittest.TestCase):
    def test_episode_se_is_not_sd_or_ci(self):
        self.assertAlmostEqual(module.episode_se(50, 96), 5.103103630798288)
        self.assertEqual(module.episode_se(0, 96), 0)
        self.assertEqual(module.episode_se(100, 96), 0)

    def test_invalid_episode_values_rejected(self):
        for value, n in ((-1, 96), (101, 96), (50, 1)):
            with self.assertRaises(ValueError):
                module.episode_se(value, n)

    def test_complete_margin_cohort(self):
        frame = module.load_margins()
        self.assertEqual(len(frame), 768)
        refined = frame[frame.arm == "refined"]
        self.assertEqual(int(refined.no_flip_certified.sum()), 184)
        self.assertEqual(int(refined.changed.sum()), 1)

    def test_no_margin_curve_dropping(self):
        frame = module.load_margins()
        self.assertEqual(len(frame.groupby(["task", "arm"])), 8)
        self.assertTrue((frame.groupby(["task", "arm"]).size() == 96).all())

    def test_changed_margin_table_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"paper/data").mkdir(parents=True)
            path = root/"paper/data/decision_geometry_scenarios.csv"
            shutil.copy(ROOT/"paper/data/decision_geometry_scenarios.csv", path)
            path.write_bytes(path.read_bytes()+b"\n")
            with self.assertRaisesRegex(ValueError, "table changed"):
                module.load_margins(root)


if __name__ == "__main__":
    unittest.main()
