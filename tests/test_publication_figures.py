"""Presentation guards: source coverage and honest uncertainty definitions."""
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
spec = importlib.util.spec_from_file_location("publication_figures", ROOT/"scripts/build_publication_figures.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PublicationFigureTests(unittest.TestCase):
    def test_benchmark_labels_and_selection_pool(self):
        self.assertEqual(module.BENCHMARK_LOCAL_LABELS, ("Unsteered", "Best edit"))
        self.assertEqual(set(module.EDIT_LABELS), set(module.ARMS) - {"native"})

    def test_benchmark_uses_fresh_count_based_rates(self):
        source, report = module.load_data()
        expected_counts = {"reach": (52, 52), "reach-wall": (28, 26),
                           "pointmaze": (83, 85), "wall": (78, 80)}
        for task, counts in expected_counts.items():
            self.assertEqual(report["results"][task]["n"], 96)
            self.assertAlmostEqual(report["results"][task]["success_percent"]["native"], counts[0]/96*100)
            self.assertAlmostEqual(module.best_observed_edit(report, task)["success_percent"], counts[1]/96*100)
        self.assertEqual(source["development_rows"][0]["values"][0], 44.79)

    def test_best_edit_retains_ties_and_randomized_winner(self):
        _, report = module.load_data()
        self.assertEqual(module.best_observed_edit(report, "reach")["arms"], ("fixed_rank4", "visual_only"))
        self.assertEqual(module.best_observed_edit(report, "reach-wall")["arms"], ("matched_random_coupling",))
        self.assertEqual(module.best_observed_edit(report, "pointmaze")["arms"], ("fixed_rank4",))
        self.assertEqual(module.best_observed_edit(report, "wall")["arms"], ("coupling_only",))

    def test_benchmark_tick_labels_do_not_collide(self):
        module.style()
        with patch.object(module, "save") as save:
            module.benchmark()
        fig = save.call_args.args[0]
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for ax in fig.axes:
                boxes = [tick.get_window_extent(renderer) for tick in ax.get_xticklabels()]
                self.assertTrue(all(a.x1 + 2 < b.x0 for a, b in zip(boxes, boxes[1:])))
        finally:
            module.plt.close(fig)

    def test_margin_legend_clears_panel_titles(self):
        module.style()
        with patch.object(module, "save") as save:
            module.margins()
        fig = save.call_args.args[0]
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend = fig.legends[0].get_window_extent(renderer)
            for ax in fig.axes:
                self.assertGreater(legend.y0, ax._left_title.get_window_extent(renderer).y1 + 2)
        finally:
            module.plt.close(fig)

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
