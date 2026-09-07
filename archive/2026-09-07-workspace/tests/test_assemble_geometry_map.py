"""Focused assembly semantics; no model loading or held-out tensor access."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("assembler", ROOT / "scripts/geometry_map/assemble_geometry_map.py")
assembler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assembler)


class AssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ROOT / "artifacts/geometry_map/reach_wall_v1"
        cls.rows, cls.sources, cls.cached, cls.causal = assembler.join_evidence(cls.root, causal_path=ROOT / "artifacts/geometry_map/reach_wall_causal_measurements_v1/causal_measurements.json")

    def test_labels_do_not_promote_proxies(self):
        names = {r["task_variable"] for r in self.rows}
        self.assertIn("cumulative_progress", names)
        self.assertIn("step_progress", names)
        self.assertNotIn("progress", names)
        self.assertIn("hand_to_goal_segment_intersects_wall", names)
        self.assertNotIn("candidate_path_crossing", names)

    def test_source_references_and_unique_rows(self):
        self.assertEqual(len({r["row_id"] for r in self.rows}), len(self.rows))
        for row in self.rows:
            self.assertTrue(row["sources"])
            for source in row["sources"]:
                self.assertIn(source["path"], self.sources)

    def test_no_unvalidated_operator(self):
        self.assertTrue(all(r["recommended_operator"] == "none_validated" for r in self.rows))
        causal_rows = [r for r in self.rows if r["causal_patch_effect"]]
        self.assertEqual(len(causal_rows), 1)
        self.assertEqual(causal_rows[0]["causal_patch_effect"]["independent_starts"], 1)

    def test_old_angles_not_imported(self):
        self.assertFalse(any("subspace_comparisons" in r for r in self.rows))
        self.assertTrue(any("common_metric_subspace_comparisons_reference" in (r["geometry"] or {}) for r in self.rows))

    def test_natural_support_not_edited_support(self):
        row = next(r for r in self.rows if r.get("site_geometry_id"))
        self.assertTrue(row["natural_activation_support_reference"])
        self.assertIsNone(row["manifold_distance"])

    def test_followups_preserve_negative_and_proxy_interpretations(self):
        rows, sources, _, _ = assembler.join_evidence(self.root)
        base = ROOT / "artifacts/geometry_map/reach_wall_support_replication_v1"
        extras = assembler.attach_validation(rows, sources, {
            "support": base / "support-output-v1/support_summary.json",
            "replication": base / "replication-output-v1/replication_summary.json",
            "specificity": self.root / "specificity-controls-v1/specificity_controls.json",
            "nonlinear_interfaces": self.root / "nonlinear-interfaces-v1/nonlinear_interfaces.json"})
        row = next(r for r in rows if r["module"] == "predictor" and r["block"] == 3
                   and r["task_variable"] == "realized_xz_direction")
        self.assertEqual(row["causal_patch_effect"]["independent_starts"], 5)
        self.assertEqual(row["manifold_distance"]["true_manifold_membership"], "not_established")
        self.assertEqual(row["specificity"]["behavioral_specificity"], "not_established")
        self.assertEqual(row["held_validation_episodes"], 75)
        self.assertEqual(row["recommended_operator"], "none_validated")
        self.assertFalse(extras["nonlinear_interfaces"]["confirmation_opened"])

    def test_plots_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "heatmap.svg"
            assembler.heatmap(self.rows, path)
            ET.parse(path)
        self.assertIsNone(assembler.finite(float("nan")))
        json.dumps(assembler.finite({"x": float("inf")}), allow_nan=False)

    def test_coordinate_followups_keep_task_and_claim_scope(self):
        rows, sources, _, _ = assembler.join_evidence(self.root)
        extras = assembler.attach_validation(rows, sources, {
            "coordinate_time": self.root/"coordinate-time-v1/coordinate_time_results.json",
            "pusht_coordinate_time": ROOT/"artifacts/geometry_map/pusht_coordinate_screen_v1/output-v1/coordinate_time_results.json"})
        new = [r for r in rows if r["task_variable"].startswith("coordinate_probe/")]
        self.assertTrue(any(r["task"] == "push-t" for r in new))
        self.assertTrue(any(r["task"] == "reach-wall" for r in new))
        self.assertTrue(all(r["recommended_operator"] == "none_validated" for r in new))
        self.assertTrue(all(r["nonlinear_gain"]["manifold_evidence"] == "not_established" for r in new))
        self.assertTrue(all(r["block"] == 3 for r in new))

    def test_single_point_control_is_visible(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/"control.svg"
            assembler.line_plot(path, "control", [("first", [(0, 6.)])], "condition", "mm")
            tree = ET.parse(path)
            self.assertEqual(len(tree.findall(".//{http://www.w3.org/2000/svg}circle")), 1)

    def test_action_response_is_not_a_probe_or_accepted_operator(self):
        rows, sources, _, _ = assembler.join_evidence(self.root)
        paths = {"action_interactions": ROOT/"artifacts/geometry_map/reach_wall_action_interactions_v1/summary-v1.json",
                 "token_time": ROOT/"artifacts/geometry_map/reach_wall_action_interactions_v1/spatial-token-time-v1/action_token_time.json"}
        extras = {k: assembler.load_source(p, sources) for k, p in paths.items()}
        assembler.attach_action_time(rows, sources, extras, paths)
        selected = [r for r in rows if r["task_variable"] == "finite_dose_action_response"]
        self.assertEqual({r["block"] for r in selected}, {0, 3, 5})
        self.assertTrue(all(r["nonlinear_gain"] is None for r in selected))
        self.assertTrue(all(r["causal_patch_effect"] is None for r in selected))
        self.assertTrue(all(r["imagined_timestep"] == [1, 2, 3, 4, 5, 6] for r in selected))
        self.assertTrue(all(r["recommended_operator"] == "none_validated" for r in selected))

    def test_residual_physics_preserves_negative_results_and_episode_unit(self):
        paths = sorted((self.root/"residual-coordinate-time-v1").glob("worker-*/summary-v1/physical_effects.json"))
        rows, sources = [], {}
        result = assembler.attach_residual_physics(rows, sources, paths)
        self.assertEqual(result["independent_starts"], 3)
        self.assertEqual(len(result["rows"]), 15)
        self.assertFalse(result["operator_validated"])
        self.assertTrue(all(c["mean_progress_vs_sham_m"] < 0 for c in result["paired_contrasts"].values()))
        self.assertEqual(rows[0]["recommended_operator"], "none_validated")
        with self.assertRaises(ValueError):
            assembler.attach_residual_physics([], {}, paths+paths[:1])


if __name__ == "__main__":
    unittest.main()
