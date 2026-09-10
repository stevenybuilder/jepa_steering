import copy
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("panel", Path(__file__).parents[1] / "scripts/geometry_map/prepare_pusht_independent.py")
panel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panel)


def fixture():
    return {"panel": "pusht_independent_reachable_v1", "episode_ids": list(range(50)),
            "starts": [{"episode": i, "environment_seed": 2026090700+i, "planner_seed": 92600+i, "template_episode": i % 10} for i in range(50)],
            "development_templates": [{"episode": i, "split": "development"} for i in range(10)],
            "action_domain": "relative_xy_times_100_unchanged_development_expert_commands"}


class PanelTests(unittest.TestCase):
    def test_fifty_independent_seeds_ten_fixed_clusters(self):
        rows, _ = panel.validate_manifest(fixture())
        self.assertEqual([sum(x["template_episode"] == i for x in rows) for i in range(10)], [5]*10)

    def test_refuses_held_template_before_loading(self):
        value = fixture()
        value["development_templates"][2]["split"] = "evaluation"
        with self.assertRaises(RuntimeError): panel.validate_manifest(value)

    def test_duplicate_seeds_rejected(self):
        value = fixture()
        value["starts"][1]["environment_seed"] = value["starts"][0]["environment_seed"]
        with self.assertRaises(RuntimeError): panel.validate_manifest(value)

    def test_no_absolute_action_substitution(self):
        value = fixture()
        value["action_domain"] = "absolute_xy"
        with self.assertRaises(RuntimeError): panel.validate_manifest(value)

    def test_native_unwrapped_bug_is_detected(self):
        value = panel.angular_errors(0, 4*math.pi+1)
        self.assertTrue(value["native_negative"])
        self.assertTrue(value["angle_threshold_disagreement"])
        self.assertAlmostEqual(value["robust_wrapped_angular_error"], 1)

    def test_canonical_wrap_agrees(self):
        value = panel.angular_errors(.02, 2*math.pi-.02)
        self.assertFalse(value["angle_threshold_disagreement"])
        self.assertAlmostEqual(value["native_angular_error"], .04)


if __name__ == "__main__": unittest.main()
