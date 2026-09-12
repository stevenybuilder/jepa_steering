import copy
import tempfile
import unittest
from pathlib import Path

import torch

from offline_study.behavioral_development import schedule
from offline_study.metaworld_component_behavior import (
    ANALYSIS, ARMS, CONTRASTS, ENGINEERING, METHOD, TASKS,
    expected_energy, validate_protocol, verify_energy,
)
from offline_study.metaworld_component_analysis import analyze, validate_panel
from offline_study.planning_native_smoke import CHECKPOINTS, SMOKE_SEED
from offline_study.protocol import sha256, write_json


def fixture():
    return {task: {arm: [{**row, "arm": arm, "seconds": 1., "initial_state_vector": [row["episode"]],
        "result": {"initial_sha256": str(row["episode"]), "goal_sha256": str(row["episode"]),
            "native_success": False, "native_state_distance": .1, "native_reward": 0., "elementary_steps": 100}}
        for row in schedule()] for arm in ARMS} for task in TASKS}


class ComponentTests(unittest.TestCase):
    def test_count_pairing_and_no_legacy_rank(self):
        self.assertEqual(len(TASKS) * len(ARMS) * len(schedule()), 768)
        self.assertEqual(len(TASKS) * (len(ARMS)-1) * len(schedule()), 576)
        self.assertFalse(any("rank" in arm for arm in (*ARMS, *ENGINEERING)))
        self.assertEqual(len(CONTRASTS) * len(TASKS), ANALYSIS["family"])
        validate_panel(fixture())
        for mutation in ("missing", "duplicate", "goal", "steps", "arm", "seed"):
            panel = fixture()
            rows = panel["reach"]["visual_only"]
            if mutation == "missing": rows.pop()
            elif mutation == "duplicate": rows[-1] = copy.deepcopy(rows[0])
            elif mutation == "goal": rows[0]["result"]["goal_sha256"] = "wrong"
            elif mutation == "steps": rows[0]["result"]["elementary_steps"] = 30
            elif mutation == "arm": rows[0]["arm"] = "joint"
            else: rows[0]["local_seed"] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_panel(panel)

    def test_native_and_shortened_forecasts_have_zero_dose(self):
        protocol = {"arms": [{"name": "native", "edits": []}, {"name": "joint", "edits": [
            {"tensor": "visual", "scale": .2}, {"tensor": "action", "scale": .3}]}]}
        bank = {"global_tensors": {"visual": torch.tensor([1., 0.]), "action": torch.tensor([0., 1.])}}
        self.assertAlmostEqual(expected_energy(protocol, bank, "joint"), .13)
        call = {"horizon": 6, "backend_calls": 1, "energy": {
            "requested_squared_l2_mean": .13, "realized_squared_l2_mean": .13}}
        verify_energy([call], protocol, bank, "joint")
        call["horizon"] = 3
        with self.assertRaises(ValueError): verify_energy([call], protocol, bank, "joint")
        call["energy"] = {"requested_squared_l2_mean": 0., "realized_squared_l2_mean": 0.}
        verify_energy([call], protocol, bank, "joint")
        call["horizon"] = 6
        verify_energy([call], protocol, bank, "native")
        with self.assertRaises(ValueError): verify_energy([call], protocol, bank, "joint")

    def test_complete_frozen_analysis_retains_nulls_and_interaction(self):
        result = analyze(fixture())
        self.assertEqual(len(result["contrasts"]), 12)
        self.assertFalse(result["automatic_candidate_selection"])
        self.assertFalse(result["fresh_confirmation"])
        self.assertTrue(all(row["simultaneous_95_interval_percentage_points"] == [0., 0.] for row in result["contrasts"]))
        panel = fixture()
        for row in panel["reach"]["joint"][:6]: row["result"]["native_success"] = True
        result = analyze(panel)
        row = next(r for r in result["contrasts"] if r["task"] == "reach" and r["contrast"] == "joint_vs_native")
        self.assertEqual(row["gain_percentage_points"], 6.25)

    def test_altered_registry_or_protected_access_rejected(self):
        p = {"method": METHOD, "arms": list(ARMS), "contrasts": CONTRASTS, "analysis": ANALYSIS,
            "episodes": schedule(), "episodes_per_task_condition": 96, "total_episode_evaluations": 768,
            "fresh_confirmation": False, "protected_access": False, "legacy_expensive_operator": False,
            "precision": "float32_strict_no_tf32", "engineering_seed": SMOKE_SEED,
            "engineering_order": list(ENGINEERING), "checkpoint_sha256": CHECKPOINTS["metaworld"],
            "tasks": {task: {} for task in TASKS}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def save(value):
                write_json(root / "protocol.json", value)
                write_json(root / "FROZEN.json", {"protocol_sha256": sha256(root / "protocol.json")})
            save(p); validate_protocol(root)
            for key, value in (("protected_access", True), ("episodes_per_task_condition", 12), ("arms", ["native"]),
                               ("legacy_expensive_operator", True)):
                changed = {**p, key: value}; save(changed)
                with self.subTest(key=key), self.assertRaises(ValueError): validate_protocol(root)


if __name__ == "__main__":
    unittest.main()
