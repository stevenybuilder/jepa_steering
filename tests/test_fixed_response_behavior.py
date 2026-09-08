import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from offline_study.behavioral_development import schedule
from offline_study.fixed_response_behavior import ANALYSIS, ARMS, CONTRASTS, METHOD, ROLE, TASKS, ObservePlanner, validate_protocol
from offline_study.fixed_response_behavior_analysis import analyze, holm, validate_panel
from offline_study.protocol import sha256, write_json


def panel_fixture():
    return {task: {arm: [{**row, "arm": arm, "seconds": 1., "initial_state_vector": [row["episode"]],
        "result": {"initial_sha256": str(row["episode"]), "goal_sha256": str(row["episode"]),
                   "elementary_steps": 100, "native_success": False,
                   "native_state_distance": .1, "native_reward": 0.}}
        for row in schedule()] for arm in ARMS} for task in TASKS}


class FixedResponseBehaviorTests(unittest.TestCase):
    def test_complete_panel_and_real_pairing_are_mandatory(self):
        original = panel_fixture()
        validate_panel(original)
        for mutation in ("missing_arm", "missing_episode", "wrong_goal", "wrong_seed", "truncated_episode"):
            panel = copy.deepcopy(original)
            row = panel["reach"]["fixed_rank4"][0]
            if mutation == "missing_arm":
                del panel["reach"]["coupling_only"]
            elif mutation == "missing_episode":
                panel["reach"]["fixed_rank4"].pop()
            elif mutation == "wrong_goal":
                row["result"]["goal_sha256"] = "different"
            elif mutation == "wrong_seed":
                row["local_seed"] += 1
            else:
                row["result"]["elementary_steps"] = 30
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_panel(panel)

    def test_new_registry_does_not_accept_old_or_reduced_panels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = {"role": ROLE, "method": METHOD, "arms": list(ARMS), "tasks": {t: {} for t in TASKS},
                "episodes": schedule(), "analysis": ANALYSIS, "episodes_per_task_condition": 96,
                "primary_contrasts_per_task": [list(c) for c in CONTRASTS], "confirmation_outcomes_authorized": False}
            def save():
                write_json(root / "protocol.json", protocol)
                write_json(root / "FROZEN.json", {"protocol_sha256": sha256(root / "protocol.json")})
            save()
            validate_protocol(root)
            protocol["confirmation_outcomes_authorized"] = True
            save()
            with self.assertRaises(ValueError):
                validate_protocol(root)
            protocol["confirmation_outcomes_authorized"] = False
            protocol["episodes_per_task_condition"] = 12
            save()
            with self.assertRaises(ValueError):
                validate_protocol(root)

    def test_observer_counts_actual_unroll_and_restores_after_error(self):
        with tempfile.TemporaryDirectory() as temp:
            original = lambda *a, **k: {"visual": torch.ones(1), "proprio": torch.ones(1)}
            model = SimpleNamespace(unroll=original)
            backend = SimpleNamespace(model=model)
            class Adapter:
                last_record = {"response_probe_rollouts": 0, "native_shadow_rollouts": 0}
                calls = 1
                def __call__(self, context, actions, **kwargs):
                    for _ in range(self.calls):
                        result = model.unroll(context, actions)
                    return result
            adapter = Adapter()
            observed = ObservePlanner(adapter, backend, Path(temp))
            observed(None, torch.ones(6, 300, 4))
            self.assertIs(model.unroll, original)
            self.assertEqual(observed.calls[0]["backend_calls"], 1)
            adapter.calls = 2
            with self.assertRaisesRegex(ValueError, "Extra forecast"):
                observed(None, torch.ones(6, 300, 4))
            self.assertIs(model.unroll, original)

    def test_analysis_retains_null_results_and_full_multiplicity(self):
        result = analyze(panel_fixture())
        self.assertEqual(len(result["contrasts"]), 8)
        self.assertFalse(result["confirmation_outcomes_authorized"])
        self.assertFalse(result["full_study_complete"])
        for task in TASKS:
            self.assertEqual(result["tasks"][task]["selected_for_later_confirmation_freeze"], "native")
        self.assertEqual(holm([.01] * 8), [.08] * 8)
        with self.assertRaises(ValueError):
            holm([.01] * 4)

    def test_success_gain_requires_random_control_and_counts_duplicate_clusters(self):
        panel = panel_fixture()
        for task in TASKS:
            for arm in ARMS:
                rows = panel[task][arm]
                rows[1]["initial_state_vector"] = rows[0]["initial_state_vector"]
                rows[1]["result"]["initial_sha256"] = rows[0]["result"]["initial_sha256"]
                rows[1]["result"]["goal_sha256"] = rows[0]["result"]["goal_sha256"]
            for row in panel[task]["fixed_rank4"]:
                row["result"]["native_success"] = True
        result = analyze(panel)
        for task in TASKS:
            self.assertEqual(result["tasks"][task]["scenario_clusters"], 95)
            self.assertEqual(result["tasks"][task]["selected_for_later_confirmation_freeze"], "fixed_rank4")
            for row in panel[task]["matched_random_fixed_rank4"]:
                row["result"]["native_success"] = True
        result = analyze(panel)
        for task in TASKS:
            self.assertEqual(result["tasks"][task]["selected_for_later_confirmation_freeze"], "native")


if __name__ == "__main__":
    unittest.main()
