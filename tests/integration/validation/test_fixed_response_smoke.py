from offline_study._paths import source_path
import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from offline_study.validation import fixed_response_smoke as smoke
from offline_study.interventions.fixed_response import ARMS, METHOD
from offline_study.core.protocol import sha256, write_json


class FixedResponseSmokeTests(unittest.TestCase):
    def test_full_cem_schedule_and_episode_length_are_not_optional(self):
        result = {"planning_calls": [{"iterations": 15, "steps_left": 20},
                                     {"iterations": 15, "steps_left": 2}],
                  "elementary_steps": 100, "published_candidate_count": 300}
        calls = [{"horizon": horizon, "candidates": count}
                 for horizon in (6, 2) for _ in range(15) for count in (300, 1)]
        smoke.verify_episode(result, calls)
        for shortened in (calls[:-1], calls[1:], calls[::-1]):
            with self.assertRaises(ValueError):
                smoke.verify_episode(result, shortened)
        with self.assertRaises(ValueError):
            smoke.verify_episode({**result, "elementary_steps": 30}, calls)

    def test_native_repeat_checks_actions_but_edits_may_change_outcomes(self):
        baseline = {"result": {"initial_sha256": "i", "goal_sha256": "g", "expert_success": 1.,
            "native_success": False, "native_state_distance": .2, "native_reward": 0.,
            "elementary_steps": 100, "observed_frames": 101, "published_candidate_count": 300},
            "action_trace": ["a"]}
        changed = copy.deepcopy(baseline)
        changed["result"]["native_success"] = True
        changed["action_trace"] = ["different"]
        smoke.verify_pair(baseline, changed)
        with self.assertRaises(ValueError):
            smoke.verify_pair(baseline, changed, native_repeat=True)
        changed = copy.deepcopy(baseline)
        changed["action_trace"] = ["different"]
        with self.assertRaisesRegex(ValueError, "actions/observations"):
            smoke.verify_pair(baseline, changed, native_repeat=True)
        changed["result"]["goal_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "stimuli"):
            smoke.verify_pair(baseline, changed)

    def test_trace_is_observational_and_checks_nonfinite_values(self):
        agent = SimpleNamespace(act=lambda obs, steps_left: obs["visual"] + steps_left)
        trace = smoke.trace_actor(agent)
        before = torch.get_rng_state().clone()
        result = agent.act({"visual": torch.zeros(2)}, steps_left=3)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertTrue(torch.equal(result, torch.full((2,), 3.)))
        self.assertEqual(trace[0]["actions_sha256"], smoke.tensor_digest(result))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            smoke.tensor_digest(torch.tensor([float("nan")]))

    def test_numerical_receipt_binds_fit_source_and_both_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            check, fit, source = [root / name for name in ("check", "fit", "source")]
            for path in (check, fit, source):
                path.mkdir()
            write_json(fit / "DONE.json", {"fixture": True})
            torch.save({}, fit / "operator_bank.pt")
            path = source / "fixed_response.py"
            shutil.copyfile(source_path(path.name), path)
            digest = hashlib.sha256(path.name.encode() + b"\0" + path.read_bytes()).hexdigest()
            contract = {"method": METHOD, "task": "mw-reach", "arms": list(ARMS),
                "fit_done_sha256": sha256(fit / "DONE.json"), "fit_bank_sha256": sha256(fit / "operator_bank.pt"),
                "planning_context": 2, "precision": "float32_strict_no_tf32", "source_sha256": digest}
            write_json(check / "contract.json", contract)
            report = {"contract_sha256": sha256(check / "contract.json"),
                "status": "fixed_map_fit_action_planning_engineering_complete", "parameters_unchanged": True,
                "scientific_efficacy_measurement": False, "backend": {"checkpoint_sha256": "checkpoint"},
                "checks": [{"arm": arm, "candidate_count": n, "static_reference_bitwise_equal": True,
                    "backend_calls": 1, "response_probes": 0, "native_shadows": 0}
                    for arm in ARMS[2:] for n in (8, 19, 300)]}
            def save():
                write_json(check / "report.json", report)
                write_json(check / "DONE.json", {"report_sha256": sha256(check / "report.json")})
            save()
            smoke.verify_numerical(check, fit, source, "reach", "checkpoint")
            with self.assertRaisesRegex(ValueError, "bind"):
                smoke.verify_numerical(check, fit, source, "reach-wall", "checkpoint")
            report["checks"].pop()
            save()
            with self.assertRaisesRegex(ValueError, "population"):
                smoke.verify_numerical(check, fit, source, "reach", "checkpoint")
            write_json(check / "FAILED.json", {"error": "fixture"})
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                smoke.verify_numerical(check, fit, source, "reach", "checkpoint")


if __name__ == "__main__":
    unittest.main()
