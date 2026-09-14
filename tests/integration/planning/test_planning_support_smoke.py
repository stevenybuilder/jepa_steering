import copy
import unittest

from offline_study.planning.planning_support import TRANSFER_POLICY
from offline_study.planning.planning_support_smoke import validate_transfer, verify_call_schedule


class SelectedPlanningSmokeTests(unittest.TestCase):
    def test_requires_full_cem_population_every_iteration(self):
        plans = [{"steps_left": h, "iterations": 15} for h in (20, 17, 14, 11, 8, 5, 2)]
        calls = [{"horizon": min(6, p["steps_left"]), "candidates": b}
                 for p in plans for _ in range(15) for b in (300, 1)]
        verify_call_schedule(plans, calls)
        with self.assertRaises(ValueError):
            verify_call_schedule(plans, calls[:-1])
        changed = copy.deepcopy(calls)
        changed[0]["candidates"] = 19
        with self.assertRaises(ValueError):
            verify_call_schedule(plans, changed)
        changed = copy.deepcopy(calls)
        changed[-1]["horizon"] = 6
        with self.assertRaises(ValueError):
            verify_call_schedule(plans, changed)

    def test_transfer_checks_bind_exact_task_arm_and_source(self):
        binding = {"protocol_sha256": "p", "bank_sha256": "b"}
        report = {"status": "primary_rank_planning_transfer_fit_only_passed",
                  "profiled_rank_arm": "rank4", "combined_planning_transfer_validated": False,
                  "diverse_fit_actions_checked": True, "parameters_unchanged": True,
                  "development_or_protected_outcomes_accessed": False,
                  "planning_precision": "float32_strict_no_tf32", "planning_context": 2,
                  "checks": [{"candidate_count": 300,
                              "full_population_forecast_bitwise_matches_source_hook_compilation": True}]}
        contract = {"policy": TRANSFER_POLICY, "profiled_rank_arm": "rank4",
                    "diagnostic_probe_batching_only": False, "source_protocol_sha256": "p",
                    "source_bank_sha256": "b", "combined_coupling_binding": None}
        validate_transfer(report, contract, "reach-wall", "rank4", binding, None)
        with self.assertRaises(ValueError):
            validate_transfer(report, contract, "reach", "rank4", binding, None)
        with self.assertRaises(ValueError):
            validate_transfer(report, contract, "reach-wall", "matched_random_rank4", binding, None)
        with self.assertRaises(ValueError):
            validate_transfer(report, contract, "reach-wall", "rank4", {**binding, "bank_sha256": "changed"}, None)
        for key, value in (("development_or_protected_outcomes_accessed", True),
                           ("diverse_fit_actions_checked", False), ("checks", [])):
            with self.assertRaises(ValueError):
                validate_transfer({**report, key: value}, contract, "reach-wall", "rank4", binding, None)
