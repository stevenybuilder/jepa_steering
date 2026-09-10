import copy
import ast
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from offline_study.author_fit import source_hash
from offline_study.behavioral_development import assigned_rows, schedule
from offline_study.planning_native_smoke import CHECKPOINTS, SMOKE_SEED
from offline_study.protocol import sha256, write_json
from offline_study.pusht_planning_check import verify_engineering
from offline_study.pusht_planning_replication import checked_cohort, TracedDataset, trace_segment


class PushTPlanningTests(unittest.TestCase):
    def test_tracing_preserves_actual_upstream_sampler_and_rng(self):
        vendor = Path(__file__).resolve().parents[1] / "vendor/jepa-wms"
        tree = ast.parse((vendor / "evals/simu_env_planning/planning/plan_evaluator.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PlanEvaluator")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sample_traj_segment_from_dset")
        namespace = {"torch": torch, "log": SimpleNamespace(info=lambda *a: None)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "actual_upstream_sampler", "exec"), namespace)
        evaluator = type("PlanEvaluator", (), {"sample_traj_segment_from_dset": namespace[method.name]})
        class Data:
            def __len__(self): return 3
            def get_seq_length(self, i): return [29, 40, 65][i]
            def __getitem__(self, i):
                n = self.get_seq_length(i)
                return ({"visual": torch.arange(n).reshape(n, 1)}, torch.arange(n*2).reshape(n, 2).float(),
                        torch.arange(n*7).reshape(n, 7).float() + i*1000, None, {"shape": "T"})
        cfg = SimpleNamespace(frameskip=5, task_specification=SimpleNamespace(env={}, task="pusht", goal_H=6))
        data = Data()
        wrapped = TracedDataset(data, {i: {"trajectory_id": f"pusht:val:{i}", "lineage_group": f"family{i}"} for i in range(3)})
        native_agent = SimpleNamespace(dset=data, local_generator=torch.Generator().manual_seed(123))
        traced_agent = SimpleNamespace(dset=wrapped, local_generator=torch.Generator().manual_seed(123))
        module = SimpleNamespace(PlanEvaluator=evaluator)
        with patch.dict("sys.modules", {"evals.simu_env_planning.planning.plan_evaluator": module}):
            for _ in range(12):
                before = evaluator().sample_traj_segment_from_dset(cfg, native_agent, 31)
                with trace_segment(wrapped) as records:
                    after = evaluator().sample_traj_segment_from_dset(cfg, traced_agent, 31)
                self.assertTrue(torch.equal(before[0]["visual"], after[0]["visual"]))
                np.testing.assert_array_equal(before[1], after[1])
                self.assertTrue(torch.equal(before[2], after[2]))
                self.assertTrue(torch.equal(native_agent.local_generator.get_state(), traced_agent.local_generator.get_state()))
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["segment_frames"], 31)
                self.assertEqual(records[0]["source_pool"], "val")
                self.assertEqual(records[0]["source_index"], wrapped.last_index)

    def test_full_released_access_is_still_replication_not_confirmation(self):
        p = Path(__file__).resolve().parents[1] / "artifacts/offline_study/utah-durable-20260907/pusht-author-replication-20260907/cohorts/pusht/cohort.json"
        if not p.exists():
            self.skipTest("Real access receipt not available in clean checkout")
        cohort = checked_cohort(p)
        self.assertEqual(len(cohort["evaluation"]), 21)
        self.assertFalse(cohort["replication_access"]["fresh_confirmation"])
        self.assertEqual(len({r["lineage_group"] for r in cohort["evaluation"]}), 21)

    def test_logical_sharding_keeps_96_total(self):
        rows = schedule()
        self.assertEqual(len(rows), 96)
        pieces = [assigned_rows(rows, [rank]) for rank in range(8)]
        self.assertTrue(all(len(x) == 12 for x in pieces))
        self.assertEqual([r for piece in pieces for r in piece], rows)
        self.assertNotIn(SMOKE_SEED, [r["environment_seed"] for r in rows])

    def test_receiving_check_rejects_changed_actions_budget_and_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract = {"task": "pusht"}
            protocol = {"planning_contract": contract, "cohort_sha256": "cohort",
                "source_sha256": source_hash(), "checkpoint_sha256": CHECKPOINTS["pusht"],
                "smoke_seed": SMOKE_SEED, "repetitions": 2, "precision": "float32_strict_no_tf32"}
            row = {"result": {"elementary_steps": 30, "published_candidate_count": 300,
                "planning_calls": [{"iterations": 30, "returned_model_actions": 6}],
                "initial_sha256": "initial", "goal_sha256": "goal", "native_success": False,
                "native_state_distance": 1.0, "native_reward": 0.0, "observed_frames": 31},
                "planned_actions": [[[0.0] * 10 for _ in range(6)]],
                "unroll_calls": [[6, 300], [6, 1]] * 30}
            def save(a, b):
                write_json(root / "protocol.json", protocol)
                for i, value in enumerate((a, b)):
                    write_json(root / f"repetition-{i}.json", value)
                report = {"status": "full_native_pusht_engineering_passed",
                    "protocol_sha256": sha256(root / "protocol.json"),
                    "fresh_confirmation": False, "same_seed_actions_and_outcomes_exact": True,
                    "parameters_unchanged": True,
                    "repetition_sha256": [sha256(root / f"repetition-{i}.json") for i in range(2)]}
                write_json(root / "report.json", report)
                write_json(root / "DONE.json", {"report_sha256": sha256(root / "report.json")})
            save(row, row)
            verify_engineering(root, contract, "cohort")
            changed = copy.deepcopy(row); changed["planned_actions"][0][0][0] = .1
            save(row, changed)
            with self.assertRaisesRegex(ValueError, "differ"):
                verify_engineering(root, contract, "cohort")
            changed = copy.deepcopy(row); changed["unroll_calls"].pop()
            save(row, changed)
            with self.assertRaisesRegex(ValueError, "CEM"):
                verify_engineering(root, contract, "cohort")
            protocol["source_sha256"] = "changed"; save(row, row)
            with self.assertRaisesRegex(ValueError, "source-bound"):
                verify_engineering(root, contract, "cohort")


if __name__ == "__main__":
    unittest.main()
