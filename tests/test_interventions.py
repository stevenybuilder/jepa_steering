import unittest
from copy import deepcopy

import torch
from torch import nn

from offline_study.intervention_runner import score_intervention_predictions, summarize_interventions
from offline_study.interventions import (
    CATEGORY_ARMS,
    CompiledEdit,
    PredictorIntervention,
    compile_edits,
    validate_frozen_protocol,
    validate_operator_bank,
)


def frozen_protocol(category="vision_action_coupling"):
    arms = []
    for name in sorted(CATEGORY_ARMS[category]):
        arms.append({
            "name": name,
            "edits": [] if name == "native" else [{
                "site": "block_condition", "horizon": 3, "block": 0,
                "tensor": "direction", "scale": 0. if name == "zero_dose" else .1,
            }],
        })
    return {
        "schema_version": 1,
        "status": "frozen",
        "category": category,
        "vendor_commit": "a" * 40,
        "manifest_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "fit_receipt_sha256": "d" * 64,
        "fit_split": "fit",
        "evaluation_split": "development",
        "tasks": ["mw-reach"],
        "hypothesis": "Predeclared fixture hypothesis",
        "dose_budget": {"kind": "fixture"},
        "frozen_at": "2026-09-07T00:00:00Z",
        "arms": arms,
        "primary_contrasts": [{
            "name": "joint_vs_native", "candidate": "joint", "control": "native",
        }],
    }


class FakeBlock(nn.Module):
    def forward(self, x, condition):
        return x + condition[:, -1, None, :]


class FakePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor_blocks = nn.ModuleList([FakeBlock() for _ in range(6)])

    def forward(self, visual, actions, proprio):
        del proprio
        x = visual[:, -1]
        for block in self.predictor_blocks:
            x = block(x, actions)
        return x


class InterventionTests(unittest.TestCase):
    def test_protocol_requires_frozen_complete_arm_registry(self):
        protocol = frozen_protocol()
        validate_frozen_protocol(protocol)
        protocol["arms"] = protocol["arms"][:-1]
        with self.assertRaisesRegex(ValueError, "missing arms"):
            validate_frozen_protocol(protocol)

    def test_operator_bank_binding_and_arm_batch_compilation(self):
        protocol = frozen_protocol()
        meta = [
            {"trajectory_id": "trajectory-a", "task": "mw-reach", "start": 5},
            {"trajectory_id": "trajectory-b", "task": "mw-reach", "start": 10},
        ]
        bank = {
            "schema_version": 1,
            "protocol_sha256": "e" * 64,
            "global_tensors": {"direction": torch.ones(3)},
            "rows": {
                f"{row['trajectory_id']}:{row['start']}": {
                    "trajectory_id": row["trajectory_id"], "start": row["start"],
                    "split": "development", "tensors": {},
                }
                for row in meta
            },
        }
        validate_operator_bank(bank, "e" * 64, meta)
        compiled = compile_edits(protocol, bank, meta, torch.device("cpu"))
        self.assertEqual(len(compiled), 1)
        arm_count = len(protocol["arms"])
        self.assertEqual(compiled[0].delta.shape, (2 * arm_count, 3))
        native = next(i for i, arm in enumerate(protocol["arms"]) if arm["name"] == "native")
        zero = next(i for i, arm in enumerate(protocol["arms"]) if arm["name"] == "zero_dose")
        self.assertTrue(torch.equal(compiled[0].delta[native], torch.zeros(3)))
        self.assertTrue(torch.equal(compiled[0].delta[zero], torch.zeros(3)))
        self.assertTrue(torch.equal(compiled[0].delta[arm_count + native], torch.zeros(3)))
        self.assertEqual(len(compiled[0].delivered_l2), 2 * arm_count)

    def test_protocol_rejects_extra_arms_and_incomplete_zero_control(self):
        protocol = frozen_protocol()
        protocol["arms"].append({"name": "post_hoc_winner", "edits": []})
        with self.assertRaisesRegex(ValueError, "unregistered arms"):
            validate_frozen_protocol(protocol)

        protocol = frozen_protocol()
        active = next(arm for arm in protocol["arms"] if arm["name"] == "joint")
        active["edits"].append({
            "site": "predictor_visual", "horizon": 4,
            "tensor": "visual_direction", "scale": .1,
        })
        with self.assertRaisesRegex(ValueError, "every registered hook"):
            validate_frozen_protocol(protocol)

    def test_protocol_task_registry_is_unique(self):
        protocol = deepcopy(frozen_protocol())
        protocol["tasks"].append(protocol["tasks"][0])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            validate_frozen_protocol(protocol)

    def test_hooks_apply_only_registered_horizon_block_and_tokens(self):
        predictor = FakePredictor()
        batch = 2
        visual = torch.zeros(batch, 2, 4, 3)
        actions = torch.zeros(batch, 2, 3)
        proprio = torch.zeros(batch, 2, 3)
        condition = CompiledEdit(
            site="block_condition", horizon=3, block=0,
            token_start=None, token_end=None, delta=torch.ones(batch, 3),
        )
        output = CompiledEdit(
            site="block_output", horizon=2, block=1,
            token_start=-2, token_end=None, delta=torch.full((batch, 2, 3), 2.),
        )
        with PredictorIntervention(predictor, [condition, output]):
            values = [predictor(visual, actions, proprio) for _ in range(6)]
        self.assertTrue(torch.equal(values[0], torch.zeros_like(values[0])))
        self.assertTrue(torch.equal(values[1][:, :2], torch.zeros(batch, 2, 3)))
        self.assertTrue(torch.equal(values[1][:, -2:], torch.full((batch, 2, 3), 2.)))
        self.assertTrue(torch.equal(values[2], torch.ones_like(values[2])))
        self.assertTrue(torch.equal(values[3], torch.zeros_like(values[3])))

    def test_contrasts_are_paired_after_trajectory_aggregation(self):
        protocol = {
            "arms": [{"name": "native"}, {"name": "candidate"}],
            "primary_contrasts": [{
                "name": "candidate_vs_native", "candidate": "candidate", "control": "native",
            }],
        }
        rows = []
        for trajectory, native, candidate in (("a", 1., 2.), ("b", 3., 2.)):
            for start in (0, 5):
                rows.extend([
                    {"task": "reach", "trajectory_id": trajectory, "start": start,
                     "arm": "native", "metrics": {"error": native}},
                    {"task": "reach", "trajectory_id": trajectory, "start": start,
                     "arm": "candidate", "metrics": {"error": candidate}},
                ])
        summary = summarize_interventions(rows, protocol)
        contrast = summary["primary_contrasts"][0]
        self.assertEqual(contrast["per_task_mean"]["reach"]["error"], 0.)
        self.assertEqual(len(contrast["per_trajectory"]), 2)

    def test_vision_action_output_and_score_interactions_are_separated(self):
        arms = ["native", "visual_only", "action_condition_only", "joint"]
        target = torch.zeros(len(arms), 7, 1)
        prediction = torch.zeros(7, len(arms), 1)
        prediction[:, 1] = 1.
        prediction[:, 2] = 2.
        prediction[:, 3] = 4.
        metrics, diagnostics = score_intervention_predictions(
            {"visual": prediction, "proprio": prediction},
            {"visual": target, "proprio": target},
            arms,
            "vision_action_coupling",
            window_count=1,
        )
        self.assertEqual(metrics[3]["visual_mse_h1"], 16.)
        self.assertEqual(diagnostics[0]["visual_output_interaction_mse_h1"], 1.)
        self.assertEqual(diagnostics[0]["visual_additive_quadratic_cross_h1"], 4.)
        self.assertEqual(diagnostics[0]["visual_factorial_mse_interaction_h1"], 11.)
        self.assertEqual(diagnostics[0]["visual_nonadditive_mse_remainder_h1"], 7.)


if __name__ == "__main__":
    unittest.main()
