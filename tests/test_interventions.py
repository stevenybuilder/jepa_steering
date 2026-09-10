import unittest
from copy import deepcopy

import torch
from torch import nn

from offline_study.intervention_runner import (
    _identity_diagnostics,
    _numerically_identical,
    score_intervention_predictions,
    summarize_interventions,
)
from offline_study.interventions import (
    CATEGORY_ARMS,
    LAYER_ARM_BLOCKS,
    RANK_ARM_RANKS,
    CompiledEdit,
    PredictorIntervention,
    compile_edits,
    validate_frozen_protocol,
    validate_operator_bank,
)


def frozen_protocol(category="vision_action_coupling"):
    arms = []
    for name in sorted(CATEGORY_ARMS[category]):
        if category == "distribution_layer" and name != "native":
            blocks = range(6) if name == "zero_dose" else sorted(LAYER_ARM_BLOCKS[name])
            edits = [{
                "site": "block_output", "horizon": 3, "block": block,
                "tensor": f"direction_b{block}",
                "scale": 0. if name == "zero_dose" else .1,
            } for block in blocks]
        elif category == "operator_rank" and name != "native":
            edits = [{
                "site": "block_output", "horizon": 3, "block": 3,
                "tensor": "direction_b3", "scale": 0. if name == "zero_dose" else .1,
            }]
        else:
            edits = [] if name == "native" else [{
                "site": "block_condition", "horizon": 3, "block": 0,
                "tensor": "direction", "scale": 0. if name == "zero_dose" else .1,
            }]
        arms.append({
            "name": name,
            "edits": edits,
        })
    primary_contrasts = {
        "vision_action_coupling": {
            "name": "joint_vs_native", "candidate": "joint", "control": "native",
        },
        "action_response_geometry": {
            "name": "cubic_vs_linear", "candidate": "cubic",
            "control": "equal_anchor_linear",
        },
        "imagined_time_routing": {
            "name": "hmm_vs_memoryless", "candidate": "hmm_filtered_gate",
            "control": "memoryless_gate",
        },
        "distribution_spatial": {
            "name": "contiguous_vs_random", "candidate": "contiguous_group",
            "control": "matched_random_contiguous_group",
        },
        "distribution_layer": {
            "name": "intermediate_zone_vs_random",
            "candidate": "intermediate_blocks2_3",
            "control": "matched_random_intermediate_blocks2_3",
        },
        "operator_rank": {
            "name": "rank8_vs_rank1", "candidate": "rank8", "control": "rank1",
        },
    }
    dose_budget = {"kind": "fixture"}
    if category == "distribution_layer":
        dose_budget.update({
            "energy_rule": "equal_total_delivered_squared_l2_per_arm",
            "capacity_rule": "fixed_total_direct_sum_rank_per_arm",
            "random_control_rule": "same_support_rank_spectrum_and_energy",
        })
    elif category == "operator_rank":
        dose_budget.update({
            "energy_rule": "equal_total_delivered_squared_l2_across_ranks",
            "random_control_rule": "same_support_rank_spectrum_and_energy",
        })
    for arm in arms:
        if arm["name"] in RANK_ARM_RANKS:
            arm["operator_rank"] = RANK_ARM_RANKS[arm["name"]]
        if arm["name"] in LAYER_ARM_BLOCKS:
            arm["operator_rank"] = 1
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
        "dose_budget": dose_budget,
        "frozen_at": "2026-09-07T00:00:00Z",
        "arms": arms,
        "primary_contrasts": [primary_contrasts[category]],
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
    def test_identity_check_accepts_cuda_scale_roundoff_but_not_real_changes(self):
        reference = torch.tensor([0., 1., -2.], dtype=torch.float32)
        roundoff = reference + torch.tensor([1e-6, -1e-5, 4e-5])
        changed = reference + torch.tensor([0., 0., 1e-3])
        self.assertTrue(_numerically_identical(reference, roundoff))
        self.assertFalse(_numerically_identical(reference, changed))
        diagnostics = _identity_diagnostics(reference, roundoff)
        self.assertAlmostEqual(diagnostics["maximum_absolute_difference"], 4e-5, places=6)
        self.assertEqual(diagnostics["different_elements"], 3)

    def test_protocol_requires_frozen_complete_arm_registry(self):
        protocol = frozen_protocol()
        validate_frozen_protocol(protocol)
        protocol["arms"] = protocol["arms"][:-1]
        with self.assertRaisesRegex(ValueError, "missing arms"):
            validate_frozen_protocol(protocol)

    def test_operator_bank_binding_and_arm_batch_compilation(self):
        protocol = frozen_protocol()
        meta = [
            {"trajectory_id": "trajectory-a", "lineage_group": "family-a",
             "task": "mw-reach", "start": 5},
            {"trajectory_id": "trajectory-b", "lineage_group": "family-b",
             "task": "mw-reach", "start": 10},
        ]
        bank = {
            "schema_version": 1,
            "protocol_sha256": "e" * 64,
            "global_tensors": {"direction": torch.ones(3)},
            "rows": {
                f"{row['trajectory_id']}:{row['start']}": {
                    "trajectory_id": row["trajectory_id"], "start": row["start"],
                    "lineage_group": row["lineage_group"],
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

    def test_layer_protocol_freezes_intermediate_block_identities(self):
        protocol = frozen_protocol("distribution_layer")
        validate_frozen_protocol(protocol)

        block2 = next(
            arm for arm in protocol["arms"] if arm["name"] == "single_block2")
        block2["edits"][0]["block"] = 1
        with self.assertRaisesRegex(ValueError, "requires zero-indexed blocks.*2"):
            validate_frozen_protocol(protocol)

    def test_layer_protocol_requires_matched_time_and_spatial_scope(self):
        protocol = frozen_protocol("distribution_layer")
        block3 = next(
            arm for arm in protocol["arms"] if arm["name"] == "single_block3")
        block3["edits"][0]["horizon"] = 4
        zero = next(arm for arm in protocol["arms"] if arm["name"] == "zero_dose")
        zero["edits"].append({
            "site": "block_output", "horizon": 4, "block": 3,
            "tensor": "direction_b3", "scale": 0.,
        })
        with self.assertRaisesRegex(ValueError, "share one edit site, horizon, and spatial scope"):
            validate_frozen_protocol(protocol)

    def test_layer_protocol_has_scope_matched_random_for_every_arm(self):
        protocol = frozen_protocol("distribution_layer")
        validate_frozen_protocol(protocol)
        random_block4 = next(
            arm for arm in protocol["arms"]
            if arm["name"] == "matched_random_single_block4")
        random_block4["edits"][0]["block"] = 3
        with self.assertRaisesRegex(ValueError, "requires zero-indexed blocks.*4"):
            validate_frozen_protocol(protocol)

    def test_layer_protocol_requires_one_shared_positive_operator_rank(self):
        protocol = frozen_protocol("distribution_layer")
        all_six = next(
            arm for arm in protocol["arms"] if arm["name"] == "all_six_blocks")
        all_six["operator_rank"] = 4
        with self.assertRaisesRegex(ValueError, "one shared positive operator_rank"):
            validate_frozen_protocol(protocol)

    def test_rank_protocol_freezes_rank_and_scope(self):
        protocol = frozen_protocol("operator_rank")
        validate_frozen_protocol(protocol)
        rank4 = next(arm for arm in protocol["arms"] if arm["name"] == "rank4")
        rank4["operator_rank"] = 3
        with self.assertRaisesRegex(ValueError, "requires operator_rank=4"):
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
        for trajectory, family, native, candidate in (
            ("a", "g1", 1., 2.), ("b", "g1", 3., 2.), ("c", "g2", 5., 8.)
        ):
            for start in (0, 5):
                rows.extend([
                    {"task": "reach", "trajectory_id": trajectory, "lineage_group": family,
                     "start": start,
                     "arm": "native", "metrics": {"error": native}},
                    {"task": "reach", "trajectory_id": trajectory, "lineage_group": family,
                     "start": start,
                     "arm": "candidate", "metrics": {"error": candidate}},
                ])
        summary = summarize_interventions(rows, protocol)
        contrast = summary["primary_contrasts"][0]
        # g1 averages (+1,-1) to zero; g2 is +3; groups receive equal weight.
        self.assertEqual(contrast["per_task_group_weighted_mean"]["reach"]["error"], 1.5)
        self.assertEqual(len(contrast["per_trajectory"]), 3)
        self.assertEqual(len(contrast["per_lineage_group"]), 2)

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
