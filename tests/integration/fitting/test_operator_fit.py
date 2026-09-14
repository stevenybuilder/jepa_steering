import math
import unittest

import torch
from torch import nn

from offline_study.interventions.interventions import validate_frozen_protocol, validate_operator_bank
from offline_study.fitting.operator_fit import fit_coupled_directions, make_operator_bank, make_protocol, NativeCouplingCapture, orthogonal_random_control, permute_visual_direction, select_fit_rows


class CaptureBlock(nn.Module):
    def forward(self, x, condition):
        return x + condition[:, -1, None, None, :]


class CapturePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor_blocks = nn.ModuleList([CaptureBlock() for _ in range(6)])

    def forward(self, visual, condition, proprio):
        del proprio
        value = visual[:, -1]
        for block in self.predictor_blocks:
            value = block(value, condition)
        return value


class OperatorFitTests(unittest.TestCase):
    def test_native_capture_reads_h3_and_block_three_without_editing(self):
        predictor = CapturePredictor()
        outputs = []
        with NativeCouplingCapture(predictor) as capture:
            for horizon in range(1, 7):
                visual = torch.full((2, 2, 1, 1, 3), float(horizon))
                condition = torch.full((2, 2, 3), float(10 + horizon))
                outputs.append(predictor(visual, condition, torch.zeros(2, 2, 3)))
        self.assertTrue(torch.equal(capture.visual, torch.full((2, 1, 1, 3), 3.)))
        self.assertTrue(torch.equal(capture.condition, torch.full((2, 3), 13.)))
        self.assertTrue(torch.equal(outputs[2], torch.full((2, 1, 1, 3), 81.)))

    def test_fit_selection_is_one_row_per_group_and_deterministic(self):
        rows = [
            {"trajectory_id": "a", "lineage_group": "g1", "task": "reach", "split": "fit"},
            {"trajectory_id": "b", "lineage_group": "g1", "task": "reach", "split": "fit"},
            {"trajectory_id": "c", "lineage_group": "g2", "task": "reach", "split": "fit"},
            {"trajectory_id": "d", "lineage_group": "g3", "task": "other", "split": "fit"},
            {"trajectory_id": "e", "lineage_group": "g4", "task": "reach", "split": "development"},
        ]
        first = select_fit_rows(rows, "reach", 10, 7)
        second = select_fit_rows(list(reversed(rows)), "reach", 10, 7)
        self.assertEqual(first, second)
        self.assertEqual({row["lineage_group"] for row in first}, {"g1", "g2"})

    def test_coupled_fit_recovers_shared_axis_and_deterministic_controls(self):
        generator = torch.Generator().manual_seed(19)
        score = torch.linspace(-2, 2, 31)
        visual_axis = torch.randn(1, 16, 16, 8, generator=generator)
        action_axis = torch.randn(11, generator=generator)
        visual = score.reshape(-1, 1, 1, 1, 1) * visual_axis
        condition = score[:, None] * action_axis
        visual_direction, action_direction, diagnostics = fit_coupled_directions(
            visual, condition, iterations=6)
        self.assertGreater(abs(torch.nn.functional.cosine_similarity(
            visual_direction.flatten(), visual_axis.flatten(), dim=0)), .999)
        self.assertGreater(abs(torch.nn.functional.cosine_similarity(
            action_direction, action_axis, dim=0)), .999)
        self.assertGreater(diagnostics["cross_covariance"], 0)
        random = orthogonal_random_control(visual_direction, 31)
        self.assertLess(abs(float(torch.dot(random.flatten(), visual_direction.flatten()))), 1e-5)
        self.assertTrue(torch.equal(random, orthogonal_random_control(visual_direction, 31)))
        permuted = permute_visual_direction(visual_direction, 23)
        self.assertAlmostEqual(float(permuted.double().norm()),
                               float(visual_direction.double().norm()), places=6)

    def test_generated_protocol_and_bank_satisfy_execution_contract(self):
        protocol = make_protocol(
            "mw-reach", "a" * 64, "b" * 64, "c" * 64, .2, .3,
            "2026-09-07T00:00:00+00:00")
        validate_frozen_protocol(protocol)
        arms = {arm["name"]: arm for arm in protocol["arms"]}
        self.assertAlmostEqual(
            arms["joint_equal_standardized_energy"]["edits"][0]["scale"],
            .2 / math.sqrt(2),
        )
        self.assertAlmostEqual(
            arms["joint_equal_standardized_energy"]["edits"][1]["scale"],
            .3 / math.sqrt(2),
        )
        self.assertAlmostEqual(
            arms["matched_random_equal_standardized_energy"]["edits"][0]["scale"],
            .2 / math.sqrt(2),
        )
        evaluation = [{"trajectory_id": "t", "lineage_group": "g", "task": "mw-reach",
                       "split": "development", "starts": [0, 5]}]
        tensors = {
            "visual_direction": torch.ones(1, 16, 16, 4),
            "action_direction": torch.ones(7),
            "permuted_visual_direction": torch.ones(1, 16, 16, 4),
            "random_visual_direction": torch.ones(1, 16, 16, 4),
            "random_action_direction": torch.ones(7),
        }
        bank = make_operator_bank("d" * 64, evaluation, tensors)
        meta = [{"trajectory_id": "t", "lineage_group": "g", "task": "mw-reach",
                 "start": start, "split": "development"} for start in (0, 5)]
        validate_operator_bank(bank, "d" * 64, meta)


if __name__ == "__main__":
    unittest.main()
