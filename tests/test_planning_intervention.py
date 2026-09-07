import unittest

import torch

from offline_study.interventions import compile_edits, window_key
from offline_study.planning_intervention import static_edits


class PlanningInterventionTests(unittest.TestCase):
    def fixture(self):
        protocol = {"category": "vision_action_coupling", "arms": [{"name": "joint", "edits": [
            {"site": "predictor_visual", "horizon": 3, "tensor": "v", "scale": .1},
            {"site": "block_condition", "block": 3, "horizon": 3, "tensor": "a", "scale": .2}]}]}
        return protocol, {"global_tensors": {"v": torch.arange(12.).reshape(3, 4), "a": torch.arange(4.)}}

    def test_matches_existing_frozen_static_compilation(self):
        protocol, bank = self.fixture()
        meta = [{"trajectory_id": "fit-smoke", "start": i} for i in range(4)]
        bank["rows"] = {window_key(row): {"tensors": {}} for row in meta}
        expected = compile_edits(protocol, bank, meta, torch.device("cpu"))
        actual = static_edits(protocol, bank, "joint", 4, 6, torch.device("cpu"))
        for before, after in zip(expected, actual):
            self.assertTrue(torch.equal(before.delta, after.delta))
            self.assertEqual(before.delivered_l2, after.delivered_l2)

    def test_only_realized_horizons_are_edited(self):
        protocol, bank = self.fixture()
        self.assertEqual(static_edits(protocol, bank, "joint", 3, 2, "cpu"), [])
        self.assertEqual(len(static_edits(protocol, bank, "joint", 3, 5, "cpu")), 2)

    def test_zero_dose_has_no_hooks_or_context_clones(self):
        protocol, bank = self.fixture()
        for edit in protocol["arms"][0]["edits"]:
            edit["scale"] = 0.
        self.assertEqual(static_edits(protocol, bank, "joint", 3, 6, "cpu"), [])

    def test_dynamic_operator_is_not_silently_reinterpreted(self):
        protocol, bank = self.fixture()
        protocol["category"] = "operator_rank"
        with self.assertRaisesRegex(ValueError, "Dynamic"):
            static_edits(protocol, bank, "joint", 3, 6, "cpu")
