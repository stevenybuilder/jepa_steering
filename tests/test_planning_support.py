import unittest

import torch

from offline_study.interventions import CompiledEdit
from offline_study.planning_support import TRANSFER_POLICY, context_values, selected_support_fields


class PlanningSupportTests(unittest.TestCase):
    def test_disjoint_singleton_expansion_and_nonpadded_tail(self):
        context = {k: torch.arange(6.).reshape(1, 1, 2, 3) for k in ("visual", "proprio")}
        result = context_values(context, 19, 16, 19)
        for key in context:
            self.assertEqual(result[key].shape[0], 3)
            self.assertTrue(torch.equal(result[key][0], context[key][0]))
        expanded = {k: torch.arange(19.).reshape(19, 1, 1) for k in context}
        result = context_values(expanded, 19, 16, 19)
        self.assertEqual(result["visual"].flatten().tolist(), [16., 17., 18.])

    def test_rejects_wrong_candidate_or_context_shape(self):
        for shape in ((2, 1, 3), (1, 2, 3)):
            with self.assertRaises(ValueError):
                context_values({k: torch.zeros(*shape) for k in ("visual", "proprio")}, 19, 0, 8)

    def test_source_arm_layout_and_counters(self):
        names = ["native", "rank4", "matched_random_rank4"]
        delta = torch.arange(18.).reshape(6, 1, 3)
        source = CompiledEdit("block_output", 3, 3, -256, None, delta, applications=1)
        result = selected_support_fields([source], names, "rank4", 2)[0]
        self.assertTrue(torch.equal(result.delta, delta[[1, 4]]))
        self.assertEqual(result.applications, 0)
        self.assertIsNone(result.realized_l2)
        with self.assertRaises(ValueError):
            selected_support_fields([source], names, "rank8", 2)

    def test_short_horizon_is_not_an_invented_target(self):
        self.assertEqual(TRANSFER_POLICY["response_horizon"], 6)
        self.assertEqual(TRANSFER_POLICY["short_horizon_policy"], "true native; no support or coupling edit below H6")
        self.assertFalse(TRANSFER_POLICY["candidate_reordering_or_padding"])
