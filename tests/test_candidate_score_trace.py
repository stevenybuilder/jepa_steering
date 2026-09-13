from types import SimpleNamespace
import unittest

import torch
from offline_study.candidate_score_trace import CandidateScoreTrace


class ToyCEM:
    iterations = 3
    num_elites = 2
    distribute_planner = False

    def __init__(self):
        self.local_generator = torch.Generator().manual_seed(31)

    def cost_function(self, actions, z_init):
        return actions.square().sum((0, 2))

    def plan(self, z_init):
        mean = torch.zeros(2, 3)
        for _ in range(self.iterations):
            actions = mean[:, None] + torch.randn(2, 5, 3, generator=self.local_generator)
            costs = self.cost_function(actions, z_init)
            idx = torch.topk(-costs, self.num_elites, dim=0).indices
            mean = actions[:, idx].mean(1)
        self._prev_mean = mean
        return SimpleNamespace(actions=mean[:1])


class TraceTests(unittest.TestCase):
    def test_identity_rng_and_actual_indices(self):
        ref, traced = ToyCEM(), ToyCEM()
        expected = ref.plan(None).actions
        original = torch.topk
        with CandidateScoreTrace(traced) as trace:
            actual = trace.run(None).actions
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(ref.local_generator.get_state(), traced.local_generator.get_state()))
        self.assertIs(torch.topk, original)
        self.assertNotIn('cost_function', traced.__dict__)
        data = trace.payload()
        self.assertEqual(len(data['iterations']), 3)
        for row in data['iterations']:
            torch.testing.assert_close(row['objective_costs'], row['candidate_actions'].square().sum((0, 2)))
            self.assertTrue(torch.equal(row['elite_indices'], torch.topk(-row['objective_costs'], 2).indices))

    def test_restoration_on_error(self):
        planner = ToyCEM()
        original = torch.topk
        with self.assertRaises(RuntimeError):
            with CandidateScoreTrace(planner):
                raise RuntimeError('test')
        self.assertIs(torch.topk, original)
        self.assertNotIn('cost_function', planner.__dict__)


if __name__ == '__main__':
    unittest.main()
