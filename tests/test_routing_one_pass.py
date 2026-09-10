import unittest

import torch

from test_fixed_response import Backend, fixture_bank
from offline_study.routing_hmm import fit_hmm, route_prefix
from offline_study.routing_intervention import ARMS, RoutedFixedResponse
from offline_study.routing_one_pass import RoutedFixedResponseOnePass


class RoutingOnePassTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous = torch.get_num_threads(); torch.set_num_threads(1)
        features = torch.randn(32, 6, 400, generator=torch.Generator().manual_seed(99))
        model = fit_hmm(features); gates = route_prefix(features[:, :2], model)
        cls.routing = {'model': model, 'normalizers': {
            key: gates[key].square().mean().sqrt() for key in ('static', 'hmm', 'memoryless')}}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous)

    def test_exact_two_pass_equivalence_all_arms_and_short_horizons(self):
        context = torch.randn(3, 260, 400, generator=torch.Generator().manual_seed(434))
        for arm in ARMS + ('zero_dose',):
            for horizon in (1, 2, 5, 6):
                actions = torch.zeros(horizon, 3, 1)
                old = RoutedFixedResponse(Backend(), fixture_bank(), self.routing, arm)
                new = RoutedFixedResponseOnePass(Backend(), fixture_bank(), self.routing, arm)
                wanted = old(context, actions); actual = new(context, actions)
                for key in wanted:
                    self.assertTrue(torch.equal(wanted[key], actual[key]), (arm, horizon, key))
                self.assertEqual(new.backend.calls, 1)
                if horizon == 6 and arm not in ('native', 'zero_dose'):
                    self.assertEqual(old.backend.calls, 2)
                    for key in ('normalized_gate', 'posterior', 'raw_gate', 'requested_l2', 'realized_l2', 'coefficients'):
                        self.assertTrue(torch.equal(new.last_record[key], old.last_record[key]), (arm, key))
                self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in new.backend.predictor.modules()))

    def test_no_cross_candidate_or_cross_call_state(self):
        context = torch.randn(3, 260, 400, generator=torch.Generator().manual_seed(343))
        actions = torch.zeros(6, 3, 1)
        new = RoutedFixedResponseOnePass(Backend(), fixture_bank(), self.routing, 'hmm_filtered_gate')
        first = new(context, actions)
        second = new(context.flip(0), actions)
        for key in first:
            torch.testing.assert_close(first[key], second[key].flip(1), rtol=0, atol=0)
        third = new(context, actions)
        for key in first:
            self.assertTrue(torch.equal(first[key], third[key]))


if __name__ == '__main__':
    unittest.main()
