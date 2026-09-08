import unittest
import torch

from test_fixed_response import Backend,fixture_bank
from offline_study.fixed_response import FixedResponseIntervention
from offline_study.routing_hmm import fit_hmm,route_prefix
from offline_study.routing_intervention import ARMS,NativePrefix,RoutedFixedResponse
from offline_study.routing_fit import family_folds,history_value


class RoutingInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous=torch.get_num_threads();torch.set_num_threads(1)
        cls.features=torch.randn(32,6,400,generator=torch.Generator().manual_seed(99))
        model=fit_hmm(cls.features);gates=route_prefix(cls.features[:,:2],model)
        cls.routing={'model':model,'normalizers':{k:gates[k].square().mean().sqrt() for k in ('static','hmm','memoryless')}}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous)

    def test_constant_exactly_matches_refined_map_and_charges_shadow(self):
        context=torch.randn(3,260,400);actions=torch.zeros(6,3,1)
        backend=Backend();bank=fixture_bank()
        actual=RoutedFixedResponse(backend,bank,self.routing,'constant_gate')
        value=actual(context,actions)
        expected=FixedResponseIntervention(Backend(),bank,'fixed_rank4')(context,actions)
        for key in value: self.assertTrue(torch.equal(value[key],expected[key]))
        self.assertEqual(backend.calls,2)
        self.assertEqual(actual.last_record['native_shadow_rollouts'],1)
        torch.testing.assert_close(actual.last_record['normalized_gate'],torch.ones(3),rtol=0,atol=0)

    def test_all_gates_and_randoms_are_causal_paired_and_independent(self):
        context=torch.randn(3,260,400);actions=torch.zeros(6,3,1)
        for arm in ARMS[1:]:
            adapter=RoutedFixedResponse(Backend(),fixture_bank(),self.routing,arm)
            result=adapter(context,actions)
            self.assertTrue(torch.equal(result['visual'][:3],context[None].expand(3,-1,-1,-1)))
            gate=adapter.last_record['normalized_gate'].clone()
            torch.testing.assert_close(adapter.last_record['requested_l2'],2*gate)
            for i in range(3):
                single=RoutedFixedResponse(Backend(),fixture_bank(),self.routing,arm)
                value=single(context[i:i+1],actions[:,i:i+1])
                torch.testing.assert_close(value['visual'],result['visual'][:,i:i+1],rtol=1e-6,atol=1e-6)
            reverse=adapter(context.flip(0),actions)
            torch.testing.assert_close(reverse['visual'].flip(1),result['visual'])
            self.assertFalse(adapter.backend.predictor._forward_pre_hooks)

    def test_short_and_zero_are_native_without_shadow(self):
        for arm in ARMS+('zero_dose',):
            for horizon in (1,2,5,6):
                if horizon==6 and arm not in ('native','zero_dose'):continue
                backend=Backend();context=torch.randn(2,256,400)
                adapter=RoutedFixedResponse(backend,fixture_bank(),self.routing,arm)
                value=adapter(context,torch.zeros(horizon,2,1))
                self.assertTrue(torch.equal(value['visual'][-1],context))
                self.assertEqual(backend.calls,1)

    def test_native_capture_does_not_store_future_or_leave_hooks(self):
        backend=Backend();context=torch.ones(2,256,400)
        with NativePrefix(backend.predictor) as capture:
            value=backend.predict(context,torch.zeros(6,2,1))
        self.assertEqual(set(capture.values),{1,2})
        self.assertTrue(torch.equal(value['visual'][-1],context))
        with self.assertRaises(ValueError):
            with NativePrefix(backend.predictor):backend.predict(context,torch.zeros(1,2,1))
        self.assertFalse(backend.predictor._forward_pre_hooks)

    def test_family_folds_keep_all_prefixes_together(self):
        metadata=[{'lineage_group':f'f{i//4}'} for i in range(32)]
        folds,assignment=family_folds(metadata)
        for i in range(0,32,4):self.assertEqual(len(set(folds[i:i+4].tolist())),1)
        report=history_value(self.features,metadata)
        self.assertEqual(len(report['family_log_density_gain_nats']),8)
        self.assertFalse(report['task_success_evidence'])
        self.assertTrue(all(r['train_families']==6 and r['test_families']==2 for r in report['folds']))


if __name__=='__main__':unittest.main()
