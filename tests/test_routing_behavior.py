import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

import test_routing_intervention as routing_fixtures
from test_fixed_response import Backend,fixture_bank
from offline_study.behavioral_development import schedule
from offline_study.fixed_response_check import CountedBackend
from offline_study.routing_behavior import ARMS,CONTRASTS,ENGINEERING,NEW_ARMS,REUSED,ObservedRouting,verify_routed_calls
from offline_study.routing_intervention import RoutedFixedResponse
from offline_study.routing_one_pass import RoutedFixedResponseOnePass
from offline_study.routing_analysis import analyze,validate_panel


class RoutingBehaviorTests(unittest.TestCase):
    def test_reference_reuse_and_contrast_counts_preserve_sample(self):
        self.assertEqual(len(ARMS),7);self.assertEqual(len(NEW_ARMS),4)
        self.assertEqual(2*len(NEW_ARMS)*len(schedule()),768)
        self.assertEqual(2*len(REUSED)*len(schedule()),576)
        self.assertEqual(len(CONTRASTS),8);self.assertEqual(len(set(CONTRASTS)),8)
        self.assertEqual(len(ENGINEERING),9)

    def test_independent_field_check_and_charged_backend_calls(self):
        fixtures=routing_fixtures.RoutingInterventionTests
        fixtures.setUpClass()
        try:
            context=torch.randn(2,256,400);actions=torch.zeros(6,2,1)
            with tempfile.TemporaryDirectory() as temporary,patch('torch.cuda.synchronize'):
                for arm in ARMS+('zero_dose',):
                    counted=CountedBackend(Backend())
                    adapter=RoutedFixedResponse(counted,fixture_bank(),fixtures.routing,arm)
                    observed=ObservedRouting(adapter,counted,Path(temporary),True)
                    observed(context,actions)
                    self.assertTrue(observed.calls[0]['independent_reference_check'])
                    self.assertEqual(observed.calls[0]['backend_calls'],1 if arm in ('native','zero_dose') else 2)
        finally:fixtures.tearDownClass()

    def test_same_pass_checks_every_call_against_frozen_two_pass(self):
        fixtures=routing_fixtures.RoutingInterventionTests
        fixtures.setUpClass()
        try:
            context=torch.randn(2,256,400);actions=torch.zeros(6,2,1)
            with tempfile.TemporaryDirectory() as temporary,patch('torch.cuda.synchronize'):
                for arm in ARMS+('zero_dose',):
                    counted=CountedBackend(Backend())
                    adapter=RoutedFixedResponseOnePass(counted,fixture_bank(),fixtures.routing,arm)
                    observed=ObservedRouting(adapter,counted,Path(temporary),True)
                    observed(context,actions);observed(context,actions)
                    for call in observed.calls:
                        self.assertEqual(call['backend_calls'],1)
                        self.assertTrue(call['two_pass_equivalence_checked'])
                        self.assertGreater(call['engineering_extra_reference_calls'],0)
                    self.assertFalse(observed.calls[1]['independent_reference_check'])
                    live=ObservedRouting(adapter,counted,Path(temporary),False)
                    live(context,actions)
                    self.assertEqual(live.calls[0]['engineering_extra_reference_calls'],0)
                    self.assertFalse(live.calls[0]['two_pass_equivalence_checked'])
        finally:fixtures.tearDownClass()

    def fixture(self):
        return {task:{arm:[{**row,'arm':arm,'device_uuid':'fixture-device-'+str(row['logical_rank']),
            'initial_state_vector':[row['episode']],'seconds':1.,'result':{'native_success':row['episode']%2==0,
            'elementary_steps':100,'published_candidate_count':300,'initial_sha256':str(row['episode']),
            'goal_sha256':'goal'+str(row['episode']),'native_state_distance':1.,'native_reward':0.}}
            for row in schedule()] for arm in ARMS} for task in ('reach','reach-wall')}

    def test_complete_panel_analysis_does_not_count_candidates(self):
        panel=self.fixture();report=analyze(panel)
        self.assertEqual(len(report['contrasts']),16)
        for task in report['tasks'].values():
            self.assertEqual(task['scenario_clusters'],96)
            self.assertTrue(all(a['episodes']==96 and a['successes']==48 for a in task['arms'].values()))
        self.assertTrue(all(r['gain_percentage_points']==0 for r in report['contrasts']))
        self.assertFalse(report['full_study_complete'])

    def test_incomplete_unpaired_wrong_device_rejected(self):
        panel=self.fixture()
        for mutation in ('missing','goal','device','seed'):
            bad=copy.deepcopy(panel);rows=bad['reach']['hmm_filtered_gate']
            if mutation=='missing':rows.pop()
            elif mutation=='goal':rows[0]['result']['goal_sha256']='wrong'
            elif mutation=='device':rows[0]['device_uuid']='other'
            else:rows[0]['local_seed']+=1
            with self.assertRaises(ValueError):validate_panel(bad)


if __name__=='__main__':unittest.main()
