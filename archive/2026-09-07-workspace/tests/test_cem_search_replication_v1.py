import hashlib
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
import cem_search_audit_v1 as core
from cem_search_replication_v1 import MANIFEST_SHA,reset_payload,source_adapter,new_source_selector,primary_contrast,raw_inventory


class Tests(unittest.TestCase):
    def stimulus(self):
        initial=torch.arange(7,dtype=torch.float32)
        raw=dict(source_id=2222,split='new_development_not_confirmation',source_offset=0,environment_seed=2026090600+2222,planner_seed=91600+2222,
            initial_state=initial,initial_state_sha256=hashlib.sha256(initial.numpy().tobytes()).hexdigest(),source_manifest_sha256=MANIFEST_SHA,
            expert_actions_raw=torch.ones(30,2),env_info={'shape':'T'})
        states=np.tile(initial.numpy(),(31,1));states[:,0]+=np.arange(31)+.5
        truth=dict(states=states,proprios=torch.from_numpy(states[:,[0,1,5,6]]).reshape(31,1,4),frames=torch.zeros(31,1,3,2,2,dtype=torch.uint8),physics=[])
        return raw,truth
    def test_actual_goal_and_original_raw_initial_separate(self):
        raw,truth=self.stimulus();r=reset_payload(raw,truth,'sha')
        core.exact(r['initial_state'],raw['initial_state'],'raw preserved');core.exact(r['goal_state'],truth['states'][-1],'actual finalgoal')
        self.assertNotEqual(float(r['initial_state'][0]),float(r['expert_states'][0,0]))
        self.assertEqual(float(r['expert_proprios'][0,0,2]),5.)
    def test_proprio_cannot_be_zeroed(self):
        raw,truth=self.stimulus();truth['proprios'].zero_()
        with self.assertRaises(ValueError):reset_payload(raw,truth,'sha')
    def test_source_split_and_seed_rejected(self):
        raw,truth=self.stimulus();raw['split']='confirmation'
        with self.assertRaises(ValueError):reset_payload(raw,truth,'sha')
        raw,truth=self.stimulus();raw['planner_seed']+=1
        with self.assertRaises(ValueError):reset_payload(raw,truth,'sha')
    def test_shard_filter_before_opening_tensor_or_receipt(self):
        with self.assertRaises(ValueError):raw_inventory(Path('/does/not/exist'),[0,1])
    def test_adapter_restoration_on_exception(self):
        original=core.select_source
        with self.assertRaises(RuntimeError):
            with source_adapter():
                self.assertIs(core.select_source,new_source_selector);raise RuntimeError('planned')
        self.assertIs(core.select_source,original)
    def test_primary_sign_and_stage_definition(self):
        report={'episode':2222,'metric_rows':[dict(stage=s,kind=k,requested_coverage_final=c,predicted_cost_by_horizon=[d],goal_xy_distance_by_horizon=[x]) for s,k,c,d,x in
            [(1,'mean',.2,1.,5),(15,'mean',.6,.8,3),(30,'mean',.5,.4,4),(30,'best',.9,.1,1)]]}
        r=primary_contrast(report)
        self.assertAlmostEqual(r['stage30_minus15_mean_requested_coverage'],-.1)
        self.assertAlmostEqual(r['stage30_minus15_mean_model_cost'],-.4)
        self.assertAlmostEqual(r['stage30_minus15_mean_xy_px'],1.)


if __name__=='__main__':unittest.main()
