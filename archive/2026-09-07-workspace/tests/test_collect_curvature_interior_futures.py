from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/geometry_map'))
from collect_curvature_interior_futures import interior_actions, select_sources, require_reset_link, compare_replays, hybrid_actions, action_sha, missing_path_actions


class InteriorTruthTests(unittest.TestCase):
    def test_missing_path_coefficients_only(self):
        center=torch.ones(30,2);delta=torch.full((30,2),.01)
        raw=torch.stack([center]+[center+s*delta for _ in range(4) for s in (1,-1)])
        coefficients=[]
        for radius in (1,4):
            for t in ((-.5,.5) if radius==1 else (-1.,-.5,.5,1.)):
                actions,meta=missing_path_actions(raw,0,radius,t)
                coefficients.append(meta['raw_delta_coefficient'])
                self.assertTrue(torch.equal(actions,center+(radius*t)*(raw[1]-center)))
        self.assertEqual(set(coefficients),{-.5,.5,-2.,2.,-4.,4.})
        with self.assertRaises(ValueError): missing_path_actions(raw,0,1,1.)
        with self.assertRaises(ValueError): missing_path_actions(raw,0,4,.25)
    def test_hybrid_prefix_suffix_no_mutation_and_h6_reuse_only(self):
        donor=torch.arange(60,dtype=torch.float32).reshape(30,2)
        center=-donor.clone();saved=donor.clone()
        for h in (1,3):
            result=hybrid_actions(donor,center,h)
            self.assertTrue(torch.equal(result[:5*h],donor[:5*h]))
            self.assertTrue(torch.equal(result[5*h:],center[5*h:]))
            self.assertNotEqual(action_sha(result),action_sha(donor))
        self.assertTrue(torch.equal(donor,saved))
        with self.assertRaises(ValueError):hybrid_actions(donor,center,6)
    def test_exact_fixed_action_definition_and_no_clipping(self):
        center=torch.full((30,2),2.,dtype=torch.float32)
        raw=torch.stack([center]+[center+s*(p+1)*.01 for p in range(4) for s in (1,-1)])
        saved=raw.clone()
        for p in range(4):
            for radius in (1,4):
                for t in (-.25,.25):
                    a,meta=interior_actions(raw,p,radius,t)
                    self.assertTrue(torch.equal(a,center+(radius*t)*(raw[1+2*p]-center)))
                    self.assertFalse(meta['clipping_applied']); self.assertGreater(float(a.min()),1.)
        self.assertTrue(torch.equal(raw,saved))
        with self.assertRaises(ValueError): interior_actions(raw,0,1,0.)

    def test_source_filter_before_load(self):
        near=dict(complete=True,outputs=[dict(path=f'near-dev-{i:03d}.pt',key=f'near-dev-{i:03d}',split='development_external') for i in range(4)])
        prepared=dict(complete=True,episodes=[dict(episode=i,split='development') for i in range(4)]+[dict(episode=20,split='confirmation')])
        self.assertEqual(len(select_sources(near,prepared)),4)
        prepared['episodes'][0]['split']='confirmation'
        with self.assertRaises(ValueError): select_sources(near,prepared)

    def test_original_raw_reset_mandatory(self):
        raw=np.arange(7.)
        reset=dict(initial_state=raw,environment_seed=123,split='development',episode=0)
        initial,seed=require_reset_link(dict(reset_input_sha256='abc'),reset,'abc',0)
        np.testing.assert_array_equal(initial,raw);self.assertIsNot(initial,raw);self.assertEqual(seed,123)
        del reset['initial_state']
        reset['states']=np.zeros((31,7))
        with self.assertRaises(ValueError): require_reset_link(dict(reset_input_sha256='abc'),reset,'abc',0)

    def test_all_physics_pixels_rewards_are_exact(self):
        value={k:np.zeros(2) for k in ('frames','states','proprios','native_applied_targets_xy','native_rewards','native_goal_success','native_reward_goal_pose')}
        value.update(physics=[dict(x=0.)],contacts=[],native_dones=[None],contact_api_available=False)
        compare_replays(value,value)
        changed=dict(value,native_rewards=np.array([0.,1e-15]))
        with self.assertRaises(RuntimeError): compare_replays(value,changed)


if __name__=='__main__':unittest.main()
