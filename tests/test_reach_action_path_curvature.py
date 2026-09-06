import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
import torch
from run_reach_action_path_curvature import frozen_directions,fixed_reach_path,checked_rows,check_original_goal,tensor_hash,EPISODES
from run_action_path_curvature import ResidualReplacement,replace_newest
from action_path_geometry import central_estimates,sampled_path_geometry

class ReachPathTests(unittest.TestCase):
    def test_qr_orthogonal_reproducible_rms(self):
        d=frozen_directions();self.assertTrue(torch.equal(d,frozen_directions()))
        flat=d.flatten(1)
        torch.testing.assert_close(flat@flat.T,torch.eye(4)*(.05**2*18),rtol=1e-6,atol=1e-8)
        torch.testing.assert_close(d.square().mean((1,2)).sqrt(),torch.full((4,),.05))
    def test_identity_gripper_repetition_and_radius(self):
        center=torch.linspace(-3,3,120).reshape(30,4);d=frozen_directions()
        p1,m=fixed_reach_path(center,d,0,1);p4,_=fixed_reach_path(center,d,0,4)
        self.assertTrue(torch.equal(p1[2],center));self.assertTrue(torch.equal(p4[:,:,3],center[None,:,3].expand(5,-1)))
        torch.testing.assert_close(p4[-1]-center,4*(p1[-1]-center),rtol=0,atol=1e-6)
        self.assertGreater(m['raw_xyz_outside_unit_box_counts'][2],0)
        self.assertFalse(m['model_action_clipping'])
        torch.testing.assert_close((p1[-1]-center)[:,:3].reshape(6,5,3),d[0,:,None].expand(6,5,3),rtol=0,atol=2e-7)
    def test_frozen_sources_reject_held_before_load(self):
        m={'complete':True,'sources':[{'episode':e,'split':'development','sealed':False} for e in EPISODES]}
        self.assertEqual(len(checked_rows(m)),4)
        m['sources'][-1]['episode']=12
        with self.assertRaises(ValueError):checked_rows(m)
    def test_reject_other_doses(self):
        with self.assertRaises(ValueError):fixed_reach_path(torch.zeros(30,4),frozen_directions(),0,2)
    def test_same_four_point_geometry_and_speed_control(self):
        t=torch.tensor([-1.,-.5,.5,1.],dtype=torch.float64)
        samples=torch.stack([2+3*t+4*t*t+5*t**3,-1+t*t],1)
        torch.testing.assert_close(central_estimates(samples)['cubic_equal_data'],torch.tensor([2.,-1.],dtype=torch.float64))
        with self.assertRaises(ValueError):central_estimates(torch.zeros(5,2))
        t=torch.tensor([-1.,-.5,0.,.5,1.],dtype=torch.float64)
        straight=sampled_path_geometry(torch.stack([t**3,2*t**3],1))
        self.assertLess(straight['max_orthogonal_fraction_of_chord'],1e-12)
        self.assertGreater(sampled_path_geometry(torch.stack([t,t*t],1))['max_orthogonal_fraction_of_chord'],.1)
    def test_newest_support_single_pulse_cleanup(self):
        x=torch.randn(7,512,400);replacement=x[:,-256:]+1
        self.assertTrue(torch.equal(replace_newest(x,x[:,-256:]),x))
        module=torch.nn.Identity()
        with ResidualReplacement(module,3,replacement):values=[module(x) for _ in range(6)]
        for i,v in enumerate(values):
            self.assertTrue(torch.equal(v[:,:256],x[:,:256]))
            self.assertEqual(torch.equal(v,x),i!=2)
        self.assertFalse(module._forward_hooks)
    def test_original_goal_contracts_without_invented_fp32_cache(self):
        encoded={k:torch.tensor([1.123456]) for k in ('visual','proprio')};raw={k:torch.tensor([1.]) for k in encoded}
        b={'initial_checks':{'goal_pixels':0.,'goal_proprio':0.},'goal':{'encoded_'+k:v.half() for k,v in encoded.items()}}
        h,mode=check_original_goal(encoded,raw,b);self.assertIn('legacy',mode)
        b['initial_checks']={'fresh_goal_sha256':h,'original_raw_goal_sha256':{k:tensor_hash(v) for k,v in raw.items()}}
        self.assertEqual(check_original_goal(encoded,raw,b)[0],h)
        with self.assertRaises(ValueError):check_original_goal({k:v+1 for k,v in encoded.items()},raw,b)

if __name__=='__main__':unittest.main()
