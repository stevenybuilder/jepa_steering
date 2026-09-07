from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from run_planner_compensation_v1 import SignedCalibration,perturbation_batch,central_jacobian,compare_replay,restore_fixed_encoded_goal,CONDITIONS,NEW_ARMS
from coordinate_calibration_operator import CoordinateCalibration
from test_coordinate_calibration import fixture

class CompensationV1Tests(unittest.TestCase):
    def test_negative_is_exact_original_postcap_opposite(self):
        original=CoordinateCalibration(fixture(),dtype=torch.float64)
        p=torch.randn(4,400,dtype=torch.float64);v=torch.randn(4,384,dtype=torch.float64)
        a,s,r=original.delta(p,v);b,t,q=SignedCalibration(original,-1).delta(p,v)
        self.assertTrue(torch.equal(b,-a));self.assertTrue(torch.equal(a.norm(dim=-1),b.norm(dim=-1)))
        self.assertTrue(torch.equal(s,t));self.assertTrue(torch.equal(r,q))
        self.assertTrue(torch.equal(original.values['correction_matrix'],-SignedCalibration(original,-1).values['correction_matrix']))
    def test_action_coordinate_order_and_no_clipping(self):
        a=torch.arange(120,dtype=torch.float64).reshape(6,20);old=a.clone();v=perturbation_batch(a,.01)
        self.assertEqual(v.shape,(6,240,20));self.assertTrue(torch.equal(a,old))
        torch.testing.assert_close(v[:,0]-a,torch.nn.functional.one_hot(torch.tensor(0),120).reshape(6,20).to(a)*.01)
        self.assertGreater(float(v.max()),1.)
        with self.assertRaises(ValueError):perturbation_batch(a[:3],.01)
    def test_finite_difference_linear_jacobian(self):
        a=torch.zeros(6,20,dtype=torch.float64);weights=torch.randn(120,3,dtype=torch.float64)
        batch=perturbation_batch(a,.01).permute(1,0,2).reshape(240,120)
        values=(batch@weights)[None].expand(6,-1,-1)
        torch.testing.assert_close(central_jacobian(values,.01),weights[None].expand(6,-1,-1))
    def test_replay_guard_and_frozen_arm_counts(self):
        a=torch.zeros(3);compare_replay(a,a,'exact')
        with self.assertRaises(RuntimeError):compare_replay(a,a+1e-4,'bad')
        self.assertEqual(len(CONDITIONS),5);self.assertEqual(len(NEW_ARMS),2)
    def test_goal_restoration_updates_actual_objective_exactly(self):
        from types import SimpleNamespace
        target={'visual':torch.ones(1,2),'proprio':torch.ones(1,4)}
        agent=SimpleNamespace(goal_state_enc=target,objective=SimpleNamespace(target_enc=target))
        agent.planner=SimpleNamespace(set_objective=lambda objective:setattr(agent.planner,'objective',objective))
        snap={'goal_visual':torch.zeros(1),'goal_proprio':torch.zeros(1),
              'goal_encoded_visual':torch.ones(1,2),'goal_encoded_proprio':torch.ones(1,4)}
        ref={k:v.clone() for k,v in snap.items()};ref['goal_encoded_proprio']+=1e-7
        result=restore_fixed_encoded_goal(agent,snap,ref)
        self.assertTrue(torch.equal(agent.planner.objective.target_enc['proprio'],ref['goal_encoded_proprio']))
        self.assertGreater(result['fresh_encoding_difference']['proprio'],0.)
        ref['goal_visual']+=1
        with self.assertRaises(RuntimeError):restore_fixed_encoded_goal(agent,snap,ref)
    def test_expanded_goal_replaced_without_alias_write(self):
        from types import SimpleNamespace
        target={'visual':torch.ones(1,2).expand(3,2),'proprio':torch.ones(1,4).expand(3,4)}
        agent=SimpleNamespace(goal_state_enc=target,objective=SimpleNamespace(target_enc=target),planner=SimpleNamespace(set_objective=lambda _:None))
        snap={'goal_visual':torch.zeros(1),'goal_proprio':torch.zeros(1),
              'goal_encoded_visual':target['visual'].clone(),'goal_encoded_proprio':target['proprio'].clone()}
        ref={k:v.clone() for k,v in snap.items()};ref['goal_encoded_proprio']+=1e-7
        restore_fixed_encoded_goal(agent,snap,ref)
        self.assertTrue(torch.equal(agent.objective.target_enc['proprio'],ref['goal_encoded_proprio']))
        self.assertTrue(agent.objective.target_enc['proprio'].is_contiguous())

if __name__=='__main__':unittest.main()
