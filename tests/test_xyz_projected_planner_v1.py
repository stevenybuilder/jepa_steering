import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from run_xyz_projected_planner_v1 import project_actions,unroll_with_projection


class Preprocessor:
    mean=torch.tensor([.1,.2,.3,.4]);std=torch.tensor([.5,.6,.7,.8])
    def denormalize_actions(self,a):return a*self.std+self.mean
    def normalize_actions(self,a):return (a-self.mean)/self.std


class ProjectionTests(unittest.TestCase):
    def test_official_coordinate_projection_and_gripper(self):
        pre=Preprocessor();actions=torch.randn(6,3,20)*4;old=actions.clone()
        result,stats=project_actions(actions,pre)
        raw=pre.denormalize_actions(old.reshape(-1,4));raw[:,:3]=raw[:,:3].clamp(-1,1)
        self.assertTrue(torch.equal(result,pre.normalize_actions(raw).reshape_as(actions)))
        self.assertTrue(torch.equal(actions,old));self.assertGreater(stats['xyz_clipped'],0)
        torch.testing.assert_close(pre.denormalize_actions(result.reshape(-1,4))[:,3],pre.denormalize_actions(old.reshape(-1,4))[:,3])
    def test_disabled_identity_uses_original_tensor(self):
        a=torch.randn(6,2,20);native=lambda z,act_suffix:(z,act_suffix)
        answer=unroll_with_projection(native,Preprocessor(),False,'z',act_suffix=a)
        self.assertIs(answer[1],a)
    def test_positional_binding(self):
        a=torch.randn(2,1,20);native=lambda z,act_suffix:act_suffix
        expected,_=project_actions(a,Preprocessor())
        self.assertTrue(torch.equal(unroll_with_projection(native,Preprocessor(),True,None,a),expected))
    def test_invalid_shape_nonfinite(self):
        for x in (torch.ones(6,20),torch.full((6,1,20),float('nan'))):
            with self.assertRaises(ValueError):project_actions(x,Preprocessor())


if __name__=='__main__':unittest.main()
