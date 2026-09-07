from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts/geometry_map"))
from action_consequence_diagnostics import action_bank,physical_metrics,actual_cost_state

class ActionConsequenceTests(unittest.TestCase):
    def test_native_cost_layout_uses_tensor_axes(self):
        from tensordict import TensorDict
        actual=TensorDict({'visual':torch.zeros(1,7,1,16,16,384),'proprio':torch.zeros(1,7,4)},batch_size=[])
        forecast=TensorDict({'visual':torch.ones(7,1,1,16,16,384),'proprio':torch.ones(7,1,4)},batch_size=[])
        result=actual_cost_state(actual,forecast)
        self.assertEqual(result['visual'].shape,(7,1,1,16,16,384));self.assertEqual(result['visual'].sum(),0)
        self.assertGreater(forecast['visual'].sum(),0)
    def test_fixed_same_state_actions(self):
        reference=torch.ones(30,2)*.04;a=action_bank(reference)
        self.assertEqual(a.shape,(7,30,2));self.assertTrue(torch.equal(a[0],reference));self.assertTrue(torch.equal(a[6],-reference))
        self.assertTrue(torch.equal(a[1],torch.zeros(30,2)));self.assertEqual(a[2,5:].abs().sum(),0)
        self.assertTrue(torch.equal(a[2],-a[3]));self.assertTrue(torch.equal(a[4],-a[5]))
    def test_wrapped_metric_not_negative(self):
        state=np.array([[0,0,0,0,4*np.pi,0,0.]])
        m=physical_metrics(state,np.zeros(7));self.assertLess(m['wrapped_angle_error'][0],1e-12)
        self.assertLess(m['native_angle_error'][0],0)
    def test_goal_progress_native_xy(self):
        m=physical_metrics(np.array([[3,4,0,0,0,0,0]]),np.zeros(7))
        self.assertEqual(m['goal_xy_distance'][0],5);self.assertEqual(m['goal_block_distance'][0],0)

if __name__=='__main__':unittest.main()
