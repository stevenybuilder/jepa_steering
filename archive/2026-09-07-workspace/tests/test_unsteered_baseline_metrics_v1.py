import importlib.util
from pathlib import Path
import unittest
import numpy as np
import sys

p=Path(__file__).resolve().parents[1]/'scripts/geometry_map/report_unsteered_baseline_metrics_v1.py'
s=importlib.util.spec_from_file_location('metrics',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
sys.path.insert(0,str(p.parent))

class BaselineTests(unittest.TestCase):
    def test_fingerprint_dtype_and_signedzero(self):
        self.assertEqual(m.fingerprint(np.array([0,1],np.float32)),m.fingerprint(np.array([-0.,1.],np.float64)))
        self.assertNotEqual(m.fingerprint([0,1]),m.fingerprint([0,2]))
    def test_native_success_not_distance_threshold(self):
        v=dict(replans=[{'state':np.zeros(39)}],goal={'state':np.ones(39)},actions_raw=np.zeros((99,4)),
            ever_success=True,final_success=False,environment_steps=100,replan_count=7,total_reward=2.,final_state_distance_to_expert_goal=0.)
        r=m.reach_metrics(v);self.assertTrue(r['native_ever_success']);self.assertFalse(r['native_final_success']);self.assertIsNone(r['final_hand_goal_distance_m']);self.assertEqual(r['raw_steps'],99)
    def test_nonfinite_scalar_rejected(self):
        v=dict(replans=[{'state':np.zeros(39)}],goal={'state':np.ones(39)},actions_raw=np.zeros((99,4)),
            ever_success=False,final_success=False,environment_steps=100,replan_count=7,total_reward=float('nan'),final_state_distance_to_expert_goal=0.)
        with self.assertRaises(ValueError):m.reach_metrics(v)
    def test_coverage_identity_and_disjoint(self):
        from diagnose_pusht_goal_coverage import coverage
        self.assertAlmostEqual(coverage([256,256,.4],[256,256,.4]),1.)
        self.assertEqual(coverage([0,0,0],[1000,1000,0]),0.)

if __name__=='__main__':unittest.main()
