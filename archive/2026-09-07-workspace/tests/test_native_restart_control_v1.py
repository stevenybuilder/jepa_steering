from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from run_native_restart_control_v1 import apply_restart_seed,SEEDS,EPISODES

class RestartTests(unittest.TestCase):
    def test_only_local_planner_rng_changes(self):
        agent=SimpleNamespace(local_gpu_generator=torch.Generator().manual_seed(7))
        initial={'planner_rng':agent.local_gpu_generator.get_state().clone(),'visual':torch.ones(2),'physics':{'qpos':torch.zeros(3)}}
        global_rng=torch.get_rng_state().clone();changed=apply_restart_seed(agent,initial,817)
        self.assertFalse(torch.equal(changed['planner_rng'],initial['planner_rng']))
        self.assertIs(changed['visual'],initial['visual']);self.assertIs(changed['physics'],initial['physics'])
        self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
        expected=torch.Generator().manual_seed(817).get_state();self.assertTrue(torch.equal(changed['planner_rng'],expected))
    def test_frozen_six_runs(self):
        self.assertEqual(SEEDS,(817,1801));self.assertEqual(EPISODES,(0,4,7))
        with self.assertRaises(ValueError):apply_restart_seed(SimpleNamespace(),{},99)

if __name__=='__main__':unittest.main()
