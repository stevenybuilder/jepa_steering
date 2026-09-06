import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT/"scripts/geometry_map"))
SPEC = importlib.util.spec_from_file_location("full_runner", ROOT/"scripts/geometry_map/run_residual_search_full_v1.py")
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class FullTests(unittest.TestCase):
    def test_only_frozen_development_starts(self):
        c = [{"id": "head", "family": "branch_scale"}]
        M.validate_design([0, 4, 7], c)
        for episodes in ([12], [8], [0, 0], []):
            with self.assertRaises(ValueError): M.validate_design(episodes, c)
        with self.assertRaises(ValueError): M.validate_design([0], c+c)
        with self.assertRaises(ValueError): M.validate_design([0], [{"id": "old", "family": "coordinate"}])

    def test_native_steps_left_keeps_released_rounding(self):
        env = SimpleNamespace(steps_left=lambda: 8)
        self.assertEqual(M.native_steps_left(env, SimpleNamespace(action_skip=1), SimpleNamespace(frameskip=5)), 1)
        env.steps_left = lambda: 98
        self.assertEqual(M.native_steps_left(env, SimpleNamespace(action_skip=1), SimpleNamespace(frameskip=5)), 19)

    def test_native_success_not_distance_proxy(self):
        states = np.zeros((99, 39)); states[:, -3:] = [1., 0., 0.]
        info = [{"success": 0.} for _ in range(99)]; info[10] = {"success": 1.}
        result = M.performance(np.zeros(3), states, [2.]*99, info)
        self.assertTrue(result["native_ever_success"])
        self.assertFalse(result["native_final_success"])
        self.assertEqual(result["native_reward_sum"], 198.)
        self.assertEqual(result["hand_goal_progress_m"], 0.)

    def test_partial_or_missing_success_fails(self):
        with self.assertRaises(RuntimeError): M.performance(np.zeros(3), np.zeros((98, 39)), [0.]*98, [{"success": 0.}]*98)
        with self.assertRaises(RuntimeError): M.performance(np.zeros(3), np.zeros((99, 39)), [0.]*99, [{}]*99)

    def test_full_adapter_respects_shortened_native_horizons(self):
        component = SimpleNamespace(step=0, records=[], norm_targets=None)
        component.reset = lambda: setattr(component, "step", 0)
        def native(z, act_suffix):
            component.step += len(act_suffix)
            return {"horizon": len(act_suffix)}
        patch = SimpleNamespace(candidate={"family": "pca_gain", "pulse": 6}, component=component,
                                native=native, norm_targets=None, records=[])
        for h in (6, 5, 3, 2, 1):
            result = M.full_patch_unroll(patch, None, act_suffix=torch.zeros(h, 300, 20))
            self.assertEqual(result["horizon"], h)
        with self.assertRaises(RuntimeError): M.full_patch_unroll(patch, None, act_suffix=torch.zeros(7, 300, 20))


if __name__ == "__main__": unittest.main()
