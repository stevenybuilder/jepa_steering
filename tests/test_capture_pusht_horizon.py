import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/"scripts/geometry_map"))
SPEC=importlib.util.spec_from_file_location("capture_pusht_horizon",ROOT/"scripts/geometry_map/capture_pusht_horizon.py")
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)

class PushHorizonTests(unittest.TestCase):
    def test_no_held_selection(self):
        manifest={"task":"pusht_original_development","checkpoint_sha256":M.CHECKPOINT,"rows":[{"episode":i,"split":"development","sealed":False} for i in range(10)]}
        self.assertEqual(len(M.select_rows(manifest,[0,1])),2)
        with self.assertRaises(ValueError):M.select_rows(manifest,[10])
        manifest["rows"][0]["sealed"]=True
        with self.assertRaises(ValueError):M.select_rows(manifest,[0])
    def test_real_context_time_is_not_horizon(self):
        x=M.temporal_labels()
        np.testing.assert_array_equal(x["imagined_step"],np.arange(1,7))
        np.testing.assert_array_equal(x["real_raw_step"],np.arange(5,31,5))
        np.testing.assert_array_equal(x["context_real_raw_step"],np.zeros(6))
    def test_full_plan_required(self):
        class Prep:
            def denormalize_actions(self,x):return x
        plan=torch.arange(60,dtype=torch.float32).reshape(6,10)
        bank={"replans":[{"planned_actions_normalized":plan,"planned_actions_raw":plan.reshape(30,2),"executed_action_count":30}],"actions_raw":plan.reshape(30,2)}
        self.assertEqual(M.validate_plan(bank,Prep())[0].shape,(6,10))
        bank["replans"][0]["planned_actions_normalized"]=plan[:3]
        with self.assertRaises(ValueError):M.validate_plan(bank,Prep())
    def test_exact_guard(self):
        M.exact(torch.zeros(2),torch.zeros(2),"same")
        with self.assertRaises(RuntimeError):M.exact(torch.ones(2),torch.zeros(2),"different")
    def test_physics_guard(self):
        M.compare_physics([{"body":[1.,2.]}],[{"body":[1.,2.]}])
        with self.assertRaises(RuntimeError):M.compare_physics([{"body":[1.,2.]}],[{"body":[1.,3.]}])
    def test_native_singleton_buffers_are_canonicalized(self):
        x=M.encoder_inputs(torch.zeros(7,1,3,16,16),torch.zeros(7,1,4))
        self.assertEqual(x["visual"].shape,(1,7,3,16,16))
        self.assertEqual(x["proprio"].shape,(1,7,4))
    def test_extra_nonsingleton_axis_rejected(self):
        with self.assertRaises(ValueError):M.encoder_inputs(torch.zeros(7,2,3,16,16),torch.zeros(7,1,4))

if __name__=="__main__":unittest.main()
