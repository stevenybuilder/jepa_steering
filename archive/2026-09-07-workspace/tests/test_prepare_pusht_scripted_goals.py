import importlib.util
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts/geometry_map"))
import prepare_pusht_scripted_goals as panel


class ControllerTests(unittest.TestCase):
    def test_direction_fixed_from_initial_coordinates(self):
        self.assertEqual(panel.unit_direction([0,0],[3,4]),[.6,.8])

    def test_degenerate_direction_is_predeclared(self):
        self.assertEqual(panel.unit_direction([4,4],[4,4]),[1.,0.])

    def test_relative_command_can_be_negative_and_is_bounded(self):
        raw,aim=panel.command([400,400],[200,200],[-1,0])
        self.assertEqual(raw,[-.25,-.25])
        self.assertEqual(aim,[170.,200.])

    def test_aim_clips_to_verified_arena(self):
        raw,aim=panel.command([490,20],[500,5],[1,-1])
        self.assertEqual(aim,[492.,20.])
        self.assertEqual(raw,[.02,0.])

    def test_command_is_relative_not_absolute(self):
        raw,aim=panel.command([250,250],[240,250],[1,0])
        self.assertEqual(aim,[270.,250.])
        self.assertEqual(raw,[.2,0.])

    def test_actual_vec2d_float32_conversion_not_float64_ideal(self):
        try:
            import numpy as np
            from pymunk import Vec2d
        except ImportError:
            self.skipTest("Native numerical runtime test runs on the simulator worker")
        position=Vec2d(350.07020496368403,174.0702038192749)
        action=np.array([-.25,-.25],dtype=np.float32)
        actual=np.asarray(position+action*100)
        ideal=np.asarray(position,dtype=np.float64)+action*100
        if int(np.__version__.split('.')[0])>=2:
            self.assertGreater(float(np.max(np.abs(actual-ideal))),1e-5)
        self.assertTrue(np.array_equal(actual,np.asarray(position+np.array(action)*100)))


if __name__=="__main__":unittest.main()
