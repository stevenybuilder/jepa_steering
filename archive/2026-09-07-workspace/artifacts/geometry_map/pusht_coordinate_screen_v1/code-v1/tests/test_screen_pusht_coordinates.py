import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT/"scripts/geometry_map"))
SPEC = importlib.util.spec_from_file_location("screen_pusht_coordinates", ROOT/"scripts/geometry_map/screen_pusht_coordinates.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class PushCoordinatesTests(unittest.TestCase):
    def test_actual_rotated_frame_roundtrip(self):
        current = np.array([[10.,20.,30.,40.],[5.,8.,2.,1.]])
        following = current+np.array([1.,2.,3.,4.])
        y, cols, origins, rotations = M.frame_targets(current, following, np.array([np.pi/2,-.4]), np.array([np.pi/2,-.4]), [0,5])
        data = dict(origins=origins, rotations=rotations)
        for frame in M.FRAMES: np.testing.assert_allclose(M.world_prediction(data, frame, y[:,cols[frame]]), following, atol=1e-12)
        np.testing.assert_allclose(y[0,cols["object_relative"]][:2], [-18.,19.], atol=1e-12)

    def test_no_confirmation_selection(self):
        self.assertFalse(M.selected({"source_id":125,"split":"confirmation"}, "fit"))
        self.assertFalse(M.selected({"source_id":125,"split":"confirmation"}, "validation"))
        with self.assertRaises(ValueError): M.selected({"source_id":125,"split":"confirmation"}, "confirmation")

    def test_freeze_required_before_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Freeze"): M.load_split(Path(folder), "validation")

    def test_motion_time_units(self):
        y, cols, _, _ = M.frame_targets(np.zeros((1,4)), np.array([[3.,4.,0.,10.]]), np.array([0.]), np.array([2*np.pi-.1]), [25])
        np.testing.assert_allclose(y[:,cols["physical_net_speed_per_raw_step"]], [[1.,2.]])
        np.testing.assert_allclose(y[:,cols["block_angular_motion_wrapped"]], [[-.1]])
        self.assertEqual(y[0,cols["observed_episode_raw_step"]][0],25)

    def test_residual_basis_one_raw_metric(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(100,20)); y = x[:,:4]
        b = M.residual_basis(x,y,.01,max_rank=8)
        self.assertEqual(b.shape,(20,8))
        np.testing.assert_allclose(b.T@b,np.eye(8),atol=1e-8)

    def test_whole_source_folds(self):
        groups = np.repeat(np.arange(100),6)
        for fold in range(5):
            train,test=set(groups[groups%5!=fold]),set(groups[groups%5==fold])
            self.assertFalse(train&test)
            self.assertEqual(len(test),20)


if __name__ == "__main__": unittest.main()
