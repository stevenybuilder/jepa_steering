import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('cem_elite_geometry',
    Path(__file__).resolve().parents[1]/'scripts/geometry_map/cem_elite_geometry.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class EliteGeometryTests(unittest.TestCase):
    def test_two_opposed_elites(self):
        r=m.summarize_elites([[-1.,0.],[1.,0.]],[0.,0.])
        self.assertEqual(r['elite_covariance_participation_rank'],1.)
        self.assertEqual(r['mean_to_nearest_elite_over_rms_radius'],1.)
        self.assertIsNone(r['diagonal_gaussian_entropy_nats'])

    def test_isotropic_square(self):
        r=m.summarize_elites([[-1,-1],[-1,1],[1,-1],[1,1]],[0,0])
        self.assertAlmostEqual(r['elite_covariance_participation_rank'],2.)
        self.assertAlmostEqual(r['diagonal_gaussian_entropy_nats'],math.log(2*math.pi*math.e*4/3))

    def test_collapse_is_defined(self):
        r=m.summarize_elites([[2,3],[2,3]],[2,3])
        self.assertEqual(r['elite_rms_radius'],0)
        self.assertEqual(r['elite_covariance_participation_rank'],0)
        self.assertIsNone(r['mean_to_nearest_elite_over_rms_radius'])

    def test_delivered_mean_checked(self):
        r=m.summarize_elites([[0],[2]],[1.1],[0])
        self.assertAlmostEqual(r['mean_arithmetic_maxabs_error'],.1)
        self.assertAlmostEqual(r['mean_update_l2'],1.1)

    def test_scale_and_translation(self):
        a=m.summarize_elites([[-2,-1],[0,0],[2,1]],[0,0])
        b=m.summarize_elites([[4,7],[10,10],[16,13]],[10,10])
        self.assertAlmostEqual(a['elite_covariance_participation_rank'],b['elite_covariance_participation_rank'])
        self.assertAlmostEqual(b['elite_rms_radius'],3*a['elite_rms_radius'])

    def test_invalid(self):
        for e,mu in [([[1]],[1]),([[1],[2]],[1,2]),([[float('nan')],[1]],[0])]:
            with self.assertRaises(ValueError):m.summarize_elites(e,mu)


if __name__=='__main__':unittest.main()
