import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from diagnose_pusht_metric_capacity import energy_parts,margin_diagnostics,scaled_labels


class MetricCapacityTests(unittest.TestCase):
    def test_frozen_label_scaling_precision(self):
        y=np.asarray([3990.841,8223.25],dtype=np.float32);scale=.00012714738536459005
        actual=scaled_labels(y,scale)
        self.assertEqual(actual.dtype,np.float64)
        np.testing.assert_array_equal(actual,y.astype(np.float64)*scale)
    def test_energy_variance_decomposition(self):
        x=np.array([[1.,3.],[2.,4.],[3.,2.]])
        native,span,complement,c,d=energy_parts(x,np.array([[1.,0.]]))
        np.testing.assert_allclose(native,span+complement)
        v=d['variance_decomposition'];self.assertAlmostEqual(v['native'],v['span']+v['complement']+v['twice_covariance'])
    def test_margin_flip_and_unchanged_complement(self):
        native=np.array([1.,1.1,2.]);span=np.array([.9,.1,.5]);comp=native-span;coords=np.sqrt(span)[:,None]
        d=margin_diagnostics(native,span,comp,coords,np.array([2.]))
        self.assertEqual(d['native_winner'],0);self.assertEqual(d['native_runner'],1)
        self.assertEqual(d['blends'][0]['selected_index'],0);self.assertEqual(d['blends'][-1]['selected_index'],1)
        self.assertAlmostEqual(d['native_pair_span_margin']+d['native_pair_complement_margin'],d['native_margin'])


if __name__=='__main__':unittest.main()
