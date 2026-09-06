import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from fit_pusht_task_metric import weighted_difference,fit_pca_directions,fit_weights,metric_cost


class TaskMetricTests(unittest.TestCase):
    def test_exact_native_weighting(self):
        v=np.arange(24).reshape(2,3,4);p=np.arange(8).reshape(2,4)
        d=weighted_difference(v,p,np.zeros_like(v),np.ones_like(p),.1)
        self.assertAlmostEqual(float(d@d),np.mean(v**2)+.1*np.mean((p-1)**2),places=12)
    def test_zero_atgoal_and_invalid_weight(self):
        x=np.ones(4);self.assertTrue(np.array_equal(weighted_difference(x,x,x,x),np.zeros(8)))
        with self.assertRaises(ValueError):weighted_difference(x,x,x,x,-1)
    def test_psd_complement_noedit_rotation(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(40,8));basis=fit_pca_directions(x,3)[0];w=np.array([0.,2.,4.])
        self.assertTrue(np.array_equal(metric_cost(x,basis,w,0),np.sum(x*x,axis=1)))
        self.assertEqual(float(metric_cost(np.zeros((1,8)),basis,w)[0]),0)
        off=x-(x@basis.T)@basis
        np.testing.assert_allclose(metric_cost(off,basis,w),np.sum(off*off,axis=1),atol=1e-12)
        q=np.linalg.qr(rng.normal(size=(3,3)))[0]
        np.testing.assert_allclose(metric_cost(x,basis,np.ones(3),1,q),np.sum(x*x,axis=1),atol=1e-12)
        self.assertTrue(np.all(metric_cost(x,basis,w)>=0))
    def test_train_fit_nonnegative_and_rankinvariant_scale(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(30,6));u=fit_pca_directions(x,3)[0]
        y=np.sum(x*x,axis=1);w,m=fit_weights(x,y,u)
        np.testing.assert_allclose(w,np.ones(3),atol=1e-10)
        scaled,_=fit_weights(x,y*100,u);np.testing.assert_allclose(w,scaled,atol=1e-10)


if __name__=='__main__':unittest.main()
