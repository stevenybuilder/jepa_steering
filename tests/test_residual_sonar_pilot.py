import unittest,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts/geometry_map'),str(ROOT/'scripts/cgs_pilot')]
from run_residual_sonar_pilot import high_progress_labels,fit_energy,raw_energy_gradient,capped_fields


class SonarTests(unittest.TestCase):
    def test_labels_are_local_progress_not_success(self):
        np.testing.assert_equal(high_progress_labels([-3,-2,-1,0,1]),[False,False,True,True,True])
        with self.assertRaises(ValueError):high_progress_labels([1,1,1])

    def test_gaussian_gradient_finite_difference(self):
        x=np.random.default_rng(45).normal(size=(10,7));fit=fit_energy(x,[np.arange(5),np.arange(5)])
        point=x[:2]+.2;energy,g=raw_energy_gradient(point,fit);basis=fit['basis'];eps=1e-5
        for j in range(fit['rank']):
            plus=raw_energy_gradient(point+eps*basis[j],fit)[0];minus=raw_energy_gradient(point-eps*basis[j],fit)[0]
            np.testing.assert_allclose((plus-minus)/(2*eps),g[:,j],rtol=1e-5,atol=1e-5)

    def test_metric_isotropic_identity_and_cap_doses(self):
        gradient=np.array([[3.,4.],[.03,.04]]);gram=np.broadcast_to(np.eye(2)*3,(2,2,2))
        fields,records=capped_fields(gradient,gram,np.ones(2))
        np.testing.assert_allclose(fields['static'],fields['local_metric'])
        np.testing.assert_allclose(np.linalg.norm(fields['static'],axis=1),[1,.05])
        np.testing.assert_allclose(np.linalg.norm(fields['static']*.5,axis=1),[.5,.025])
        self.assertEqual(records['fields']['static']['cap_saturated'],[True,False])

    def test_zero_gradient_stays_zero(self):
        fields,_=capped_fields(np.zeros((1,3)),np.zeros((1,3,3)),np.ones(1))
        for x in fields.values():np.testing.assert_equal(x,0)

    def test_direction_only_ablation_has_matched_nonzero_budget(self):
        gradient=np.array([[.03,.04],[0.,0.]])
        gram=np.broadcast_to(np.eye(2),(2,2,2))
        fields,records=capped_fields(gradient,gram,np.array([2.,2.]),normalize_direction=True)
        for field in fields.values():
            np.testing.assert_allclose(np.linalg.norm(field,axis=1),[2.,0.])
            np.testing.assert_allclose(field[0]/2,[-.6,-.8])
        self.assertTrue(records['fields']['static']['direction_normalized'])


if __name__=='__main__':unittest.main()
