import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from run_density_geometry_pilot import KNOTS,natural_weights,pca_delta,check_path_coordinate,local_linear_weights,orthonormalize_rows


class DensityGeometryTests(unittest.TestCase):
    def test_natural_endpoints_and_affine(self):
        np.testing.assert_allclose(natural_weights(KNOTS),np.eye(4),atol=1e-14)
        x=np.linspace(-4,4,151);w=natural_weights(x)
        np.testing.assert_allclose(w@KNOTS,x,atol=1e-14)
        np.testing.assert_allclose(w.sum(1),1,atol=1e-14)

    def test_nonzero_curvature_without_midpoint(self):
        weights=natural_weights([0])[0]
        self.assertLess(float(weights@(KNOTS**2)),float((KNOTS[[0,-1]]**2).mean()))
        self.assertEqual(len(weights),4)
        with self.assertRaises(ValueError):natural_weights([5])

    def test_local_linear(self):
        x=np.linspace(-4,4,33);weights=local_linear_weights(x)
        np.testing.assert_allclose(weights@KNOTS,x,atol=1e-14)
        np.testing.assert_allclose(local_linear_weights([0]),[[0,.5,.5,0]])

    def test_svd_roundoff_cleanup_preserves_span(self):
        torch.manual_seed(18);original=torch.linalg.qr(torch.randn(100,8,dtype=torch.float64)).Q.T
        drifted=original*torch.linspace(.999,1.001,8)[:,None]
        corrected=orthonormalize_rows(drifted)
        torch.testing.assert_close(corrected@corrected.T,torch.eye(8,dtype=torch.float64),atol=1e-13,rtol=0)
        torch.testing.assert_close(corrected.T@corrected,original.T@original,atol=1e-13,rtol=0)

    def test_complement_unchanged(self):
        torch.manual_seed(42);basis=torch.linalg.qr(torch.randn(20,4,dtype=torch.float64)).Q.T
        x=torch.randn(3,20,dtype=torch.float64);target=torch.randn_like(x);mean=torch.randn(20,dtype=torch.float64)
        delta=pca_delta(x,target,mean,basis)
        torch.testing.assert_close(delta-(delta@basis.T)@basis,torch.zeros_like(delta),atol=1e-13,rtol=0)
        torch.testing.assert_close((x+delta-mean)@basis.T,(target-mean)@basis.T)
        torch.testing.assert_close(pca_delta(x,x,mean,basis),torch.zeros_like(x),atol=0,rtol=0)

    def test_intrinsic_action_coordinate(self):
        center=torch.randn(9,30,2);delta=torch.randn(4,9,30,2)*.01
        knots=center[None,None]+torch.tensor(KNOTS).float()[:,None,None,None,None]*delta[None]
        self.assertTrue(check_path_coordinate(center,knots)['local_input_coordinate_identified'])
        with self.assertRaises(ValueError):check_path_coordinate(center,center[None,None].expand(4,4,-1,-1,-1).clone())


if __name__=='__main__':unittest.main()
