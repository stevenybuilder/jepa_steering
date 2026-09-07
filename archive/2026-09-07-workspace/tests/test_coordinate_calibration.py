from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"/"geometry_map"))
from coordinate_calibration_operator import calibration_matrix, radial_cap, rotated_sham, CoordinateCalibration, CoordinateCalibrationHook


def fixture(dtype=torch.float64):
    generator = torch.Generator().manual_seed(31)
    observer = torch.randn(3,384,generator=generator,dtype=dtype)*.01
    covariance = torch.diag(torch.linspace(.5,2,384,dtype=dtype))
    q_scale = torch.tensor([.2,.5,.1],dtype=dtype)
    jacobian = observer/q_scale[:,None]
    return {"p3_mean": torch.zeros(400,dtype=dtype), "p3_scale": torch.ones(400,dtype=dtype),
            "p3_coef": torch.randn(3,400,generator=generator,dtype=dtype)*.01,
            "p3_intercept": torch.ones(3,dtype=dtype), "q_mean": torch.tensor([.1,.2,.3],dtype=dtype),
            "q_scale": q_scale, "observer_mean": torch.zeros(384,dtype=dtype),
            "observer_scale": torch.ones(384,dtype=dtype), "observer_coef": observer,
            "observer_intercept": torch.zeros(3,dtype=dtype),
            "correction_matrix": calibration_matrix(covariance,jacobian), "beta": .5,"cap_radius": .2}


class CalibrationTests(unittest.TestCase):
    def test_covariance_formula_solves_primal_normal_equation(self):
        dtype=torch.float64
        sigma=torch.tensor([[2.,.3],[.3,.7]],dtype=dtype)
        jac=torch.tensor([[1.,2.]],dtype=dtype)
        residual=torch.tensor([.8],dtype=dtype)
        delta=calibration_matrix(sigma,jac,.1)@residual
        gradient=jac.T@(jac@delta-residual)+.1*torch.linalg.solve(sigma,delta)
        torch.testing.assert_close(gradient,torch.zeros_like(gradient),atol=1e-12,rtol=0)
        self.assertGreater(float((jac@delta)*residual),0)

    def test_observer_jacobian_finite_difference_and_q_units(self):
        artifact=fixture(); operator=CoordinateCalibration(artifact,dtype=torch.float64)
        h=torch.randn(2,384,dtype=torch.float64); direction=torch.randn_like(h)
        eps=1e-5
        numeric=(operator.observer_physical(h+eps*direction)-operator.observer_physical(h-eps*direction))/(2*eps)/artifact['q_scale']
        jac=artifact['observer_coef']/artifact['observer_scale'][None,:]/artifact['q_scale'][:,None]
        torch.testing.assert_close(numeric,direction@jac.T,atol=1e-9,rtol=1e-9)

    def test_radial_cap_after_beta_and_exact_rotated_norm(self):
        delta=torch.randn(300,384)
        capped,saturated=radial_cap(.5*delta,.1)
        self.assertTrue(saturated.all())
        self.assertLessEqual(float(capped.norm(dim=-1).max()),.1000001)
        sham=rotated_sham(capped)
        self.assertTrue(torch.equal(sham.norm(dim=-1),capped.norm(dim=-1)))
        self.assertFalse(torch.equal(sham,capped))

    def test_zero_residual_and_beta_zero_are_identity(self):
        artifact=fixture(); artifact['p3_coef'].zero_(); artifact['p3_intercept'].zero_()
        artifact['observer_coef'].zero_(); artifact['observer_intercept']=artifact['q_mean'].clone()
        operator=CoordinateCalibration(artifact,dtype=torch.float64)
        delta,_,_=operator.delta(torch.randn(2,400,dtype=torch.float64),torch.randn(2,384,dtype=torch.float64))
        self.assertTrue(delta.eq(0).all())

    def test_native_tuple_hook_shape_centered_structure_identity_and_cleanup(self):
        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.block=torch.nn.Identity(); self.last=None
            def forward_pred(self, visual):
                b,t=visual.shape[:2]
                self.block(torch.ones(b,t*256,400,dtype=visual.dtype))
                self.last=(visual,torch.ones(b,t,1,4),torch.ones(b,t,1,16))
                return self.last
        wm=Fake(); original=wm.forward_pred
        operator=CoordinateCalibration(fixture(),dtype=torch.float64)
        visual=torch.randn(2,2,1,16,16,384,dtype=torch.float64)
        with CoordinateCalibrationHook(wm,wm.block,operator,beta=0):
            self.assertIs(wm.forward_pred(visual),wm.last)
        self.assertEqual(wm.forward_pred,original)
        self.assertEqual(len(wm.block._forward_hooks),0)
        with CoordinateCalibrationHook(wm,wm.block,operator) as hook:
            result=wm.forward_pred(visual)
            self.assertTrue(torch.equal(result[0][:,0],visual[:,0]))
            self.assertIs(result[1],wm.last[1]); self.assertIs(result[2],wm.last[2])
            a=result[0][:,-1].reshape(2,256,384); b=visual[:,-1].reshape(2,256,384)
            torch.testing.assert_close(a-a.mean(1,keepdim=True),b-b.mean(1,keepdim=True),atol=1e-12,rtol=0)
            self.assertEqual(hook.calls,1)
        self.assertEqual(len(wm.block._forward_hooks),0)

    def test_bad_covariance_or_shape_rejected(self):
        with self.assertRaises(torch.linalg.LinAlgError):
            calibration_matrix(torch.diag(torch.tensor([1.,-1.])),torch.ones(1,2))
        operator=CoordinateCalibration(fixture(),dtype=torch.float64)
        with self.assertRaises(ValueError):
            operator.delta(torch.zeros(400),torch.zeros(384))


if __name__ == '__main__':
    unittest.main()
