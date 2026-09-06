from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/geometry_map'))
from finite_response_consequence import correction,FirstDelta

class FiniteResponseTests(unittest.TestCase):
    def test_measured_linear_response_solution(self):
        b=torch.eye(4)[:,:2];g=torch.tensor([[2.,0.],[0.,3.]])
        d=correction(torch.tensor([[2.,3.]]),b,g,1.)
        torch.testing.assert_close(d,torch.tensor([[2**-.5,2**-.5,0.,0.]]))
    def test_first_only_identity_and_cleanup(self):
        layer=torch.nn.Identity();x=torch.ones(1,256,400);delta=torch.ones(1,400)*.01
        with FirstDelta(layer,delta) as hook:
            first=layer(x);second=layer(x)
        torch.testing.assert_close(first,x+.01);self.assertTrue(torch.equal(second,x));self.assertTrue(torch.equal(layer(x),x))
    def test_negative_norm_control(self):
        b=torch.eye(4)[:,:2];g=torch.eye(2);e=torch.ones(1,2)
        self.assertTrue(torch.equal(correction(e,b,g,-.5),-correction(e,b,g,.5)))

if __name__=='__main__':unittest.main()
