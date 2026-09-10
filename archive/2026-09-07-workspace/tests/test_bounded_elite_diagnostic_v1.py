import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from run_bounded_elite_diagnostic_v1 import canonicalize_scratch
from run_xyz_projected_planner_v1 import project_actions


class Preprocessor:
    def denormalize_actions(self,a):return a*.5+.2
    def normalize_actions(self,a):return (a-.2)/.5


class BoundedScratchTests(unittest.TestCase):
    def test_disabled_is_exact_identity(self):
        a=torch.randn(6,300,20);old=a.clone()
        self.assertIs(canonicalize_scratch(a,torch.zeros_like(a),False),a)
        self.assertTrue(torch.equal(a,old))
    def test_bounded_first_effective_input_parity(self):
        nominal=torch.randn(6,300,20)*3;preserved=nominal.clone();expected,_=project_actions(nominal,Preprocessor())
        scratch=nominal.clone();ptr=scratch.data_ptr();canonicalize_scratch(scratch,expected,True)
        self.assertEqual(ptr,scratch.data_ptr());self.assertTrue(torch.equal(scratch,expected))
        self.assertTrue(torch.equal(nominal,preserved))
    def test_view_rejected_before_mutation(self):
        owner=torch.randn(7,300,20);old=owner.clone()
        with self.assertRaises(ValueError):canonicalize_scratch(owner[:6],torch.zeros_like(owner[:6]),True)
        self.assertTrue(torch.equal(owner,old))
    def test_shape_rejected(self):
        with self.assertRaises(ValueError):canonicalize_scratch(torch.ones(6,300,20),torch.ones(1),True)


if __name__=='__main__':unittest.main()
