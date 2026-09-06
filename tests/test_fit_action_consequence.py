from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts/geometry_map"))
from fit_action_consequence import subspaces,linear_map,predict,ranking

class FitActionTests(unittest.TestCase):
    def test_matched_orthonormal_subspaces(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(4,7,6,400));q=rng.normal(size=(4,7,6,6))
        bases,_,_=subspaces(x,q)
        for b in bases.values():np.testing.assert_allclose(b.T@b,np.eye(2),atol=1e-10)
        np.testing.assert_allclose(bases['random'].T@bases['readable'],0,atol=1e-10)
    def test_centering_excludes_same_state_readability(self):
        rng=np.random.default_rng(3);base=rng.normal(size=(4,1,6,400));x=base+rng.normal(size=(4,7,6,400))*.1
        centered=x-x.mean(1,keepdims=True)
        np.testing.assert_allclose(centered.mean(1),0,atol=1e-12)
    def test_physical_rank_regret(self):
        r=ranking(np.array([2.,0.,1.]),np.array([0.,2.,1.]),np.array([0.,4.,2.]))
        self.assertEqual(r['selected_candidate'],1);self.assertEqual(r['actual_encoded_regret'],2)

if __name__=='__main__':unittest.main()
