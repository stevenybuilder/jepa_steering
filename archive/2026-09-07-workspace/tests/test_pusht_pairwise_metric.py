import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from run_pusht_pairwise_metric import pairwise_objective,make_pair_groups,native_pair_temperature


class PairwiseTests(unittest.TestCase):
    def test_gradient_matches_finite_difference(self):
        rng=np.random.default_rng(19);f=[rng.uniform(size=(8,3)),rng.uniform(size=(5,3))]
        c=[rng.uniform(size=8),rng.uniform(size=5)];y=[rng.uniform(size=8),rng.uniform(size=5)]
        groups=make_pair_groups(f,c,y);w=np.array([.4,1.2,1.7]);value,grad=pairwise_objective(w,groups,.3,.13)
        fd=[]
        for k in range(3):
            plus=w.copy();minus=w.copy();plus[k]+=1e-6;minus[k]-=1e-6
            fd.append((pairwise_objective(plus,groups,.3,.13)[0]-pairwise_objective(minus,groups,.3,.13)[0])/2e-6)
        np.testing.assert_allclose(grad,fd,rtol=1e-7,atol=1e-8)
    def test_better_pair_cost_lower_reduces_loss(self):
        groups=make_pair_groups([np.array([[1.],[0.]])],[np.zeros(2)],[np.array([0.,1.])])
        self.assertLess(pairwise_objective(np.array([.1]),groups,1.,0)[0],pairwise_objective(np.array([1.]),groups,1.,0)[0])
    def test_tie_zero_direction_and_state_equal_weight(self):
        groups=make_pair_groups([np.array([[1.],[0.]])],[np.array([0.,1.])],[np.array([.3,.3])])
        self.assertEqual(groups[0]['preference'][0],.5)
        self.assertEqual(pairwise_objective(np.ones(1),groups,1.,0)[1][0],0.)
        doubled={**groups[0],**{k:np.concatenate([groups[0][k],groups[0][k]]) for k in ('features','offset','preference')}}
        a=pairwise_objective(np.array([.4]),groups,1.,.1)
        b=pairwise_objective(np.array([.4]),[doubled],1.,.1)
        self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    def test_temperature_training_only_and_scale(self):
        train=[np.array([1.,2.,3.]),np.array([0.,2.,4.])]
        tau=native_pair_temperature(train)
        self.assertAlmostEqual(native_pair_temperature([x*7 for x in train]),7*tau)
        with self.assertRaises(ValueError):native_pair_temperature([np.ones(4)])


if __name__=='__main__':unittest.main()
