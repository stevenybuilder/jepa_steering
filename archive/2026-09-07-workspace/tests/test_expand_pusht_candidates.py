import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from run_expand_pusht_candidates import expanded_actions,spearman,summarize


class Prep:
    def normalize_actions(self,x): return (x-.2)/.3
    def denormalize_actions(self,x): return x*.3+.2


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.raw=torch.arange(540,dtype=torch.float32).reshape(9,30,2)/1000
        self.normal=Prep().normalize_actions(self.raw).reshape(9,6,10)
    def test_original_nine_and_zero_exact(self):
        raw,norm,meta,dirs=expanded_actions(self.raw,self.normal,0,Prep())
        self.assertTrue(torch.equal(raw[:9],self.raw));self.assertTrue(torch.equal(norm[:9],self.normal))
        self.assertEqual(raw.shape,(64,30,2));self.assertTrue(torch.equal(raw[63],torch.zeros(30,2)))
        self.assertEqual(len(meta),64);np.testing.assert_allclose((dirs**2).mean((1,2)),1,atol=1e-14)
    def test_antithetic_dose_and_reproducible(self):
        raw,norm,meta,_=expanded_actions(self.raw,self.normal,2,Prep())
        other=expanded_actions(self.raw,self.normal,2,Prep())[0];self.assertTrue(torch.equal(raw,other))
        for i in range(9,63,2):
            torch.testing.assert_close(norm[i]+norm[i+1],2*self.normal[0],atol=4e-7,rtol=0)
            self.assertAlmostEqual(float((norm[i]-self.normal[0]).square().mean().sqrt()),meta[i]['normalized_rms_radius'],places=6)
    def test_no_hidden_clip_and_forbidden_sources(self):
        raw=expanded_actions(self.raw,self.normal,0,Prep())[0]
        self.assertLess(float(raw.min()),0)
        with self.assertRaises(ValueError): expanded_actions(self.raw,self.normal,4,Prep())
    def test_tie_spearman_and_oracle(self):
        self.assertAlmostEqual(spearman([0,0,2],[1,1,3]),1)
        self.assertIsNone(spearman([0,0],[1,2]))
        rows=[{'final_state':[float(i+1)]*7,'native_full_objective_cost':float(i)} for i in range(64)]
        s=summarize(rows,np.zeros(7),np.ones(7)*100)
        self.assertEqual(s['full7d_state_l2']['oracle_index'],0)
        self.assertEqual(s['joint_xy_distance_px']['new64_oracle_gain_vs_original9'],0)


if __name__=='__main__': unittest.main()
