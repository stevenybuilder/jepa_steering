import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
import numpy as np
import torch
from run_action_ranking_pilot import safe_rows,rank_correlation,donor_plans,local_chord_delta,norm_sham,score

class RankingTests(unittest.TestCase):
    def test_rank_sign_ties_and_constants(self):
        a=np.arange(9,dtype=float)
        self.assertAlmostEqual(rank_correlation(a,a),1)
        self.assertAlmostEqual(rank_correlation(a,-a),-1)
        self.assertIsNone(rank_correlation(a,np.zeros(9)))
        tied=np.array([0,0,1,1,2,2,3,3,4.]);self.assertAlmostEqual(rank_correlation(tied,tied),1)
    def test_forbid_missing_and_held_sources(self):
        with self.assertRaises(ValueError):safe_rows({'complete':True,'outputs':[]})
    def test_symmetric_donors_keep_candidate_centers(self):
        g=torch.Generator().manual_seed(5);center=torch.randn(30,2,generator=g);d=torch.randn(4,30,2,generator=g)*.01
        raw=torch.stack([center]+[v for x in d for v in (center+x,center-x)])
        normalized=raw.reshape(9,6,10);rr,nn=donor_plans(raw,normalized)
        torch.testing.assert_close((rr[0]+rr[1])/2,raw)
        self.assertEqual(tuple(nn.shape),(8,9,6,10))
    def test_smoother_and_sham_norm(self):
        recipient=torch.ones(9,256,400);donors=torch.stack([recipient+i for i in (-4,-3,-2,-1,1,2,3,4)])
        self.assertTrue(torch.equal(local_chord_delta(donors,recipient),torch.zeros_like(recipient)))
        delta=torch.randn_like(recipient)*.01;s=norm_sham(delta,0)
        torch.testing.assert_close(s.flatten(1).norm(dim=1),delta.flatten(1).norm(dim=1))
        self.assertTrue(torch.equal(s,norm_sham(delta,0)))
    def test_true_physics_selection_not_latent_proxy(self):
        states=np.zeros((9,31,7));states[:,-1,0]=np.arange(9);goal=np.zeros(7)
        result=score(-np.arange(9),states,goal)
        self.assertEqual(result['selected_action_index'],8)
        self.assertEqual(result['selected_actual_progress_px'],-8)
        self.assertEqual(result['original_native_planner_actual_progress_px'],0)
        self.assertAlmostEqual(result['cost_vs_physical_distance_spearman'],-1)

if __name__=='__main__':unittest.main()
