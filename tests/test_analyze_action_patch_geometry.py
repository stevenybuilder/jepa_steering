from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/geometry_map'))
from analyze_action_patch_geometry import head_metrics

class PatchAnalysisTests(unittest.TestCase):
    def test_equal_orthogonal_heads(self):
        n=np.ones((1,1,1,16));c=np.eye(16)[None,None,None]
        m=head_metrics(dict(head_norm=n,head_cosine=c,head_pooled_norm=n/16,head_pooled_cosine=c))
        self.assertAlmostEqual(float(m['effective_head_count'].item()),16.)
        self.assertAlmostEqual(float(m['max_head_energy_fraction'].item()),1/16)
        self.assertAlmostEqual(float(m['total_head_sum_energy_to_individual_energy'].item()),1.)
        self.assertAlmostEqual(float(m['pooling_retained_energy_fraction'].item()),1.)
    def test_one_dominant_head(self):
        n=np.zeros((1,1,1,16));n[...,0]=2.;c=np.eye(16)[None,None,None]
        m=head_metrics(dict(head_norm=n,head_cosine=c,head_pooled_norm=n/16,head_pooled_cosine=c))
        self.assertAlmostEqual(float(m['effective_head_count'].item()),1.)

if __name__=='__main__':unittest.main()
