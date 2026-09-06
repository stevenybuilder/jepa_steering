from pathlib import Path
import sys
import unittest
import torch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/geometry_map'))
from intervene_head_spatial_mean import component,FixedNormHead,specifications,batch_context

class SpatialHeadTests(unittest.TestCase):
    def test_components_reconstruct_and_center(self):
        x=torch.randn(2,256,400)
        torch.testing.assert_close(component(x,'spatial')+component(x,'mean'),x)
        torch.testing.assert_close(component(x,'spatial').mean(1),torch.zeros(2,400),atol=1e-7,rtol=0)
    def test_all_heads_horizons_common_norm(self):
        torch.manual_seed(8);layer=torch.nn.Linear(400,400);block=SimpleNamespace(attn=SimpleNamespace(proj=layer,num_heads=16))
        x=torch.randn(2,256,400);gate=torch.randn(2,256,400)
        for head in range(16):
            for h in (1,3,6):
                for kind in ('spatial','mean'):
                    for sham in (False,True):
                        hook=FixedNormHead(block,head,h,kind,2.,sham);hook.step=h
                        delta,record=hook._delta(x,gate)
                        torch.testing.assert_close(record['requested_residual_norm'],torch.full((2,),2.),rtol=2e-6,atol=2e-5)
    def test_fixed_grid_count(self):
        self.assertEqual(len(specifications(16)),192);self.assertEqual(len(specifications(8)),96)
        self.assertEqual({r['horizon'] for r in specifications(16)},{1,3,6})
    def test_native_context_batch_before_time(self):
        x={'visual':torch.ones(1,1,1,16,16,384),'proprio':torch.ones(1,1,256,16)}
        y=batch_context(x,9)
        self.assertEqual(y['visual'].shape,(9,1,1,16,16,384))
        self.assertEqual(y['proprio'].shape,(9,1,256,16))

if __name__=='__main__':unittest.main()
