import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from run_patch_policy_action_spatial import response_variants, central_current_condition, CurrentConditionResponse


class Block(torch.nn.Module):
    def forward(self,x,z,**kwargs):
        condition=z.repeat_interleave(256,dim=1)
        return x+torch.tanh(x)*condition


class Tests(unittest.TestCase):
    def test_norm_mean_permutation_and_zero(self):
        torch.manual_seed(8);d=torch.randn(3,256,400)+.2;d[0]=0
        v,available,stats=response_variants(d)
        self.assertTrue(available.all());self.assertTrue(torch.equal(v['dense'],d))
        for value in v.values():
            self.assertTrue(torch.equal(value[0],d[0]))
            torch.testing.assert_close(value.double().flatten(1).norm(dim=-1),d.double().flatten(1).norm(dim=-1),rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(v['spatial_permutation'].mean(1),d.mean(1))
        self.assertTrue(torch.equal(v['mean'][:,0],v['mean'][:,-1]))

    def test_degenerate_mean_unavailable(self):
        d=torch.ones(2,256,4);d[:,128:]=-1;d[0]=0
        v,a,stats=response_variants(d)
        self.assertEqual(a.tolist(),[True,False]);self.assertEqual(stats['mean_scale'],[0.,0.])
        self.assertEqual(v['mean'].count_nonzero(),0)

    def test_condition_current_only_ownership(self):
        source=torch.arange(4*3*5.).reshape(4,3,5);z=source[:,:2];saved=source.clone()
        changed=central_current_condition(z)
        self.assertEqual(changed.stride(),z.stride());self.assertTrue(torch.equal(source,saved))
        self.assertTrue(torch.equal(changed[:,:-1],z[:,:-1]));self.assertTrue(torch.equal(changed[:,-1],z[0:1,-1].expand(4,5)))

    def test_hook_native_identity_and_reset_removal(self):
        torch.manual_seed(2);block=Block();x=torch.randn(3,512,400);z=torch.randn(3,2,400)
        original=block(x,z)
        with CurrentConditionResponse(block) as cap:
            outputs=[block(x,z) for _ in range(6)]
        self.assertTrue(all(torch.equal(o,original) for o in outputs));self.assertEqual(len(block._forward_hooks),0)
        for dose in (0.,.1,-.1):
            with CurrentConditionResponse(block,cap.record,'dense',dose) as edit:
                outputs=[block(x,z) for _ in range(6)]
            self.assertEqual(edit.calls,6)
            for i,o in enumerate(outputs):
                if i!=2 or dose==0:self.assertTrue(torch.equal(o,original))
                else:
                    self.assertTrue(torch.equal(o[:,:256],original[:,:256]));self.assertTrue(torch.equal(o[0],original[0]))
        self.assertEqual(len(block._forward_hooks),0);self.assertTrue(torch.equal(block(x,z),original))

    def test_shapes_rejected(self):
        with self.assertRaises(ValueError):response_variants(torch.zeros(2,128,400))
        with self.assertRaises(ValueError):central_current_condition(torch.zeros(1,2,400))

    def test_first_horizon_single_frame_and_empty_past(self):
        torch.manual_seed(7);block=Block();x=torch.randn(3,256,400);z=torch.randn(3,1,400)
        original=block(x,z)
        with CurrentConditionResponse(block,horizon=1) as cap:
            outputs=[block(x,z) for _ in range(6)]
        self.assertTrue(all(torch.equal(o,original) for o in outputs))
        self.assertEqual(cap.record['counter_z'][:,:-1].numel(),0)
        for dose in (0.,-.1,.1):
            with CurrentConditionResponse(block,cap.record,'spatial_permutation',dose,horizon=1):
                outputs=[block(x,z) for _ in range(6)]
            self.assertTrue(torch.equal(outputs[0][0],original[0]))
            for o in outputs[1:]:self.assertTrue(torch.equal(o,original))
            if dose==0:self.assertTrue(torch.equal(outputs[0],original))


if __name__=='__main__':unittest.main()
