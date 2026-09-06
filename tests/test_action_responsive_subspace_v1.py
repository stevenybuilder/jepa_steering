import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from action_responsive_subspace_v1 import fit_geometry,prepare_edits,match_delivered,InputPatch,norm


class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(10)
    def test_recovers_known_rank4_and_salient_patches(self):
        u=torch.linalg.qr(torch.randn(384,4,dtype=torch.float64)).Q
        c=torch.randn(7,256,4,dtype=torch.float64);c[:,64:]*=.01
        d=(c@u.T).float();d[0]=0
        variants,bases,masks,stats=fit_geometry(d,17)
        self.assertGreater(stats['retained_centered_variance'],.9999)
        self.assertGreater(float((bases['targeted'].T@u).square().sum()),3.999)
        self.assertEqual(int(masks['targeted'][:64].sum()),64)
        self.assertEqual(int(masks['random_patches'].sum()),64)
        self.assertTrue(torch.equal(variants['targeted'][0],torch.zeros_like(d[0])))
    def test_delivered_matching_signs_and_complement(self):
        d=torch.randn(5,256,384);d[0]=0;base=20*torch.randn_like(d)
        variants,bases,masks,_=fit_geometry(d,19)
        edited,stats=prepare_edits(base,variants,bases,masks)
        for name,out in edited.items():
            kind=name.rsplit('_',1)[0]
            self.assertTrue(torch.equal(out[0],base[0]));self.assertTrue(torch.equal(out[:,~masks[kind]],base[:,~masks[kind]]))
            torch.testing.assert_close(torch.tensor(stats[name]['delivered_l2']),torch.tensor(stats[name]['matched_target_l2']),rtol=2e-6,atol=2e-6)
    def test_zero_is_exact(self):
        base=torch.randn(4,256,384);zero=torch.zeros_like(base)
        out,delta,_=match_delivered(base,zero,torch.zeros(4,dtype=torch.float64))
        self.assertTrue(torch.equal(out,base));self.assertTrue(torch.equal(delta,zero))
    def test_impossible_match_rejected(self):
        with self.assertRaises(ValueError):match_delivered(torch.zeros(2,3,4),torch.zeros(2,3,4),torch.ones(2))
    def test_truevisual_rejects_mixed400(self):
        with self.assertRaises(ValueError):fit_geometry(torch.randn(5,256,400),9)
    def test_repeat_geometry_no_outcome_argument(self):
        d=torch.randn(5,256,384);d[0]=0
        first=fit_geometry(d,99);second=fit_geometry(d,99)
        for k in first[0]:self.assertTrue(torch.equal(first[0][k],second[0][k]))
    def test_hook_preserves_past_proprio_action_and_resets(self):
        v=torch.randn(3,2,1,16,16,384);a=torch.randn(3,2,10);p=torch.randn(3,2,256,16)
        edited=v[:,-1].reshape(3,256,384).clone();edited[1:]+=.1
        hook=InputPatch(None,(v,a,p),edited)
        for _ in range(2):self.assertIsNone(hook.hook(None,(v,a,p)))
        out=hook.hook(None,(v,a,p));self.assertIs(out[1],a);self.assertIs(out[2],p)
        self.assertTrue(torch.equal(out[0][:,:-1],v[:,:-1]));self.assertTrue(torch.equal(out[0][0],v[0]))
        fresh=InputPatch(None,(v,a,p),edited);self.assertEqual(fresh.calls,0)


if __name__=='__main__':unittest.main()
