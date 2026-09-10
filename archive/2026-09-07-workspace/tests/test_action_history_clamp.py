from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/geometry_map'))
from run_action_history_clamp import clamp_arguments, PredictorInputs


class HistoryClampTests(unittest.TestCase):
    def make(self,t=2):
        g=torch.Generator().manual_seed(12)
        args=(torch.randn(7,t,1,2,2,3,generator=g),torch.randn(7,t,2,generator=g),torch.randn(7,t,4,2,generator=g))
        central=tuple(x[2:3].clone() for x in args)
        return args,central
    def test_state_clamp_leaves_actions_and_source_storage(self):
        args,central=self.make();saved=[x.clone() for x in args]
        result=clamp_arguments(args,central)
        self.assertIs(result[1],args[1])
        for i in (0,2): self.assertTrue(torch.equal(result[i],central[i].expand_as(result[i])))
        for a,b in zip(args,saved): self.assertTrue(torch.equal(a,b))
    def test_past_action_clamp_keeps_newest_and_central(self):
        args,central=self.make(3);result=clamp_arguments(args,central,True)
        self.assertTrue(torch.equal(result[1][:,-1],args[1][:,-1]))
        self.assertTrue(torch.equal(result[1][:,:-1],central[1][:,:-1].expand_as(result[1][:,:-1])))
        for i in range(3): self.assertTrue(torch.equal(result[i][2:3],central[i]))
    def test_window_one_exact_duplicate_and_stride(self):
        args,central=self.make(1)
        first=clamp_arguments(args,central);second=clamp_arguments(args,central,True)
        for a,b,x in zip(first,second,args):
            self.assertTrue(torch.equal(a,b));self.assertEqual(a.stride(),x.stride())
    def test_native_sliding_window_strides_and_zero_change(self):
        full,_=self.make(6)
        args=tuple(x[:,-2:] for x in full)
        central=tuple(x[2:3].clone() for x in args)
        result=clamp_arguments(args,central,True)
        for a,b in zip(args,result): self.assertEqual(a.stride(),b.stride())
        equal=tuple(x[2:3].repeat(7,*([1]*(x.ndim-1))) for x in args)
        identity=clamp_arguments(equal,central,True)
        for a,b in zip(equal,identity): self.assertTrue(torch.equal(a,b))
    def test_capture_cleanup_and_six_call_guard(self):
        class Fake(torch.nn.Module):
            def forward(self,v,a,p):return v+a.mean()+p.mean()
        model=Fake();args,_=self.make(1);expected=model(*args)
        with PredictorInputs(model) as cap:
            for _ in range(6): self.assertTrue(torch.equal(expected,model(*args)))
        self.assertEqual(len(cap.values),6);self.assertFalse(model._forward_pre_hooks)
        with self.assertRaises(ValueError):
            with PredictorInputs(model):model(*args)
        self.assertFalse(model._forward_pre_hooks)


if __name__=='__main__':unittest.main()
