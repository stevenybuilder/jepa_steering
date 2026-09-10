import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/geometry_map'))
from cem_search_audit_v1 import elite_statistics,NativeTrace,CostObserver,exact


class FakePlanner:
    num_elites=2;momentum_mean=0.;momentum_std=0.;var_scale=1.
    def __init__(self):self.local_generator=torch.Generator().manual_seed(5)
    def unroll(self,z,act_suffix):
        x=act_suffix.square().sum(-1)
        return {'visual':x[...,None],'proprio':x[...,None]}
    def objective(self,out,actions,keepdims=False):
        value=out['visual'][...,0].sum(0)
        return out['visual'][...,0] if keepdims else value
    def cost_function(self,actions,z):return self.objective(self.unroll(z,actions),actions)


class Tests(unittest.TestCase):
    def test_elite_mean_is_not_best(self):
        a=torch.tensor([[[0.],[1.],[3.]]]);cost=torch.tensor([5.,0.,1.])
        idx,mean,std,best=elite_statistics(a,cost,2)
        self.assertEqual(best,1);exact(idx,torch.tensor([1,2]),'selected elites');exact(mean,torch.tensor([[2.]]),'mean')
        self.assertNotEqual(float(mean[0,0]),float(a[0,best,0]))

    def test_cost_trace_identity_and_restoration(self):
        p=FakePlanner();cost_method=p.cost_function;unroll_method=p.unroll
        a=torch.randn(6,5,2);a[:,0]=0;before=a.clone();reference=p.cost_function(a,None)
        with NativeTrace(p) as trace:
            actual=p.cost_function(a,None);idx,mean,_,_=elite_statistics(a,actual,2)
            out=p.unroll(None,act_suffix=mean[:,None])
        exact(actual,reference,'cost');exact(a,before,'actions');self.assertEqual(p.cost_function,cost_method);self.assertEqual(p.unroll,unroll_method)
        exact(trace.rows[0]['mean_after'],mean,'mean');self.assertIn('mean_forecast',trace.stages[1])

    def test_exception_restores_owned_methods(self):
        p=FakePlanner();original=p.unroll
        with self.assertRaises(RuntimeError):
            with NativeTrace(p):raise RuntimeError('planned')
        self.assertEqual(p.unroll,original)

    def test_candidate_zero_previous_mean_guard(self):
        p=FakePlanner();a=torch.randn(6,5,2);a[:,0]=0
        with NativeTrace(p):
            costs=p.cost_function(a,None);_,mean,_,_=elite_statistics(a,costs,2);p.unroll(None,act_suffix=mean[:,None])
            with self.assertRaises(ValueError):p.cost_function(a+20,None)

    def test_shape_rejection(self):
        with self.assertRaises(ValueError):elite_statistics(torch.ones(6,300,10),torch.ones(300,1),10)

    def test_passive_observer_returns_same_tensor(self):
        value=torch.arange(300,dtype=torch.float32);o=CostObserver(lambda *a,**kw:value)
        self.assertIs(o(None,torch.zeros(6,300,10)),value);exact(o.costs[0],value,'observer')


if __name__=='__main__':unittest.main()
